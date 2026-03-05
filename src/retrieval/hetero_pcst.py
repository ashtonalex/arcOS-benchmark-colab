"""PCST subgraph extraction adapted for PyG HeteroData."""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig

try:
    import pcst_fast
    HAS_PCST = True
except ImportError:
    HAS_PCST = False


class HeteroPCST:
    """Extracts subgraphs from HeteroData using Prize-Collecting Steiner Tree."""

    def __init__(self, config: BenchmarkConfig, verbose: bool = False):
        self.budget = config.pcst_budget
        self.cost = config.pcst_cost
        self.pruning = config.pcst_pruning
        self.temporal_cost_weight = config.pcst_temporal_cost_weight
        self.existence_prize = config.pcst_existence_prize
        self.spread_factor = config.pcst_prize_spread_factor
        self.verbose = verbose

    def extract(self, data: HeteroData, prizes: Dict[int, float], root: Optional[int] = None) -> HeteroData:
        """Extract a subgraph from HeteroData using PCST.

        Args:
            data: Full video scene graph as HeteroData.
            prizes: Dict mapping node indices to prize values.
            root: Optional root node for PCST.
        Returns:
            HeteroData subgraph with `selected_nodes` attribute containing original node indices.
        """
        num_nodes = data["object"].num_nodes
        prize_array = np.zeros(num_nodes, dtype=np.float64)
        for node_idx, prize in prizes.items():
            if 0 <= node_idx < num_nodes:
                prize_array[node_idx] = prize

        edges, costs, edge_type_map = self._flatten_edges(data, num_nodes)

        # Step 1: Existence prizes — give relay nodes a small base prize
        if self.existence_prize > 0:
            for i in range(num_nodes):
                if prize_array[i] == 0.0:
                    prize_array[i] = self.existence_prize

        # Step 2: Prize spreading — 1-hop neighbors of seeds get stepping-stone prizes
        if self.spread_factor > 0 and len(edges) > 0:
            prize_array = self._spread_prizes(prize_array, edges)

        # Step 3: Auto-select root as highest-prize node
        if root is None and num_nodes > 0:
            root = int(np.argmax(prize_array))

        if self.verbose:
            n_prized = int(np.count_nonzero(prize_array))
            prized_vals = prize_array[prize_array > 0]
            edge_counts = {et[1]: data[et].edge_index.shape[1] for et in data.edge_types}
            if n_prized:
                print(f"  [PCST] num_nodes={num_nodes}, edges={len(edges)} "
                      f"({edge_counts}), prizes={n_prized} "
                      f"(min={prized_vals.min():.3f}, max={prized_vals.max():.3f}, "
                      f"mean={prized_vals.mean():.3f})")
            else:
                print(f"  [PCST] num_nodes={num_nodes}, edges={len(edges)} "
                      f"({edge_counts}), prizes=0")
            print(f"[PCST-DEBUG] Nodes: {num_nodes} | Edges: {len(edges)}")
            total_prizes = float(prize_array.sum())
            total_costs = float(costs.sum()) if len(costs) > 0 else 0.0
            print(f"[PCST-DEBUG] Total Prizes: {total_prizes:.4f} | Total Edge Costs: {total_costs:.4f}")
            anchor = root if root is not None else int(np.argmax(prize_array))
            p_anchor = float(prize_array[anchor])
            if len(edges) > 0:
                mask = (edges[:, 0] == anchor) | (edges[:, 1] == anchor)
                for edge, c_ij in zip(edges[mask][:10], costs[mask][:10]):
                    j = int(edge[1]) if int(edge[0]) == anchor else int(edge[0])
                    p_j = float(prize_array[j])
                    label = "FAIL (cost>prize)" if c_ij > p_j else "OK"
                    print(f"  Node {anchor} Prize={p_anchor:.4f} | Edge→{j}: cost={c_ij:.4f} | Neighbor Prize={p_j:.4f} | {label}")

        if len(edges) == 0:
            return self._select_top_prized(data, prizes)

        try:
            selected_nodes = self._run_pcst(num_nodes, edges, costs, prize_array, root)
        except Exception:
            selected_nodes = self._bfs_fallback(edges, prizes, num_nodes)

        # Min-size fallback: if PCST result is too small, retry then BFS
        min_size = min(len([p for p in prizes.values() if p > 0]), 3)
        if len(selected_nodes) < min_size:
            if self.verbose:
                print(f"  [PCST] Too few nodes ({len(selected_nodes)} < {min_size}), retrying with pruning='none'")
            try:
                selected_nodes = self._run_pcst(
                    num_nodes, edges, costs, prize_array, root, pruning_override="none"
                )
            except Exception:
                pass
        if len(selected_nodes) < min_size:
            if self.verbose:
                print(f"  [PCST] Still too few ({len(selected_nodes)}), falling back to BFS")
            selected_nodes = self._bfs_fallback(edges, prizes, num_nodes)

        if self.verbose:
            print(f"  [PCST] PCST output: {len(selected_nodes)} nodes")
            if len(selected_nodes) == 1:
                survivor = selected_nodes[0]
                p_surv = float(prize_array[survivor])
                print(f"[PCST-DEBUG] 1-node result! Survivor={survivor} Prize={p_surv:.4f}")
                if len(edges) > 0:
                    mask = (edges[:, 0] == survivor) | (edges[:, 1] == survivor)
                    for edge, c_ij in zip(edges[mask][:10], costs[mask][:10]):
                        j = int(edge[1]) if int(edge[0]) == survivor else int(edge[0])
                        p_j = float(prize_array[j])
                        label = "FAIL" if c_ij > p_j else "OK"
                        print(f"  Survivor {survivor} (p={p_surv:.4f}) → Neighbor {j} (p={p_j:.4f}) cost={c_ij:.4f} | Break-even: {label}")

        if len(selected_nodes) > self.budget:
            scored = [(n, prize_array[n]) for n in selected_nodes]
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [n for n, _ in scored[:self.budget]]

        selected_nodes = sorted(set(selected_nodes))

        if self.verbose:
            print(f"  [PCST] final selected: {len(selected_nodes)} nodes")

        return self._slice_heterodata(data, selected_nodes)

    def _spread_prizes(self, prize_array, edges):
        """Give 1-hop neighbors of prized nodes a fraction of the prize."""
        boosted = prize_array.copy()
        threshold = self.existence_prize if self.existence_prize > 0 else 0.0
        prized_nodes = set(int(i) for i in np.where(prize_array > threshold)[0])
        for src, dst in edges:
            src_i, dst_i = int(src), int(dst)
            if src_i in prized_nodes:
                spread_val = prize_array[src_i] * self.spread_factor
                if boosted[dst_i] < spread_val:
                    boosted[dst_i] = spread_val
            if dst_i in prized_nodes:
                spread_val = prize_array[dst_i] * self.spread_factor
                if boosted[src_i] < spread_val:
                    boosted[src_i] = spread_val
        return boosted

    def _flatten_edges(self, data, num_nodes):
        edges = []
        costs = []
        edge_type_map = []
        for etype in data.edge_types:
            ei = data[etype].edge_index
            num_edges = ei.shape[1]
            type_name = etype[1]
            cost = self.cost * self.temporal_cost_weight if type_name == "temporal" else self.cost
            for i in range(num_edges):
                src, dst = int(ei[0, i]), int(ei[1, i])
                edges.append([src, dst])
                costs.append(cost)
                edge_type_map.append(etype)
        return (
            np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32),
            np.array(costs, dtype=np.float64) if costs else np.zeros(0, dtype=np.float64),
            edge_type_map,
        )

    def _run_pcst(self, num_nodes, edges, costs, prizes, root, pruning_override=None):
        if not HAS_PCST:
            raise ImportError("pcst_fast not available")
        root_idx = root if root is not None else -1
        pruning = pruning_override if pruning_override is not None else self.pruning
        vertices, selected_edges = pcst_fast.pcst_fast(
            edges.astype(np.int32), prizes, costs, root_idx, 1, pruning, 0,
        )
        if len(vertices) == 0:
            raise ValueError("PCST returned empty result")
        return vertices.tolist()

    def _bfs_fallback(self, edges, prizes, num_nodes):
        if not prizes:
            return list(range(min(self.budget, num_nodes)))
        adj = {i: [] for i in range(num_nodes)}
        for src, dst in edges:
            adj[int(src)].append(int(dst))
            adj[int(dst)].append(int(src))
        start = max(prizes, key=prizes.get)
        visited = set()
        queue = [start]
        visited.add(start)
        while queue and len(visited) < self.budget:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited and len(visited) < self.budget:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return list(visited)

    def _select_top_prized(self, data, prizes):
        if not prizes:
            selected = list(range(min(self.budget, data["object"].num_nodes)))
        else:
            sorted_nodes = sorted(prizes, key=prizes.get, reverse=True)
            selected = sorted_nodes[:self.budget]
        return self._slice_heterodata(data, selected)

    def _slice_heterodata(self, data, selected_nodes):
        selected = sorted(selected_nodes)
        selected_set = set(selected)
        old_to_new = {old: new for new, old in enumerate(selected)}
        sub = HeteroData()
        idx = torch.tensor(selected, dtype=torch.long)
        sub["object"].x = data["object"].x[idx]
        if hasattr(data["object"], "frame_id"):
            sub["object"].frame_id = data["object"].frame_id[idx]
        if hasattr(data["object"], "object_class"):
            sub["object"].object_class = data["object"].object_class[idx]
        for etype in data.edge_types:
            ei = data[etype].edge_index
            mask = torch.tensor(
                [int(ei[0, i]) in selected_set and int(ei[1, i]) in selected_set for i in range(ei.shape[1])],
                dtype=torch.bool,
            )
            new_src = [old_to_new[int(ei[0, i])] for i in range(ei.shape[1]) if mask[i]]
            new_dst = [old_to_new[int(ei[1, i])] for i in range(ei.shape[1]) if mask[i]]
            if new_src:
                sub[etype].edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
            else:
                sub[etype].edge_index = torch.zeros((2, 0), dtype=torch.long)
            if hasattr(data[etype], "edge_attr") and mask.any():
                sub[etype].edge_attr = data[etype].edge_attr[mask]
        sub.selected_nodes = torch.tensor(selected, dtype=torch.long)
        if hasattr(data, "video_id"):
            sub.video_id = data.video_id

        # Preserve human-readable names needed by labeling and verbalization
        if hasattr(data, "object_names") and data.object_names is not None:
            sub.object_names = [data.object_names[i] for i in selected]

        # Copy spatial_predicates (ordered by spatial edges in the original graph)
        spatial_etype = ("object", "spatial_rel", "object")
        if (
            hasattr(data, "spatial_predicates")
            and data.spatial_predicates
            and spatial_etype in data.edge_types
        ):
            ei = data[spatial_etype].edge_index
            kept = [
                i for i in range(ei.shape[1])
                if int(ei[0, i]) in selected_set and int(ei[1, i]) in selected_set
            ]
            sub.spatial_predicates = [data.spatial_predicates[i] for i in kept]

        return sub
