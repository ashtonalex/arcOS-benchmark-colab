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

    def __init__(self, config: BenchmarkConfig):
        self.budget = config.pcst_budget
        self.cost = config.pcst_cost
        self.pruning = config.pcst_pruning
        self.temporal_cost_weight = config.pcst_temporal_cost_weight

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

        if len(edges) == 0:
            return self._select_top_prized(data, prizes)

        try:
            selected_nodes = self._run_pcst(num_nodes, edges, costs, prize_array, root)
        except Exception:
            selected_nodes = self._bfs_fallback(edges, prizes, num_nodes)

        if len(selected_nodes) > self.budget:
            scored = [(n, prize_array[n]) for n in selected_nodes]
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [n for n, _ in scored[:self.budget]]

        selected_nodes = sorted(set(selected_nodes))
        return self._slice_heterodata(data, selected_nodes)

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
            np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64),
            np.array(costs, dtype=np.float64) if costs else np.zeros(0, dtype=np.float64),
            edge_type_map,
        )

    def _run_pcst(self, num_nodes, edges, costs, prizes, root):
        if not HAS_PCST:
            raise ImportError("pcst_fast not available")
        root_idx = root if root is not None else -1
        vertices, selected_edges = pcst_fast.pcst_fast(
            edges.astype(np.int64), prizes, costs, root_idx, 1, self.pruning, 0,
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
        return sub
