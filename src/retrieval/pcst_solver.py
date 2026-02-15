"""
PCST subgraph extraction using Prize-Collecting Steiner Tree.

Key design: Localize first, then run PCST on a small neighborhood.
Running PCST on the full unified graph (1M+ nodes) is too slow and causes
aggressive pruning (only root node returned). Instead, we BFS from seed
nodes to extract a ~300-node local neighborhood, then run PCST on that.
"""

from typing import List, Dict, Set, Tuple
import networkx as nx
import numpy as np
from collections import deque


def _diagnose_pcst_fast():
    """One-time diagnostic: determine pcst_fast's return format.

    Runs a tiny 4-node problem where we KNOW the correct answer,
    then checks whether the return values are indices or labels.

    Returns:
        "indices" — pcst_fast returns filtered node/edge indices (correct API)
        "labels"  — pcst_fast returns filtered cluster labels (broken API)
    """
    import pcst_fast

    #  Triangle: 0—1—2—0, with node 3 disconnected via edge to 1
    #  Prizes: [10, 10, 10, 0]  → nodes 0-2 high value, node 3 worthless
    #  With root=0, pruning='none': should select nodes {0, 1, 2} at minimum
    probe_edges  = np.array([[0, 1], [1, 2], [2, 0], [1, 3]], dtype=np.int64)
    probe_prizes = np.array([10.0, 10.0, 10.0, 0.0], dtype=np.float64)
    probe_costs  = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float64)

    verts, edges = pcst_fast.pcst_fast(
        probe_edges, probe_prizes, probe_costs,
        0,        # root
        1,        # num_clusters
        "none",   # pruning (keep everything reachable)
        0         # verbosity
    )
    verts = np.asarray(verts)

    # If indices: verts ⊂ {0,1,2,3} with distinct values, e.g. [0,1,2]
    # If labels:  verts = [0,0,0] (cluster label 0 repeated)
    n_unique = len(np.unique(verts))

    if n_unique >= 2:
        return "indices"
    else:
        return "labels"


# Module-level cache
_pcst_format = None


def _pcst_solve(edges_array, prize_array, costs_array, root, num_clusters,
                pruning, verbosity, num_nodes, num_edges):
    """Solve PCST and return (selected_node_indices, selected_edge_indices).

    Handles both pcst_fast return formats transparently:
    - "indices" format: use returned arrays directly
    - "labels" format: count selected, prune lowest-prize leaves to match
    """
    global _pcst_format
    import pcst_fast

    # One-time format detection
    if _pcst_format is None:
        _pcst_format = _diagnose_pcst_fast()
        print(f"  [pcst] Detected return format: '{_pcst_format}'")

    raw_v, raw_e = pcst_fast.pcst_fast(
        edges_array, prize_array, costs_array,
        root, num_clusters, pruning, verbosity
    )
    raw_v = np.asarray(raw_v)
    raw_e = np.asarray(raw_e)

    if _pcst_format == "indices":
        # Happy path: raw_v contains actual node indices
        return raw_v.astype(np.int64), raw_e.astype(np.int64)

    # --- "labels" format: raw_v has the correct COUNT but not indices ---
    # pcst_fast selected len(raw_v) nodes out of num_nodes.
    # We reconstruct which nodes by solving:
    #   "keep exactly len(raw_v) nodes, rooted at `root`, maximising
    #    total prize minus total edge cost, connected."
    #
    # For a small graph (≤500 nodes) the fastest correct approach is:
    # 1. Build MST of the local graph
    # 2. Iteratively prune lowest-prize leaves until we reach target count
    # This is a standard PCST approximation on trees.

    target_n = len(raw_v)
    target_e = len(raw_e)

    if target_n >= num_nodes:
        # PCST kept everything
        return np.arange(num_nodes, dtype=np.int64), np.arange(num_edges, dtype=np.int64)

    # Build NetworkX graph from edge array for MST computation
    import networkx as nx
    G_local = nx.Graph()
    G_local.add_nodes_from(range(num_nodes))

    edges_list = edges_array.tolist() if hasattr(edges_array, 'tolist') else list(edges_array)
    for i, (u, v) in enumerate(edges_list):
        u, v = int(u), int(v)
        # Use edge cost as weight; keep edge index for later
        if not G_local.has_edge(u, v) or G_local[u][v].get("weight", float("inf")) > float(costs_array[i]):
            G_local.add_edge(u, v, weight=float(costs_array[i]), edge_idx=i)

    # MST from the local graph
    if not nx.is_connected(G_local):
        # Use the component containing root
        comp = nx.node_connected_component(G_local, int(root))
        G_local = G_local.subgraph(comp).copy()
        # Recount: can't keep more nodes than the component has
        target_n = min(target_n, len(G_local))

    mst = nx.minimum_spanning_tree(G_local)

    # Iterative leaf pruning: remove lowest-prize leaves until |nodes| == target
    while len(mst) > target_n:
        leaves = [n for n in mst.nodes() if mst.degree(n) <= 1 and n != int(root)]
        if not leaves:
            break
        worst = min(leaves, key=lambda n: float(prize_array[n]))
        mst.remove_node(worst)

    # Ensure connectivity after pruning
    if not nx.is_connected(mst):
        comp = nx.node_connected_component(mst, int(root))
        mst = mst.subgraph(comp).copy()

    selected_node_indices = np.array(sorted(mst.nodes()), dtype=np.int64)

    # Map MST edges back to original edge indices
    selected_edge_set = set()
    for u, v, data in mst.edges(data=True):
        if "edge_idx" in data:
            selected_edge_set.add(data["edge_idx"])
    selected_edge_indices = np.array(sorted(selected_edge_set), dtype=np.int64)

    return selected_node_indices, selected_edge_indices


def normalize_pcst_inputs(
    edges: np.ndarray,
    prizes: np.ndarray,
    costs: np.ndarray,
    root: int,
    num_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Ensure all pcst_fast inputs have correct dtypes (int64/float64)."""
    return (
        np.asarray(edges, dtype=np.int64),
        np.asarray(prizes, dtype=np.float64),
        np.asarray(costs, dtype=np.float64),
        int(root),
        int(num_clusters),
    )


class PCSTSolver:
    """Extract connected subgraphs using PCST algorithm."""

    def __init__(self, cost: float = 5.0, budget: int = 70, local_budget: int = 500,
                 pruning: str = "strong", base_prize_ratio: float = 1.5):
        """
        Initialize PCST solver.

        Args:
            cost: Edge cost parameter for PCST
            budget: Maximum nodes in extracted subgraph
            local_budget: Max nodes for BFS localization before PCST
            pruning: PCST pruning strategy ('none', 'gw', or 'strong')
            base_prize_ratio: base_prize = cost * ratio (intermediates survive pruning)
        """
        self.cost = cost
        self.budget = budget
        self.local_budget = local_budget
        self.pruning = pruning
        self.base_prize_ratio = base_prize_ratio
        # Cache for undirected graph conversion of the full graph
        self._cached_graph_id = None
        self._cached_undirected: nx.Graph = None

    def _get_undirected(self, G: nx.DiGraph) -> nx.Graph:
        """Return cached undirected conversion of G, recomputing only if G changed."""
        if self._cached_graph_id != id(G):
            self._cached_undirected = G.to_undirected()
            self._cached_graph_id = id(G)
        return self._cached_undirected

    def extract_subgraph(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        prizes: Dict[str, float]
    ) -> nx.DiGraph:
        """
        Extract connected subgraph using PCST algorithm.

        Pipeline:
        1. Localize: BFS from seeds to extract ~local_budget node neighborhood
        2. PCST: Run on the small local graph (fast and effective)
        3. Validate: Ensure minimum size and connectivity
        4. Fallback: BFS if PCST produces poor results

        Args:
            G: Full NetworkX graph (can be 1M+ nodes)
            seed_nodes: List of seed node names (from k-NN search)
            prizes: Dictionary mapping node_name -> prize value

        Returns:
            Connected NetworkX subgraph (≤ budget nodes)
        """
        if not seed_nodes:
            return nx.DiGraph()

        # Filter seed nodes to those actually in graph
        valid_seeds = [n for n in seed_nodes if n in G]
        if not valid_seeds:
            print("⚠ No valid seed nodes in graph, returning empty subgraph")
            return nx.DiGraph()

        # Step 1: Localize — extract small neighborhood around seeds
        local_graph = self._localize(G, valid_seeds)
        print(f"  Localized: {len(local_graph)} nodes, {local_graph.number_of_edges()} edges")

        # Step 2: Try PCST on the local graph
        try:
            subgraph = self._pcst_extract(local_graph, valid_seeds, prizes)
        except Exception as e:
            print(f"  PCST failed ({e}), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds, self.budget)

        # Step 3: Validate — PCST should return at least a few nodes
        if len(subgraph) < min(len(valid_seeds), 3):
            print(f"  PCST result too small ({len(subgraph)} nodes), retrying with pruning='none'")
            try:
                subgraph = self._pcst_extract(local_graph, valid_seeds, prizes, pruning_override="none")
            except Exception as e:
                print(f"  PCST retry failed ({e})")
                subgraph = nx.DiGraph()

        # Step 3b: If still too small after retry, fall back to BFS
        if len(subgraph) < min(len(valid_seeds), 3):
            print(f"  PCST still too small ({len(subgraph)} nodes), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds, self.budget)

        # Step 4: Enforce budget via iterative leaf pruning
        if len(subgraph) > self.budget:
            pre_trim = len(subgraph)
            subgraph = self._trim_to_budget(subgraph, prizes)
            print(f"  Trimmed: {pre_trim} -> {len(subgraph)} nodes")

        # Step 5: Ensure weak connectivity
        if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
            pre_comp = len(subgraph)
            subgraph = self._largest_component(subgraph)
            print(f"  Largest component: {pre_comp} -> {len(subgraph)} nodes")

        print(f"  Final: {len(subgraph)} nodes, {subgraph.number_of_edges()} edges, "
              f"connected={nx.is_weakly_connected(subgraph) if len(subgraph) > 0 else 'N/A'}")

        return subgraph

    def _localize(self, G: nx.DiGraph, seed_nodes: List[str]) -> nx.DiGraph:
        """
        Multi-source BFS to extract a local neighborhood around seeds.

        This reduces the graph from 1M+ nodes to ~local_budget nodes,
        making PCST fast and effective.
        """
        G_undirected = self._get_undirected(G)

        visited = set()
        queue = deque()

        for seed in seed_nodes:
            if seed in G_undirected:
                visited.add(seed)
                queue.append(seed)

        while queue and len(visited) < self.local_budget:
            node = queue.popleft()
            for neighbor in G_undirected.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= self.local_budget:
                        break

        return G.subgraph(list(visited)).copy()

    def _pcst_extract(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        prizes: Dict[str, float],
        pruning_override: str = None
    ) -> nx.DiGraph:
        """
        Extract subgraph using pcst_fast library.

        Builds fresh node/edge arrays from the input graph (expected to be
        a small localized graph, ~300 nodes). No caching needed.
        """
        # Build undirected version of (small) local graph
        G_und = G.to_undirected()
        nodes = list(G_und.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = {i: n for n, i in node_to_idx.items()}

        # Build edge arrays
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_und.edges()]
        if not edges:
            return G.subgraph([s for s in seed_nodes if s in G]).copy()

        edges_array = np.array(edges, dtype=np.int64)
        costs_array = np.full(len(edges), self.cost, dtype=np.float64)

        # Build prize array — base prize for ALL nodes so intermediates survive pruning
        num_nodes = len(nodes)
        base_prize = self.cost * self.base_prize_ratio  # e.g. 1.0 * 1.5 = 1.5
        seed_floor = max(5.0, self.cost * 5.0)  # Seeds clearly outweigh edge cost
        prize_array = np.full(num_nodes, base_prize, dtype=np.float64)
        for node, prize in prizes.items():
            if node in node_to_idx:
                prize_array[node_to_idx[node]] = base_prize + max(prize, 0.0)

        # Ensure seed nodes have meaningful prizes
        for seed in seed_nodes:
            if seed in node_to_idx:
                s_idx = node_to_idx[seed]
                if prize_array[s_idx] < seed_floor:
                    prize_array[s_idx] = seed_floor

        # Root the tree at the highest-prize seed node
        valid_seeds_in_local = [s for s in seed_nodes if s in node_to_idx]
        if not valid_seeds_in_local:
            return nx.DiGraph()

        best_seed = max(valid_seeds_in_local, key=lambda s: prizes.get(s, 0.0))
        root = node_to_idx[best_seed]

        # Diagnostics
        effective_pruning = pruning_override if pruning_override else self.pruning
        nonzero_prizes = np.count_nonzero(prize_array > base_prize)
        print(f"  PCST input: {num_nodes} nodes, {len(edges)} edges, "
              f"cost={self.cost:.2f}, base_prize={base_prize:.2f}, "
              f"seed_floor={seed_floor:.1f}, pruning='{effective_pruning}', "
              f"{nonzero_prizes} high-prize nodes, root={best_seed[:30]}...")

        # Normalize dtypes for pcst_fast (requires int64/float64)
        edges_array, prize_array, costs_array, root, num_clusters = normalize_pcst_inputs(
            edges_array, prize_array, costs_array, root, 1
        )

        # Run PCST — handles both "indices" and "labels" return formats
        selected_node_indices, selected_edge_indices = _pcst_solve(
            edges_array, prize_array, costs_array,
            root, num_clusters, effective_pruning, 0,
            num_nodes, len(edges)
        )

        print(f"  PCST output: {len(selected_node_indices)} nodes, {len(selected_edge_indices)} edges")

        # Sanity checks
        assert len(selected_node_indices) == 0 or selected_node_indices.max() < num_nodes, \
            f"Node index {selected_node_indices.max()} out of range [0, {num_nodes})"
        n_unique = len(np.unique(selected_node_indices))
        assert n_unique == len(selected_node_indices), \
            f"Duplicate node indices: {len(selected_node_indices)} total but only {n_unique} unique"

        unique_indices = sorted(int(i) for i in selected_node_indices)
        selected_node_names = [idx_to_node[i] for i in unique_indices if i in idx_to_node]

        print(f"  Mapped: {len(selected_node_indices)} node indices "
              f"-> {len(selected_node_names)} named, G has {len(G)} nodes")

        if not selected_node_names:
            selected_node_names = valid_seeds_in_local

        # Extract subgraph (preserve original directed edges)
        subgraph = G.subgraph(selected_node_names).copy()

        if len(subgraph) != len(selected_node_names):
            # Debug: find which names are missing from G
            g_nodes = set(G.nodes())
            missing = [n for n in selected_node_names if n not in g_nodes]
            print(f"  WARNING: subgraph={len(subgraph)} != selected={len(selected_node_names)}, "
                  f"{len(missing)} names missing from G")
            if missing[:3]:
                sample_g = list(g_nodes)[:2]
                print(f"    Missing sample: {missing[:3]}")
                print(f"    Missing type: {type(missing[0])}, G node type: {type(sample_g[0]) if sample_g else 'N/A'}")

        return subgraph

    def _bfs_fallback(self, G: nx.DiGraph, seed_nodes: List[str], budget: int) -> nx.DiGraph:
        """
        Fallback: BFS expansion from the highest-degree seed node.

        Uses single-source BFS from the best seed to guarantee connectivity,
        then adds remaining seeds and their immediate neighbors.
        """
        G_undirected = self._get_undirected(G)

        # Start BFS from the seed with highest degree (most connections)
        best_seed = max(
            [s for s in seed_nodes if s in G_undirected],
            key=lambda s: G_undirected.degree(s),
            default=None
        )
        if best_seed is None:
            return nx.DiGraph()

        # Single-source BFS from best seed (guarantees connectivity)
        visited = {best_seed}
        queue = deque([best_seed])

        while queue and len(visited) < budget:
            node = queue.popleft()
            for neighbor in G_undirected.neighbors(node):
                if neighbor not in visited and len(visited) < budget:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Add remaining seed nodes and their 1-hop neighbors if space
        for seed in seed_nodes:
            if seed in G_undirected and seed not in visited and len(visited) < budget:
                visited.add(seed)
                for neighbor in G_undirected.neighbors(seed):
                    if neighbor not in visited and len(visited) < budget:
                        visited.add(neighbor)

        # Extract subgraph (preserve original directed edges)
        subgraph = G.subgraph(list(visited)).copy()

        return subgraph

    def _trim_to_budget(self, subgraph: nx.DiGraph, prizes: Dict[str, float]) -> nx.DiGraph:
        """Trim subgraph to budget by iteratively removing lowest-prize leaf nodes.

        Unlike naive top-K selection which scatters nodes and destroys connectivity,
        iterative leaf pruning preserves the tree structure by only removing nodes
        from the periphery.
        """
        if len(subgraph) <= self.budget:
            return subgraph

        G = subgraph.copy()
        G_und = G.to_undirected()

        while len(G) > self.budget:
            # Find leaf nodes (degree 1 in undirected view)
            leaves = [n for n in G_und.nodes() if G_und.degree(n) <= 1]
            if not leaves:
                # No leaves — graph is fully connected with no periphery; fall back
                # to removing lowest-prize node
                worst = min(G.nodes(), key=lambda n: prizes.get(n, 0.0))
                G.remove_node(worst)
                G_und.remove_node(worst)
                continue

            # Remove the leaf with the lowest prize
            worst_leaf = min(leaves, key=lambda n: prizes.get(n, 0.0))
            G.remove_node(worst_leaf)
            G_und.remove_node(worst_leaf)

        return G

    def _largest_component(self, subgraph: nx.DiGraph) -> nx.DiGraph:
        """Return the largest weakly connected component."""
        components = list(nx.weakly_connected_components(subgraph))
        if not components:
            return subgraph
        largest = max(components, key=len)
        return subgraph.subgraph(list(largest)).copy()

    def validate_subgraph(self, subgraph: nx.DiGraph) -> bool:
        """Check if subgraph is connected and within budget."""
        if len(subgraph) == 0:
            return True

        if not nx.is_weakly_connected(subgraph):
            return False

        if len(subgraph) > self.budget:
            return False

        return True
