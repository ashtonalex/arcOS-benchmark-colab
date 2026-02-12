"""
PCST subgraph extraction using Prize-Collecting Steiner Tree.

Key design: Localize first, then run PCST on a small neighborhood.
Running PCST on the full unified graph (1M+ nodes) is too slow and causes
aggressive pruning (only root node returned). Instead, we BFS from seed
nodes to extract a ~300-node local neighborhood, then run PCST on that.
"""

from typing import List, Dict, Set
import networkx as nx
import numpy as np
from collections import deque


class PCSTSolver:
    """Extract connected subgraphs using PCST algorithm."""

    def __init__(self, cost: float = 1.0, budget: int = 50, local_budget: int = 300):
        """
        Initialize PCST solver.

        Args:
            cost: Edge cost parameter for PCST
            budget: Maximum nodes in extracted subgraph
            local_budget: Max nodes for BFS localization before PCST
        """
        self.cost = cost
        self.budget = budget
        self.local_budget = local_budget
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

        # Step 2: Try PCST on the local graph
        try:
            subgraph = self._pcst_extract(local_graph, valid_seeds, prizes)
        except Exception as e:
            print(f"⚠ PCST failed ({e}), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds, self.budget)

        # Step 3: Validate — PCST should return at least a few nodes
        if len(subgraph) < min(len(valid_seeds), 3):
            print(f"⚠ PCST result too small ({len(subgraph)} nodes), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds, self.budget)

        # Step 4: Enforce budget
        if len(subgraph) > self.budget:
            subgraph = self._trim_to_budget(subgraph, prizes)

        # Step 5: Ensure weak connectivity
        if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
            subgraph = self._largest_component(subgraph)

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
        prizes: Dict[str, float]
    ) -> nx.DiGraph:
        """
        Extract subgraph using pcst_fast library.

        Builds fresh node/edge arrays from the input graph (expected to be
        a small localized graph, ~300 nodes). No caching needed.
        """
        try:
            import pcst_fast
        except ImportError:
            raise ImportError("pcst_fast not available, using BFS fallback")

        # Build undirected version of (small) local graph
        G_und = G.to_undirected()
        nodes = list(G_und.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = {i: n for n, i in node_to_idx.items()}

        # Build edge arrays
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_und.edges()]
        if not edges:
            return G.subgraph([s for s in seed_nodes if s in G]).copy()

        edges_array = np.array(edges, dtype=np.int32)
        costs_array = np.full(len(edges), self.cost, dtype=np.float64)

        # Build prize array
        num_nodes = len(nodes)
        prize_array = np.zeros(num_nodes, dtype=np.float64)
        for node, prize in prizes.items():
            if node in node_to_idx:
                prize_array[node_to_idx[node]] = prize

        # Ensure seed nodes have non-zero prizes
        for seed in seed_nodes:
            if seed in node_to_idx:
                s_idx = node_to_idx[seed]
                if prize_array[s_idx] == 0:
                    prize_array[s_idx] = 1.0

        # Root the tree at the highest-prize seed node
        valid_seeds_in_local = [s for s in seed_nodes if s in node_to_idx]
        if not valid_seeds_in_local:
            return nx.DiGraph()

        best_seed = max(valid_seeds_in_local, key=lambda s: prizes.get(s, 0.0))
        root = node_to_idx[best_seed]

        # Run PCST with no pruning (GW pruning is too aggressive on sparse graphs)
        selected_nodes, selected_edges = pcst_fast.pcst_fast(
            edges_array,
            prize_array,
            costs_array,
            root,
            1,       # num_clusters (ignored with root)
            'none',  # pruning
            0        # verbosity
        )

        # Convert back to node names
        selected_node_names = [idx_to_node[i] for i in selected_nodes if i in idx_to_node]

        if not selected_node_names:
            selected_node_names = valid_seeds_in_local

        # Extract subgraph (preserve original directed edges)
        subgraph = G.subgraph(selected_node_names).copy()

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
        """Trim subgraph to budget by keeping highest-prize nodes."""
        if len(subgraph) <= self.budget:
            return subgraph

        nodes_with_prizes = [
            (node, prizes.get(node, 0.0))
            for node in subgraph.nodes()
        ]
        nodes_with_prizes.sort(key=lambda x: x[1], reverse=True)

        keep_nodes = [node for node, _ in nodes_with_prizes[:self.budget]]
        trimmed = subgraph.subgraph(keep_nodes).copy()

        return trimmed

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
