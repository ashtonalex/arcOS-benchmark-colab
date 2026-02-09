"""
PCST subgraph extraction using Prize-Collecting Steiner Tree.
"""

from typing import List, Dict, Set
import networkx as nx
import numpy as np
from collections import deque


class PCSTSolver:
    """Extract connected subgraphs using PCST algorithm."""

    def __init__(self, cost: float = 1.0, budget: int = 50):
        """
        Initialize PCST solver.

        Args:
            cost: Edge cost parameter for PCST
            budget: Maximum nodes in extracted subgraph
        """
        self.cost = cost
        self.budget = budget

    def extract_subgraph(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        prizes: Dict[str, float]
    ) -> nx.DiGraph:
        """
        Extract connected subgraph using PCST algorithm.

        Args:
            G: Full NetworkX graph
            seed_nodes: List of seed node names (from k-NN search)
            prizes: Dictionary mapping node_name -> prize value

        Returns:
            Connected NetworkX subgraph (≤ budget nodes)
        """
        # Validate inputs
        if not seed_nodes:
            return nx.DiGraph()

        # Filter seed nodes to those actually in graph
        valid_seeds = [n for n in seed_nodes if n in G]
        if not valid_seeds:
            print("⚠ No valid seed nodes in graph, returning empty subgraph")
            return nx.DiGraph()

        # Try PCST extraction
        try:
            subgraph = self._pcst_extract(G, valid_seeds, prizes)
        except Exception as e:
            print(f"⚠ PCST failed ({e}), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds, self.budget)

        # Validate result
        if not self.validate_subgraph(subgraph):
            print(f"⚠ PCST produced invalid subgraph, falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds, self.budget)

        # Enforce budget if needed
        if len(subgraph) > self.budget:
            subgraph = self._trim_to_budget(subgraph, prizes)

        return subgraph

    def _pcst_extract(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        prizes: Dict[str, float]
    ) -> nx.DiGraph:
        """
        Extract subgraph using pcst_fast library.

        Args:
            G: Full graph
            seed_nodes: Seed nodes
            prizes: Node prizes

        Returns:
            Extracted subgraph
        """
        try:
            import pcst_fast
        except ImportError:
            raise ImportError("pcst_fast not available, using BFS fallback")

        # Convert to undirected for PCST (required by algorithm)
        G_undirected = G.to_undirected()

        # Build node-to-index mapping
        all_nodes = list(G_undirected.nodes())
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}

        # Build edge list: [(node_i, node_j, cost)]
        edges = []
        costs = []
        for u, v in G_undirected.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edges.append((u_idx, v_idx))
            costs.append(self.cost)

        if not edges:
            # No edges, return seed nodes only
            return G.subgraph(seed_nodes).copy()

        # Build prize array
        prize_array = np.zeros(len(all_nodes))
        for node, prize in prizes.items():
            if node in node_to_idx:
                prize_array[node_to_idx[node]] = prize

        # Ensure seed nodes have non-zero prizes
        for seed in seed_nodes:
            if seed in node_to_idx and prize_array[node_to_idx[seed]] == 0:
                prize_array[node_to_idx[seed]] = 1.0

        # Convert to numpy arrays
        edges_array = np.array(edges, dtype=np.int32)
        costs_array = np.array(costs, dtype=np.float64)
        prize_array = prize_array.astype(np.float64)

        # Create virtual root (connect to all nodes with high cost to discourage use)
        root = -1  # Virtual root index
        num_clusters = 1  # Single connected component
        pruning = 'gw'  # Goemans-Williamson pruning

        # Run PCST
        selected_nodes, selected_edges = pcst_fast.pcst_fast(
            edges_array,
            prize_array,
            costs_array,
            root,
            num_clusters,
            pruning,
            0  # verbosity level
        )

        # Convert back to node names
        selected_node_names = [idx_to_node[i] for i in selected_nodes if i in idx_to_node]

        if not selected_node_names:
            # PCST failed, fall back to seeds
            selected_node_names = seed_nodes

        # Extract subgraph (preserve original directed edges)
        subgraph = G.subgraph(selected_node_names).copy()

        return subgraph

    def _bfs_fallback(self, G: nx.DiGraph, seed_nodes: List[str], budget: int) -> nx.DiGraph:
        """
        Fallback: BFS expansion from seed nodes up to budget.

        Args:
            G: Full graph
            seed_nodes: Starting nodes
            budget: Max nodes

        Returns:
            Subgraph with ≤ budget nodes
        """
        # Multi-source BFS
        visited = set()
        queue = deque(seed_nodes)
        visited.update(seed_nodes)

        # Convert to undirected for BFS traversal
        G_undirected = G.to_undirected()

        while queue and len(visited) < budget:
            node = queue.popleft()

            if node not in G_undirected:
                continue

            # Expand neighbors
            for neighbor in G_undirected.neighbors(node):
                if neighbor not in visited and len(visited) < budget:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Extract subgraph (preserve original directed edges)
        subgraph = G.subgraph(list(visited)).copy()

        return subgraph

    def _trim_to_budget(self, subgraph: nx.DiGraph, prizes: Dict[str, float]) -> nx.DiGraph:
        """
        Trim subgraph to budget by keeping highest-prize nodes.

        Args:
            subgraph: Subgraph to trim
            prizes: Node prizes

        Returns:
            Trimmed subgraph with ≤ budget nodes
        """
        if len(subgraph) <= self.budget:
            return subgraph

        # Sort nodes by prize (descending)
        nodes_with_prizes = [
            (node, prizes.get(node, 0.0))
            for node in subgraph.nodes()
        ]
        nodes_with_prizes.sort(key=lambda x: x[1], reverse=True)

        # Keep top budget nodes
        keep_nodes = [node for node, _ in nodes_with_prizes[:self.budget]]

        # Extract subgraph
        trimmed = subgraph.subgraph(keep_nodes).copy()

        return trimmed

    def validate_subgraph(self, subgraph: nx.DiGraph) -> bool:
        """
        Check if subgraph is connected and within budget.

        Args:
            subgraph: Subgraph to validate

        Returns:
            True if valid, False otherwise
        """
        if len(subgraph) == 0:
            return True  # Empty is technically valid

        # Check weak connectivity (for directed graphs)
        if not nx.is_weakly_connected(subgraph):
            return False

        # Check size constraint
        if len(subgraph) > self.budget:
            return False

        return True
