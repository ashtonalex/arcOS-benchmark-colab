"""
PCST subgraph extraction using Prize-Collecting Steiner Tree.

Key design: Localize first, then run PCST on a small neighborhood.
Running PCST on the full unified graph (1M+ nodes) is too slow.
BFS from seed nodes extracts a ~300-node local neighborhood,
then PCST selects the optimal prize-maximizing subtree from that.

Prize structure follows G-Retriever: prizes are raw cosine similarity
scores (0 to 1) with zero base. Only nodes scored by the retriever
get nonzero prizes. Intermediate relay nodes survive only if they
connect high-prize nodes cheaply enough to justify the edge cost.
"""

from typing import List, Dict
import networkx as nx
import numpy as np
from collections import deque


class PCSTSolver:
    """Extract connected subgraphs using Prize-Collecting Steiner Tree."""

    def __init__(self, cost: float = 0.3, budget: int = 70,
                 local_budget: int = 300, pruning: str = "gw"):
        """
        Initialize PCST solver.

        Args:
            cost: Uniform edge cost for PCST. Controls selectivity:
                  higher = more aggressive pruning (fewer nodes kept).
                  Should be tuned relative to prize scale (cosine sim 0-1).
            budget: Maximum nodes in extracted subgraph.
            local_budget: Max nodes for BFS localization before PCST.
            pruning: PCST pruning strategy ('none', 'gw', or 'strong').
        """
        self.cost = cost
        self.budget = budget
        self.local_budget = local_budget
        self.pruning = pruning
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
        prizes: Dict[str, float],
        root_entities: List[str] = None
    ) -> nx.DiGraph:
        """
        Extract connected subgraph using PCST.

        Pipeline:
        1. Localize: BFS from seeds to ~local_budget node neighborhood
        2. PCST: Select optimal subtree on the local graph
        3. Budget: Trim if over budget via leaf pruning
        4. Connectivity: Ensure weak connectivity

        Args:
            G: Full NetworkX graph (can be 1M+ nodes)
            seed_nodes: Seed node names from k-NN search
            prizes: node_name -> cosine similarity score (0 to 1)

        Returns:
            Connected NetworkX subgraph (<= budget nodes)
        """
        if not seed_nodes:
            return nx.DiGraph()

        valid_seeds = [n for n in seed_nodes if n in G]
        if not valid_seeds:
            print("Warning: no valid seed nodes in graph")
            return nx.DiGraph()

        # Step 1: Localize — reduce graph to small neighborhood
        local_graph = self._localize(G, valid_seeds)
        print(f"  Localized: {len(local_graph)} nodes, "
              f"{local_graph.number_of_edges()} edges")

        # Step 2: PCST on local graph
        try:
            subgraph = self._pcst_extract(local_graph, valid_seeds, prizes,
                                          root_entities=root_entities)
        except Exception as e:
            print(f"  PCST failed ({e}), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds)

        # Step 3: Validate minimum size — retry with relaxed pruning
        if len(subgraph) < min(len(valid_seeds), 3):
            print(f"  PCST too small ({len(subgraph)} nodes), "
                  f"retrying with pruning='none'")
            try:
                subgraph = self._pcst_extract(
                    local_graph, valid_seeds, prizes, pruning_override="none",
                    root_entities=root_entities)
            except Exception:
                subgraph = nx.DiGraph()

        # Step 3b: BFS fallback if still too small
        if len(subgraph) < min(len(valid_seeds), 3):
            print(f"  Still too small ({len(subgraph)} nodes), BFS fallback")
            subgraph = self._bfs_fallback(G, valid_seeds)

        # Step 4: Enforce budget
        if len(subgraph) > self.budget:
            pre = len(subgraph)
            subgraph = self._trim_to_budget(subgraph, prizes)
            print(f"  Trimmed: {pre} -> {len(subgraph)} nodes")

        # Step 5: Ensure weak connectivity
        if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
            pre = len(subgraph)
            subgraph = self._largest_component(subgraph)
            print(f"  Largest component: {pre} -> {len(subgraph)} nodes")

        connected = (nx.is_weakly_connected(subgraph)
                     if len(subgraph) > 0 else "N/A")
        print(f"  Final: {len(subgraph)} nodes, "
              f"{subgraph.number_of_edges()} edges, connected={connected}")

        return subgraph

    def _localize(self, G: nx.DiGraph, seed_nodes: List[str]) -> nx.DiGraph:
        """Multi-source BFS to extract local neighborhood around seeds."""
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
        pruning_override: str = None,
        root_entities: List[str] = None
    ) -> nx.DiGraph:
        """
        Run PCST on a localized graph via pcst_fast.

        Prize assignment (G-Retriever style):
        - Zero base prize for all nodes
        - Nodes with retriever similarity scores get their raw score as prize
        - PCST finds the subtree maximizing sum(prizes) - sum(edge_costs)
        - Intermediate nodes (prize=0) are kept only if they cheaply connect
          high-prize nodes
        """
        import pcst_fast

        G_und = G.to_undirected()
        nodes = list(G_und.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # Build edge arrays
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_und.edges()]
        if not edges:
            return G.subgraph([s for s in seed_nodes if s in G]).copy()

        edges_array = np.array(edges, dtype=np.int64)
        costs_array = np.full(len(edges), self.cost, dtype=np.float64)

        # Prizes: raw similarity scores, zero base
        prize_array = np.zeros(num_nodes, dtype=np.float64)
        for node, score in prizes.items():
            if node in node_to_idx:
                prize_array[node_to_idx[node]] = max(score, 0.0)

        # Root selection: prefer topic entity, fall back to highest-prize seed
        valid_seeds_in_local = [s for s in seed_nodes if s in node_to_idx]
        if not valid_seeds_in_local:
            return nx.DiGraph()

        root_node = None
        if root_entities:
            for entity in root_entities:
                if entity in node_to_idx:
                    root_node = entity
                    break

        if root_node is None:
            root_node = max(valid_seeds_in_local,
                            key=lambda s: prizes.get(s, 0.0))

        root = int(node_to_idx[root_node])

        effective_pruning = pruning_override or self.pruning
        scored_nodes = int(np.count_nonzero(prize_array))
        print(f"  PCST input: {num_nodes} nodes, {len(edges)} edges, "
              f"cost={self.cost:.2f}, pruning='{effective_pruning}', "
              f"{scored_nodes} scored nodes, "
              f"root={root_node[:30]}...")

        # Run pcst_fast
        result_nodes, result_edges = pcst_fast.pcst_fast(
            edges_array, prize_array, costs_array,
            root, 1, effective_pruning, 0
        )
        result_nodes = np.asarray(result_nodes, dtype=np.int64)

        # Detect labels-vs-indices return format:
        # Indices are unique node IDs (no duplicates possible).
        # Labels are cluster membership IDs (many duplicates like 0,0,0,1,0).
        # Previous heuristic (len == num_nodes) failed on disconnected graphs
        # where labels covered only one component.
        n_unique = len(np.unique(result_nodes))

        if n_unique < len(result_nodes):
            # Labels format: duplicates present, extract root's cluster
            root_label = result_nodes[root] if root < len(result_nodes) else 0
            selected = np.where(result_nodes == root_label)[0]
            print(f"  PCST returned labels format, "
                  f"extracted {len(selected)} nodes in root cluster")
        else:
            # Indices format: all unique values, use directly
            selected = result_nodes
            print(f"  PCST output: {len(selected)} nodes (indices format)")

        # Map indices back to node names
        selected_names = [nodes[i] for i in selected
                          if 0 <= i < num_nodes]

        if not selected_names:
            selected_names = valid_seeds_in_local

        subgraph = G.subgraph(selected_names).copy()
        return subgraph

    def _bfs_fallback(self, G: nx.DiGraph, seed_nodes: List[str]) -> nx.DiGraph:
        """BFS expansion from highest-degree seed as PCST fallback."""
        G_undirected = self._get_undirected(G)

        best_seed = max(
            [s for s in seed_nodes if s in G_undirected],
            key=lambda s: G_undirected.degree(s),
            default=None
        )
        if best_seed is None:
            return nx.DiGraph()

        visited = {best_seed}
        queue = deque([best_seed])

        while queue and len(visited) < self.budget:
            node = queue.popleft()
            for neighbor in G_undirected.neighbors(node):
                if neighbor not in visited and len(visited) < self.budget:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Add remaining seeds and 1-hop neighbors if space
        for seed in seed_nodes:
            if seed in G_undirected and seed not in visited:
                if len(visited) < self.budget:
                    visited.add(seed)
                    for neighbor in G_undirected.neighbors(seed):
                        if neighbor not in visited and len(visited) < self.budget:
                            visited.add(neighbor)

        return G.subgraph(list(visited)).copy()

    def _trim_to_budget(self, subgraph: nx.DiGraph,
                        prizes: Dict[str, float]) -> nx.DiGraph:
        """Trim to budget by iteratively removing lowest-prize leaves."""
        if len(subgraph) <= self.budget:
            return subgraph

        G = subgraph.copy()
        G_und = G.to_undirected()

        while len(G) > self.budget:
            leaves = [n for n in G_und.nodes() if G_und.degree(n) <= 1]
            if not leaves:
                worst = min(G.nodes(), key=lambda n: prizes.get(n, 0.0))
                G.remove_node(worst)
                G_und.remove_node(worst)
                continue

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
