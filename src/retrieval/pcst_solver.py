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

from typing import List, Dict, Optional, Set
from itertools import chain
import networkx as nx
import numpy as np
from collections import deque


class PCSTSolver:
    """Extract connected subgraphs using Prize-Collecting Steiner Tree."""

    def __init__(self, cost: float = 0.1, budget: int = 70,
                 local_budget: int = 500, pruning: str = "gw",
                 edge_weight_alpha: float = 0.0,
                 bridge_components: bool = True,
                 bridge_max_hops: int = 4,
                 verbose: bool = True):
        """
        Initialize PCST solver.

        Args:
            cost: Base edge cost for PCST. Controls selectivity:
                  higher = more aggressive pruning (fewer nodes kept).
                  Should be tuned relative to prize scale (cosine sim 0-1).
                  Default 0.1 allows traversal of ~4 edges for a 0.4-prize node.
            budget: Maximum nodes in extracted subgraph.
            local_budget: Max nodes for BFS localization before PCST.
            pruning: PCST pruning strategy ('none', 'gw', or 'strong').
            edge_weight_alpha: Query-aware edge cost scaling in [0, 1].
                0.0 = uniform costs (default, backward-compatible).
                Higher = more differentiation based on query-relation similarity.
            bridge_components: If True, attempt to bridge disconnected PCST
                components via shortest paths before falling back to largest.
            bridge_max_hops: Max intermediate relay nodes when bridging.
            verbose: Print debug info per retrieval. Set False for batch loops.
        """
        self.cost = cost
        self.budget = budget
        self.local_budget = local_budget
        self.pruning = pruning
        self.edge_weight_alpha = edge_weight_alpha
        self.bridge_components = bridge_components
        self.bridge_max_hops = bridge_max_hops
        self.verbose = verbose

    def extract_subgraph(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        prizes: Dict[str, float],
        root_entities: List[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        relation_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> nx.DiGraph:
        """
        Extract connected subgraph using PCST.

        Pipeline:
        1. Localize: BFS from seeds to ~local_budget node neighborhood
        2. PCST: Select optimal subtree on the local graph
        3. Budget: Trim if over budget via leaf pruning
        4. Connectivity: Bridge disconnected components or keep largest

        Args:
            G: Full NetworkX graph (can be 1M+ nodes)
            seed_nodes: Seed node names from k-NN search
            prizes: node_name -> cosine similarity score (0 to 1)
            root_entities: Preferred root entities for PCST
            query_embedding: Query vector for edge weight normalization
            relation_embeddings: Dict mapping relation names to embedding vectors

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
        #   Prioritize root entities so PCST root is well-connected
        local_graph = self._localize(G, valid_seeds, root_entities=root_entities)
        if self.verbose:
            print(f"  Localized: {len(local_graph)} nodes, "
                  f"{local_graph.number_of_edges()} edges")

        # Step 2: PCST on local graph
        try:
            subgraph = self._pcst_extract(local_graph, valid_seeds, prizes,
                                          root_entities=root_entities,
                                          query_embedding=query_embedding,
                                          relation_embeddings=relation_embeddings)
        except Exception as e:
            if self.verbose:
                print(f"  PCST failed ({e}), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds,
                                          root_entities=root_entities)

        # Step 3: Validate minimum size — retry with relaxed pruning
        if len(subgraph) < min(len(valid_seeds), 3):
            if self.verbose:
                print(f"  PCST too small ({len(subgraph)} nodes), "
                      f"retrying with pruning='none'")
            try:
                subgraph = self._pcst_extract(
                    local_graph, valid_seeds, prizes, pruning_override="none",
                    root_entities=root_entities,
                    query_embedding=query_embedding,
                    relation_embeddings=relation_embeddings)
            except Exception:
                subgraph = nx.DiGraph()

        # Step 3b: BFS fallback if still too small
        if len(subgraph) < min(len(valid_seeds), 3):
            if self.verbose:
                print(f"  Still too small ({len(subgraph)} nodes), BFS fallback")
            subgraph = self._bfs_fallback(G, valid_seeds,
                                          root_entities=root_entities)

        # Step 4: Enforce budget
        if len(subgraph) > self.budget:
            pre = len(subgraph)
            subgraph = self._trim_to_budget(subgraph, prizes)
            if self.verbose:
                print(f"  Trimmed: {pre} -> {len(subgraph)} nodes")

        # Step 5: Ensure weak connectivity
        if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
            if self.bridge_components:
                pre = len(subgraph)
                subgraph = self._bridge_components(subgraph, local_graph, prizes)
                if self.verbose:
                    print(f"  Bridged: {pre} -> {len(subgraph)} nodes")
                # Re-trim after bridging added relay nodes
                if len(subgraph) > self.budget:
                    subgraph = self._trim_to_budget(subgraph, prizes)
                # Final fallback: if still disconnected, keep largest
                if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
                    pre = len(subgraph)
                    subgraph = self._largest_component(subgraph)
                    if self.verbose:
                        print(f"  Largest component fallback: {pre} -> {len(subgraph)} nodes")
            else:
                pre = len(subgraph)
                subgraph = self._largest_component(subgraph)
                if self.verbose:
                    print(f"  Largest component: {pre} -> {len(subgraph)} nodes")

        connected = (nx.is_weakly_connected(subgraph)
                     if len(subgraph) > 0 else "N/A")
        if self.verbose:
            print(f"  Final: {len(subgraph)} nodes, "
                  f"{subgraph.number_of_edges()} edges, connected={connected}")

        return subgraph

    def _localize(self, G: nx.DiGraph, seed_nodes: List[str],
                   root_entities: Optional[List[str]] = None) -> nx.DiGraph:
        """Two-phase BFS to extract local neighborhood focused on the root.

        Phase 1: BFS from root entities using ~60% of local_budget.
                 This ensures the PCST root has a dense, well-connected
                 neighborhood so the solver can build meaningful trees.
        Phase 2: BFS from remaining seeds using the rest of the budget.
                 Brings in k-NN seed neighborhoods for prize coverage.

        Traverses both edge directions to treat the graph as undirected,
        without creating an expensive undirected copy of the full graph.
        """
        visited = set()
        queue = deque()

        # Identify valid root nodes for phase 1
        root_nodes = []
        if root_entities:
            root_nodes = [r for r in root_entities if r in G]

        if root_nodes:
            # Phase 1: BFS from root entities (60% of budget)
            root_budget = int(self.local_budget * 0.6)

            for root in root_nodes:
                visited.add(root)
                queue.append(root)

            while queue and len(visited) < root_budget:
                node = queue.popleft()
                for neighbor in chain(G.successors(node), G.predecessors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(visited) >= root_budget:
                            break

            # Phase 2: BFS from remaining seeds (40% of budget)
            queue.clear()
            for seed in seed_nodes:
                if seed in G and seed not in visited:
                    visited.add(seed)
                    queue.append(seed)

            while queue and len(visited) < self.local_budget:
                node = queue.popleft()
                for neighbor in chain(G.successors(node), G.predecessors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(visited) >= self.local_budget:
                            break
        else:
            # No root entities — uniform BFS from all seeds (original behavior)
            for seed in seed_nodes:
                if seed in G:
                    visited.add(seed)
                    queue.append(seed)

            while queue and len(visited) < self.local_budget:
                node = queue.popleft()
                for neighbor in chain(G.successors(node), G.predecessors(node)):
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
        root_entities: List[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        relation_embeddings: Optional[Dict[str, np.ndarray]] = None
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

        # Edge costs: uniform or query-aware
        if (self.edge_weight_alpha > 0
                and query_embedding is not None
                and relation_embeddings is not None):
            costs_array = self._compute_query_aware_costs(
                G, G_und, edges, query_embedding, relation_embeddings)
        else:
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
        if self.verbose:
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
        # - Labels format: one entry per node (len == num_nodes), values are
        #   cluster IDs (-1 = unselected, 0+ = cluster membership).
        # - Indices format: subset of selected node indices (len < num_nodes),
        #   values are node IDs in [0, num_nodes).
        # With root >= 0 and num_clusters=1, pcst_fast should return indices,
        # but some versions return labels regardless.
        if len(result_nodes) == num_nodes:
            # Labels format: extract root's cluster
            root_label = int(result_nodes[root])
            if root_label < 0:
                # Root marked unselected — find the largest non-negative cluster
                positive = result_nodes[result_nodes >= 0]
                if len(positive) > 0:
                    root_label = int(np.bincount(positive.astype(np.intp)).argmax())
                    selected = np.where(result_nodes == root_label)[0]
                else:
                    selected = np.array([root], dtype=np.int64)
            else:
                selected = np.where(result_nodes == root_label)[0]
            if self.verbose:
                print(f"  PCST returned labels format, "
                      f"extracted {len(selected)} nodes in root cluster")
        else:
            # Indices format: deduplicate and clamp to valid range
            selected = np.unique(result_nodes)
            selected = selected[(selected >= 0) & (selected < num_nodes)]
            if self.verbose:
                print(f"  PCST output: {len(selected)} nodes (indices format)")

        # Always include root node in result
        if root not in selected:
            selected = np.append(selected, root)

        # Map indices back to node names
        selected_names = [nodes[i] for i in selected
                          if 0 <= i < num_nodes]

        if not selected_names:
            selected_names = valid_seeds_in_local

        subgraph = G.subgraph(selected_names).copy()
        return subgraph

    def _bfs_fallback(self, G: nx.DiGraph, seed_nodes: List[str],
                      root_entities: List[str] = None) -> nx.DiGraph:
        """BFS expansion from topic entity (or highest-degree seed) as PCST fallback.

        Traverses both edge directions without creating an undirected copy.
        """
        # Prefer topic entity as BFS root, fall back to highest-degree seed
        best_seed = None
        if root_entities:
            for entity in root_entities:
                if entity in G:
                    best_seed = entity
                    break

        if best_seed is None:
            best_seed = max(
                [s for s in seed_nodes if s in G],
                key=lambda s: G.degree(s),
                default=None
            )
        if best_seed is None:
            return nx.DiGraph()

        visited = {best_seed}
        queue = deque([best_seed])

        while queue and len(visited) < self.budget:
            node = queue.popleft()
            for neighbor in chain(G.successors(node), G.predecessors(node)):
                if neighbor not in visited and len(visited) < self.budget:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Add remaining seeds and 1-hop neighbors if space
        for seed in seed_nodes:
            if seed in G and seed not in visited:
                if len(visited) < self.budget:
                    visited.add(seed)
                    for neighbor in chain(G.successors(seed), G.predecessors(seed)):
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

    def _compute_query_aware_costs(
        self,
        G_dir: nx.DiGraph,
        G_und: nx.Graph,
        edges: list,
        query_embedding: np.ndarray,
        relation_embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute per-edge costs scaled by query-relation cosine similarity.

        Formula: cost_i = base_cost * (1 - alpha * cosine_sim(query, relation_i))
        High similarity -> lower cost -> PCST prefers this edge.
        """
        alpha = self.edge_weight_alpha
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return np.full(len(edges), self.cost, dtype=np.float64)
        q_unit = query_embedding / q_norm

        costs = np.full(len(edges), self.cost, dtype=np.float64)
        nodes = list(G_und.nodes())

        for i, (u_idx, v_idx) in enumerate(edges):
            u, v = nodes[u_idx], nodes[v_idx]
            # Look up relation from the directed graph
            rel = None
            if G_dir.has_edge(u, v):
                rel = G_dir[u][v].get("relation")
            elif G_dir.has_edge(v, u):
                rel = G_dir[v][u].get("relation")

            if rel and rel in relation_embeddings:
                r_emb = relation_embeddings[rel]
                r_norm = np.linalg.norm(r_emb)
                if r_norm > 0:
                    cos_sim = float(np.dot(q_unit, r_emb / r_norm))
                    cos_sim = max(cos_sim, 0.0)  # clamp negative similarities
                    costs[i] = self.cost * (1.0 - alpha * cos_sim)

        return costs

    def _bridge_components(
        self,
        subgraph: nx.DiGraph,
        local_graph: nx.DiGraph,
        prizes: Dict[str, float]
    ) -> nx.DiGraph:
        """Bridge disconnected PCST components via shortest paths in local_graph.

        For each smaller component that contains at least one prized node,
        find the shortest path to the main (largest) component through the
        local graph. If the path is within bridge_max_hops, add intermediate
        relay nodes to connect the components.
        """
        components = list(nx.weakly_connected_components(subgraph))
        if len(components) <= 1:
            return subgraph

        # Identify main component (largest)
        main_comp = max(components, key=len)
        main_nodes: Set[str] = set(main_comp)
        smaller_comps = [c for c in components if c is not main_comp]

        # Build undirected view of local graph for shortest-path search
        local_und = local_graph.to_undirected(as_view=True)

        all_nodes = set(subgraph.nodes())

        for comp in smaller_comps:
            # Only bridge components with at least one prized node
            has_prize = any(prizes.get(n, 0.0) > 0 for n in comp)
            if not has_prize:
                continue

            # BFS from comp nodes to find shortest path to main_nodes
            best_path = None
            visited: Set[str] = set(comp)
            # (node, path_from_comp_boundary)
            frontier = deque()
            for n in comp:
                if n in local_und:
                    for nbr in local_und.neighbors(n):
                        if nbr not in visited:
                            visited.add(nbr)
                            frontier.append((nbr, [nbr]))

            while frontier:
                node, path = frontier.popleft()
                if node in main_nodes:
                    best_path = path
                    break
                if len(path) >= self.bridge_max_hops:
                    continue
                if node not in local_und:
                    continue
                for nbr in local_und.neighbors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        frontier.append((nbr, path + [nbr]))

            if best_path is not None:
                all_nodes.update(best_path)

        # Rebuild subgraph from local_graph (relay nodes only exist there)
        valid_nodes = [n for n in all_nodes if n in local_graph]
        return local_graph.subgraph(valid_nodes).copy()

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
