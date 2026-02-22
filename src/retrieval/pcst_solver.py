"""
PCST subgraph extraction using Prize-Collecting Steiner Tree.

Key design: Localize first, then run PCST on a small neighborhood.
Running PCST on the full unified graph (1M+ nodes) is too slow.
BFS from seed nodes extracts a ~500-node local neighborhood,
then PCST selects the optimal prize-maximizing subtree from that.

Prize structure follows G-Retriever: prizes are raw cosine similarity
scores (0 to 1) with zero base. Only nodes scored by the retriever
get nonzero prizes. Intermediate relay nodes survive only if they
connect high-prize nodes cheaply enough to justify the edge cost.

Critical insight: k-NN seeds are semantically similar to the query
but may be graph-distant from the root entity. The localized graph
can be disconnected, leaving the root's component with no prized
nodes. We fix this by computing LOCAL prizes: cosine similarity
between the query embedding and entity embeddings for all nodes
in the root's connected component.
"""

from typing import List, Dict, Optional, Set
from itertools import chain
import networkx as nx
import numpy as np
from collections import deque
import re



class PCSTSolver:
    """Extract connected subgraphs using Prize-Collecting Steiner Tree."""

    def __init__(self, cost: float = 0.1, budget: int = 70,
                 local_budget: int = 500, pruning: str = "gw",
                 edge_weight_alpha: float = 0.0,
                 bridge_components: bool = True,
                 bridge_max_hops: int = 4,
                 local_prize_threshold: float = 0.12,
                 existence_prize: float = 0.0,
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
            local_prize_threshold: Min cosine similarity for local prize
                assignment. Nodes in root's component with query similarity
                above this threshold get prizes.
            existence_prize: Small base prize for ALL nodes in root's
                component. Prevents PCST from aggressively pruning relay
                nodes that connect high-prize seeds. Should be << cost
                so relay chains only survive when they connect valuable
                endpoints. Default 0.02 means ~5 relay nodes can be
                "free" if they bridge a 0.1-cost edge to a 0.4-prize node.
            verbose: Print debug info per retrieval. Set False for batch loops.
        """
        self.cost = cost
        self.budget = budget
        self.local_budget = local_budget
        self.pruning = pruning
        self.edge_weight_alpha = edge_weight_alpha
        self.bridge_components = bridge_components
        self.bridge_max_hops = bridge_max_hops
        self.local_prize_threshold = local_prize_threshold
        self.existence_prize = existence_prize
        self.verbose = verbose

    def extract_subgraph(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        prizes: Dict[str, float],
        root_entities: List[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        relation_embeddings: Optional[Dict[str, np.ndarray]] = None,
        entity_embeddings: Optional[Dict[str, np.ndarray]] = None,
        question: str = None,
    ) -> nx.DiGraph:
        """
        Extract connected subgraph using PCST.

        Pipeline:
        1. Localize: BFS from root to ~local_budget node neighborhood
        2. Extract root's connected component
        3. Compute local prizes via query-entity cosine similarity
        4. PCST: Select optimal subtree on the root's component
        5. Budget: Trim if over budget via leaf pruning
        6. Connectivity: Bridge disconnected components or keep largest

        Args:
            G: Full NetworkX graph (can be 1M+ nodes)
            seed_nodes: Seed node names from k-NN search
            prizes: node_name -> cosine similarity score (0 to 1)
            root_entities: Preferred root entities for PCST
            query_embedding: Query vector for edge weight normalization
                and local prize computation
            relation_embeddings: Dict mapping relation names to embedding vectors
            entity_embeddings: Dict mapping entity names to embedding vectors.
                Used to compute local prizes within root's component.

        Returns:
            Connected NetworkX subgraph (<= budget nodes)
        """
        if not seed_nodes:
            return nx.DiGraph()

        valid_seeds = [n for n in seed_nodes if n in G]
        if not valid_seeds:
            print("Warning: no valid seed nodes in graph")
            return nx.DiGraph()

        # Step 1: Localize — BFS from root entity to fill local_budget
        local_graph = self._localize(G, valid_seeds, root_entities=root_entities)
        if self.verbose:
            print(f"  Root entities (input): {root_entities}")
            print(f"  Localized: {len(local_graph)} nodes, "
                  f"{local_graph.number_of_edges()} edges")

        # Step 2: Extract root's connected component for PCST
        root_node = self._pick_root(local_graph, valid_seeds, root_entities, prizes)
        if self.verbose:
            print(f"  Selected root: {root_node}")
        
        # MODIFIED: Use the full local graph instead of just the root's component.
        # This allows us to bridge disjoint components via "virtual edges" later.
        pcst_graph = local_graph
        n_components = nx.number_weakly_connected_components(pcst_graph)

        # Step 3: Compute local prizes
        local_prizes = self._compute_local_prizes(
            pcst_graph, prizes, query_embedding, entity_embeddings,
            question=question)

        if self.verbose:
            global_in_comp = sum(1 for n in prizes if n in pcst_graph)
            local_semantic = sum(1 for n in local_prizes
                                 if n not in prizes
                                 and local_prizes[n] > self.existence_prize)
            existence_only = sum(1 for n in local_prizes
                                 if local_prizes[n] == self.existence_prize
                                 and n not in prizes)
            print(f"  Root component: {len(pcst_graph)} nodes "
                  f"({n_components} components in localized graph)")
            print(f"  Prizes: {global_in_comp} global (k-NN) + "
                  f"{local_semantic} local (query-entity sim) + "
                  f"{existence_only} existence in root component")

        # Step 4: PCST on localized graph (with virtual edges if needed)
        try:
            subgraph = self._pcst_extract(pcst_graph, valid_seeds, local_prizes,
                                          root_entities=root_entities,
                                          query_embedding=query_embedding,
                                          relation_embeddings=relation_embeddings)
        except Exception as e:
            if self.verbose:
                print(f"  PCST failed ({e}), falling back to BFS")
            subgraph = self._bfs_fallback(G, valid_seeds,
                                          root_entities=root_entities)

        # Step 5: Validate minimum size — retry with relaxed pruning
        if len(subgraph) < min(len(valid_seeds), 3):
            if self.verbose:
                print(f"  PCST too small ({len(subgraph)} nodes), "
                      f"retrying with pruning='none'")
            try:
                subgraph = self._pcst_extract(
                    pcst_graph, valid_seeds, local_prizes,
                    pruning_override="none",
                    root_entities=root_entities,
                    query_embedding=query_embedding,
                    relation_embeddings=relation_embeddings)
            except Exception:
                subgraph = nx.DiGraph()

        # Step 5b: BFS fallback if still too small
        if len(subgraph) < min(len(valid_seeds), 3):
            if self.verbose:
                print(f"  Still too small ({len(subgraph)} nodes), BFS fallback")
            subgraph = self._bfs_fallback(G, valid_seeds,
                                          root_entities=root_entities)

        # Step 6: Enforce budget
        if len(subgraph) > self.budget:
            pre = len(subgraph)
            subgraph = self._trim_to_budget(subgraph, local_prizes)
            if self.verbose:
                print(f"  Trimmed: {pre} -> {len(subgraph)} nodes")

        # Step 7: Ensure weak connectivity
        if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
            if self.bridge_components:
                pre = len(subgraph)
                subgraph = self._bridge_components(subgraph, pcst_graph,
                                                   local_prizes)
                if self.verbose:
                    print(f"  Bridged: {pre} -> {len(subgraph)} nodes")
                if len(subgraph) > self.budget:
                    subgraph = self._trim_to_budget(subgraph, local_prizes)
                if len(subgraph) > 1 and not nx.is_weakly_connected(subgraph):
                    pre = len(subgraph)
                    subgraph = self._largest_component(subgraph)
                    if self.verbose:
                        print(f"  Largest component fallback: "
                              f"{pre} -> {len(subgraph)} nodes")
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

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def _localize(self, G: nx.DiGraph, seed_nodes: List[str],
                   root_entities: Optional[List[str]] = None) -> nx.DiGraph:
        """BFS from root entity to extract dense local neighborhood.

        Allocates the full local_budget to BFS from the root entity.
        This ensures PCST has a single large connected component
        centered on the root, rather than scattered disconnected islands
        around k-NN seeds.

        Remaining seeds are added as individual nodes (no BFS expansion)
        to preserve their prize eligibility without fragmenting the graph.

        Traverses both edge directions to treat the graph as undirected,
        without creating an expensive undirected copy of the full graph.
        """
        visited = set()
        queue = deque()

        # BFS from root entities (primary)
        root_nodes = []
        if root_entities:
            root_nodes = [r for r in root_entities if r in G]

        if root_nodes:
            for root in root_nodes:
                visited.add(root)
                queue.append(root)
        else:
            # No root — start from first valid seed
            for seed in seed_nodes:
                if seed in G:
                    visited.add(seed)
                    queue.append(seed)
                    break

        # BFS expansion from root(s)
        while queue and len(visited) < self.local_budget:
            node = queue.popleft()
            for neighbor in chain(G.successors(node), G.predecessors(node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= self.local_budget:
                        break

        # Add remaining seed nodes (no expansion) so they can receive prizes
        for seed in seed_nodes:
            if seed in G and seed not in visited:
                visited.add(seed)

        return G.subgraph(list(visited)).copy()

    # ------------------------------------------------------------------
    # Root component extraction
    # ------------------------------------------------------------------

    def _pick_root(self, G: nx.DiGraph, seed_nodes: List[str],
                   root_entities: Optional[List[str]],
                   prizes: Dict[str, float]) -> str:
        """Select root node from the localized graph."""
        if root_entities:
            for entity in root_entities:
                if entity in G:
                    return entity

        valid = [s for s in seed_nodes if s in G]
        if valid:
            return max(valid, key=lambda s: prizes.get(s, 0.0))
        return list(G.nodes())[0]

    def _root_component(self, G: nx.DiGraph,
                        root_node: str) -> tuple:
        """Extract the weakly connected component containing root_node.

        Returns:
            (component_subgraph, total_num_components)
        """
        components = list(nx.weakly_connected_components(G))
        n_components = len(components)

        for comp in components:
            if root_node in comp:
                return G.subgraph(list(comp)).copy(), n_components

        # Shouldn't happen, but fallback to full graph
        return G.copy(), n_components

    # ------------------------------------------------------------------
    # Local prize computation
    # ------------------------------------------------------------------

    def _compute_local_prizes(
        self,
        root_component: nx.DiGraph,
        global_prizes: Dict[str, float],
        query_embedding: Optional[np.ndarray],
        entity_embeddings: Optional[Dict[str, np.ndarray]],
        question: str = None,
    ) -> Dict[str, float]:
        """Compute prizes for nodes in root's component.

        Merges two sources:
        1. Global prizes (from k-NN search) — for nodes that happen to
           be in the root's component.
        2. Local prizes — cosine similarity between query embedding and
           each node's entity embedding. This finds relevant nodes that
           the global k-NN search missed because they weren't in the
           top-K globally but ARE near the root in graph space.

        Local prizes are essential: without them, the root's component
        typically has zero prized nodes (only the root with prize 1.0),
        causing PCST to return just the root every time.

        Returns:
            Dict[node_name, prize] for all nodes in root_component
        """
        prizes = {}
        comp_nodes = set(root_component.nodes())

        # 1. Copy global prizes for nodes in this component
        for node, score in global_prizes.items():
            if node in comp_nodes:
                prizes[node] = max(score, 0.0)

        # 2. Compute local prizes via query-entity cosine similarity
        if query_embedding is not None and entity_embeddings is not None:
            q_norm = np.linalg.norm(query_embedding)
            if q_norm > 0:
                q_unit = query_embedding / q_norm

                for node in comp_nodes:
                    if node in prizes:
                        continue  # global prize already set
                    emb = entity_embeddings.get(node)
                    if emb is None:
                        continue
                    e_norm = np.linalg.norm(emb)
                    if e_norm == 0:
                        continue
                    cos_sim = float(np.dot(q_unit, emb / e_norm))
                    if cos_sim >= self.local_prize_threshold:
                        prizes[node] = cos_sim

        # 3. Existence prize
        if self.existence_prize > 0:
            for node in comp_nodes:
                if node not in prizes:
                    prizes[node] = self.existence_prize

        # 4. Answer-Type boosting
        if question:
            q_lower = question.lower()
            target_types = []
            if any(x in q_lower for x in ['when', 'year', 'date', 'how old', 'birth']):
                target_types.append('date')
            if any(x in q_lower for x in ['how many', 'count', 'population', 'price', 'cost']):
                target_types.append('number')

            if target_types:
                date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
                # Simple number pattern: integer or strictly float, avoid "Model T"
                number_pattern = re.compile(r'^-?\d+(\.\d+)?$')
                
                for node in comp_nodes:
                    boost = 0.0
                    if 'date' in target_types and date_pattern.match(node):
                        boost = 0.5
                    elif 'number' in target_types and number_pattern.match(node):
                        boost = 0.5
                    
                    if boost > 0:
                        prizes[node] = max(prizes.get(node, 0.0), boost)

        return prizes

    # ------------------------------------------------------------------
    # PCST core
    # ------------------------------------------------------------------

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

        # Update for pcst_fast compatibility:
        # Standard pcst_fast (and many pybind11 wrappers) expect int32/int64 based on compilation.
        # If int64 produced garbage (169 items collapsing to 1 node), it suggests a type mismatch
        # where the C++ side read zeros/garbage. Reverting to int32 as a fix.
        # Original comment about "Forcing int32 caused... zeros" might have been environment-specific
        # or related to a different version. Given current failure, we swap to int32.
        edges_array = np.array(edges, dtype=np.int32)

        # Edge costs: uniform or query-aware
        if (self.edge_weight_alpha > 0
                and query_embedding is not None
                and relation_embeddings is not None):
            costs_array = self._compute_query_aware_costs(
                G, G_und, edges, query_embedding, relation_embeddings)
        else:
            costs_array = np.full(len(edges), self.cost, dtype=np.float64)

        # pcst_fast requires strictly positive edge costs — enforce a floor.
        # cost=0 causes the GW algorithm to degenerate (returns only root).
        costs_array = np.maximum(costs_array, 1e-9)

        # Prizes: from merged global + local prizes
        prize_array = np.zeros(num_nodes, dtype=np.float64)
        for node, score in prizes.items():
            if node in node_to_idx:
                prize_array[node_to_idx[node]] = max(score, 0.0)

        # Root selection and Virtual Edges
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

        # Virtual Edges: Connect root to top high-prize nodes in disjoint components
        # Calculate components relative to undirected graph
        # But for PCST, we pass a list of edges.
        # If the graph is disconnected, pcst_fast sees disconnected components.
        # We need to explicitly add edges.
        
        # Identify nodes with prizes that are NOT reachable from root?
        # That requires a full traversal check. Faster to just add virtual edges
        # to ALL high-prize nodes ( > cost used for virtual edge).
        
        virtual_edge_cost = self.cost * 10
        virtual_edges = []
        
        # Only add virtual edges if they have a prize big enough to justify the cost
        # (or at least close to it, so they are considered).
        # We limit to top-K prize nodes to avoid exploding edge count.
        top_prized = sorted(
            [(n, p) for n, p in prizes.items() if n in node_to_idx and n != root_node],
            key=lambda x: x[1], reverse=True
        )[:20]  # Check top 20 candidates

        for node_name, prize_val in top_prized:
            target_idx = node_to_idx[node_name]
            # Check if reachable? (Optional optimization, skipping for now for speed)
            # Add virtual edge
            virtual_edges.append((root, target_idx))

        if virtual_edges:
            if self.verbose:
                print(f"  Adding {len(virtual_edges)} virtual edges to bridge components")
            
            # Extend edges_array and costs_array
            v_edges_arr = np.array(virtual_edges, dtype=np.int32)
            edges_array = np.vstack([edges_array, v_edges_arr])
            
            v_costs = np.full(len(virtual_edges), virtual_edge_cost, dtype=np.float64)
            costs_array = np.concatenate([costs_array, v_costs])


        effective_pruning = pruning_override or self.pruning
        scored_nodes = int(np.count_nonzero(prize_array))
        if self.verbose:
            print(f"  PCST input: {num_nodes} nodes, {len(edges)} edges, "
                  f"cost={self.cost:.4g}, pruning='{effective_pruning}', "
                  f"{scored_nodes} scored nodes, "
                  f"root={root_node[:30]}...")

        # Run pcst_fast
        try:
            result_nodes, result_edges = pcst_fast.pcst_fast(
                edges_array, prize_array, costs_array,
                root, 1, effective_pruning, 0
            )
        except Exception as e:
            if self.verbose:
                print(f"  pcst_fast crashed: {e}")
            return nx.DiGraph()

        result_nodes = np.asarray(result_nodes, dtype=np.int64)
        
        if self.verbose:
            print(f"  pcst_fast raw output: {len(result_nodes)} items")
            if len(result_nodes) > 0:
                print(f"  Sample raw output: {result_nodes[:20]}")
                unique_sample = np.unique(result_nodes)
                print(f"  Unique values in output: {len(unique_sample)}")
                if len(unique_sample) < 20:
                     print(f"  Unique values: {unique_sample}")

        # Detect labels-vs-indices return format:
        # pcst_fast with root >= 0 should return indices, but some versions
        # return a labels array of length num_nodes.
        
        selected = np.array([], dtype=np.int64)
        
        if len(result_nodes) == num_nodes:
            # Labels format: extract root's cluster
            root_label = int(result_nodes[root])
            if root_label < 0:
                # Root assigned to -1 (unselected)? Try to find any positive cluster
                positive = result_nodes[result_nodes >= 0]
                if len(positive) > 0:
                    # Pick largest cluster
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
            # Indices format
            # Sanity check: if we got indices, are they valid?
            if len(result_nodes) > 0:
                min_idx = np.min(result_nodes)
                max_idx = np.max(result_nodes)
                if min_idx < 0 or max_idx >= num_nodes:
                    if self.verbose:
                        print(f"  Warning: pcst_fast returned invalid indices "
                              f"[{min_idx}, {max_idx}] for num_nodes={num_nodes}")
                    # Filter invalid
                    selected = result_nodes[(result_nodes >= 0) & (result_nodes < num_nodes)]
                    selected = np.unique(selected)
                else:
                    selected = np.unique(result_nodes)
            else:
                selected = np.array([], dtype=np.int64)

            if self.verbose:
                print(f"  PCST output: {len(selected)} nodes (indices format)")

        # Always include root
        if root not in selected:
            selected = np.append(selected, root)
        
        # Determine names
        selected_names = []
        for i in selected:
            if 0 <= i < num_nodes:
                selected_names.append(nodes[i])
        
        if not selected_names:
            selected_names = valid_seeds_in_local
            
        subgraph = G.subgraph(selected_names).copy()
        return subgraph

    # ------------------------------------------------------------------
    # Fallback and trimming
    # ------------------------------------------------------------------

    def _bfs_fallback(self, G: nx.DiGraph, seed_nodes: List[str],
                      root_entities: List[str] = None) -> nx.DiGraph:
        """BFS expansion from topic entity (or highest-degree seed) as PCST fallback.

        Traverses both edge directions without creating an undirected copy.
        """
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
                    for neighbor in chain(G.successors(seed),
                                          G.predecessors(seed)):
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

    # ------------------------------------------------------------------
    # Query-aware edge costs
    # ------------------------------------------------------------------

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
                    cos_sim = max(cos_sim, 0.0)
                    costs[i] = self.cost * (1.0 - alpha * cos_sim)

        return costs

    # ------------------------------------------------------------------
    # Component bridging
    # ------------------------------------------------------------------

    def _bridge_components(
        self,
        subgraph: nx.DiGraph,
        local_graph: nx.DiGraph,
        prizes: Dict[str, float]
    ) -> nx.DiGraph:
        """Bridge disconnected PCST components via shortest paths in local_graph."""
        components = list(nx.weakly_connected_components(subgraph))
        if len(components) <= 1:
            return subgraph

        main_comp = max(components, key=len)
        main_nodes: Set[str] = set(main_comp)
        smaller_comps = [c for c in components if c is not main_comp]

        local_und = local_graph.to_undirected(as_view=True)
        all_nodes = set(subgraph.nodes())

        for comp in smaller_comps:
            has_prize = any(prizes.get(n, 0.0) > 0 for n in comp)
            if not has_prize:
                continue

            best_path = None
            visited: Set[str] = set(comp)
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
