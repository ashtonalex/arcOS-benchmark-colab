"""
Retriever orchestration layer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import networkx as nx
import numpy as np
from pathlib import Path
from collections import deque

from ..config import BenchmarkConfig
from ..utils.checkpoints import save_checkpoint, load_checkpoint, checkpoint_exists
from .embeddings import TextEmbedder
from .faiss_index import EntityIndex
from .pcst_solver import PCSTSolver


@dataclass
class RetrievedSubgraph:
    """Container for retrieval results."""
    subgraph: nx.DiGraph
    question: str
    seed_entities: List[str]
    similarity_scores: Dict[str, float]
    num_nodes: int
    num_edges: int
    retrieval_time_ms: float
    pcst_used: bool
    has_answer: Optional[bool] = None


class Retriever:
    """High-level API for retrieving query-relevant subgraphs."""

    def __init__(
        self,
        config: BenchmarkConfig,
        unified_graph: nx.DiGraph,
        embedder: TextEmbedder,
        entity_index: EntityIndex,
        pcst_solver: PCSTSolver,
        entity_embeddings: Dict[str, np.ndarray],
        relation_embeddings: Dict[str, np.ndarray]
    ):
        """
        Initialize retriever with all components.

        Args:
            config: Benchmark configuration
            unified_graph: Full knowledge graph
            embedder: Text embedding model
            entity_index: FAISS k-NN index
            pcst_solver: PCST subgraph extractor
            entity_embeddings: Dict mapping entity names to embedding vectors
            relation_embeddings: Dict mapping relation names to embedding vectors
        """
        self.config = config
        self.unified_graph = unified_graph
        self.embedder = embedder
        self.text_embedder = embedder  # Alias for compatibility with GNN module
        self.entity_index = entity_index
        self.pcst_solver = pcst_solver
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

    def retrieve(
        self,
        question: str,
        q_entity: Optional[List[str]] = None,
        answer_entities: Optional[List[str]] = None,
    ) -> RetrievedSubgraph:
        """
        Main retrieval method: question -> subgraph.

        Args:
            question: Natural language question
            q_entity: Optional list of known topic entity names from the
                dataset.  When provided these are used as primary seeds,
                bypassing the k-NN search for seed selection.
            answer_entities: Optional list of expected answer entities.
                When provided, ``RetrievedSubgraph.has_answer`` is set to
                True if any answer entity appears as a node in the returned
                subgraph.

        Returns:
            RetrievedSubgraph with extracted subgraph and metadata
        """
        start_time = time.time()

        # 1. Embed query
        query_embedding = self.embedder.embed_texts([question], show_progress=False)[0]

        # 2. k-NN search (always run — used for secondary seeds + prizes)
        top_k_results = self.entity_index.search(query_embedding, k=self.config.top_k_entities)

        # 3. Build seed list — q_entity first (primary), then k-NN (secondary)
        seed_entities = []
        similarity_scores = {}
        q_entity_names = set()

        if q_entity:
            entities = q_entity if isinstance(q_entity, list) else [q_entity]
            for entity in entities:
                if entity in self.unified_graph and entity not in similarity_scores:
                    seed_entities.append(entity)
                    similarity_scores[entity] = 1.0
                    q_entity_names.add(entity)

        for entity, score in top_k_results:
            if entity not in similarity_scores:
                seed_entities.append(entity)
                similarity_scores[entity] = score

        # 3b. Filter lexical noise from k-NN seeds
        #   k-NN embedding search can return false positives like
        #   "Audierne" for a query about "Audi". Remove seeds whose
        #   names are suspiciously similar substrings of other seeds
        #   but have low similarity scores.
        if q_entity_names:
            seed_entities = self._filter_noisy_seeds(
                seed_entities, q_entity_names, similarity_scores)

        # 4. Prizes — raw cosine similarity scores (0 to 1)
        #
        #   Prizes and edge costs on the same scale gives PCST meaningful
        #   signal. A node with cosine sim 0.2 doesn't justify a 0.3 edge cost.
        #   No 1-hop neighbor prizes — PCST keeps relay nodes organically
        #   when they cheaply connect high-prize targets.
        prizes = {}
        similarity_threshold = 0.4

        # Topic entities: perfect relevance
        for entity in q_entity_names:
            prizes[entity] = 1.0

        # k-NN entities: raw cosine similarity, filtered by threshold
        for entity, score in top_k_results:
            if entity not in prizes and score >= similarity_threshold:
                prizes[entity] = float(score)

        # 5. Extract subgraph
        pcst_used = True
        try:
            subgraph = self.pcst_solver.extract_subgraph(
                self.unified_graph,
                seed_entities,
                prizes,
                root_entities=list(q_entity_names) if q_entity_names else None,
                query_embedding=query_embedding,
                relation_embeddings=self.relation_embeddings,
                entity_embeddings=self.entity_embeddings,
            )
        except Exception as e:
            print(f"⚠ PCST extraction failed: {e}, using BFS fallback")
            subgraph = self.pcst_solver._bfs_fallback(
                self.unified_graph,
                seed_entities,
                root_entities=list(q_entity_names) if q_entity_names else None
            )
            pcst_used = False

        # 6. Calculate timing
        end_time = time.time()
        retrieval_time_ms = (end_time - start_time) * 1000

        # 7. Build result
        has_answer: Optional[bool] = None
        if answer_entities is not None:
            subgraph_nodes = set(subgraph.nodes())
            has_answer = any(ent in subgraph_nodes for ent in answer_entities)

        result = RetrievedSubgraph(
            subgraph=subgraph,
            question=question,
            seed_entities=seed_entities,
            similarity_scores=similarity_scores,
            num_nodes=len(subgraph),
            num_edges=subgraph.number_of_edges(),
            retrieval_time_ms=retrieval_time_ms,
            pcst_used=pcst_used,
            has_answer=has_answer,
        )

        return result

    @staticmethod
    def _filter_noisy_seeds(
        seed_entities: List[str],
        q_entity_names: set,
        similarity_scores: Dict[str, float],
        min_score: float = 0.35,
    ) -> List[str]:
        """Remove k-NN seeds that are likely lexical false positives.

        Heuristic: if a seed's name is a near-substring of a topic entity
        (or vice versa) but they are NOT the same entity, it's probably
        a lexical coincidence (e.g. "Audierne" for "Audi").  Drop it
        unless its similarity score is high enough to override.

        Also drops any seed below *min_score* that isn't a topic entity.
        """
        filtered = []
        for entity in seed_entities:
            # Always keep topic entities
            if entity in q_entity_names:
                filtered.append(entity)
                continue

            score = similarity_scores.get(entity, 0.0)

            # Drop low-scoring seeds outright
            if score < min_score:
                continue

            # Check for suspicious substring overlap with topic entities
            e_lower = entity.lower()
            is_noisy = False
            for q_ent in q_entity_names:
                q_lower = q_ent.lower()
                # If one is a prefix/substring of the other but they differ
                if (e_lower != q_lower
                        and len(e_lower) > 3 and len(q_lower) > 3
                        and (e_lower.startswith(q_lower)
                             or q_lower.startswith(e_lower))):
                    # Only keep if score is very high
                    if score < 0.6:
                        is_noisy = True
                        break

            if not is_noisy:
                filtered.append(entity)

        return filtered

    @classmethod
    def build_from_checkpoint_or_new(
        cls,
        config: BenchmarkConfig,
        unified_graph: nx.DiGraph
    ) -> "Retriever":
        """
        Factory method: load from checkpoints or build fresh.

        Args:
            config: Benchmark configuration
            unified_graph: Full knowledge graph

        Returns:
            Fully initialized Retriever instance
        """
        print("=" * 60)
        print("BUILDING RETRIEVAL PIPELINE")
        print("=" * 60)

        # Define checkpoint paths
        entity_embeddings_path = config.get_checkpoint_path("entity_embeddings.pkl")
        relation_embeddings_path = config.get_checkpoint_path("relation_embeddings.pkl")
        faiss_index_path = config.get_checkpoint_path("faiss_index.bin")
        entity_mapping_path = config.get_checkpoint_path("entity_mapping.pkl")

        # 1. Initialize embedder
        print("\n[1/4] Initializing text embedder...")
        embedder = TextEmbedder(
            model_name=config.embedding_model,
            device="cuda"
        )

        # 2. Load or compute entity embeddings
        print("\n[2/4] Loading/computing entity embeddings...")
        if checkpoint_exists(entity_embeddings_path):
            print(f"Loading cached embeddings from {entity_embeddings_path.name}")
            entity_embeddings = load_checkpoint(entity_embeddings_path, format="pickle")
            print(f"✓ Loaded {len(entity_embeddings)} entity embeddings")
        else:
            print("Computing entity embeddings (this may take several minutes)...")
            entity_embeddings = embedder.embed_graph_entities(unified_graph, batch_size=32)
            save_checkpoint(entity_embeddings, entity_embeddings_path, format="pickle")

        # 3. Load or compute relation embeddings (for future use)
        print("\n[3/4] Loading/computing relation embeddings...")
        if checkpoint_exists(relation_embeddings_path):
            print(f"Loading cached relation embeddings from {relation_embeddings_path.name}")
            relation_embeddings = load_checkpoint(relation_embeddings_path, format="pickle")
            print(f"✓ Loaded {len(relation_embeddings)} relation embeddings")
        else:
            print("Computing relation embeddings...")
            relation_embeddings = embedder.embed_relations(unified_graph, batch_size=32)
            save_checkpoint(relation_embeddings, relation_embeddings_path, format="pickle")

        # 4. Load or build FAISS index
        print("\n[4/4] Loading/building FAISS index...")
        entity_index = EntityIndex(embedding_dim=config.embedding_dim)

        if checkpoint_exists(faiss_index_path) and checkpoint_exists(entity_mapping_path):
            print(f"Loading cached FAISS index from {faiss_index_path.name}")
            entity_index.load(faiss_index_path, entity_mapping_path)
        else:
            print("Building FAISS index...")
            entity_index.build(entity_embeddings)
            entity_index.save(faiss_index_path, entity_mapping_path)

        # 5. Initialize PCST solver (no state to checkpoint)
        print("\nInitializing PCST solver...")
        pcst_solver = PCSTSolver(
            cost=config.pcst_cost,
            budget=config.pcst_budget,
            local_budget=config.pcst_local_budget,
            pruning=config.pcst_pruning,
            edge_weight_alpha=config.pcst_edge_weight_alpha,
            bridge_components=config.pcst_bridge_components,
            bridge_max_hops=config.pcst_bridge_max_hops,
        )
        print(f"✓ PCST solver ready (cost: {config.pcst_cost}, budget: {config.pcst_budget}, "
              f"local: {config.pcst_local_budget}, pruning: {config.pcst_pruning}, "
              f"alpha: {config.pcst_edge_weight_alpha}, bridge: {config.pcst_bridge_components})")

        # 6. Create retriever instance
        retriever = cls(
            config=config,
            unified_graph=unified_graph,
            embedder=embedder,
            entity_index=entity_index,
            pcst_solver=pcst_solver,
            entity_embeddings=entity_embeddings,
            relation_embeddings=relation_embeddings
        )

        print("\n" + "=" * 60)
        print("✓ RETRIEVAL PIPELINE READY")
        print("=" * 60)

        return retriever

    # ------------------------------------------------------------------
    # Hit-rate diagnostics
    # ------------------------------------------------------------------

    def check_entity_coverage(
        self, dataset, limit: Optional[int] = None
    ) -> Dict[str, object]:
        """
        Check how many answer entities from *dataset* exist in the unified graph.

        Separates "graph doesn't contain the answer" (coverage gap) from
        "retrieval didn't find the answer" (retrieval gap).

        Args:
            dataset: HuggingFace Dataset with ``a_entity`` field.
            limit: Max examples to check (None = all).

        Returns:
            Dict with keys:
                total: number of examples checked
                reachable: examples whose a_entity has >= 1 node in unified graph
                unreachable: examples with no a_entity node in unified graph
                reachable_rate: reachable / total (0-1)
                missing_entities: set of a_entity values absent from the graph
        """
        graph_nodes = set(self.unified_graph.nodes())
        total = 0
        reachable = 0
        missing_entities: set = set()

        n = len(dataset) if limit is None else min(limit, len(dataset))
        for i in range(n):
            example = dataset[i]
            answer_entities = example.get("a_entity", [])
            if isinstance(answer_entities, str):
                answer_entities = [answer_entities]

            total += 1
            if any(ent in graph_nodes for ent in answer_entities):
                reachable += 1
            else:
                missing_entities.update(answer_entities)

        rate = reachable / total if total > 0 else 0.0
        return {
            "total": total,
            "reachable": reachable,
            "unreachable": total - reachable,
            "reachable_rate": rate,
            "missing_entities": missing_entities,
        }

    @staticmethod
    def node_hit(
        subgraph: nx.DiGraph,
        answer_entities: List[str],
    ) -> bool:
        """True if any answer entity is a node in *subgraph*."""
        nodes = set(subgraph.nodes())
        return any(ent in nodes for ent in answer_entities)

    @staticmethod
    def evidence_hit(
        subgraph: nx.DiGraph,
        q_entities: List[str],
        answer_entities: List[str],
        max_hops: int = 2,
    ) -> bool:
        """
        True if the subgraph contains a path of <= *max_hops* edges from
        any q_entity to any a_entity (in either direction).

        This is stricter than ``node_hit`` — the answer node must be
        *reachable from the question entity* within the subgraph, meaning
        the connecting evidence triples are present and verbalisable.
        """
        nodes = set(subgraph.nodes())
        q_in = [q for q in q_entities if q in nodes]
        a_set = set(a for a in answer_entities if a in nodes)

        if not q_in or not a_set:
            return False

        # BFS from each q_entity on the undirected view (relations are
        # evidence regardless of edge direction).
        undirected = subgraph.to_undirected(as_view=True)
        for start in q_in:
            visited = {start}
            frontier = deque([(start, 0)])
            while frontier:
                node, depth = frontier.popleft()
                if node in a_set:
                    return True
                if depth >= max_hops:
                    continue
                for nbr in undirected.neighbors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        frontier.append((nbr, depth + 1))
        return False

    def evaluate_hit_rates(
        self,
        dataset,
        limit: Optional[int] = None,
        max_hops: int = 2,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """
        Run retrieval on *dataset* and report coverage-aware hit rates.

        Returns a dict with:
            total, reachable, reachable_rate  — coverage stats
            node_hits, node_hit_rate_raw      — classic metric (all examples)
            node_hit_rate_cond                — hits / reachable (retrieval quality)
            evidence_hits, evidence_hit_rate_raw, evidence_hit_rate_cond
            per_example — list of per-example dicts for detailed inspection
        """
        graph_nodes = set(self.unified_graph.nodes())
        n = len(dataset) if limit is None else min(limit, len(dataset))

        reachable = 0
        node_hits = 0
        evidence_hits = 0
        per_example: List[Dict] = []

        for i in range(n):
            example = dataset[i]
            question = example["question"]
            answer_entities = example.get("a_entity", [])
            if isinstance(answer_entities, str):
                answer_entities = [answer_entities]
            q_entities = example.get("q_entity", [])
            if isinstance(q_entities, str):
                q_entities = [q_entities]

            is_reachable = any(ent in graph_nodes for ent in answer_entities)
            if is_reachable:
                reachable += 1

            result = self.retrieve(question, q_entity=q_entities)

            n_hit = self.node_hit(result.subgraph, answer_entities)
            e_hit = self.evidence_hit(
                result.subgraph, q_entities, answer_entities, max_hops=max_hops
            )

            if n_hit:
                node_hits += 1
            if e_hit:
                evidence_hits += 1

            per_example.append({
                "question": question,
                "q_entity": q_entities,
                "a_entity": answer_entities,
                "reachable": is_reachable,
                "node_hit": n_hit,
                "evidence_hit": e_hit,
                "num_nodes": result.num_nodes,
                "retrieval_time_ms": result.retrieval_time_ms,
            })

        node_raw = node_hits / n if n > 0 else 0.0
        node_cond = node_hits / reachable if reachable > 0 else 0.0
        ev_raw = evidence_hits / n if n > 0 else 0.0
        ev_cond = evidence_hits / reachable if reachable > 0 else 0.0

        summary = {
            "total": n,
            "reachable": reachable,
            "reachable_rate": reachable / n if n > 0 else 0.0,
            "node_hits": node_hits,
            "node_hit_rate_raw": node_raw,
            "node_hit_rate_cond": node_cond,
            "evidence_hits": evidence_hits,
            "evidence_hit_rate_raw": ev_raw,
            "evidence_hit_rate_cond": ev_cond,
            "per_example": per_example,
        }

        if verbose:
            print("=" * 60)
            print("HIT RATE EVALUATION")
            print("=" * 60)
            print(f"Examples evaluated:  {n}")
            print(f"Answer in graph:     {reachable}/{n} "
                  f"({summary['reachable_rate']:.1%})")
            print()
            print(f"--- Node-level hit (a_entity is a subgraph node) ---")
            print(f"  Raw hit rate:      {node_hits}/{n} ({node_raw:.1%})")
            print(f"  Conditioned:       {node_hits}/{reachable} ({node_cond:.1%})")
            print()
            print(f"--- Evidence hit (q_entity -> a_entity within "
                  f"{max_hops} hops) ---")
            print(f"  Raw hit rate:      {evidence_hits}/{n} ({ev_raw:.1%})")
            print(f"  Conditioned:       {evidence_hits}/{reachable} "
                  f"({ev_cond:.1%})")
            print("=" * 60)

        return summary

    def evaluate_batch(
        self,
        examples: List[Dict],
        verbose: bool = True,
    ) -> Dict[str, object]:
        """
        Retrieve subgraphs for a list of example dicts and return aggregate
        hit statistics.

        Each element of *examples* must have:
            ``question`` (str), ``q_entity`` (str | List[str]),
            ``a_entity``  (str | List[str]).

        This is a lightweight alternative to ``evaluate_hit_rates`` that
        accepts a plain list of dicts instead of a HuggingFace Dataset
        object, and stores ``has_answer`` on every ``RetrievedSubgraph``
        via the new ``answer_entities`` argument to ``retrieve()``.

        Returns:
            Dict with keys:
                total, hits, hit_rate,
                avg_time_ms, avg_nodes,
                results  — list of RetrievedSubgraph objects
        """
        hits = 0
        total_time_ms = 0.0
        total_nodes = 0
        results: List[RetrievedSubgraph] = []

        for example in examples:
            question = example["question"]
            answer_entities = example.get("a_entity", [])
            if isinstance(answer_entities, str):
                answer_entities = [answer_entities]
            q_entities = example.get("q_entity", [])
            if isinstance(q_entities, str):
                q_entities = [q_entities]

            result = self.retrieve(
                question,
                q_entity=q_entities,
                answer_entities=answer_entities,
            )
            results.append(result)

            if result.has_answer:
                hits += 1
            total_time_ms += result.retrieval_time_ms
            total_nodes += result.num_nodes

        n = len(examples)
        hit_rate = hits / n if n > 0 else 0.0
        avg_time = total_time_ms / n if n > 0 else 0.0
        avg_nodes = total_nodes / n if n > 0 else 0.0

        summary = {
            "total": n,
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_time_ms": avg_time,
            "avg_nodes": avg_nodes,
            "results": results,
        }

        if verbose:
            print("=" * 60)
            print("BATCH EVALUATION")
            print("=" * 60)
            print(f"Examples:       {n}")
            print(f"Hit rate:       {hits}/{n} ({hit_rate:.1%})")
            print(f"Avg time:       {avg_time:.1f} ms")
            print(f"Avg nodes:      {avg_nodes:.1f}")
            print("=" * 60)

        return summary
