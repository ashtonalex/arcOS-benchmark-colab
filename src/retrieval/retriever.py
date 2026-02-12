"""
Retriever orchestration layer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import time
import networkx as nx
import numpy as np
from pathlib import Path

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
        self, question: str, q_entity: Optional[List[str]] = None
    ) -> RetrievedSubgraph:
        """
        Main retrieval method: question -> subgraph.

        Args:
            question: Natural language question
            q_entity: Optional list of known topic entity names from the
                dataset.  When provided these are used as primary seeds,
                bypassing the k-NN search for seed selection.

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

        # 4. Prizes — logarithmic scaling to prevent top-rank dominance
        #
        #   Old linear:  rank 1 → 150, rank 15 → 10   (15:1 ratio)
        #   New log:     rank 1 →  55, rank 15 → 14   ( 4:1 ratio)
        #   q_entity:    fixed 100  (always highest)
        prizes = {}
        top_k = self.config.top_k_entities

        # Topic entities get a fixed high prize
        for entity in q_entity_names:
            prizes[entity] = 100.0

        # k-NN entities: log-scaled rank prize + similarity prize
        for rank, (entity, score) in enumerate(top_k_results, start=1):
            if entity not in prizes:
                rank_prize = math.log1p(top_k - rank + 1) * 20.0
                score_prize = max(float(score), 0.0) * 20.0
                prizes[entity] = rank_prize + score_prize

        # 1-hop neighbor prizes (enough to justify edge cost of 0.1)
        max_neighbors_per_seed = 20
        for seed in seed_entities:
            if seed in self.unified_graph:
                count = 0
                for neighbor in self.unified_graph.successors(seed):
                    if neighbor not in prizes:
                        prizes[neighbor] = 5.0
                        count += 1
                        if count >= max_neighbors_per_seed:
                            break
                count = 0
                for neighbor in self.unified_graph.predecessors(seed):
                    if neighbor not in prizes:
                        prizes[neighbor] = 5.0
                        count += 1
                        if count >= max_neighbors_per_seed:
                            break

        # 5. Extract subgraph
        pcst_used = True
        try:
            subgraph = self.pcst_solver.extract_subgraph(
                self.unified_graph,
                seed_entities,
                prizes
            )
        except Exception as e:
            print(f"⚠ PCST extraction failed: {e}, using BFS fallback")
            subgraph = self.pcst_solver._bfs_fallback(
                self.unified_graph,
                seed_entities,
                self.config.pcst_budget
            )
            pcst_used = False

        # 6. Calculate timing
        end_time = time.time()
        retrieval_time_ms = (end_time - start_time) * 1000

        # 7. Build result
        result = RetrievedSubgraph(
            subgraph=subgraph,
            question=question,
            seed_entities=seed_entities,
            similarity_scores=similarity_scores,
            num_nodes=len(subgraph),
            num_edges=subgraph.number_of_edges(),
            retrieval_time_ms=retrieval_time_ms,
            pcst_used=pcst_used
        )

        return result

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
            local_budget=config.pcst_local_budget
        )
        print(f"✓ PCST solver ready (cost: {config.pcst_cost}, budget: {config.pcst_budget} nodes, local: {config.pcst_local_budget})")

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
