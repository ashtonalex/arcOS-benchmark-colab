"""Orchestrates per-video retrieval: embed query -> k-NN -> PCST -> subgraph."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig
from src.retrieval.video_index import VideoIndex
from src.retrieval.hetero_pcst import HeteroPCST


@dataclass
class RetrievalResult:
    subgraph: HeteroData
    question: str
    seed_indices: List[int]
    similarity_scores: Dict[int, float]
    num_nodes: int
    num_edges: int
    retrieval_time_ms: float
    pcst_used: bool


class VideoRetriever:
    """Retrieves relevant subgraph from a video scene graph given a question."""

    def __init__(self, config: BenchmarkConfig, embedder=None):
        self.config = config
        self.embedder = embedder
        self.pcst = HeteroPCST(config)

    def retrieve(
        self,
        question: str,
        scene_graph: HeteroData,
        answer_nodes: Optional[List[int]] = None,
    ) -> RetrievalResult:
        start = time.time()
        query_emb = self.embedder.embed_texts([question])[0]
        index = VideoIndex(embedding_dim=self.config.embedding_dim)
        index.build(scene_graph)
        results = index.search(query_emb, k=self.config.top_k_seeds)
        seed_indices = [r[0] for r in results]
        prizes = {idx: score for idx, score in results}
        pcst_used = True
        try:
            subgraph = self.pcst.extract(scene_graph, prizes)
        except Exception:
            pcst_used = False
            subgraph = self.pcst.extract(scene_graph, prizes)
        elapsed = (time.time() - start) * 1000
        total_edges = sum(subgraph[et].edge_index.shape[1] for et in subgraph.edge_types)
        return RetrievalResult(
            subgraph=subgraph,
            question=question,
            seed_indices=seed_indices,
            similarity_scores=prizes,
            num_nodes=subgraph["object"].num_nodes,
            num_edges=total_edges,
            retrieval_time_ms=elapsed,
            pcst_used=pcst_used,
        )
