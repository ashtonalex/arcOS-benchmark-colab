"""Per-video FAISS index for k-NN seed selection on HeteroData."""

from typing import List, Tuple
import numpy as np
import faiss
from torch_geometric.data import HeteroData


class VideoIndex:
    """FAISS index for a single video's object node embeddings."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self._num_nodes = 0

    def build(self, data: HeteroData) -> None:
        embeddings = data["object"].x.numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self._num_nodes = len(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        k = min(k, self._num_nodes)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        scores, indices = self.index.search(query, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def __len__(self) -> int:
        return self._num_nodes
