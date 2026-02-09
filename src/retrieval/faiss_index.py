"""
FAISS index construction and k-NN search.
"""

from typing import Dict, List, Tuple
import numpy as np
import faiss
from pathlib import Path
import pickle


class EntityIndex:
    """FAISS index for k-NN entity retrieval."""

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        # Use IndexFlatIP for exact inner product search
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.id_to_name = {}  # FAISS integer ID -> entity name
        self.name_to_id = {}  # entity name -> FAISS integer ID

    def build(self, entity_embeddings: Dict[str, np.ndarray]):
        """
        Build index from entity embeddings.

        Args:
            entity_embeddings: Dictionary mapping entity_name -> embedding vector
        """
        if not entity_embeddings:
            raise ValueError("Cannot build index from empty embeddings")

        # Create consistent ordering
        entity_names = sorted(entity_embeddings.keys())
        embeddings_array = np.array([entity_embeddings[name] for name in entity_names], dtype=np.float32)

        # Normalize vectors for cosine similarity (L2 norm)
        # After normalization, inner product = cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add to FAISS index
        self.index.add(embeddings_array)

        # Store mappings
        self.id_to_name = {i: name for i, name in enumerate(entity_names)}
        self.name_to_id = {name: i for i, name in enumerate(entity_names)}

        print(f"✓ Built FAISS index with {len(entity_names)} entities")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k nearest entities to query.

        Args:
            query_embedding: Query vector (shape: (embedding_dim,))
            k: Number of neighbors to retrieve

        Returns:
            List of (entity_name, similarity_score) tuples, sorted by score descending
        """
        if self.index.ntotal == 0:
            return []

        # Ensure query is 2D array with shape (1, embedding_dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query vector
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Clamp k to index size
        k = min(k, self.index.ntotal)

        # Search
        similarities, indices = self.index.search(query_embedding, k)

        # Map FAISS IDs back to entity names
        results = []
        for i in range(k):
            entity_id = int(indices[0, i])
            similarity = float(similarities[0, i])
            entity_name = self.id_to_name[entity_id]
            results.append((entity_name, similarity))

        return results

    def save(self, index_path: Path, mapping_path: Path):
        """
        Serialize index to disk.

        Args:
            index_path: Path to save FAISS index
            mapping_path: Path to save ID mappings
        """
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save mappings
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'id_to_name': self.id_to_name,
                'name_to_id': self.name_to_id,
                'embedding_dim': self.embedding_dim
            }, f)

        print(f"✓ Saved FAISS index to {index_path}")
        print(f"✓ Saved entity mappings to {mapping_path}")

    def load(self, index_path: Path, mapping_path: Path):
        """
        Load index from disk.

        Args:
            index_path: Path to FAISS index file
            mapping_path: Path to ID mappings file
        """
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load mappings
        with open(mapping_path, 'rb') as f:
            data = pickle.load(f)
            self.id_to_name = data['id_to_name']
            self.name_to_id = data['name_to_id']
            self.embedding_dim = data['embedding_dim']

        print(f"✓ Loaded FAISS index from {index_path}")
        print(f"  - {len(self.id_to_name)} entities indexed")

    def __len__(self) -> int:
        """Return number of entities in index."""
        return self.index.ntotal
