"""
Text embedding module using Sentence-Transformers.
"""

from typing import List, Dict
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch


class TextEmbedder:
    """Generates sentence embeddings for entities, relations, and queries."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Load Sentence-Transformer model.

        Args:
            model_name: HuggingFace model identifier
            device: "cuda" or "cpu"
        """
        # Auto-detect device if cuda requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU for embeddings")
            device = "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"✓ Loaded embedding model: {model_name}")
        print(f"  - Device: {self.device}")
        print(f"  - Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Batch-encode texts to embeddings.

        Args:
            texts: List of strings to embed
            batch_size: Batch size for encoding
            show_progress: Show tqdm progress bar

        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize separately for FAISS
        )

        return embeddings

    def embed_graph_entities(self, G: nx.DiGraph, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Embed all node entities in graph.

        Args:
            G: NetworkX graph
            batch_size: Batch size for encoding

        Returns:
            Dictionary mapping node_name -> embedding vector
        """
        # Extract unique node names
        node_names = list(G.nodes())

        if not node_names:
            return {}

        print(f"Embedding {len(node_names)} entities...")

        # Batch-encode all entities
        embeddings = self.embed_texts(node_names, batch_size=batch_size, show_progress=True)

        # Create mapping
        entity_embeddings = {
            node: embeddings[i]
            for i, node in enumerate(node_names)
        }

        print(f"✓ Embedded {len(entity_embeddings)} entities")

        return entity_embeddings

    def embed_relations(self, G: nx.DiGraph, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Embed all unique relation strings.

        Args:
            G: NetworkX graph with 'relation' edge attributes
            batch_size: Batch size for encoding

        Returns:
            Dictionary mapping relation -> embedding vector
        """
        # Extract unique relations
        relations = set()
        for _, _, data in G.edges(data=True):
            if 'relation' in data:
                relations.add(data['relation'])

        relation_list = sorted(list(relations))

        if not relation_list:
            return {}

        print(f"Embedding {len(relation_list)} unique relations...")

        # Batch-encode all relations
        embeddings = self.embed_texts(relation_list, batch_size=batch_size, show_progress=True)

        # Create mapping
        relation_embeddings = {
            rel: embeddings[i]
            for i, rel in enumerate(relation_list)
        }

        print(f"✓ Embedded {len(relation_embeddings)} relations")

        return relation_embeddings
