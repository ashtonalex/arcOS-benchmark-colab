"""
Text embedding module using Sentence-Transformers.
"""

from typing import List, Dict
from collections import defaultdict
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Freebase relation prefixes that are administrative/meta (no semantic value).
# Kept in sync with graph_builder._JUNK_RELATION_PREFIXES — those filter
# triples at graph construction; these filter during enrichment text
# generation for any relations that survive into the graph.
_SKIP_RELATION_PREFIXES = (
    'freebase.valuenotation',
    'freebase.type_profile',
    'type.object',
    'kg.object_profile',
    'rdf-schema#',
)


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

    def _clean_relation(self, relation: str) -> str:
        """Extract meaningful name from Freebase dot-notation relation."""
        parts = relation.split('.')
        return parts[-1].replace('_', ' ')

    def _build_enriched_texts(
        self, G: nx.DiGraph, node_names: List[str], max_relations: int = 8
    ) -> List[str]:
        """
        Build enriched text representations by appending relation context.

        Transforms bare entity names like "Cleveland" into
        "Cleveland | containedby, time zone, adjoins" so the embedding
        model captures what each entity IS, not just its name.

        Args:
            G: NetworkX graph with 'relation' edge attributes
            node_names: Ordered list of node names to enrich
            max_relations: Max relation tags per entity

        Returns:
            List of enriched text strings (same order as node_names)
        """
        # Single pass over edges: collect relations per node
        node_relations: Dict[str, set] = defaultdict(set)
        for u, v, data in G.edges(data=True):
            rel = data.get('relation', '')
            if not rel or any(rel.startswith(p) for p in _SKIP_RELATION_PREFIXES):
                continue
            node_relations[u].add(rel)
            node_relations[v].add(rel)

        # Build enriched text for each node
        texts = []
        enriched_count = 0
        for node in node_names:
            relations = node_relations.get(node)
            if relations:
                cleaned = sorted(set(
                    self._clean_relation(r) for r in relations
                ))[:max_relations]
                texts.append(f"{node} | {', '.join(cleaned)}")
                enriched_count += 1
            else:
                texts.append(node)

        print(f"  Enriched {enriched_count}/{len(node_names)} entities with relation context")
        # Show samples
        for t in texts[:3]:
            print(f"    sample: {t[:100]}")

        return texts

    def embed_graph_entities(self, G: nx.DiGraph, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Embed all node entities in graph using relation-enriched text.

        Each entity is embedded as "entity_name | rel1, rel2, ..." rather
        than the bare name, giving the embedding model semantic context.

        Args:
            G: NetworkX graph
            batch_size: Batch size for encoding

        Returns:
            Dictionary mapping node_name -> embedding vector
        """
        node_names = list(G.nodes())

        if not node_names:
            return {}

        print(f"Embedding {len(node_names)} entities (with relation enrichment)...")
        texts = self._build_enriched_texts(G, node_names)

        embeddings = self.embed_texts(texts, batch_size=batch_size, show_progress=True)

        entity_embeddings = {
            node: embeddings[i]
            for i, node in enumerate(node_names)
        }

        print(f"✓ Embedded {len(entity_embeddings)} enriched entities")

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

        # Clean Freebase dot-notation into natural language before embedding
        # e.g. "people.person.sibling_s" -> "sibling s"
        # This ensures relation embeddings are semantically comparable to
        # natural language query embeddings for edge weight normalization.
        cleaned_list = [self._clean_relation(r) for r in relation_list]

        # Batch-encode cleaned relations
        embeddings = self.embed_texts(cleaned_list, batch_size=batch_size, show_progress=True)

        # Create mapping
        relation_embeddings = {
            rel: embeddings[i]
            for i, rel in enumerate(relation_list)
        }

        print(f"✓ Embedded {len(relation_embeddings)} relations")

        return relation_embeddings
