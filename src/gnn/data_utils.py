"""
Data utilities for converting NetworkX graphs to PyTorch Geometric format.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx

from ..config import BenchmarkConfig
from ..retrieval.embeddings import TextEmbedder


@dataclass
class GNNOutput:
    """Output from GNN encoder."""

    node_embeddings: torch.Tensor  # [num_nodes, hidden_dim]
    attention_scores: Dict[str, float]  # {node_name: score}
    graph_embedding: torch.Tensor  # [hidden_dim]


class SubgraphConverter:
    """
    Converts NetworkX subgraphs to PyTorch Geometric Data objects.

    Uses entity/relation embeddings from Phase 2 and question embeddings
    from TextEmbedder for query conditioning.
    """

    def __init__(
        self,
        entity_embeddings: Dict[str, np.ndarray],
        relation_embeddings: Dict[str, np.ndarray],
        text_embedder: TextEmbedder,
        config: BenchmarkConfig,
    ):
        """
        Args:
            entity_embeddings: Dict mapping entity names to 384-dim embeddings
            relation_embeddings: Dict mapping relation names to 384-dim embeddings
            text_embedder: TextEmbedder for encoding questions
            config: BenchmarkConfig instance
        """
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.text_embedder = text_embedder
        self.config = config
        self.embedding_dim = 384  # Phase 2 uses all-MiniLM-L6-v2

    def convert(
        self,
        subgraph: nx.DiGraph,
        question: str,
        answer_entities: Optional[List[str]] = None,
    ) -> Data:
        """
        Convert NetworkX subgraph to PyG Data object.

        Args:
            subgraph: NetworkX DiGraph from retrieval
            question: Natural language question text
            answer_entities: List of answer entity names (for training labels)

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, embedding_dim]
                - edge_index: COO format edges [2, num_edges]
                - edge_attr: Edge features [num_edges, embedding_dim]
                - query_embedding: Question embedding [embedding_dim]
                - y: Node labels [num_nodes] (1 if answer, 0 otherwise)
                - node_names: List of node names (for attention mapping)
        """
        # Get sorted node list for deterministic ordering
        nodes = sorted(subgraph.nodes())
        num_nodes = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Build node features from entity embeddings
        node_features = []
        for node in nodes:
            if node in self.entity_embeddings:
                node_features.append(self.entity_embeddings[node])
            else:
                # Unknown entity: use zero vector
                node_features.append(np.zeros(self.embedding_dim))

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Build edge index (COO format) and edge features
        edge_list = []
        edge_features = []

        for src, dst, edge_data in subgraph.edges(data=True):
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])

            # Get relation embedding
            relation = edge_data.get("relation", "")
            if relation in self.relation_embeddings:
                edge_features.append(self.relation_embeddings[relation])
            else:
                # Unknown relation: use zero vector
                edge_features.append(np.zeros(self.embedding_dim))

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        else:
            # Empty graph edge case
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.embedding_dim), dtype=torch.float32)

        # Encode question
        query_embedding = self.text_embedder.embed_texts([question])[0]
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)  # [1, dim] for proper PyG batching

        # Build labels if answer entities provided
        if answer_entities is not None:
            y = torch.zeros(num_nodes, dtype=torch.float32)
            for i, node in enumerate(nodes):
                if node in answer_entities:
                    y[i] = 1.0
        else:
            y = None

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            query_embedding=query_embedding,
            y=y,
            node_names=nodes,  # Store for attention mapping
        )

        return data

    def convert_batch(
        self,
        subgraphs: List[nx.DiGraph],
        questions: List[str],
        answer_entities_list: Optional[List[List[str]]] = None,
    ) -> List[Data]:
        """
        Convert multiple subgraphs to PyG Data objects.

        Args:
            subgraphs: List of NetworkX DiGraphs
            questions: List of question texts
            answer_entities_list: List of answer entity lists (optional)

        Returns:
            List of PyG Data objects
        """
        if answer_entities_list is None:
            answer_entities_list = [None] * len(subgraphs)

        data_list = []
        for subgraph, question, answer_entities in zip(
            subgraphs, questions, answer_entities_list
        ):
            data = self.convert(subgraph, question, answer_entities)
            data_list.append(data)

        return data_list
