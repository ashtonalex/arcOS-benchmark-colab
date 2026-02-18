"""
Graph pooling layers for aggregating node embeddings to graph-level representations.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

from ..config import BenchmarkConfig


class AttentionPooling(nn.Module):
    """
    Attention-based graph pooling using PyG GlobalAttention.

    Learns to weight nodes by importance and aggregate to graph-level embedding.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Args:
            config: BenchmarkConfig with GNN hyperparameters
        """
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim

        # Gate network: computes attention scores
        gate_nn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Feature network: transforms embeddings before pooling
        feat_nn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.global_attention = GlobalAttention(gate_nn=gate_nn, nn=feat_nn)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool node embeddings to graph-level representation.

        Args:
            x: Node embeddings [num_nodes_total, hidden_dim]
            batch: Batch assignment [num_nodes_total] (which graph each node belongs to)

        Returns:
            graph_embedding: [batch_size, hidden_dim]
            attention_scores: [num_nodes_total] (normalized per graph)
        """
        # GlobalAttention returns (pooled, attention_weights)
        graph_embedding = self.global_attention(x, batch)

        # Extract attention scores from gate network
        attention_scores = self.global_attention.gate_nn(x).squeeze(-1)  # [num_nodes]

        # Softmax per graph
        attention_scores = self._softmax_per_graph(attention_scores, batch)

        return graph_embedding, attention_scores

    def _softmax_per_graph(
        self, scores: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply softmax normalization per graph in batch.

        Args:
            scores: Raw scores [num_nodes_total]
            batch: Batch assignment [num_nodes_total]

        Returns:
            normalized_scores: [num_nodes_total]
        """
        return pyg_softmax(scores, batch)


class MeanPooling(nn.Module):
    """
    Simple mean pooling baseline.

    Averages all node embeddings to get graph-level representation.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool node embeddings via mean.

        Args:
            x: Node embeddings [num_nodes_total, hidden_dim]
            batch: Batch assignment [num_nodes_total]

        Returns:
            graph_embedding: [batch_size, hidden_dim]
            attention_scores: [num_nodes_total] (uniform weights)
        """
        graph_embedding = global_mean_pool(x, batch)

        # Uniform attention (all nodes equally important), normalized per graph
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1
        counts = torch.zeros(num_graphs, device=x.device)
        counts.scatter_add_(0, batch, torch.ones(num_nodes, device=x.device))
        attention_scores = 1.0 / counts[batch]

        return graph_embedding, attention_scores


class MaxPooling(nn.Module):
    """
    Max pooling baseline.

    Takes element-wise maximum across node embeddings.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool node embeddings via max.

        Args:
            x: Node embeddings [num_nodes_total, hidden_dim]
            batch: Batch assignment [num_nodes_total]

        Returns:
            graph_embedding: [batch_size, hidden_dim]
            attention_scores: [num_nodes_total] (binary: 1 if max, 0 otherwise)
        """
        graph_embedding = global_max_pool(x, batch)

        # Binary attention: 1 for nodes that match graph max in any dimension
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1
        attention_scores = torch.zeros(num_nodes, device=x.device)

        for graph_id in range(num_graphs):
            mask = batch == graph_id
            graph_nodes = x[mask]
            max_vals = graph_nodes.max(dim=0)[0]  # [hidden_dim]
            matches = (graph_nodes == max_vals.unsqueeze(0)).any(dim=1).float()
            attention_scores[mask] = matches

        # Normalize per graph
        sum_per_graph = torch.zeros(num_graphs, device=x.device)
        sum_per_graph.scatter_add_(0, batch, attention_scores)
        safe_sum = torch.clamp(sum_per_graph, min=1.0)
        attention_scores = attention_scores / safe_sum[batch]

        return graph_embedding, attention_scores
