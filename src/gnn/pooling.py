"""
Graph pooling layers for aggregating node embeddings to graph-level representations.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

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
        # Compute softmax within each graph
        max_scores = torch.zeros_like(scores)
        for graph_id in batch.unique():
            mask = batch == graph_id
            max_scores[mask] = scores[mask].max()

        exp_scores = torch.exp(scores - max_scores)

        # Sum per graph
        sum_exp = torch.zeros(batch.max().item() + 1, device=scores.device)
        sum_exp.scatter_add_(0, batch, exp_scores)

        # Normalize
        normalized = exp_scores / sum_exp[batch]

        return normalized


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

        # Uniform attention (all nodes equally important)
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1
        attention_scores = torch.ones(num_nodes, device=x.device)

        # Normalize per graph
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            count = mask.sum().item()
            attention_scores[mask] = 1.0 / count

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

        # Binary attention: 1 for nodes that contributed to max
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1
        attention_scores = torch.zeros(num_nodes, device=x.device)

        for graph_id in range(num_graphs):
            mask = batch == graph_id
            graph_nodes = x[mask]
            max_vals = graph_nodes.max(dim=0)[0]  # [hidden_dim]

            # Mark nodes that match max in any dimension
            for i, node_emb in enumerate(graph_nodes):
                if torch.any(node_emb == max_vals):
                    attention_scores[mask.nonzero()[i]] = 1.0

        # Normalize
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            total = attention_scores[mask].sum()
            if total > 0:
                attention_scores[mask] /= total

        return graph_embedding, attention_scores
