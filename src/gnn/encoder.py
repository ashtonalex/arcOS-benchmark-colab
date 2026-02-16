"""
GNN encoder architectures: GATv2 and GraphSAGE.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv

from ..config import BenchmarkConfig


class GATv2Encoder(nn.Module):
    """
    Graph Attention Network v2 encoder with query conditioning.

    Architecture:
        Input (384-dim) → Projection (hidden_dim) →
        GATv2Conv Layer 1 (multi-head) → Dropout → LayerNorm → Residual →
        GATv2Conv Layer 2 → ... → GATv2Conv Layer N →
        Output: (node_embeddings, attention_weights)
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Args:
            config: BenchmarkConfig with GNN hyperparameters
        """
        super().__init__()
        self.config = config
        self.input_dim = 384  # Phase 2 embedding dimension
        self.hidden_dim = config.gnn_hidden_dim
        self.num_layers = config.gnn_num_layers
        self.num_heads = config.gnn_num_heads
        self.dropout = config.gnn_dropout

        # Input projection (384 → hidden_dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Query projection (384 → hidden_dim) for conditioning
        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(self.num_layers):
            # Input dimension for first layer is hidden_dim
            # Output from multi-head is hidden_dim (concat=False uses mean)
            gat = GATv2Conv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=self.num_heads,
                dropout=self.dropout,
                edge_dim=self.input_dim,  # Edge features are 384-dim
                concat=False,  # Average heads instead of concat
                add_self_loops=True,
            )
            self.gat_layers.append(gat)
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GATv2 encoder.

        Args:
            x: Node features [num_nodes, 384]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, 384]
            query_embedding: Query embedding [384]

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            attention_weights: [num_nodes] (averaged across heads and layers)
        """
        # Project input features
        h = self.input_proj(x)  # [num_nodes, hidden_dim]

        # Query conditioning: broadcast and add to all nodes
        query_proj = self.query_proj(query_embedding.view(-1))  # [hidden_dim]
        h = h + query_proj.unsqueeze(0)  # Broadcasting

        # Collect attention weights across layers
        all_attention_weights = []

        # Pass through GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_in = h

            # GAT layer returns (output, (edge_index, attention_weights))
            h_out, (_, attn) = gat(
                h, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )

            # Dropout
            h_out = self.dropout_layer(h_out)

            # Layer normalization
            h_out = norm(h_out)

            # Residual connection
            h = h_in + h_out

            # ReLU activation (except last layer)
            if i < self.num_layers - 1:
                h = F.relu(h)

            # Collect attention weights
            # attn is [num_edges, num_heads], average to [num_edges]
            attn_avg = attn.mean(dim=1)  # [num_edges]
            all_attention_weights.append(attn_avg)

        # Aggregate attention weights to node level
        # For each node, average the attention of all incoming edges across layers
        node_attention = self._aggregate_attention_to_nodes(
            edge_index, all_attention_weights, x.size(0)
        )

        return h, node_attention

    def _aggregate_attention_to_nodes(
        self,
        edge_index: torch.Tensor,
        attention_weights_list: list,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Aggregate edge attention weights to node-level scores.

        For each node, compute the mean attention of all incoming edges,
        averaged across all layers.

        Args:
            edge_index: [2, num_edges]
            attention_weights_list: List of [num_edges] tensors (one per layer)
            num_nodes: Number of nodes

        Returns:
            node_attention: [num_nodes]
        """
        device = edge_index.device

        # Average attention across layers
        if len(attention_weights_list) == 0:
            return torch.ones(num_nodes, device=device) / num_nodes

        avg_attention = torch.stack(attention_weights_list, dim=0).mean(
            dim=0
        )  # [num_edges]

        # Aggregate to target nodes (edge_index[1] are destination nodes)
        node_attention = torch.zeros(num_nodes, device=device)
        node_counts = torch.zeros(num_nodes, device=device)

        target_nodes = edge_index[1]  # [num_edges]

        # Scatter add attention to target nodes
        node_attention.scatter_add_(0, target_nodes, avg_attention)
        node_counts.scatter_add_(0, target_nodes, torch.ones_like(avg_attention))

        # Average by count (avoid division by zero)
        node_counts = torch.clamp(node_counts, min=1.0)
        node_attention = node_attention / node_counts

        # Normalize to sum to 1
        node_attention = node_attention / (node_attention.sum() + 1e-8)

        return node_attention


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder with learned attention head.

    Alternative to GATv2 for ablation studies. Uses mean aggregation
    and adds a learned attention layer on top for interface compatibility.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Args:
            config: BenchmarkConfig with GNN hyperparameters
        """
        super().__init__()
        self.config = config
        self.input_dim = 384
        self.hidden_dim = config.gnn_hidden_dim
        self.num_layers = config.gnn_num_layers
        self.dropout = config.gnn_dropout

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Query projection
        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # SAGE layers
        self.sage_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            sage = SAGEConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                aggr="mean",
            )
            self.sage_layers.append(sage)
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))

        # Attention head for interface compatibility
        self.attention_head = nn.Linear(self.hidden_dim, 1)

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GraphSAGE encoder.

        Args:
            x: Node features [num_nodes, 384]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features (not used by SAGE)
            query_embedding: Query embedding [384]

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            attention_weights: [num_nodes]
        """
        # Project input features
        h = self.input_proj(x)

        # Query conditioning
        query_proj = self.query_proj(query_embedding.view(-1))
        h = h + query_proj.unsqueeze(0)

        # Pass through SAGE layers with residual connections
        for i, (sage, norm) in enumerate(zip(self.sage_layers, self.layer_norms)):
            h_in = h

            # SAGE layer
            h_out = sage(h, edge_index)

            # Dropout
            h_out = self.dropout_layer(h_out)

            # Layer normalization
            h_out = norm(h_out)

            # Residual connection
            h = h_in + h_out

            # ReLU activation (except last layer)
            if i < self.num_layers - 1:
                h = F.relu(h)

        # Compute attention weights using learned head
        attention_logits = self.attention_head(h).squeeze(-1)  # [num_nodes]
        attention_weights = F.softmax(attention_logits, dim=0)  # [num_nodes]

        return h, attention_weights
