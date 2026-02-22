"""Heterogeneous GATv2 encoder using HeteroConv for video scene graphs."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv
from src.config import BenchmarkConfig


class HeteroGATv2Encoder(nn.Module):
    """Type-aware GATv2 encoder for HeteroData scene graphs."""

    SPATIAL_EDGE = ("object", "spatial_rel", "object")
    TEMPORAL_EDGE = ("object", "temporal", "object")

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim
        self.num_layers = config.gnn_num_layers
        self.num_heads = config.gnn_num_heads
        self.dropout = config.gnn_dropout
        self.input_dim = config.embedding_dim

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = self.hidden_dim
            head_dim = self.hidden_dim // self.num_heads
            conv = HeteroConv(
                {
                    self.SPATIAL_EDGE: GATv2Conv(
                        in_dim, head_dim, heads=self.num_heads,
                        concat=True, dropout=self.dropout, add_self_loops=False,
                    ),
                    self.TEMPORAL_EDGE: GATv2Conv(
                        in_dim, head_dim, heads=self.num_heads,
                        concat=True, dropout=self.dropout, add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.dropout_layer = nn.Dropout(self.dropout)
        self.pool_gate = nn.Linear(self.hidden_dim, 1)

    def forward(self, data: HeteroData, query_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(data["object"].x)
        q = self.query_proj(query_embedding.unsqueeze(0))
        x = x + q.expand_as(x)

        edge_index_dict = {}
        for etype in [self.SPATIAL_EDGE, self.TEMPORAL_EDGE]:
            if etype in data.edge_types and data[etype].edge_index.shape[1] > 0:
                edge_index_dict[etype] = data[etype].edge_index

        all_attn_weights = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if not edge_index_dict:
                break
            x_out = conv({"object": x}, edge_index_dict)
            x_new = x_out.get("object", x)
            x = norm(x + self.dropout_layer(x_new))
            layer_attn = self._extract_attention(conv, data, edge_index_dict)
            if layer_attn is not None:
                all_attn_weights.append(layer_attn)

        num_nodes = x.shape[0]
        if all_attn_weights:
            attn_scores = self._aggregate_attention(all_attn_weights, num_nodes, edge_index_dict)
        else:
            attn_scores = torch.ones(num_nodes, device=x.device) / num_nodes

        gate = torch.sigmoid(self.pool_gate(x)).squeeze(-1)
        graph_emb = (x * gate.unsqueeze(-1)).sum(dim=0) / (gate.sum() + 1e-8)

        return x, attn_scores, graph_emb

    def _extract_attention(self, conv, data, edge_index_dict):
        attn_per_edge = {}
        for etype, subconv in conv.convs.items():
            if etype in edge_index_dict:
                if hasattr(subconv, '_alpha') and subconv._alpha is not None:
                    attn_per_edge[etype] = subconv._alpha.detach()
        return attn_per_edge if attn_per_edge else None

    def _aggregate_attention(self, all_attn_weights, num_nodes, edge_index_dict):
        node_scores = torch.zeros(num_nodes)
        for layer_attn in all_attn_weights:
            for etype, alpha in layer_attn.items():
                if etype in edge_index_dict:
                    ei = edge_index_dict[etype]
                    avg_alpha = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
                    dst = ei[1]
                    for j in range(len(dst)):
                        d = int(dst[j])
                        if d < num_nodes:
                            node_scores[d] += float(avg_alpha[j])
        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()
        return node_scores
