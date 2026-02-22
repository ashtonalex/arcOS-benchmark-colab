import pytest
import torch
from torch_geometric.data import HeteroData
from src.gnn.hetero_encoder import HeteroGATv2Encoder
from src.config import BenchmarkConfig


def make_test_subgraph(num_nodes=10, dim=384):
    data = HeteroData()
    data["object"].x = torch.randn(num_nodes, dim)
    data["object"].frame_id = torch.arange(num_nodes)
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    data["object", "spatial_rel", "object"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    even = list(range(0, num_nodes, 2))
    data["object", "temporal", "object"].edge_index = torch.tensor([even[:-1], even[1:]], dtype=torch.long)
    return data


def test_encoder_output_shapes():
    config = BenchmarkConfig(gnn_hidden_dim=256, gnn_num_layers=3, gnn_num_heads=4)
    encoder = HeteroGATv2Encoder(config)
    data = make_test_subgraph(num_nodes=10)
    query_emb = torch.randn(384)
    node_emb, attn_scores, graph_emb = encoder(data, query_emb)
    assert node_emb.shape == (10, 256)
    assert attn_scores.shape == (10,)
    assert graph_emb.shape == (256,)


def test_encoder_attention_normalized():
    config = BenchmarkConfig()
    encoder = HeteroGATv2Encoder(config)
    data = make_test_subgraph(num_nodes=10)
    query_emb = torch.randn(384)
    _, attn_scores, _ = encoder(data, query_emb)
    assert attn_scores.min() >= 0.0
    assert attn_scores.max() <= 1.0


def test_encoder_no_temporal_edges():
    config = BenchmarkConfig()
    encoder = HeteroGATv2Encoder(config)
    data = HeteroData()
    data["object"].x = torch.randn(5, 384)
    data["object", "spatial_rel", "object"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data["object", "temporal", "object"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    query_emb = torch.randn(384)
    node_emb, attn_scores, graph_emb = encoder(data, query_emb)
    assert node_emb.shape == (5, 256)


def test_encoder_gradient_flows():
    config = BenchmarkConfig()
    encoder = HeteroGATv2Encoder(config)
    data = make_test_subgraph(num_nodes=8)
    query_emb = torch.randn(384)
    node_emb, attn_scores, graph_emb = encoder(data, query_emb)
    loss = graph_emb.sum()
    loss.backward()
    for param in encoder.parameters():
        if param.requires_grad:
            assert param.grad is not None
