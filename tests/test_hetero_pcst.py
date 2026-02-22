import pytest
import torch
import numpy as np
from torch_geometric.data import HeteroData
from src.retrieval.hetero_pcst import HeteroPCST
from src.config import BenchmarkConfig


def make_chain_graph(n=20):
    data = HeteroData()
    rng = np.random.RandomState(42)
    data["object"].x = torch.tensor(rng.randn(n, 384), dtype=torch.float32)
    data["object"].frame_id = torch.arange(n)

    src = list(range(n - 1))
    dst = list(range(1, n))
    data["object", "spatial_rel", "object"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    even = list(range(0, n, 2))
    t_src = even[:-1]
    t_dst = even[1:]
    data["object", "temporal", "object"].edge_index = torch.tensor([t_src, t_dst], dtype=torch.long)
    data["object", "temporal", "object"].edge_attr = torch.ones(len(t_src), 2, dtype=torch.float32)
    data.video_id = "test"
    return data


def test_hetero_pcst_returns_heterodata():
    config = BenchmarkConfig(pcst_budget=10)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 1.0, 5: 0.8, 10: 0.6}
    sub = solver.extract(data, prizes)
    assert isinstance(sub, HeteroData)


def test_hetero_pcst_respects_budget():
    config = BenchmarkConfig(pcst_budget=10)
    solver = HeteroPCST(config)
    data = make_chain_graph(50)
    prizes = {i: 1.0 for i in range(0, 50, 5)}
    sub = solver.extract(data, prizes)
    assert sub["object"].num_nodes <= 10


def test_hetero_pcst_includes_seed_nodes():
    config = BenchmarkConfig(pcst_budget=20)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 1.0, 5: 0.8}
    sub = solver.extract(data, prizes)
    selected = sub.selected_nodes.tolist()
    assert 0 in selected
    assert 5 in selected


def test_hetero_pcst_preserves_edge_types():
    config = BenchmarkConfig(pcst_budget=15)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 1.0, 2: 0.8, 4: 0.6}
    sub = solver.extract(data, prizes)
    assert ("object", "spatial_rel", "object") in sub.edge_types
    assert ("object", "temporal", "object") in sub.edge_types


def test_hetero_pcst_bfs_fallback():
    config = BenchmarkConfig(pcst_budget=5)
    solver = HeteroPCST(config)
    data = make_chain_graph(10)
    prizes = {0: 1.0}
    sub = solver.extract(data, prizes)
    assert sub["object"].num_nodes > 0
    assert sub["object"].num_nodes <= 5
