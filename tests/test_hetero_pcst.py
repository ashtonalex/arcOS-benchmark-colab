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


def test_hetero_pcst_verbose_logs_to_stdout(capsys):
    """verbose=True prints PCST diagnostics without crashing."""
    config = BenchmarkConfig(pcst_budget=10)
    solver = HeteroPCST(config, verbose=True)
    data = make_chain_graph(10)
    prizes = {0: 0.8, 5: 0.6}
    solver.extract(data, prizes)
    captured = capsys.readouterr()
    assert "nodes" in captured.out
    assert "edges" in captured.out
    assert "prizes" in captured.out
    assert "PCST output" in captured.out


def test_hetero_pcst_edges_are_int32():
    """Edge array passed to pcst_fast must be int32 (Colab pcst_fast build requirement).

    This test patches pcst_fast.pcst_fast to capture the actual dtype of the edges
    array passed in, so we can assert int32 even on Windows where int64 also works.
    """
    import unittest.mock as mock
    config = BenchmarkConfig(pcst_budget=20)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 0.8, 10: 0.6}

    captured_dtype = {}

    def fake_pcst(edges, prizes, costs, root, n_clusters, pruning, verbosity):
        captured_dtype["edges"] = edges.dtype
        # Return a valid result: first two nodes
        import numpy as np
        return np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64)

    import src.retrieval.hetero_pcst as pcst_module
    with mock.patch.object(pcst_module, "pcst_fast", create=True) as mock_lib:
        pcst_module.HAS_PCST = True
        mock_lib.pcst_fast.side_effect = fake_pcst
        solver.extract(data, prizes)

    assert captured_dtype["edges"] == np.int32, (
        f"Expected int32 edges for pcst_fast compatibility, got {captured_dtype['edges']}"
    )
