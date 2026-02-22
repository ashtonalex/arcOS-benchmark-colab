import pytest
import numpy as np
import torch
from torch_geometric.data import HeteroData
from src.retrieval.video_index import VideoIndex


def make_mock_heterodata(num_nodes=20, dim=384):
    data = HeteroData()
    rng = np.random.RandomState(42)
    data["object"].x = torch.tensor(rng.randn(num_nodes, dim), dtype=torch.float32)
    data["object"].frame_id = torch.arange(num_nodes)
    data.video_id = "test_vid"
    return data


def test_video_index_build():
    data = make_mock_heterodata(num_nodes=20)
    index = VideoIndex()
    index.build(data)
    assert len(index) == 20


def test_video_index_search_returns_top_k():
    data = make_mock_heterodata(num_nodes=50)
    index = VideoIndex()
    index.build(data)
    query = np.random.randn(384).astype(np.float32)
    results = index.search(query, k=10)
    assert len(results) == 10
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_video_index_search_scores_descending():
    data = make_mock_heterodata(num_nodes=50)
    index = VideoIndex()
    index.build(data)
    query = np.random.randn(384).astype(np.float32)
    results = index.search(query, k=10)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_video_index_k_larger_than_nodes():
    data = make_mock_heterodata(num_nodes=5)
    index = VideoIndex()
    index.build(data)
    query = np.random.randn(384).astype(np.float32)
    results = index.search(query, k=20)
    assert len(results) == 5
