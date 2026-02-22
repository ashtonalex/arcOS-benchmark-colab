import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch_geometric.data import HeteroData
from src.retrieval.video_retriever import VideoRetriever, RetrievalResult
from src.config import BenchmarkConfig


def make_mock_scene_graph(num_nodes=30, dim=384):
    data = HeteroData()
    rng = np.random.RandomState(42)
    data["object"].x = torch.tensor(rng.randn(num_nodes, dim), dtype=torch.float32)
    data["object"].frame_id = torch.arange(num_nodes)
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    data["object", "spatial_rel", "object"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    even = list(range(0, num_nodes, 2))
    data["object", "temporal", "object"].edge_index = torch.tensor([even[:-1], even[1:]], dtype=torch.long)
    data["object", "temporal", "object"].edge_attr = torch.ones(len(even) - 1, 2, dtype=torch.float32)
    data.video_id = "test_vid"
    return data


def test_retriever_returns_result():
    config = BenchmarkConfig(pcst_budget=10, top_k_seeds=5)
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = np.random.randn(1, 384).astype(np.float32)
    retriever = VideoRetriever(config, embedder=mock_embedder)
    scene_graph = make_mock_scene_graph()
    result = retriever.retrieve("What is the person holding?", scene_graph)
    assert isinstance(result, RetrievalResult)
    assert isinstance(result.subgraph, HeteroData)
    assert result.subgraph["object"].num_nodes <= 10
    assert result.subgraph["object"].num_nodes > 0


def test_retrieval_result_has_metadata():
    config = BenchmarkConfig(pcst_budget=10, top_k_seeds=5)
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = np.random.randn(1, 384).astype(np.float32)
    retriever = VideoRetriever(config, embedder=mock_embedder)
    scene_graph = make_mock_scene_graph()
    result = retriever.retrieve("test question", scene_graph)
    assert result.question == "test question"
    assert result.num_nodes > 0
    assert result.retrieval_time_ms >= 0
    assert isinstance(result.seed_indices, list)
