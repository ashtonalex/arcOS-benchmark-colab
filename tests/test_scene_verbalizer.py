import pytest
import torch
from torch_geometric.data import HeteroData
from src.verbalization.scene_verbalizer import SceneVerbalizer
from src.config import BenchmarkConfig


def make_annotated_subgraph():
    data = HeteroData()
    data["object"].x = torch.randn(4, 384)
    data["object"].frame_id = torch.tensor([0, 0, 1, 1])
    data.object_names = ["person", "cup", "person", "table"]
    data.spatial_predicates = ["holding", "sitting_on"]
    data["object", "spatial_rel", "object"].edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    data["object", "temporal", "object"].edge_index = torch.tensor([[0], [2]], dtype=torch.long)
    data.video_id = "test"
    return data


def test_verbalizer_returns_string():
    config = BenchmarkConfig()
    verbalizer = SceneVerbalizer(config)
    data = make_annotated_subgraph()
    attn = torch.tensor([0.9, 0.7, 0.5, 0.3])
    text = verbalizer.verbalize(data, attn)
    assert isinstance(text, str)
    assert len(text) > 0


def test_verbalizer_respects_token_budget():
    config = BenchmarkConfig(top_k_triples=2)
    verbalizer = SceneVerbalizer(config)
    data = make_annotated_subgraph()
    attn = torch.tensor([0.9, 0.7, 0.5, 0.3])
    text = verbalizer.verbalize(data, attn)
    lines = [l for l in text.strip().split("\n") if l.strip()]
    assert len(lines) <= 2


def test_verbalizer_unweighted_mode():
    config = BenchmarkConfig()
    verbalizer = SceneVerbalizer(config)
    data = make_annotated_subgraph()
    text = verbalizer.verbalize_unweighted(data)
    assert isinstance(text, str)
    assert len(text) > 0
