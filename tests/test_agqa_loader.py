import pytest
from unittest.mock import patch, MagicMock
from src.data.agqa_loader import AGQALoader
from src.config import BenchmarkConfig


def make_mock_agqa_sample():
    return {
        "question": "What did the person put down before sitting?",
        "answer": "cup",
        "video_id": "3MSZA",
        "program": [
            {"op": "filter", "args": ["put_down"]},
            {"op": "before", "args": ["sit"]},
            {"op": "query", "args": ["object"]},
        ],
    }


def test_agqa_loader_init():
    config = BenchmarkConfig()
    loader = AGQALoader(config)
    assert loader.subset_size == 50000


def test_agqa_loader_parse_sample():
    config = BenchmarkConfig()
    loader = AGQALoader(config)
    raw = make_mock_agqa_sample()
    parsed = loader.parse_sample(raw)
    assert parsed["question"] == "What did the person put down before sitting?"
    assert parsed["answer"] == "cup"
    assert parsed["video_id"] == "3MSZA"
    assert isinstance(parsed["program"], list)


def test_agqa_loader_get_video_ids():
    config = BenchmarkConfig()
    loader = AGQALoader(config)
    samples = [make_mock_agqa_sample(), make_mock_agqa_sample()]
    samples[1]["video_id"] = "ABC12"
    video_ids = loader.get_unique_video_ids(samples)
    assert video_ids == {"3MSZA", "ABC12"}


def test_agqa_loader_subsample():
    config = BenchmarkConfig(agqa_subset_size=2)
    loader = AGQALoader(config)
    samples = [make_mock_agqa_sample() for _ in range(10)]
    for i, s in enumerate(samples):
        s["video_id"] = f"vid_{i}"
    subsampled = loader.subsample(samples)
    assert len(subsampled) == 2
