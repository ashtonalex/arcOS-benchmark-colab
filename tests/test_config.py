import pytest
from src.config import BenchmarkConfig


def test_video_scene_graph_config_defaults():
    config = BenchmarkConfig()
    assert config.agqa_subset_size == 50000
    assert config.ag_frame_sample_rate == 3
    assert config.ag_num_object_classes == 36
    assert config.ag_num_relation_types == 26
    assert config.top_k_seeds == 10
    assert config.pcst_temporal_cost_weight == 0.5
    assert config.gnn_batch_size == 128
    assert config.gnn_encoder_type == "hetero_gatv2"
    assert "attention_precision" in config.metrics
    assert "retrieval_hit_rate" in config.metrics


def test_video_config_validation():
    with pytest.raises(ValueError):
        BenchmarkConfig(agqa_subset_size=-1)
    with pytest.raises(ValueError):
        BenchmarkConfig(ag_frame_sample_rate=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(top_k_seeds=0)
