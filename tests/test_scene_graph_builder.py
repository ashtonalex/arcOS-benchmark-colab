import pytest
import torch
import numpy as np
from torch_geometric.data import HeteroData
from src.data.scene_graph_builder import SceneGraphBuilder
from src.config import BenchmarkConfig


def make_mock_ag_annotations():
    return {
        "video_id": "3MSZA",
        "frames": [
            {
                "frame_id": 0,
                "objects": [
                    {"object_id": 0, "class": "person", "bbox": [10, 20, 100, 200]},
                    {"object_id": 1, "class": "cup", "bbox": [50, 60, 80, 90]},
                ],
                "relations": [
                    {"subject_id": 0, "object_id": 1, "predicate": "holding"},
                ],
            },
            {
                "frame_id": 1,
                "objects": [
                    {"object_id": 0, "class": "person", "bbox": [12, 22, 102, 202]},
                    {"object_id": 1, "class": "cup", "bbox": [55, 65, 85, 95]},
                ],
                "relations": [
                    {"subject_id": 0, "object_id": 1, "predicate": "not_contacting"},
                ],
            },
            {
                "frame_id": 2,
                "objects": [
                    {"object_id": 0, "class": "person", "bbox": [14, 24, 104, 204]},
                ],
                "relations": [],
            },
        ],
    }


def test_builder_produces_heterodata():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    assert isinstance(data, HeteroData)


def test_builder_node_counts():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    assert data["object"].num_nodes == 5


def test_builder_spatial_edges():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    edge_type = ("object", "spatial_rel", "object")
    assert edge_type in data.edge_types
    assert data[edge_type].edge_index.shape[1] == 2


def test_builder_temporal_edges():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    edge_type = ("object", "temporal", "object")
    assert edge_type in data.edge_types
    assert data[edge_type].edge_index.shape[1] == 3


def test_builder_node_features():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    assert data["object"].x.shape == (5, 384)
    assert data["object"].frame_id.shape == (5,)


def test_builder_metadata():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    assert data.video_id == "3MSZA"
