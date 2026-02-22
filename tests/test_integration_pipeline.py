"""End-to-end smoke test for the video scene graph pipeline.

Does NOT require GPU, AGQA download, or OpenRouter API.
Uses mock data to validate the full data flow.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig
from src.data.scene_graph_builder import SceneGraphBuilder
from src.retrieval.video_retriever import VideoRetriever
from src.gnn.hetero_encoder import HeteroGATv2Encoder
from src.verbalization.scene_verbalizer import SceneVerbalizer
from src.evaluation.benchmark import BenchmarkEvaluator


def make_ag_annotations():
    return {
        "video_id": "SMOKE",
        "frames": [
            {
                "frame_id": i,
                "objects": [
                    {"object_id": 0, "class": "person", "bbox": [10, 20, 100, 200]},
                    {"object_id": 1, "class": "cup", "bbox": [50, 60, 80, 90]},
                    {"object_id": 2, "class": "table", "bbox": [0, 100, 200, 150]},
                ],
                "relations": [
                    {"subject_id": 0, "object_id": 1, "predicate": "holding"},
                    {"subject_id": 1, "object_id": 2, "predicate": "on"},
                ],
            }
            for i in range(5)
        ],
    }


def test_full_pipeline_smoke():
    config = BenchmarkConfig(
        pcst_budget=10,
        top_k_seeds=5,
        top_k_triples=5,
        gnn_hidden_dim=64,
        gnn_num_layers=2,
        gnn_num_heads=2,
    )

    # 1. Build scene graph
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_ag_annotations()
    scene_graph = builder.build(ag)
    assert isinstance(scene_graph, HeteroData)
    assert scene_graph["object"].num_nodes == 15  # 3 objects x 5 frames

    # 2. Retrieve subgraph
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = np.random.randn(1, 384).astype(np.float32)
    retriever = VideoRetriever(config, embedder=mock_embedder)
    result = retriever.retrieve("What is the person holding?", scene_graph)
    assert result.subgraph["object"].num_nodes <= 10
    assert result.subgraph["object"].num_nodes > 0

    # 3. GNN encoding
    config_gnn = BenchmarkConfig(
        gnn_hidden_dim=64, gnn_num_layers=2, gnn_num_heads=2, embedding_dim=384
    )
    encoder = HeteroGATv2Encoder(config_gnn)
    query_emb = torch.randn(384)
    node_emb, attn_scores, graph_emb = encoder(result.subgraph, query_emb)
    assert attn_scores.shape[0] == result.subgraph["object"].num_nodes

    # 4. Verbalize
    n = result.subgraph["object"].num_nodes
    result.subgraph.object_names = ["person", "cup", "table"] * (n // 3 + 1)
    result.subgraph.object_names = result.subgraph.object_names[:n]
    verbalizer = SceneVerbalizer(config)
    text = verbalizer.verbalize(result.subgraph, attn_scores)
    assert isinstance(text, str)
    assert len(text) > 0

    # 5. Evaluate (mock LLM response)
    evaluator = BenchmarkEvaluator()
    em = evaluator.exact_match("cup", "cup")
    assert em == 1.0

    print(f"\nSmoke test passed!")
    print(f"  Scene graph: {scene_graph['object'].num_nodes} nodes")
    print(f"  Subgraph: {result.subgraph['object'].num_nodes} nodes")
    print(f"  GNN output: {node_emb.shape}")
    print(f"  Verbalization:\n{text}")
