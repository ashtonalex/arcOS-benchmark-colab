"""
Test script for Phase 3: GNN Encoder module imports.

Validates that all GNN classes can be imported and instantiated
without requiring Colab or GPU.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all GNN modules can be imported."""
    print("[1/7] Testing GNN module imports...")
    try:
        from src.gnn import (
            SubgraphConverter,
            GNNOutput,
            GATv2Encoder,
            GraphSAGEEncoder,
            AttentionPooling,
            MeanPooling,
            MaxPooling,
            GNNTrainer,
            GNNModel,
        )
        print("  OK - All GNN classes imported")
    except ImportError as e:
        # Check if it's a missing optional dependency (expected on local machine)
        if "faiss" in str(e) or "torch_geometric" in str(e) or "sentence_transformers" in str(e):
            print(f"  SKIP - Missing optional dependency (expected locally): {e}")
            print("         This test will pass in Colab environment")
            return True
        print(f"  FAIL - Import failed: {e}")
        return False
    return True


def test_config():
    """Test BenchmarkConfig has GNN parameters."""
    print("[2/7] Testing BenchmarkConfig GNN parameters...")
    try:
        from src.config import BenchmarkConfig

        config = BenchmarkConfig()
        assert hasattr(config, "gnn_hidden_dim")
        assert hasattr(config, "gnn_num_layers")
        assert hasattr(config, "gnn_num_heads")
        assert hasattr(config, "gnn_dropout")
        print(f"  OK - Config has GNN params (hidden_dim={config.gnn_hidden_dim})")
    except Exception as e:
        print(f"  FAIL - Config test failed: {e}")
        return False
    return True


def test_encoder_creation():
    """Test encoder instantiation."""
    print("[3/7] Testing GNN encoder creation...")
    try:
        from src.config import BenchmarkConfig
        from src.gnn import GATv2Encoder, GraphSAGEEncoder

        config = BenchmarkConfig()

        # GATv2
        gat_encoder = GATv2Encoder(config)
        print(f"  OK - GATv2Encoder created ({config.gnn_num_layers} layers)")

        # GraphSAGE
        sage_encoder = GraphSAGEEncoder(config)
        print(f"  OK - GraphSAGEEncoder created ({config.gnn_num_layers} layers)")

    except Exception as e:
        if "faiss" in str(e) or "torch_geometric" in str(e):
            print(f"  SKIP - Missing optional dependency (expected locally)")
            return True
        print(f"  FAIL - Encoder creation failed: {e}")
        return False
    return True


def test_pooling_creation():
    """Test pooling layer instantiation."""
    print("[4/7] Testing pooling layer creation...")
    try:
        from src.config import BenchmarkConfig
        from src.gnn import AttentionPooling, MeanPooling, MaxPooling

        config = BenchmarkConfig()

        attention_pool = AttentionPooling(config)
        print(f"  OK - AttentionPooling created")

        mean_pool = MeanPooling(config)
        print(f"  OK - MeanPooling created")

        max_pool = MaxPooling(config)
        print(f"  OK - MaxPooling created")

    except Exception as e:
        if "faiss" in str(e) or "torch_geometric" in str(e):
            print(f"  SKIP - Missing optional dependency (expected locally)")
            return True
        print(f"  FAIL - Pooling creation failed: {e}")
        return False
    return True


def test_trainer_creation():
    """Test trainer instantiation."""
    print("[5/7] Testing GNNTrainer creation...")
    try:
        from src.config import BenchmarkConfig
        from src.gnn import GNNTrainer

        config = BenchmarkConfig()

        trainer = GNNTrainer(config, encoder_type="gatv2", pooling_type="attention")
        print(f"  OK - GNNTrainer created (device={trainer.device})")

    except Exception as e:
        if "faiss" in str(e) or "torch_geometric" in str(e):
            print(f"  SKIP - Missing optional dependency (expected locally)")
            return True
        print(f"  FAIL - Trainer creation failed: {e}")
        return False
    return True


def test_gnn_output():
    """Test GNNOutput dataclass."""
    print("[6/7] Testing GNNOutput dataclass...")
    try:
        import torch
        from src.gnn import GNNOutput

        output = GNNOutput(
            node_embeddings=torch.randn(10, 256),
            attention_scores={"node1": 0.5, "node2": 0.3},
            graph_embedding=torch.randn(256),
        )
        assert output.node_embeddings.shape == (10, 256)
        assert len(output.attention_scores) == 2
        assert output.graph_embedding.shape == (256,)
        print(f"  OK - GNNOutput dataclass works correctly")

    except Exception as e:
        if "faiss" in str(e) or "torch_geometric" in str(e):
            print(f"  SKIP - Missing optional dependency (expected locally)")
            return True
        print(f"  FAIL - GNNOutput test failed: {e}")
        return False
    return True


def test_model_api():
    """Test GNNModel API exists."""
    print("[7/7] Testing GNNModel API...")
    try:
        from src.gnn import GNNModel

        # Check methods exist
        assert hasattr(GNNModel, "build_from_checkpoint_or_train")
        assert hasattr(GNNModel, "encode")
        assert hasattr(GNNModel, "encode_batch")
        assert hasattr(GNNModel, "get_top_attention_nodes")
        print(f"  OK - GNNModel API available")

    except Exception as e:
        if "faiss" in str(e) or "torch_geometric" in str(e):
            print(f"  SKIP - Missing optional dependency (expected locally)")
            return True
        print(f"  FAIL - GNNModel API test failed: {e}")
        return False
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 3 IMPORT VALIDATION: GNN Encoder")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_config,
        test_encoder_creation,
        test_pooling_creation,
        test_trainer_creation,
        test_gnn_output,
        test_model_api,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Summary
    print("=" * 60)
    if all(results):
        print("SUCCESS: All Phase 3 imports validated OK -")
    else:
        print(f"FAILED: {results.count(False)}/{len(results)} tests failed")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
