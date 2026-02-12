"""
Quick validation script to test Phase 1 module imports.

This script verifies that all modules can be imported successfully
without requiring Colab or GPU access.
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

print("=" * 60)
print("Phase 1 Import Validation")
print("=" * 60)

# Test config
print("\n1. Testing config module...")
try:
    from src.config import BenchmarkConfig
    config = BenchmarkConfig()
    print("   ✓ BenchmarkConfig imported and instantiated")
    print(f"   ✓ Default seed: {config.seed}")
    print(f"   ✓ Checkpoint dir: {config.checkpoint_dir}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test utils.seeds
print("\n2. Testing utils.seeds module...")
try:
    from src.utils.seeds import set_seeds
    print("   ✓ set_seeds imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test utils.checkpoints
print("\n3. Testing utils.checkpoints module...")
try:
    from src.utils.checkpoints import (
        ensure_drive_mounted,
        checkpoint_exists,
        save_checkpoint,
        load_checkpoint,
        create_checkpoint_dirs,
    )
    print("   ✓ All checkpoint functions imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test data.dataset_loader
print("\n4. Testing data.dataset_loader module...")
try:
    from src.data.dataset_loader import RoGWebQSPLoader
    print("   ✓ RoGWebQSPLoader imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test data.graph_builder
print("\n5. Testing data.graph_builder module...")
try:
    from src.data.graph_builder import GraphBuilder
    print("   ✓ GraphBuilder imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test package imports
print("\n6. Testing package-level imports...")
try:
    from src.utils import set_seeds as set_seeds_pkg
    from src.data import RoGWebQSPLoader as Loader, GraphBuilder as Builder
    print("   ✓ Package-level imports work")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test configuration validation
print("\n7. Testing configuration validation...")
try:
    # Valid config
    valid_config = BenchmarkConfig(seed=42)
    print("   ✓ Valid config accepted")

    # Invalid config (should raise ValueError)
    try:
        invalid_config = BenchmarkConfig(seed=-1)
        print("   ✗ Invalid config not caught")
        sys.exit(1)
    except ValueError:
        print("   ✓ Invalid config rejected")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test graph builder basic functionality
print("\n8. Testing GraphBuilder basic functionality...")
try:
    builder = GraphBuilder(directed=True)
    triples = [
        ["entity1", "relation1", "entity2"],
        ["entity2", "relation2", "entity3"],
    ]
    graph = builder.build_from_triples(triples)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    print("   ✓ Graph construction works")
    print(f"   ✓ Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL IMPORTS VALID - Phase 1 modules ready")
print("=" * 60)
print("\nNext steps:")
print("1. Upload src/ directory to Google Colab")
print("2. Upload notebooks/arcOS_benchmark.ipynb to Colab")
print("3. Run notebook cells 1-8 sequentially")
print("4. Verify Phase 1 success criteria in Cell 8")
