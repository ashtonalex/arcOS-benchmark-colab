"""
Central configuration system for arcOS Benchmark.

All hyperparameters, paths, and settings are managed here with Pydantic validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for the entire benchmark pipeline."""

    # ========== Reproducibility ==========
    seed: int = 42
    deterministic: bool = True

    # ========== Google Drive Paths ==========
    drive_root: str = "/content/drive/MyDrive/arcOS_benchmark"

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for saving checkpoints (datasets, graphs, models)."""
        return Path(self.drive_root) / "checkpoints"

    @property
    def results_dir(self) -> Path:
        """Directory for saving evaluation results."""
        return Path(self.drive_root) / "results"

    def get_checkpoint_path(self, filename: str) -> Path:
        """Get full path for a checkpoint file."""
        return self.checkpoint_dir / filename

    def get_results_path(self, filename: str) -> Path:
        """Get full path for a results file."""
        return self.results_dir / filename

    # ========== Dataset ==========
    dataset_name: str = "rmanluo/RoG-webqsp"
    split_train: str = "train"
    split_val: str = "validation"
    split_test: str = "test"

    # Dataset size limits (for faster iteration during development)
    max_train_examples: Optional[int] = 600  # ~1/5 of 2826, None = use all
    max_val_examples: Optional[int] = 50     # ~1/5 of 246, None = use all
    max_test_examples: Optional[int] = None  # Keep test set full for final eval

    # Expected dataset sizes (for validation) - after slicing
    expected_train_size: int = 900
    expected_val_size: int = 90
    expected_test_size: int = 1628  # Full test set (actual size from dataset)

    # ========== Graph Construction ==========
    graph_directed: bool = True
    # Reduced min sizes for 1/5 dataset (was 10000/30000 for full)
    unified_graph_min_nodes: int = 2000
    unified_graph_min_edges: int = 6000

    # ========== Retrieval (Phase 2) ==========
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k_entities: int = 15
    pcst_budget: int = 70  # Max nodes in extracted subgraph
    pcst_local_budget: int = 500  # BFS neighborhood size before PCST
    pcst_cost: float = 0.1  # Edge cost for PCST (tuned to cosine sim prize scale 0-1)
    pcst_pruning: str = "gw"  # PCST pruning strategy: 'none', 'gw', or 'strong'
    pcst_edge_weight_alpha: float = 0.5  # Query-aware edge cost scaling [0,1]. 0=uniform costs (default)
    pcst_bridge_components: bool = True  # Bridge disconnected PCST components via shortest paths
    pcst_bridge_max_hops: int = 4  # Max relay hops when bridging disconnected components

    # ========== GNN (Phase 3) ==========
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1
    gnn_pooling: str = "attention"  # attention or mean

    # ========== Verbalization (Phase 4) ==========
    top_k_triples: int = 15  # Max triples to verbalize
    verbalization_format: str = "natural"  # natural or structured

    # ========== LLM (Phase 5) ==========
    llm_provider: str = "openrouter"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    llm_api_base: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0

    # ========== Training (Phase 6) ==========
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 10
    patience: int = 5
    gradient_clip: float = 1.0

    # ========== Evaluation (Phase 7) ==========
    metrics: list = field(default_factory=lambda: ["exact_match", "f1", "hits@1"])

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.top_k_entities <= 0:
            raise ValueError(f"top_k_entities must be positive, got {self.top_k_entities}")
        if self.pcst_budget <= 0:
            raise ValueError(f"pcst_budget must be positive, got {self.pcst_budget}")
        if not 0 <= self.gnn_dropout <= 1:
            raise ValueError(f"gnn_dropout must be in [0, 1], got {self.gnn_dropout}")
        if not 0 <= self.pcst_edge_weight_alpha <= 1:
            raise ValueError(f"pcst_edge_weight_alpha must be in [0, 1], got {self.pcst_edge_weight_alpha}")
        if self.pcst_bridge_max_hops <= 0:
            raise ValueError(f"pcst_bridge_max_hops must be positive, got {self.pcst_bridge_max_hops}")
        if self.pcst_pruning not in ["none", "gw", "strong"]:
            raise ValueError(f"pcst_pruning must be 'none', 'gw', or 'strong', got {self.pcst_pruning}")
        if self.gnn_pooling not in ["attention", "mean"]:
            raise ValueError(f"gnn_pooling must be 'attention' or 'mean', got {self.gnn_pooling}")
        if self.verbalization_format not in ["natural", "structured"]:
            raise ValueError(f"verbalization_format must be 'natural' or 'structured', got {self.verbalization_format}")

    def print_summary(self):
        """Print human-readable configuration summary."""
        print("=" * 60)
        print("arcOS Benchmark Configuration")
        print("=" * 60)
        print(f"Seed: {self.seed} (deterministic={self.deterministic})")
        print(f"Dataset: {self.dataset_name}")
        print(f"Drive root: {self.drive_root}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Results dir: {self.results_dir}")
        print("\n--- Retrieval ---")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Top-K entities: {self.top_k_entities}")
        print(f"PCST budget: {self.pcst_budget}")
        print(f"PCST local budget: {self.pcst_local_budget}")
        print(f"PCST edge cost: {self.pcst_cost}")
        print(f"PCST pruning: {self.pcst_pruning}")
        print("\n--- GNN ---")
        print(f"Hidden dim: {self.gnn_hidden_dim}")
        print(f"Num layers: {self.gnn_num_layers}")
        print(f"Num heads: {self.gnn_num_heads}")
        print(f"Pooling: {self.gnn_pooling}")
        print("\n--- LLM ---")
        print(f"Model: {self.llm_model}")
        print(f"Provider: {self.llm_provider}")
        print(f"Temperature: {self.llm_temperature}")
        print("=" * 60)
