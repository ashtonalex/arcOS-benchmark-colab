# Video Scene Graph Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Freebase KG QA pipeline with video scene graph QA using AGQA + Action Genome, benchmarking GNN-augmented LLM vs pure LLM inference.

**Architecture:** AGQA QA pairs map to Action Genome scene graphs stored as PyG HeteroData (object nodes, spatial + temporal edges). Per-video FAISS indices feed k-NN seeds to a simplified PCST solver. A HeteroConv GATv2 encoder produces attention-weighted node scores for verbalization into LLM hard prompts.

**Tech Stack:** PyTorch Geometric (HeteroData, HeteroConv, GATv2Conv), pcst_fast, FAISS, sentence-transformers, OpenRouter API

**Design Doc:** `docs/plans/2026-02-22-video-scene-graph-retrieval-design.md`

---

## Task 1: Update BenchmarkConfig for Video Scene Graphs

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from src.config import BenchmarkConfig


def test_video_scene_graph_config_defaults():
    """New video scene graph config fields have correct defaults."""
    config = BenchmarkConfig()

    # Data source
    assert config.agqa_subset_size == 50000
    assert config.ag_frame_sample_rate == 3  # fps
    assert config.ag_num_object_classes == 36
    assert config.ag_num_relation_types == 26

    # Per-video retrieval
    assert config.top_k_seeds == 10
    assert config.pcst_budget == 50
    assert config.pcst_temporal_cost_weight == 0.5

    # HeteroGNN
    assert config.gnn_batch_size == 128
    assert config.gnn_encoder_type == "hetero_gatv2"

    # Benchmark
    assert "attention_precision" in config.metrics
    assert "retrieval_hit_rate" in config.metrics


def test_video_config_validation():
    """Validation rejects invalid video config values."""
    with pytest.raises(ValueError):
        BenchmarkConfig(agqa_subset_size=-1)
    with pytest.raises(ValueError):
        BenchmarkConfig(ag_frame_sample_rate=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(top_k_seeds=0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — missing attributes

**Step 3: Write minimal implementation**

Add to `BenchmarkConfig` in `src/config.py` after the existing fields:

```python
    # ========== Video Scene Graph (AGQA + Action Genome) ==========
    agqa_subset_size: int = 50000
    ag_frame_sample_rate: int = 3          # frames per second to sample
    ag_num_object_classes: int = 36
    ag_num_relation_types: int = 26

    # ========== Per-Video Retrieval ==========
    top_k_seeds: int = 10
    pcst_budget: int = 50                  # overrides old default of 70
    pcst_temporal_cost_weight: float = 0.5 # temporal edge cost multiplier

    # ========== HeteroGNN ==========
    gnn_batch_size: int = 128
    gnn_encoder_type: str = "hetero_gatv2"
```

Update `metrics` default:
```python
    metrics: list = field(default_factory=lambda: [
        "exact_match", "f1", "hits@1", "retrieval_hit_rate", "attention_precision"
    ])
```

Add validation in `__post_init__`:
```python
    if self.agqa_subset_size < 1:
        raise ValueError("agqa_subset_size must be positive")
    if self.ag_frame_sample_rate < 1:
        raise ValueError("ag_frame_sample_rate must be positive")
    if self.top_k_seeds < 1:
        raise ValueError("top_k_seeds must be positive")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_config.py src/config.py
git commit -m "feat: add video scene graph config fields to BenchmarkConfig"
```

---

## Task 2: AGQA Dataset Loader

**Files:**
- Create: `src/data/agqa_loader.py`
- Test: `tests/test_agqa_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_agqa_loader.py
import pytest
from unittest.mock import patch, MagicMock
from src.data.agqa_loader import AGQALoader
from src.config import BenchmarkConfig


def make_mock_agqa_sample():
    """Create a realistic AGQA QA pair."""
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agqa_loader.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/data/agqa_loader.py
"""AGQA dataset loader for video QA benchmarking."""

from typing import Any, Dict, List, Set
import random
from src.config import BenchmarkConfig


class AGQALoader:
    """Loads and parses AGQA QA pairs."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.subset_size = config.agqa_subset_size

    def parse_sample(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a raw AGQA sample into a standardized format."""
        return {
            "question": raw["question"],
            "answer": raw["answer"],
            "video_id": raw["video_id"],
            "program": raw["program"],
        }

    def get_unique_video_ids(self, samples: List[Dict[str, Any]]) -> Set[str]:
        """Extract unique video IDs from a list of samples."""
        return {s["video_id"] for s in samples}

    def subsample(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Subsample to configured subset size, preserving random seed."""
        if len(samples) <= self.subset_size:
            return samples
        rng = random.Random(self.config.seed)
        return rng.sample(samples, self.subset_size)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agqa_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/agqa_loader.py tests/test_agqa_loader.py
git commit -m "feat: add AGQA dataset loader with parsing and subsampling"
```

---

## Task 3: Action Genome Scene Graph → PyG HeteroData Builder

**Files:**
- Create: `src/data/scene_graph_builder.py`
- Test: `tests/test_scene_graph_builder.py`

**Step 1: Write the failing test**

```python
# tests/test_scene_graph_builder.py
import pytest
import torch
import numpy as np
from torch_geometric.data import HeteroData
from src.data.scene_graph_builder import SceneGraphBuilder
from src.config import BenchmarkConfig


def make_mock_ag_annotations():
    """Mock Action Genome annotations for one video (3 frames, 2 objects each)."""
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
    # 3 frames: 2 + 2 + 1 = 5 object nodes
    assert data["object"].num_nodes == 5


def test_builder_spatial_edges():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    edge_type = ("object", "spatial_rel", "object")
    assert edge_type in data.edge_types
    # 1 relation in frame 0 + 1 in frame 1 + 0 in frame 2 = 2
    assert data[edge_type].edge_index.shape[1] == 2


def test_builder_temporal_edges():
    config = BenchmarkConfig()
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_mock_ag_annotations()
    data = builder.build(ag)
    edge_type = ("object", "temporal", "object")
    assert edge_type in data.edge_types
    # object 0 appears in frames 0,1,2 → 2 temporal edges
    # object 1 appears in frames 0,1 → 1 temporal edge
    # Total: 3 temporal edges
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scene_graph_builder.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/data/scene_graph_builder.py
"""Converts Action Genome annotations to PyG HeteroData."""

from typing import Any, Dict, List, Optional
import torch
import numpy as np
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig


class SceneGraphBuilder:
    """Builds PyG HeteroData from Action Genome frame annotations."""

    def __init__(
        self,
        config: BenchmarkConfig,
        embedding_dim: int = 384,
        embedder=None,
    ):
        self.config = config
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self._class_cache: Dict[str, np.ndarray] = {}

    def build(self, ag_annotations: Dict[str, Any]) -> HeteroData:
        """Convert one video's Action Genome annotations to HeteroData.

        Args:
            ag_annotations: Dict with keys 'video_id' and 'frames'.
                Each frame has 'frame_id', 'objects' (list of dicts with
                'object_id', 'class', 'bbox'), and 'relations' (list of
                dicts with 'subject_id', 'object_id', 'predicate').

        Returns:
            PyG HeteroData with object nodes and spatial/temporal edges.
        """
        data = HeteroData()
        frames = ag_annotations["frames"]

        # --- Collect all object instances ---
        node_list = []  # (frame_id, object_id, class_name, bbox)
        node_index_map = {}  # (frame_id, object_id) → global_node_idx

        global_idx = 0
        for frame in frames:
            fid = frame["frame_id"]
            for obj in frame["objects"]:
                oid = obj["object_id"]
                node_list.append((fid, oid, obj["class"], obj.get("bbox")))
                node_index_map[(fid, oid)] = global_idx
                global_idx += 1

        num_nodes = len(node_list)

        # --- Node features ---
        x = self._embed_object_classes([n[2] for n in node_list])
        data["object"].x = torch.tensor(x, dtype=torch.float32)
        data["object"].frame_id = torch.tensor(
            [n[0] for n in node_list], dtype=torch.long
        )
        data["object"].object_class = torch.tensor(
            [hash(n[2]) % (2**31) for n in node_list], dtype=torch.long
        )

        # --- Spatial edges (within-frame relations) ---
        spatial_src, spatial_dst = [], []
        spatial_relations = []
        for frame in frames:
            fid = frame["frame_id"]
            for rel in frame["relations"]:
                src_key = (fid, rel["subject_id"])
                dst_key = (fid, rel["object_id"])
                if src_key in node_index_map and dst_key in node_index_map:
                    spatial_src.append(node_index_map[src_key])
                    spatial_dst.append(node_index_map[dst_key])
                    spatial_relations.append(rel["predicate"])

        if spatial_src:
            data["object", "spatial_rel", "object"].edge_index = torch.tensor(
                [spatial_src, spatial_dst], dtype=torch.long
            )
        else:
            data["object", "spatial_rel", "object"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

        # --- Temporal edges (same object across consecutive frames) ---
        temporal_src, temporal_dst = [], []
        temporal_attrs = []

        # Group by object_id across frames
        obj_frames: Dict[int, List[int]] = {}
        for fid, oid, _, bbox in node_list:
            obj_frames.setdefault(oid, []).append((fid, node_index_map[(fid, oid)]))

        for oid, appearances in obj_frames.items():
            appearances.sort(key=lambda x: x[0])
            for i in range(len(appearances) - 1):
                fid_a, idx_a = appearances[i]
                fid_b, idx_b = appearances[i + 1]
                temporal_src.append(idx_a)
                temporal_dst.append(idx_b)
                temporal_attrs.append([fid_b - fid_a, 0.0])  # [frame_delta, bbox_iou placeholder]

        if temporal_src:
            data["object", "temporal", "object"].edge_index = torch.tensor(
                [temporal_src, temporal_dst], dtype=torch.long
            )
            data["object", "temporal", "object"].edge_attr = torch.tensor(
                temporal_attrs, dtype=torch.float32
            )
        else:
            data["object", "temporal", "object"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )
            data["object", "temporal", "object"].edge_attr = torch.zeros(
                (0, 2), dtype=torch.float32
            )

        # --- Metadata ---
        data.video_id = ag_annotations["video_id"]
        data.num_frames = len(frames)

        return data

    def _embed_object_classes(self, class_names: List[str]) -> np.ndarray:
        """Embed object class names. Uses embedder if available, else random."""
        if self.embedder is not None:
            # Cache embeddings per class to avoid recomputation
            unique_classes = list(set(class_names))
            to_embed = [c for c in unique_classes if c not in self._class_cache]
            if to_embed:
                embeddings = self.embedder.embed_texts(to_embed, show_progress=False)
                for cls, emb in zip(to_embed, embeddings):
                    self._class_cache[cls] = emb
            return np.array([self._class_cache[c] for c in class_names])
        else:
            # Deterministic fallback for testing
            rng = np.random.RandomState(42)
            cache = {}
            for c in set(class_names):
                cache[c] = rng.randn(self.embedding_dim).astype(np.float32)
            return np.array([cache[c] for c in class_names])
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scene_graph_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/scene_graph_builder.py tests/test_scene_graph_builder.py
git commit -m "feat: add Action Genome scene graph to PyG HeteroData builder"
```

---

## Task 4: Per-Video FAISS Index

**Files:**
- Create: `src/retrieval/video_index.py`
- Test: `tests/test_video_index.py`

**Step 1: Write the failing test**

```python
# tests/test_video_index.py
import pytest
import numpy as np
import torch
from torch_geometric.data import HeteroData
from src.retrieval.video_index import VideoIndex


def make_mock_heterodata(num_nodes=20, dim=384):
    """Create a HeteroData with random object embeddings."""
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
    # Results are (node_idx, score) tuples
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_video_index.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/retrieval/video_index.py
"""Per-video FAISS index for k-NN seed selection on HeteroData."""

from typing import List, Tuple
import numpy as np
import faiss
from torch_geometric.data import HeteroData


class VideoIndex:
    """FAISS index for a single video's object node embeddings."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self._num_nodes = 0

    def build(self, data: HeteroData) -> None:
        """Build FAISS index from HeteroData object node features."""
        embeddings = data["object"].x.numpy().astype(np.float32)
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self._num_nodes = len(embeddings)

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Search for top-k nearest object nodes.

        Args:
            query_embedding: 1D array of shape (embedding_dim,)
            k: number of results

        Returns:
            List of (node_index, similarity_score) tuples, descending by score.
        """
        k = min(k, self._num_nodes)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores, indices = self.index.search(query, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def __len__(self) -> int:
        return self._num_nodes
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_video_index.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/retrieval/video_index.py tests/test_video_index.py
git commit -m "feat: add per-video FAISS index for k-NN seed selection"
```

---

## Task 5: PCST Adapter for HeteroData

**Files:**
- Create: `src/retrieval/hetero_pcst.py`
- Test: `tests/test_hetero_pcst.py`

**Step 1: Write the failing test**

```python
# tests/test_hetero_pcst.py
import pytest
import torch
import numpy as np
from torch_geometric.data import HeteroData
from src.retrieval.hetero_pcst import HeteroPCST
from src.config import BenchmarkConfig


def make_chain_graph(n=20):
    """Create a linear chain HeteroData: 0-1-2-...-n with spatial edges
    and temporal edges linking even nodes to the next even node."""
    data = HeteroData()
    rng = np.random.RandomState(42)
    data["object"].x = torch.tensor(rng.randn(n, 384), dtype=torch.float32)
    data["object"].frame_id = torch.arange(n)

    # Spatial: chain 0→1→2→...→(n-1)
    src = list(range(n - 1))
    dst = list(range(1, n))
    data["object", "spatial_rel", "object"].edge_index = torch.tensor(
        [src, dst], dtype=torch.long
    )

    # Temporal: 0→2→4→...
    even = list(range(0, n, 2))
    t_src = even[:-1]
    t_dst = even[1:]
    data["object", "temporal", "object"].edge_index = torch.tensor(
        [t_src, t_dst], dtype=torch.long
    )
    data["object", "temporal", "object"].edge_attr = torch.ones(
        len(t_src), 2, dtype=torch.float32
    )

    data.video_id = "test"
    return data


def test_hetero_pcst_returns_heterodata():
    config = BenchmarkConfig(pcst_budget=10)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 1.0, 5: 0.8, 10: 0.6}
    sub = solver.extract(data, prizes)
    assert isinstance(sub, HeteroData)


def test_hetero_pcst_respects_budget():
    config = BenchmarkConfig(pcst_budget=10)
    solver = HeteroPCST(config)
    data = make_chain_graph(50)
    prizes = {i: 1.0 for i in range(0, 50, 5)}
    sub = solver.extract(data, prizes)
    assert sub["object"].num_nodes <= 10


def test_hetero_pcst_includes_seed_nodes():
    config = BenchmarkConfig(pcst_budget=20)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 1.0, 5: 0.8}
    sub = solver.extract(data, prizes)
    # The selected_nodes mask should include the prized nodes
    selected = sub.selected_nodes.tolist()
    assert 0 in selected
    assert 5 in selected


def test_hetero_pcst_preserves_edge_types():
    config = BenchmarkConfig(pcst_budget=15)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 1.0, 2: 0.8, 4: 0.6}
    sub = solver.extract(data, prizes)
    assert ("object", "spatial_rel", "object") in sub.edge_types
    assert ("object", "temporal", "object") in sub.edge_types


def test_hetero_pcst_bfs_fallback():
    """PCST should fall back to BFS on error and still return valid subgraph."""
    config = BenchmarkConfig(pcst_budget=5)
    solver = HeteroPCST(config)
    data = make_chain_graph(10)
    prizes = {0: 1.0}
    # Even if PCST fails internally, BFS fallback should work
    sub = solver.extract(data, prizes)
    assert sub["object"].num_nodes > 0
    assert sub["object"].num_nodes <= 5
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hetero_pcst.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/retrieval/hetero_pcst.py
"""PCST subgraph extraction adapted for PyG HeteroData."""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig

try:
    import pcst_fast
    HAS_PCST = True
except ImportError:
    HAS_PCST = False


class HeteroPCST:
    """Extracts subgraphs from HeteroData using Prize-Collecting Steiner Tree."""

    def __init__(self, config: BenchmarkConfig):
        self.budget = config.pcst_budget
        self.cost = config.pcst_cost
        self.pruning = config.pcst_pruning
        self.temporal_cost_weight = config.pcst_temporal_cost_weight

    def extract(
        self,
        data: HeteroData,
        prizes: Dict[int, float],
        root: Optional[int] = None,
    ) -> HeteroData:
        """Extract a subgraph from HeteroData using PCST.

        Args:
            data: Full video scene graph as HeteroData.
            prizes: Dict mapping node indices to prize values.
            root: Optional root node for PCST.

        Returns:
            HeteroData subgraph with `selected_nodes` attribute
            containing original node indices.
        """
        num_nodes = data["object"].num_nodes

        # Build prize array
        prize_array = np.zeros(num_nodes, dtype=np.float64)
        for node_idx, prize in prizes.items():
            if 0 <= node_idx < num_nodes:
                prize_array[node_idx] = prize

        # Flatten all edge types into single edge list
        edges, costs, edge_type_map = self._flatten_edges(data, num_nodes)

        if len(edges) == 0:
            return self._select_top_prized(data, prizes)

        # Try PCST, fall back to BFS on failure
        try:
            selected_nodes = self._run_pcst(
                num_nodes, edges, costs, prize_array, root
            )
        except Exception:
            selected_nodes = self._bfs_fallback(
                edges, prizes, num_nodes
            )

        # Trim to budget
        if len(selected_nodes) > self.budget:
            scored = [(n, prize_array[n]) for n in selected_nodes]
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [n for n, _ in scored[: self.budget]]

        selected_nodes = sorted(set(selected_nodes))
        return self._slice_heterodata(data, selected_nodes)

    def _flatten_edges(self, data, num_nodes):
        """Flatten all edge types into a single edge list with costs."""
        edges = []
        costs = []
        edge_type_map = []

        for etype in data.edge_types:
            ei = data[etype].edge_index
            num_edges = ei.shape[1]
            type_name = etype[1]  # relation name

            if type_name == "temporal":
                cost = self.cost * self.temporal_cost_weight
            else:
                cost = self.cost

            for i in range(num_edges):
                src, dst = int(ei[0, i]), int(ei[1, i])
                edges.append([src, dst])
                costs.append(cost)
                edge_type_map.append(etype)

        return (
            np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64),
            np.array(costs, dtype=np.float64) if costs else np.zeros(0, dtype=np.float64),
            edge_type_map,
        )

    def _run_pcst(self, num_nodes, edges, costs, prizes, root):
        """Run pcst_fast and return selected node indices."""
        if not HAS_PCST:
            raise ImportError("pcst_fast not available")

        root_idx = root if root is not None else -1
        vertices, selected_edges = pcst_fast.pcst_fast(
            edges.astype(np.int64),
            prizes,
            costs,
            root_idx,
            1,  # num_clusters
            self.pruning,
            0,  # verbosity
        )

        # pcst_fast returns selected vertex indices
        if len(vertices) == 0:
            raise ValueError("PCST returned empty result")

        return vertices.tolist()

    def _bfs_fallback(self, edges, prizes, num_nodes):
        """BFS expansion from highest-prize nodes."""
        if not prizes:
            return list(range(min(self.budget, num_nodes)))

        # Build adjacency
        adj = {i: [] for i in range(num_nodes)}
        for src, dst in edges:
            adj[int(src)].append(int(dst))
            adj[int(dst)].append(int(src))

        # BFS from top-prize node
        start = max(prizes, key=prizes.get)
        visited = set()
        queue = [start]
        visited.add(start)

        while queue and len(visited) < self.budget:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited and len(visited) < self.budget:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return list(visited)

    def _select_top_prized(self, data, prizes):
        """Fallback: select top-budget nodes by prize when no edges exist."""
        if not prizes:
            selected = list(range(min(self.budget, data["object"].num_nodes)))
        else:
            sorted_nodes = sorted(prizes, key=prizes.get, reverse=True)
            selected = sorted_nodes[: self.budget]
        return self._slice_heterodata(data, selected)

    def _slice_heterodata(self, data, selected_nodes):
        """Create a subgraph HeteroData from selected node indices."""
        selected = sorted(selected_nodes)
        selected_set = set(selected)
        old_to_new = {old: new for new, old in enumerate(selected)}

        sub = HeteroData()

        # Slice node features
        idx = torch.tensor(selected, dtype=torch.long)
        sub["object"].x = data["object"].x[idx]
        if hasattr(data["object"], "frame_id"):
            sub["object"].frame_id = data["object"].frame_id[idx]
        if hasattr(data["object"], "object_class"):
            sub["object"].object_class = data["object"].object_class[idx]

        # Slice edges per type
        for etype in data.edge_types:
            ei = data[etype].edge_index
            mask = torch.tensor(
                [
                    int(ei[0, i]) in selected_set and int(ei[1, i]) in selected_set
                    for i in range(ei.shape[1])
                ],
                dtype=torch.bool,
            )
            new_src = [old_to_new[int(ei[0, i])] for i in range(ei.shape[1]) if mask[i]]
            new_dst = [old_to_new[int(ei[1, i])] for i in range(ei.shape[1]) if mask[i]]

            if new_src:
                sub[etype].edge_index = torch.tensor(
                    [new_src, new_dst], dtype=torch.long
                )
            else:
                sub[etype].edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Preserve edge attributes
            if hasattr(data[etype], "edge_attr") and mask.any():
                sub[etype].edge_attr = data[etype].edge_attr[mask]

        # Store original indices for ground truth mapping
        sub.selected_nodes = torch.tensor(selected, dtype=torch.long)

        # Copy metadata
        if hasattr(data, "video_id"):
            sub.video_id = data.video_id

        return sub
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hetero_pcst.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/retrieval/hetero_pcst.py tests/test_hetero_pcst.py
git commit -m "feat: add PCST adapter for HeteroData subgraph extraction"
```

---

## Task 6: Video Retriever Orchestration

**Files:**
- Create: `src/retrieval/video_retriever.py`
- Test: `tests/test_video_retriever.py`

**Step 1: Write the failing test**

```python
# tests/test_video_retriever.py
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
    data["object", "spatial_rel", "object"].edge_index = torch.tensor(
        [src, dst], dtype=torch.long
    )
    even = list(range(0, num_nodes, 2))
    data["object", "temporal", "object"].edge_index = torch.tensor(
        [even[:-1], even[1:]], dtype=torch.long
    )
    data["object", "temporal", "object"].edge_attr = torch.ones(
        len(even) - 1, 2, dtype=torch.float32
    )
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_video_retriever.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/retrieval/video_retriever.py
"""Orchestrates per-video retrieval: embed query → k-NN → PCST → subgraph."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig
from src.retrieval.video_index import VideoIndex
from src.retrieval.hetero_pcst import HeteroPCST


@dataclass
class RetrievalResult:
    subgraph: HeteroData
    question: str
    seed_indices: List[int]
    similarity_scores: Dict[int, float]
    num_nodes: int
    num_edges: int
    retrieval_time_ms: float
    pcst_used: bool


class VideoRetriever:
    """Retrieves relevant subgraph from a video scene graph given a question."""

    def __init__(self, config: BenchmarkConfig, embedder=None):
        self.config = config
        self.embedder = embedder
        self.pcst = HeteroPCST(config)

    def retrieve(
        self,
        question: str,
        scene_graph: HeteroData,
        answer_nodes: Optional[List[int]] = None,
    ) -> RetrievalResult:
        """Retrieve a subgraph relevant to the question.

        Args:
            question: Natural language question.
            scene_graph: Full video scene graph as HeteroData.
            answer_nodes: Optional ground truth node indices for evaluation.

        Returns:
            RetrievalResult with extracted subgraph and metadata.
        """
        start = time.time()

        # 1. Embed query
        query_emb = self.embedder.embed_texts([question])[0]

        # 2. Build per-video index and search
        index = VideoIndex(embedding_dim=self.config.embedding_dim)
        index.build(scene_graph)
        results = index.search(query_emb, k=self.config.top_k_seeds)

        seed_indices = [r[0] for r in results]
        prizes = {idx: score for idx, score in results}

        # 3. PCST extraction
        pcst_used = True
        try:
            subgraph = self.pcst.extract(scene_graph, prizes)
        except Exception:
            pcst_used = False
            subgraph = self.pcst.extract(scene_graph, prizes)

        elapsed = (time.time() - start) * 1000

        # Count edges across all types
        total_edges = sum(
            subgraph[et].edge_index.shape[1] for et in subgraph.edge_types
        )

        return RetrievalResult(
            subgraph=subgraph,
            question=question,
            seed_indices=seed_indices,
            similarity_scores=prizes,
            num_nodes=subgraph["object"].num_nodes,
            num_edges=total_edges,
            retrieval_time_ms=elapsed,
            pcst_used=pcst_used,
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_video_retriever.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/retrieval/video_retriever.py tests/test_video_retriever.py
git commit -m "feat: add VideoRetriever orchestrating per-video k-NN and PCST"
```

---

## Task 7: HeteroGATv2 Encoder

**Files:**
- Create: `src/gnn/hetero_encoder.py`
- Test: `tests/test_hetero_encoder.py`

**Step 1: Write the failing test**

```python
# tests/test_hetero_encoder.py
import pytest
import torch
from torch_geometric.data import HeteroData
from src.gnn.hetero_encoder import HeteroGATv2Encoder
from src.config import BenchmarkConfig


def make_test_subgraph(num_nodes=10, dim=384):
    data = HeteroData()
    data["object"].x = torch.randn(num_nodes, dim)
    data["object"].frame_id = torch.arange(num_nodes)

    # Spatial edges
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    data["object", "spatial_rel", "object"].edge_index = torch.tensor(
        [src, dst], dtype=torch.long
    )

    # Temporal edges
    even = list(range(0, num_nodes, 2))
    data["object", "temporal", "object"].edge_index = torch.tensor(
        [even[:-1], even[1:]], dtype=torch.long
    )

    return data


def test_encoder_output_shapes():
    config = BenchmarkConfig(gnn_hidden_dim=256, gnn_num_layers=3, gnn_num_heads=4)
    encoder = HeteroGATv2Encoder(config)
    data = make_test_subgraph(num_nodes=10)
    query_emb = torch.randn(384)

    node_emb, attn_scores, graph_emb = encoder(data, query_emb)

    assert node_emb.shape == (10, 256)
    assert attn_scores.shape == (10,)
    assert graph_emb.shape == (256,)


def test_encoder_attention_normalized():
    config = BenchmarkConfig()
    encoder = HeteroGATv2Encoder(config)
    data = make_test_subgraph(num_nodes=10)
    query_emb = torch.randn(384)

    _, attn_scores, _ = encoder(data, query_emb)

    # Scores should be in [0, 1]
    assert attn_scores.min() >= 0.0
    assert attn_scores.max() <= 1.0


def test_encoder_no_temporal_edges():
    """Encoder works when only spatial edges exist."""
    config = BenchmarkConfig()
    encoder = HeteroGATv2Encoder(config)
    data = HeteroData()
    data["object"].x = torch.randn(5, 384)
    data["object", "spatial_rel", "object"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]], dtype=torch.long
    )
    data["object", "temporal", "object"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    query_emb = torch.randn(384)

    node_emb, attn_scores, graph_emb = encoder(data, query_emb)
    assert node_emb.shape == (5, 256)


def test_encoder_gradient_flows():
    config = BenchmarkConfig()
    encoder = HeteroGATv2Encoder(config)
    data = make_test_subgraph(num_nodes=8)
    query_emb = torch.randn(384)

    node_emb, attn_scores, graph_emb = encoder(data, query_emb)
    loss = graph_emb.sum()
    loss.backward()

    for param in encoder.parameters():
        if param.requires_grad:
            assert param.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hetero_encoder.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/gnn/hetero_encoder.py
"""Heterogeneous GATv2 encoder using HeteroConv for video scene graphs."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv
from src.config import BenchmarkConfig


class HeteroGATv2Encoder(nn.Module):
    """Type-aware GATv2 encoder for HeteroData scene graphs.

    Uses HeteroConv to wrap per-edge-type GATv2Conv layers,
    providing direct access to attention weights for verbalization.
    """

    SPATIAL_EDGE = ("object", "spatial_rel", "object")
    TEMPORAL_EDGE = ("object", "temporal", "object")

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim
        self.num_layers = config.gnn_num_layers
        self.num_heads = config.gnn_num_heads
        self.dropout = config.gnn_dropout
        self.input_dim = config.embedding_dim  # 384

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Per-layer HeteroConv with GATv2Conv per edge type
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = self.hidden_dim
            # GATv2Conv: heads split the output dim
            head_dim = self.hidden_dim // self.num_heads

            conv = HeteroConv(
                {
                    self.SPATIAL_EDGE: GATv2Conv(
                        in_dim, head_dim, heads=self.num_heads,
                        concat=True, dropout=self.dropout,
                        add_self_loops=False,
                    ),
                    self.TEMPORAL_EDGE: GATv2Conv(
                        in_dim, head_dim, heads=self.num_heads,
                        concat=True, dropout=self.dropout,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.dropout_layer = nn.Dropout(self.dropout)

        # Attention pooling for graph embedding
        self.pool_gate = nn.Linear(self.hidden_dim, 1)

    def forward(
        self, data: HeteroData, query_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            data: HeteroData subgraph with 'object' nodes and edge types.
            query_embedding: Query embedding tensor of shape (input_dim,).

        Returns:
            Tuple of:
                - node_embeddings: (num_nodes, hidden_dim)
                - attention_scores: (num_nodes,) normalized to [0, 1]
                - graph_embedding: (hidden_dim,)
        """
        x = self.input_proj(data["object"].x)

        # Query conditioning: broadcast and add
        q = self.query_proj(query_embedding.unsqueeze(0))  # (1, hidden_dim)
        x = x + q.expand_as(x)

        # Build edge_index dict
        edge_index_dict = {}
        for etype in [self.SPATIAL_EDGE, self.TEMPORAL_EDGE]:
            if etype in data.edge_types and data[etype].edge_index.shape[1] > 0:
                edge_index_dict[etype] = data[etype].edge_index

        # Collect attention weights across layers
        all_attn_weights = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if not edge_index_dict:
                # No edges — skip message passing
                break

            x_out = conv({"object": x}, edge_index_dict)
            x_new = x_out.get("object", x)

            # Residual + norm + dropout
            x = norm(x + self.dropout_layer(x_new))

            # Extract attention weights from each GATv2Conv
            layer_attn = self._extract_attention(conv, data, edge_index_dict)
            if layer_attn is not None:
                all_attn_weights.append(layer_attn)

        # Aggregate attention to node-level scores
        num_nodes = x.shape[0]
        if all_attn_weights:
            attn_scores = self._aggregate_attention(
                all_attn_weights, num_nodes, edge_index_dict
            )
        else:
            attn_scores = torch.ones(num_nodes, device=x.device) / num_nodes

        # Attention pooling for graph embedding
        gate = torch.sigmoid(self.pool_gate(x)).squeeze(-1)  # (num_nodes,)
        graph_emb = (x * gate.unsqueeze(-1)).sum(dim=0) / (gate.sum() + 1e-8)

        return x, attn_scores, graph_emb

    def _extract_attention(self, conv, data, edge_index_dict):
        """Extract attention weights from a HeteroConv layer."""
        attn_per_edge = {}
        for etype, subconv in conv.convs.items():
            if etype in edge_index_dict:
                # GATv2Conv stores alpha after forward pass via return_attention_weights
                # We re-run with return_attention_weights=True
                # But this is expensive, so we use the stored _alpha attribute
                if hasattr(subconv, '_alpha') and subconv._alpha is not None:
                    attn_per_edge[etype] = subconv._alpha.detach()
        return attn_per_edge if attn_per_edge else None

    def _aggregate_attention(self, all_attn_weights, num_nodes, edge_index_dict):
        """Aggregate edge attention weights to node-level scores."""
        node_scores = torch.zeros(num_nodes)

        for layer_attn in all_attn_weights:
            for etype, alpha in layer_attn.items():
                if etype in edge_index_dict:
                    ei = edge_index_dict[etype]
                    # Average attention heads: alpha is (num_edges, num_heads)
                    avg_alpha = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
                    # Scatter to destination nodes
                    dst = ei[1]
                    for j in range(len(dst)):
                        d = int(dst[j])
                        if d < num_nodes:
                            node_scores[d] += float(avg_alpha[j])

        # Normalize to [0, 1]
        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()

        return node_scores
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hetero_encoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/gnn/hetero_encoder.py tests/test_hetero_encoder.py
git commit -m "feat: add HeteroGATv2 encoder with per-edge-type attention"
```

---

## Task 8: Scene Graph Verbalizer

**Files:**
- Create: `src/verbalization/scene_verbalizer.py`
- Test: `tests/test_scene_verbalizer.py`

**Step 1: Write the failing test**

```python
# tests/test_scene_verbalizer.py
import pytest
import torch
from torch_geometric.data import HeteroData
from src.verbalization.scene_verbalizer import SceneVerbalizer
from src.config import BenchmarkConfig


def make_annotated_subgraph():
    """Subgraph with object class names and spatial relations."""
    data = HeteroData()
    data["object"].x = torch.randn(4, 384)
    data["object"].frame_id = torch.tensor([0, 0, 1, 1])
    # Store class names as metadata
    data.object_names = ["person", "cup", "person", "table"]
    data.spatial_predicates = ["holding", "sitting_on"]

    data["object", "spatial_rel", "object"].edge_index = torch.tensor(
        [[0, 2], [1, 3]], dtype=torch.long
    )
    data["object", "temporal", "object"].edge_index = torch.tensor(
        [[0], [2]], dtype=torch.long
    )
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
    # Should not have more lines than top_k_triples
    lines = [l for l in text.strip().split("\n") if l.strip()]
    assert len(lines) <= 2


def test_verbalizer_unweighted_mode():
    """Unweighted mode for baseline comparison (no GNN attention)."""
    config = BenchmarkConfig()
    verbalizer = SceneVerbalizer(config)
    data = make_annotated_subgraph()
    text = verbalizer.verbalize_unweighted(data)
    assert isinstance(text, str)
    assert len(text) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scene_verbalizer.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/verbalization/scene_verbalizer.py
"""Verbalizes scene graph subgraphs into natural language for LLM prompts."""

from typing import List, Optional
import torch
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig


class SceneVerbalizer:
    """Converts attention-weighted scene graph subgraphs to text."""

    def __init__(self, config: BenchmarkConfig):
        self.top_k = config.top_k_triples

    def verbalize(
        self,
        data: HeteroData,
        attention_scores: torch.Tensor,
    ) -> str:
        """Verbalize subgraph using GNN attention scores to rank triples.

        Args:
            data: HeteroData subgraph with object_names and spatial_predicates.
            attention_scores: Per-node attention scores from GNN.

        Returns:
            Natural language description of top-K attention-weighted triples.
        """
        triples = self._extract_triples(data)
        scored = self._score_triples(triples, attention_scores)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]
        return self._format_triples(top)

    def verbalize_unweighted(self, data: HeteroData) -> str:
        """Verbalize subgraph without attention weighting (baseline)."""
        triples = self._extract_triples(data)
        # Uniform scoring
        scored = [(t, 1.0) for t in triples]
        top = scored[: self.top_k]
        return self._format_triples(top)

    def _extract_triples(self, data: HeteroData) -> List[dict]:
        """Extract subject-predicate-object triples from HeteroData."""
        triples = []
        names = getattr(data, "object_names", None)
        predicates = getattr(data, "spatial_predicates", None)

        # Spatial relations
        etype = ("object", "spatial_rel", "object")
        if etype in data.edge_types:
            ei = data[etype].edge_index
            for i in range(ei.shape[1]):
                src, dst = int(ei[0, i]), int(ei[1, i])
                subj = names[src] if names and src < len(names) else f"object_{src}"
                obj = names[dst] if names and dst < len(names) else f"object_{dst}"
                pred = predicates[i] if predicates and i < len(predicates) else "related_to"
                frame = int(data["object"].frame_id[src]) if hasattr(data["object"], "frame_id") else -1
                triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "frame": frame,
                    "src_idx": src,
                    "dst_idx": dst,
                })

        # Temporal relations
        etype = ("object", "temporal", "object")
        if etype in data.edge_types:
            ei = data[etype].edge_index
            for i in range(ei.shape[1]):
                src, dst = int(ei[0, i]), int(ei[1, i])
                name = names[src] if names and src < len(names) else f"object_{src}"
                frame_a = int(data["object"].frame_id[src]) if hasattr(data["object"], "frame_id") else -1
                frame_b = int(data["object"].frame_id[dst]) if hasattr(data["object"], "frame_id") else -1
                triples.append({
                    "subject": name,
                    "predicate": "appears_across_frames",
                    "object": f"frame {frame_a} → {frame_b}",
                    "frame": frame_a,
                    "src_idx": src,
                    "dst_idx": dst,
                })

        return triples

    def _score_triples(self, triples, attention_scores):
        """Score triples by average attention of their source and destination."""
        scored = []
        for t in triples:
            src_score = float(attention_scores[t["src_idx"]]) if t["src_idx"] < len(attention_scores) else 0.0
            dst_score = float(attention_scores[t["dst_idx"]]) if t["dst_idx"] < len(attention_scores) else 0.0
            avg = (src_score + dst_score) / 2.0
            scored.append((t, avg))
        return scored

    def _format_triples(self, scored_triples):
        """Format scored triples as numbered natural language lines."""
        lines = []
        for i, (t, score) in enumerate(scored_triples, 1):
            if t["predicate"] == "appears_across_frames":
                line = f"{i}. {t['subject']} {t['object']}"
            else:
                line = f"{i}. {t['subject']} {t['predicate']} {t['object']} (frame {t['frame']})"
            lines.append(line)
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scene_verbalizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/verbalization/__init__.py src/verbalization/scene_verbalizer.py tests/test_scene_verbalizer.py
git commit -m "feat: add scene graph verbalizer with attention-weighted and baseline modes"
```

---

## Task 9: Benchmark Evaluator

**Files:**
- Create: `src/evaluation/benchmark.py`
- Test: `tests/test_benchmark.py`

**Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import pytest
import torch
from src.evaluation.benchmark import BenchmarkEvaluator


def test_exact_match():
    evaluator = BenchmarkEvaluator()
    assert evaluator.exact_match("cup", "cup") == 1.0
    assert evaluator.exact_match("Cup", "cup") == 1.0
    assert evaluator.exact_match("glass", "cup") == 0.0


def test_f1_score():
    evaluator = BenchmarkEvaluator()
    assert evaluator.f1("the red cup", "the red cup") == 1.0
    assert evaluator.f1("the red cup", "the blue cup") > 0.0
    assert evaluator.f1("xyz", "abc") == 0.0


def test_retrieval_hit_rate():
    evaluator = BenchmarkEvaluator()
    selected_nodes = [0, 1, 2, 5, 8]
    answer_nodes = [2, 5]
    rate = evaluator.retrieval_hit_rate(selected_nodes, answer_nodes)
    assert rate == 1.0  # Both answer nodes found

    rate = evaluator.retrieval_hit_rate(selected_nodes, [2, 99])
    assert rate == 0.5  # Only one of two found


def test_attention_precision():
    evaluator = BenchmarkEvaluator()
    attn = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7])
    answer_nodes = [0, 2]  # Nodes 0 and 2 are answers
    precision = evaluator.attention_precision(attn, answer_nodes, top_k=3)
    # Top 3 by attention: nodes 0 (0.9), 2 (0.8), 4 (0.7)
    # 2 of 3 are answer nodes → precision = 2/3
    assert abs(precision - 2 / 3) < 1e-6


def test_aggregate_metrics():
    evaluator = BenchmarkEvaluator()
    results = [
        {"exact_match": 1.0, "f1": 1.0, "retrieval_hit_rate": 1.0},
        {"exact_match": 0.0, "f1": 0.5, "retrieval_hit_rate": 0.5},
    ]
    agg = evaluator.aggregate(results)
    assert agg["exact_match"] == 0.5
    assert agg["f1"] == 0.75
    assert agg["retrieval_hit_rate"] == 0.75
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/evaluation/benchmark.py
"""Benchmark evaluation metrics for video QA."""

from typing import Dict, List
import torch


class BenchmarkEvaluator:
    """Computes QA and retrieval metrics for the benchmark."""

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Case-insensitive exact match."""
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    def f1(self, prediction: str, ground_truth: str) -> float:
        """Token-level F1 between prediction and ground truth."""
        pred_tokens = set(prediction.strip().lower().split())
        gt_tokens = set(ground_truth.strip().lower().split())
        if not pred_tokens or not gt_tokens:
            return 0.0
        common = pred_tokens & gt_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def retrieval_hit_rate(
        self, selected_nodes: List[int], answer_nodes: List[int]
    ) -> float:
        """Fraction of answer nodes found in retrieved subgraph."""
        if not answer_nodes:
            return 0.0
        selected_set = set(selected_nodes)
        hits = sum(1 for n in answer_nodes if n in selected_set)
        return hits / len(answer_nodes)

    def attention_precision(
        self,
        attention_scores: torch.Tensor,
        answer_nodes: List[int],
        top_k: int = 5,
    ) -> float:
        """Fraction of top-K attended nodes that are answer nodes."""
        if top_k == 0:
            return 0.0
        _, top_indices = torch.topk(attention_scores, min(top_k, len(attention_scores)))
        top_set = set(top_indices.tolist())
        answer_set = set(answer_nodes)
        hits = len(top_set & answer_set)
        return hits / len(top_set)

    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate per-example metrics into averages."""
        if not results:
            return {}
        keys = results[0].keys()
        return {k: sum(r[k] for r in results) / len(results) for k in keys}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/benchmark.py tests/test_benchmark.py
git commit -m "feat: add benchmark evaluator with EM, F1, hit rate, attention precision"
```

---

## Task 10: Create `__init__.py` Files for New Modules

**Files:**
- Create: `src/verbalization/__init__.py`
- Create: `src/evaluation/__init__.py`

**Step 1: Create empty init files**

```python
# src/verbalization/__init__.py
# src/evaluation/__init__.py
```

(These are already handled in the commit steps of Tasks 8 and 9, but listed here for completeness.)

**Step 2: Commit** (if not already committed above)

```bash
git add src/verbalization/__init__.py src/evaluation/__init__.py
git commit -m "chore: add __init__.py for verbalization and evaluation modules"
```

---

## Task 11: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/test_integration_pipeline.py`

**Step 1: Write the integration test**

```python
# tests/test_integration_pipeline.py
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
        gnn_hidden_dim=64,  # Small for testing
        gnn_num_layers=2,
        gnn_num_heads=2,
    )

    # 1. Build scene graph
    builder = SceneGraphBuilder(config, embedding_dim=384)
    ag = make_ag_annotations()
    scene_graph = builder.build(ag)
    assert isinstance(scene_graph, HeteroData)
    assert scene_graph["object"].num_nodes == 15  # 3 objects × 5 frames

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
    # Attach mock object names
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
```

**Step 2: Run the integration test**

Run: `python -m pytest tests/test_integration_pipeline.py -v -s`
Expected: PASS with printed output

**Step 3: Commit**

```bash
git add tests/test_integration_pipeline.py
git commit -m "test: add end-to-end smoke test for video scene graph pipeline"
```

---

## Task 12: Update CLAUDE.md and Memory

**Files:**
- Modify: `CLAUDE.md`
- Modify: `C:\Users\User\.claude\projects\C--Users-User-arcOS-benchmark-colab\memory\MEMORY.md`

**Step 1: Update CLAUDE.md**

Add a new section documenting the video scene graph architecture alongside the existing Freebase documentation. Update the module structure to include new modules.

**Step 2: Update memory**

Update MEMORY.md to reflect the architectural pivot and new module paths.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for video scene graph architecture"
```

---

## Dependency Graph

```
Task 1 (Config) ← Task 2 (AGQA Loader)
Task 1 (Config) ← Task 3 (Scene Graph Builder)
Task 1 (Config) ← Task 4 (Video Index)
Task 1 (Config) ← Task 5 (PCST Adapter)
Task 4 + Task 5 ← Task 6 (Video Retriever)
Task 1 (Config) ← Task 7 (HeteroGATv2 Encoder)
Task 1 (Config) ← Task 8 (Verbalizer)
Task 1 (Config) ← Task 9 (Evaluator)
Task 8 + Task 9 ← Task 10 (Init files — concurrent with 8/9)
Tasks 3,6,7,8,9 ← Task 11 (Integration test)
Task 11 ← Task 12 (Docs update)
```

**Parallel execution waves:**
- Wave 1: Task 1
- Wave 2: Tasks 2, 3, 4, 5 (all depend only on Task 1)
- Wave 3: Tasks 6, 7, 8, 9, 10 (6 depends on 4+5; others depend on Task 1)
- Wave 4: Task 11
- Wave 5: Task 12
