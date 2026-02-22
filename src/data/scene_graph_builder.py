"""Converts Action Genome annotations to PyG HeteroData."""

from typing import Any, Dict, List, Optional
import torch
import numpy as np
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig


class SceneGraphBuilder:
    """Builds PyG HeteroData from Action Genome frame annotations."""

    def __init__(self, config: BenchmarkConfig, embedding_dim: int = 384, embedder=None):
        self.config = config
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self._class_cache: Dict[str, np.ndarray] = {}

    def build(self, ag_annotations: Dict[str, Any]) -> HeteroData:
        data = HeteroData()
        frames = ag_annotations["frames"]

        node_list = []
        node_index_map = {}
        global_idx = 0
        for frame in frames:
            fid = frame["frame_id"]
            for obj in frame["objects"]:
                oid = obj["object_id"]
                node_list.append((fid, oid, obj["class"], obj.get("bbox")))
                node_index_map[(fid, oid)] = global_idx
                global_idx += 1

        x = self._embed_object_classes([n[2] for n in node_list])
        data["object"].x = torch.tensor(x, dtype=torch.float32)
        data["object"].frame_id = torch.tensor([n[0] for n in node_list], dtype=torch.long)
        data["object"].object_class = torch.tensor(
            [hash(n[2]) % (2**31) for n in node_list], dtype=torch.long
        )

        spatial_src, spatial_dst = [], []
        for frame in frames:
            fid = frame["frame_id"]
            for rel in frame["relations"]:
                src_key = (fid, rel["subject_id"])
                dst_key = (fid, rel["object_id"])
                if src_key in node_index_map and dst_key in node_index_map:
                    spatial_src.append(node_index_map[src_key])
                    spatial_dst.append(node_index_map[dst_key])

        if spatial_src:
            data["object", "spatial_rel", "object"].edge_index = torch.tensor(
                [spatial_src, spatial_dst], dtype=torch.long
            )
        else:
            data["object", "spatial_rel", "object"].edge_index = torch.zeros((2, 0), dtype=torch.long)

        temporal_src, temporal_dst = [], []
        temporal_attrs = []
        obj_frames: Dict[int, List] = {}
        for fid, oid, _, bbox in node_list:
            obj_frames.setdefault(oid, []).append((fid, node_index_map[(fid, oid)]))

        for oid, appearances in obj_frames.items():
            appearances.sort(key=lambda x: x[0])
            for i in range(len(appearances) - 1):
                fid_a, idx_a = appearances[i]
                fid_b, idx_b = appearances[i + 1]
                temporal_src.append(idx_a)
                temporal_dst.append(idx_b)
                temporal_attrs.append([fid_b - fid_a, 0.0])

        if temporal_src:
            data["object", "temporal", "object"].edge_index = torch.tensor(
                [temporal_src, temporal_dst], dtype=torch.long
            )
            data["object", "temporal", "object"].edge_attr = torch.tensor(
                temporal_attrs, dtype=torch.float32
            )
        else:
            data["object", "temporal", "object"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["object", "temporal", "object"].edge_attr = torch.zeros((0, 2), dtype=torch.float32)

        data.video_id = ag_annotations["video_id"]
        data.num_frames = len(frames)
        return data

    def _embed_object_classes(self, class_names: List[str]) -> np.ndarray:
        if self.embedder is not None:
            unique_classes = list(set(class_names))
            to_embed = [c for c in unique_classes if c not in self._class_cache]
            if to_embed:
                embeddings = self.embedder.embed_texts(to_embed, show_progress=False)
                for cls, emb in zip(to_embed, embeddings):
                    self._class_cache[cls] = emb
            return np.array([self._class_cache[c] for c in class_names])
        else:
            rng = np.random.RandomState(42)
            cache = {}
            for c in set(class_names):
                cache[c] = rng.randn(self.embedding_dim).astype(np.float32)
            return np.array([cache[c] for c in class_names])
