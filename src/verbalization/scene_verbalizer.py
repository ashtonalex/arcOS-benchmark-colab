"""Verbalizes scene graph subgraphs into natural language for LLM prompts."""

from typing import List, Optional
import torch
from torch_geometric.data import HeteroData
from src.config import BenchmarkConfig


class SceneVerbalizer:
    """Converts attention-weighted scene graph subgraphs to text."""

    def __init__(self, config: BenchmarkConfig):
        self.top_k = config.top_k_triples

    def verbalize(self, data: HeteroData, attention_scores: torch.Tensor) -> str:
        triples = self._extract_triples(data)
        scored = self._score_triples(triples, attention_scores)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self.top_k]
        return self._format_triples(top)

    def verbalize_unweighted(self, data: HeteroData) -> str:
        triples = self._extract_triples(data)
        scored = [(t, 1.0) for t in triples]
        top = scored[:self.top_k]
        return self._format_triples(top)

    def _extract_triples(self, data: HeteroData) -> List[dict]:
        triples = []
        names = getattr(data, "object_names", None)
        predicates = getattr(data, "spatial_predicates", None)

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
                    "subject": subj, "predicate": pred, "object": obj,
                    "frame": frame, "src_idx": src, "dst_idx": dst,
                })

        etype = ("object", "temporal", "object")
        if etype in data.edge_types:
            ei = data[etype].edge_index
            for i in range(ei.shape[1]):
                src, dst = int(ei[0, i]), int(ei[1, i])
                name = names[src] if names and src < len(names) else f"object_{src}"
                frame_a = int(data["object"].frame_id[src]) if hasattr(data["object"], "frame_id") else -1
                frame_b = int(data["object"].frame_id[dst]) if hasattr(data["object"], "frame_id") else -1
                triples.append({
                    "subject": name, "predicate": "appears_across_frames",
                    "object": f"frame {frame_a} \u2192 {frame_b}",
                    "frame": frame_a, "src_idx": src, "dst_idx": dst,
                })
        return triples

    def _score_triples(self, triples, attention_scores):
        scored = []
        for t in triples:
            src_score = float(attention_scores[t["src_idx"]]) if t["src_idx"] < len(attention_scores) else 0.0
            dst_score = float(attention_scores[t["dst_idx"]]) if t["dst_idx"] < len(attention_scores) else 0.0
            avg = (src_score + dst_score) / 2.0
            scored.append((t, avg))
        return scored

    def _format_triples(self, scored_triples):
        lines = []
        for i, (t, score) in enumerate(scored_triples, 1):
            if t["predicate"] == "appears_across_frames":
                line = f"{i}. {t['subject']} {t['object']}"
            else:
                line = f"{i}. {t['subject']} {t['predicate']} {t['object']} (frame {t['frame']})"
            lines.append(line)
        return "\n".join(lines)
