"""Benchmark evaluation metrics for video QA."""

from typing import Dict, List
import torch


class BenchmarkEvaluator:
    """Computes QA and retrieval metrics for the benchmark."""

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    def f1(self, prediction: str, ground_truth: str) -> float:
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

    def retrieval_hit_rate(self, selected_nodes: List[int], answer_nodes: List[int]) -> float:
        if not answer_nodes:
            return 0.0
        selected_set = set(selected_nodes)
        hits = sum(1 for n in answer_nodes if n in selected_set)
        return hits / len(answer_nodes)

    def attention_precision(self, attention_scores: torch.Tensor, answer_nodes: List[int], top_k: int = 5) -> float:
        if top_k == 0:
            return 0.0
        _, top_indices = torch.topk(attention_scores, min(top_k, len(attention_scores)))
        top_set = set(top_indices.tolist())
        answer_set = set(answer_nodes)
        hits = len(top_set & answer_set)
        return hits / len(top_set)

    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        keys = results[0].keys()
        return {k: sum(r[k] for r in results) / len(results) for k in keys}
