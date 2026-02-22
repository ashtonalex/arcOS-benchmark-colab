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
    assert rate == 1.0
    rate = evaluator.retrieval_hit_rate(selected_nodes, [2, 99])
    assert rate == 0.5


def test_attention_precision():
    evaluator = BenchmarkEvaluator()
    attn = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7])
    answer_nodes = [0, 2]
    precision = evaluator.attention_precision(attn, answer_nodes, top_k=3)
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
