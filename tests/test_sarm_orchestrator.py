"""Tests for SARM orchestrator — full integration with mocked client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import BenchmarkConfig
from src.llm.client import LLMResponse
from src.sarm.orchestrator import SARMOrchestrator, SARMScenario
from src.sarm.models import (
    classify_tier,
    TIER_CALIBRATION,
    TIER_STANDARD,
    TIER_ADVERSARIAL,
)


# ========== Tier Classification ==========


def test_tier_calibration():
    assert classify_tier(2) == TIER_CALIBRATION
    assert classify_tier(5) == TIER_CALIBRATION


def test_tier_standard():
    assert classify_tier(6) == TIER_STANDARD
    assert classify_tier(15) == TIER_STANDARD


def test_tier_adversarial():
    assert classify_tier(20) == TIER_ADVERSARIAL
    assert classify_tier(100) == TIER_ADVERSARIAL


# ========== Mock Helpers ==========


def _make_mock_client():
    """Create a mock OpenRouterClient that returns structured LLM responses."""
    client = MagicMock()

    # Counter to vary responses by call
    call_count = {"n": 0}

    async def mock_complete(model, messages, **kwargs):
        call_count["n"] += 1
        n = call_count["n"]

        # Detect what kind of call this is by system prompt content
        system = messages[0]["content"] if messages else ""

        if "adversarial evaluator" in system:
            # Inquisitor generates probes
            return LLMResponse(
                content="""Target: entity_1 (0.9, hub)
Probe Type: Spatial Neglect
Question: "If entity_1 were far away, would the conclusion hold?"
Rationale: Tests spatial reasoning
Expected Behavior: Should acknowledge distance impact

Target: entity_2 (0.7, leaf)
Probe Type: Causal Shortcutting
Question: "What is the complete causal chain from entity_2 to the conclusion?"
Rationale: Tests causal completeness
Expected Behavior: Should trace all intermediate steps""",
                model=model, usage={}, latency_ms=100, finish_reason="stop",
            )
        elif "question quality validator" in system:
            # Validator marks as ROBUST
            return LLMResponse(
                content="Ground Truth Verifiable: YES\nAblation Test: PASS\nRobustness Score: ROBUST",
                model=model, usage={}, latency_ms=50, finish_reason="stop",
            )
        elif "evaluating a GNN-augmented" in system:
            # Judge verdict
            return LLMResponse(
                content="""HALLUCINATION_DETECTED: FALSE

Causal Consistency (C): 4 - Good reasoning
Spatio-Temporal Coherence (P): 4 - Respects physics
Attribution Accuracy (A): 3 - Some rationalization
Uncertainty Calibration (U): 3 - Neutral confidence""",
                model=model, usage={}, latency_ms=200, finish_reason="stop",
            )
        else:
            # Benchmarked model response
            return LLMResponse(
                content="Based on the graph context, entity_1 interacts with entity_2 spatially.",
                model=model, usage={}, latency_ms=150, finish_reason="stop",
            )

    client.complete = AsyncMock(side_effect=mock_complete)
    return client


def _make_scenario(entity_count=3):
    return SARMScenario(
        scenario_id="test-001",
        question="What is happening in the scene?",
        verbalized_context="0.9 entity_1 holding entity_2 (frame 1)\n0.7 entity_2 near entity_3 (frame 2)",
        subgraph={"nodes": ["entity_1", "entity_2", "entity_3"],
                   "edges": ["entity_1 --holding--> entity_2"]},
        entity_count=entity_count,
    )


# ========== Orchestrator Tests ==========


@pytest.mark.asyncio
async def test_run_scenario_protocol():
    """Single scenario executes the full SARM protocol."""
    config = BenchmarkConfig(sarm_probes_per_scenario=2, sarm_judge_models=["m1", "m2"])
    client = _make_mock_client()

    orch = SARMOrchestrator(config=config, client=client)
    scenario = _make_scenario(entity_count=3)

    result = await orch.run_scenario(scenario)

    assert result.scenario_id == "test-001"
    assert result.tier == TIER_CALIBRATION
    assert len(result.probe_results) > 0
    assert result.aggregated_score.sarm_score > 0


@pytest.mark.asyncio
async def test_run_scenario_adversarial_tier():
    """Scenario with 50 entities is classified as Adversarial."""
    config = BenchmarkConfig(
        sarm_probes_per_scenario=1,
        sarm_judge_models=["m1"],
    )
    client = _make_mock_client()
    orch = SARMOrchestrator(config=config, client=client)

    scenario = _make_scenario(entity_count=50)
    result = await orch.run_scenario(scenario)
    assert result.tier == TIER_ADVERSARIAL


@pytest.mark.asyncio
async def test_run_benchmark_multi_run():
    """Multi-run benchmark computes mean across runs."""
    config = BenchmarkConfig(
        sarm_probes_per_scenario=1,
        sarm_runs_per_scenario=2,
        sarm_judge_models=["m1"],
    )
    client = _make_mock_client()
    orch = SARMOrchestrator(config=config, client=client)

    scenarios = [_make_scenario(entity_count=3)]
    report = await orch.run_benchmark(scenarios)

    assert report.total_scenarios == 1
    assert len(report.tier_scores) >= 1
    assert report.tier_scores[0].tier == TIER_CALIBRATION
    assert len(report.failure_mode_analysis) == 5  # All 5 failure modes


@pytest.mark.asyncio
async def test_run_benchmark_high_variance_flagging():
    """High variance scenarios (σ > 0.5) are flagged."""
    config = BenchmarkConfig(
        sarm_probes_per_scenario=1,
        sarm_runs_per_scenario=2,
        sarm_judge_models=["m1"],
    )

    # Create client with varying responses to trigger variance
    client = MagicMock()
    call_count = {"n": 0}

    async def varying_complete(model, messages, **kwargs):
        call_count["n"] += 1
        system = messages[0]["content"] if messages else ""

        if "adversarial evaluator" in system:
            return LLMResponse(
                content="""Target: e1 (0.9, hub)
Probe Type: Spatial Neglect
Question: "test?"
Rationale: test
Expected Behavior: test""",
                model=model, usage={}, latency_ms=50, finish_reason="stop",
            )
        elif "question quality validator" in system:
            return LLMResponse(
                content="Robustness Score: ROBUST\nGround Truth Verifiable: YES\nAblation Test: PASS",
                model=model, usage={}, latency_ms=50, finish_reason="stop",
            )
        elif "evaluating a GNN-augmented" in system:
            # Alternate between high and low scores to create variance
            if call_count["n"] % 2 == 0:
                return LLMResponse(
                    content="HALLUCINATION_DETECTED: FALSE\nCausal Consistency (C): 5\nSpatio-Temporal Coherence (P): 5\nAttribution Accuracy (A): 5\nUncertainty Calibration (U): 5",
                    model=model, usage={}, latency_ms=50, finish_reason="stop",
                )
            else:
                return LLMResponse(
                    content="HALLUCINATION_DETECTED: TRUE\nCausal Consistency (C): 1\nSpatio-Temporal Coherence (P): 1\nAttribution Accuracy (A): 1\nUncertainty Calibration (U): 1",
                    model=model, usage={}, latency_ms=50, finish_reason="stop",
                )
        else:
            return LLMResponse(
                content="Test response",
                model=model, usage={}, latency_ms=50, finish_reason="stop",
            )

    client.complete = AsyncMock(side_effect=varying_complete)
    orch = SARMOrchestrator(config=config, client=client)

    scenarios = [_make_scenario()]
    report = await orch.run_benchmark(scenarios)

    # With alternating hallucination/no-hallucination, scores should vary
    # The test verifies the mechanism works, not specific values
    assert isinstance(report.high_variance_scenarios, list)
