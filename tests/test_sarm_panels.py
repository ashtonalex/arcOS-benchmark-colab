"""Tests for SARM panel classes — mock LLM responses."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.sarm.panels import (
    InquisitorGenerator,
    InquisitorValidator,
    Judge,
    _parse_probes,
    _parse_validation,
    _parse_judge_verdict,
)
from src.sarm.models import Probe, JudgeVerdict
from src.llm.client import LLMResponse


# ========== Probe Parsing ==========


def test_parse_well_formed_probes():
    """Parser extracts well-formed probe blocks."""
    text = """Target: person_42 (0.92, hub)
Probe Type: Spatial Neglect
Question: "If person_42 were relocated 50m away, would the interaction hold?"
Rationale: Tests spatial awareness of proximity-dependent interaction
Expected Behavior: Model should acknowledge distance invalidates interaction

Target: cup_7 (0.45, leaf)
Probe Type: Spurious Attribution
Question: "You cited cup_7 as critical. If cup_7 were removed, would your conclusion change?"
Rationale: Tests whether low-attention node is truly necessary
Expected Behavior: Model should recognize cup_7 is not causally critical"""

    probes = _parse_probes(text)
    assert len(probes) == 2
    assert probes[0].target_node == "person_42"
    assert probes[0].probe_type == "Spatial Neglect"
    assert probes[0].attention_score == pytest.approx(0.92)
    assert probes[0].role_in_subgraph == "hub"
    assert probes[1].target_node == "cup_7"
    assert probes[1].probe_type == "Spurious Attribution"


def test_parse_partial_probes():
    """Parser skips blocks missing required fields."""
    text = """Target: node_1 (0.5, leaf)
Probe Type: Temporal Confusion
Question: "What if the timing changed?"
Rationale: Tests temporal reasoning

Some random text here without proper format

Target: node_2
Question: "Missing probe type"
"""
    probes = _parse_probes(text)
    # First block is complete, second is missing Probe Type
    assert len(probes) == 1
    assert probes[0].target_node == "node_1"


def test_parse_empty_returns_empty():
    """Empty input returns empty list."""
    assert _parse_probes("") == []
    assert _parse_probes("random text with no probes") == []


# ========== Validation Parsing ==========


def test_parse_validation_robust():
    """Validator parses ROBUST result correctly."""
    probe = Probe(
        target_node="n1", probe_type="Spatial Neglect",
        question="q?", rationale="r", expected_behavior="e"
    )
    text = """
Failure Mode: Spatial Neglect
Ground Truth Verifiable: YES - distance is explicitly stated
Ablation Test: PASS - without relocation, answer is trivially "yes"
Robustness Score: ROBUST
"""
    result = _parse_validation(text, probe)
    assert result is not None
    assert result.robustness_score == "ROBUST"
    assert result.ground_truth_verifiable is True
    assert result.ablation_test_passed is True


def test_parse_validation_not_robust_returns_none():
    """NOT ROBUST probes return None."""
    probe = Probe(
        target_node="n1", probe_type="Spatial Neglect",
        question="q?", rationale="r", expected_behavior="e"
    )
    text = """
Robustness Score: NOT ROBUST
"""
    result = _parse_validation(text, probe)
    assert result is None


def test_parse_validation_malformed_defaults_not_robust():
    """Malformed output defaults to NOT ROBUST → returns None."""
    probe = Probe(
        target_node="n1", probe_type="Spatial Neglect",
        question="q?", rationale="r", expected_behavior="e"
    )
    result = _parse_validation("totally invalid output", probe)
    assert result is None


# ========== Judge Verdict Parsing ==========


def test_parse_judge_scores():
    """Judge parser extracts all scores correctly."""
    text = """HALLUCINATION_DETECTED: FALSE

Causal Consistency (C): 4 - Reasoning mostly holds under counterfactual
Spatio-Temporal Coherence (P): 5 - Perfect spatial reasoning
Attribution Accuracy (A): 3 - Some rationalization observed
Uncertainty Calibration (U): 4 - Appropriate caution shown"""

    verdict = _parse_judge_verdict(text, "test/model")
    assert verdict.hallucination_detected is False
    assert verdict.causal_consistency == 4.0
    assert verdict.spatiotemporal_coherence == 5.0
    assert verdict.attribution_accuracy == 3.0
    assert verdict.uncertainty_calibration == 4.0
    assert verdict.model_id == "test/model"


def test_parse_judge_hallucination_detected():
    """Judge parser detects hallucination with rule citation."""
    text = """HALLUCINATION_DETECTED: TRUE
Rule 2: Model asserts "person is holding book" but no such triple exists

Causal Consistency (C): 2 - Logic breaks
Spatio-Temporal Coherence (P): 1 - Geometric hallucination
Attribution Accuracy (A): 1 - Blind guessing
Uncertainty Calibration (U): 1 - Total delusion"""

    verdict = _parse_judge_verdict(text, "test/model")
    assert verdict.hallucination_detected is True
    assert verdict.hallucination_rule == "Rule 2"
    assert verdict.causal_consistency == 2.0


def test_parse_judge_missing_scores_default_to_3():
    """Missing scores default to 3.0 (conservative neutral)."""
    text = """HALLUCINATION_DETECTED: FALSE
Some text without proper score format"""

    verdict = _parse_judge_verdict(text, "m")
    assert verdict.causal_consistency == 3.0
    assert verdict.spatiotemporal_coherence == 3.0
    assert verdict.attribution_accuracy == 3.0
    assert verdict.uncertainty_calibration == 3.0


# ========== InquisitorGenerator ==========


@pytest.mark.asyncio
async def test_inquisitor_generates_probes():
    """InquisitorGenerator produces Probe objects from LLM output."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=LLMResponse(
        content="""Target: person_1 (0.9, hub)
Probe Type: Spatial Neglect
Question: "If person_1 were 100m away, would the interaction hold?"
Rationale: Tests spatial reasoning
Expected Behavior: Should acknowledge distance change""",
        model="test", usage={}, latency_ms=100, finish_reason="stop",
    ))

    gen = InquisitorGenerator(model="test/model", client=mock_client)
    probes = await gen.generate(
        question="What is happening?",
        verbalized_context="0.9 person holding cup",
        subgraph={"nodes": ["person_1", "cup_1"], "edges": ["person_1 --holding--> cup_1"]},
        num_probes=1,
    )

    assert len(probes) == 1
    assert probes[0].target_node == "person_1"


# ========== InquisitorValidator ==========


@pytest.mark.asyncio
async def test_validator_filters_to_robust():
    """InquisitorValidator returns only ROBUST probes."""
    mock_client = MagicMock()
    # First probe: ROBUST, Second probe: NOT ROBUST
    mock_client.complete = AsyncMock(side_effect=[
        LLMResponse(
            content="Ground Truth Verifiable: YES\nAblation Test: PASS\nRobustness Score: ROBUST",
            model="test", usage={}, latency_ms=50, finish_reason="stop",
        ),
        LLMResponse(
            content="Ground Truth Verifiable: NO\nAblation Test: FAIL\nRobustness Score: NOT ROBUST",
            model="test", usage={}, latency_ms=50, finish_reason="stop",
        ),
    ])

    val = InquisitorValidator(model="test/model", client=mock_client)
    probes = [
        Probe(target_node="n1", probe_type="Spatial Neglect",
              question="q1?", rationale="r1", expected_behavior="e1"),
        Probe(target_node="n2", probe_type="Temporal Confusion",
              question="q2?", rationale="r2", expected_behavior="e2"),
    ]
    validated = await val.validate(probes, "context", "question")
    assert len(validated) == 1
    assert validated[0].probe.target_node == "n1"


# ========== Judge ==========


@pytest.mark.asyncio
async def test_judge_evaluates():
    """Judge produces a JudgeVerdict from LLM output."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=LLMResponse(
        content="""HALLUCINATION_DETECTED: FALSE

Causal Consistency (C): 4 - Good
Spatio-Temporal Coherence (P): 3 - Adequate
Attribution Accuracy (A): 4 - Aligned
Uncertainty Calibration (U): 3 - Neutral""",
        model="test", usage={}, latency_ms=200, finish_reason="stop",
    ))

    judge = Judge(model="test/model", client=mock_client)
    probe = Probe(target_node="n1", probe_type="Spatial Neglect",
                  question="q?", rationale="r", expected_behavior="e")

    verdict = await judge.evaluate(
        question="What happened?",
        verbalized_context="0.9 person holding cup",
        base_response="Person interacted with cup.",
        probe=probe,
        probe_response="If moved, interaction wouldn't hold.",
    )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.causal_consistency == 4.0
    assert verdict.spatiotemporal_coherence == 3.0
    assert not verdict.hallucination_detected
