"""Tests for SARM scoring functions — pure math with known values."""

import math

import pytest

from src.sarm.scorer import compute_phi, compute_sarm_score, aggregate_judges, aggregate_scenarios
from src.sarm.models import (
    JudgeVerdict,
    ScenarioResult,
    AggregatedScore,
    TIER_CALIBRATION,
    TIER_STANDARD,
    TIER_ADVERSARIAL,
)


# ========== compute_phi ==========


def test_phi_perfect_scores():
    """C=5, P=5 → Φ = 1.0"""
    assert compute_phi(5.0, 5.0, False) == pytest.approx(1.0)


def test_phi_four_four():
    """C=4, P=4 → Φ = 0.8"""
    assert compute_phi(4.0, 4.0, False) == pytest.approx(0.8)


def test_phi_three_three():
    """C=3, P=3 → Φ = 0.6"""
    assert compute_phi(3.0, 3.0, False) == pytest.approx(0.6)


def test_phi_imbalanced():
    """C=5, P=1 → Φ ≈ 0.447"""
    expected = math.sqrt(5.0 * 1.0) / 5.0
    assert compute_phi(5.0, 1.0, False) == pytest.approx(expected)


def test_phi_hallucination_is_zero():
    """Hallucination → Φ = 0.0 regardless of scores."""
    assert compute_phi(5.0, 5.0, True) == 0.0


def test_phi_out_of_range_raises():
    """Scores outside [1.0, 5.0] raise ValueError."""
    with pytest.raises(ValueError):
        compute_phi(0.5, 3.0, False)
    with pytest.raises(ValueError):
        compute_phi(3.0, 5.5, False)
    with pytest.raises(ValueError):
        compute_phi(6.0, 3.0, False)


# ========== compute_sarm_score ==========


def test_sarm_perfect_score():
    """All 5s with Φ=1.0 → S_SARM = 5.0"""
    phi = compute_phi(5.0, 5.0, False)
    score = compute_sarm_score(5.0, 5.0, 5.0, 5.0, phi)
    assert score == pytest.approx(5.0)


def test_sarm_hallucination_zero():
    """Hallucination → S_SARM = 0.0"""
    phi = compute_phi(5.0, 5.0, True)
    score = compute_sarm_score(5.0, 5.0, 5.0, 5.0, phi)
    assert score == 0.0


def test_sarm_custom_weights():
    """Custom weights produce expected result."""
    weights = {"C": 0.25, "P": 0.25, "A": 0.25, "U": 0.25}
    phi = compute_phi(4.0, 4.0, False)  # 0.8
    score = compute_sarm_score(4.0, 4.0, 4.0, 4.0, phi, weights)
    # weighted = 0.25*4 * 4 = 4.0; score = 4.0 * 0.8 = 3.2
    assert score == pytest.approx(3.2)


def test_sarm_out_of_range_raises():
    """Scores outside [1.0, 5.0] raise ValueError."""
    with pytest.raises(ValueError):
        compute_sarm_score(0.0, 3.0, 3.0, 3.0, 0.6)


# ========== aggregate_judges ==========


def test_aggregate_mean_scores():
    """Mean of 3 judges is computed correctly."""
    verdicts = [
        JudgeVerdict(model_id="m1", hallucination_detected=False,
                     causal_consistency=4.0, spatiotemporal_coherence=4.0,
                     attribution_accuracy=4.0, uncertainty_calibration=4.0),
        JudgeVerdict(model_id="m2", hallucination_detected=False,
                     causal_consistency=5.0, spatiotemporal_coherence=5.0,
                     attribution_accuracy=5.0, uncertainty_calibration=5.0),
        JudgeVerdict(model_id="m3", hallucination_detected=False,
                     causal_consistency=3.0, spatiotemporal_coherence=3.0,
                     attribution_accuracy=3.0, uncertainty_calibration=3.0),
    ]
    agg = aggregate_judges(verdicts)
    assert agg.causal_consistency == pytest.approx(4.0)
    assert agg.spatiotemporal_coherence == pytest.approx(4.0)
    assert agg.attribution_accuracy == pytest.approx(4.0)
    assert agg.uncertainty_calibration == pytest.approx(4.0)


def test_aggregate_hallucination_any():
    """Hallucination is flagged if ANY judge detects one."""
    verdicts = [
        JudgeVerdict(model_id="m1", hallucination_detected=False,
                     causal_consistency=4.0, spatiotemporal_coherence=4.0,
                     attribution_accuracy=4.0, uncertainty_calibration=4.0),
        JudgeVerdict(model_id="m2", hallucination_detected=True,
                     causal_consistency=4.0, spatiotemporal_coherence=4.0,
                     attribution_accuracy=4.0, uncertainty_calibration=4.0),
    ]
    agg = aggregate_judges(verdicts)
    assert agg.hallucination_detected is True
    assert agg.sarm_score == 0.0


def test_aggregate_disagreement_flags():
    """Flag dimensions with spread > threshold."""
    verdicts = [
        JudgeVerdict(model_id="m1", hallucination_detected=False,
                     causal_consistency=5.0, spatiotemporal_coherence=3.0,
                     attribution_accuracy=3.0, uncertainty_calibration=3.0),
        JudgeVerdict(model_id="m2", hallucination_detected=False,
                     causal_consistency=2.0, spatiotemporal_coherence=3.0,
                     attribution_accuracy=3.0, uncertainty_calibration=3.0),
    ]
    agg = aggregate_judges(verdicts, threshold=1.0)
    # C spread = 3.0, should be flagged
    assert "C" in agg.disagreement_flags
    # P spread = 0.0, should not be flagged
    assert "P" not in agg.disagreement_flags


def test_aggregate_empty_raises():
    """Empty verdicts list raises ValueError."""
    with pytest.raises(ValueError):
        aggregate_judges([])


# ========== aggregate_scenarios ==========


def test_aggregate_scenarios_by_tier():
    """Scenarios are grouped and aggregated by tier."""
    results = [
        ScenarioResult(
            scenario_id="s1", question="q", entity_count=3,
            tier=TIER_CALIBRATION, probe_results=[],
            aggregated_score=AggregatedScore(
                causal_consistency=5.0, spatiotemporal_coherence=5.0,
                attribution_accuracy=5.0, uncertainty_calibration=5.0,
                phi=1.0, sarm_score=5.0, hallucination_detected=False,
            ),
        ),
        ScenarioResult(
            scenario_id="s2", question="q", entity_count=10,
            tier=TIER_STANDARD, probe_results=[],
            aggregated_score=AggregatedScore(
                causal_consistency=3.0, spatiotemporal_coherence=3.0,
                attribution_accuracy=3.0, uncertainty_calibration=3.0,
                phi=0.6, sarm_score=1.8, hallucination_detected=False,
            ),
        ),
    ]
    tiers = aggregate_scenarios(results)
    assert len(tiers) == 2
    assert tiers[0].tier == TIER_CALIBRATION
    assert tiers[0].mean_sarm == pytest.approx(5.0)
    assert tiers[1].tier == TIER_STANDARD
    assert tiers[1].mean_sarm == pytest.approx(1.8)
