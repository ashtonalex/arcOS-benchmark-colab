"""
SARM scoring functions (Section 5).

Pure math — no I/O, no external dependencies.
"""

import math
from typing import Optional

from .models import (
    JudgeVerdict,
    AggregatedScore,
    ScenarioResult,
    TierScore,
    TIER_CALIBRATION,
    TIER_STANDARD,
    TIER_ADVERSARIAL,
)


def _validate_score(value: float, name: str) -> None:
    """Raise ValueError if score is outside [1.0, 5.0]."""
    if not (1.0 <= value <= 5.0):
        raise ValueError(f"{name} must be in [1.0, 5.0], got {value}")


def compute_phi(C: float, P: float, hallucination: bool) -> float:
    """Compute the Critical Failure Multiplier (Section 5.3).

    Φ = 0.0 if hallucination, else sqrt(C*P)/5
    """
    if hallucination:
        return 0.0
    _validate_score(C, "C")
    _validate_score(P, "P")
    return math.sqrt(C * P) / 5.0


def compute_sarm_score(
    C: float,
    P: float,
    A: float,
    U: float,
    phi: float,
    weights: Optional[dict] = None,
) -> float:
    """Compute S_SARM (Section 5.1).

    S_SARM = (w_C*C + w_P*P + w_A*A + w_U*U) * Φ
    """
    if weights is None:
        weights = {"C": 0.40, "P": 0.30, "A": 0.20, "U": 0.10}

    _validate_score(C, "C")
    _validate_score(P, "P")
    _validate_score(A, "A")
    _validate_score(U, "U")

    weighted = weights["C"] * C + weights["P"] * P + weights["A"] * A + weights["U"] * U
    return weighted * phi


def aggregate_judges(
    verdicts: list[JudgeVerdict],
    threshold: float = 1.0,
) -> AggregatedScore:
    """Aggregate verdicts from multiple judges.

    Returns mean scores, flags disagreements > threshold per dimension.
    Hallucination is flagged if ANY judge detects one.
    """
    if not verdicts:
        raise ValueError("Must have at least one verdict")

    cs = [v.causal_consistency for v in verdicts]
    ps = [v.spatiotemporal_coherence for v in verdicts]
    a_s = [v.attribution_accuracy for v in verdicts]
    us = [v.uncertainty_calibration for v in verdicts]

    mean_c = sum(cs) / len(cs)
    mean_p = sum(ps) / len(ps)
    mean_a = sum(a_s) / len(a_s)
    mean_u = sum(us) / len(us)

    hallucination = any(v.hallucination_detected for v in verdicts)

    disagreement_flags = {}
    for name, scores in [("C", cs), ("P", ps), ("A", a_s), ("U", us)]:
        spread = max(scores) - min(scores)
        if spread > threshold:
            disagreement_flags[name] = spread

    phi = compute_phi(mean_c, mean_p, hallucination)
    sarm = compute_sarm_score(mean_c, mean_p, mean_a, mean_u, phi)

    return AggregatedScore(
        causal_consistency=mean_c,
        spatiotemporal_coherence=mean_p,
        attribution_accuracy=mean_a,
        uncertainty_calibration=mean_u,
        phi=phi,
        sarm_score=sarm,
        hallucination_detected=hallucination,
        disagreement_flags=disagreement_flags,
    )


def aggregate_scenarios(results: list[ScenarioResult]) -> list[TierScore]:
    """Aggregate scenario results into per-tier scores (Section 7.2)."""
    tiers = {}
    for r in results:
        tier = r.tier
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append(r)

    tier_scores = []
    for tier_name in [TIER_CALIBRATION, TIER_STANDARD, TIER_ADVERSARIAL]:
        scenarios = tiers.get(tier_name, [])
        if not scenarios:
            continue

        cs = [s.aggregated_score.causal_consistency for s in scenarios]
        ps = [s.aggregated_score.spatiotemporal_coherence for s in scenarios]
        a_s = [s.aggregated_score.attribution_accuracy for s in scenarios]
        us = [s.aggregated_score.uncertainty_calibration for s in scenarios]
        phis = [s.aggregated_score.phi for s in scenarios]
        sarms = [s.aggregated_score.sarm_score for s in scenarios]

        n = len(scenarios)
        mean_sarm = sum(sarms) / n
        std_sarm = math.sqrt(sum((s - mean_sarm) ** 2 for s in sarms) / n) if n > 1 else 0.0

        tier_scores.append(TierScore(
            tier=tier_name,
            mean_c=sum(cs) / n,
            mean_p=sum(ps) / n,
            mean_a=sum(a_s) / n,
            mean_u=sum(us) / n,
            mean_phi=sum(phis) / n,
            mean_sarm=mean_sarm,
            std_sarm=std_sarm,
            scenario_count=n,
        ))

    return tier_scores
