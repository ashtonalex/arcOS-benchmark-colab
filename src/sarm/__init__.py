"""
SARM (Spatio-temporal Adversarial Reasoning Methodology) benchmark layer.

Evaluates whether GNN-LLM pipelines make correct decisions for the right
reasons via adversarial counterfactual probing.
"""

from .models import (
    Probe,
    ValidatedProbe,
    JudgeVerdict,
    AggregatedScore,
    ProbeResult,
    ScenarioResult,
    BenchmarkReport,
)
from .scorer import compute_phi, compute_sarm_score, aggregate_judges, aggregate_scenarios
from .panels import InquisitorGenerator, InquisitorValidator, Judge
from .orchestrator import SARMOrchestrator, SARMScenario

__all__ = [
    "Probe",
    "ValidatedProbe",
    "JudgeVerdict",
    "AggregatedScore",
    "ProbeResult",
    "ScenarioResult",
    "BenchmarkReport",
    "compute_phi",
    "compute_sarm_score",
    "aggregate_judges",
    "aggregate_scenarios",
    "InquisitorGenerator",
    "InquisitorValidator",
    "Judge",
    "SARMOrchestrator",
    "SARMScenario",
]
