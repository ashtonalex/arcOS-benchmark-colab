"""
SARM benchmark orchestrator — full protocol execution (Section 8).

Consumes pipeline output (verbalized context, subgraph, question) and
produces benchmark scores. Never writes back to the pipeline.
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Optional

from src.config import BenchmarkConfig
from src.llm.client import OpenRouterClient, LLMRequest
from .models import (
    Probe,
    ProbeResult,
    ScenarioResult,
    BenchmarkReport,
    AggregatedScore,
    TierScore,
    FailureModeAnalysis,
    classify_tier,
    ALL_FAILURE_MODES,
)
from .prompts import BENCHMARKED_MODEL_PROMPT
from .scorer import aggregate_judges, aggregate_scenarios
from .panels import InquisitorGenerator, InquisitorValidator, Judge


@dataclass
class SARMScenario:
    """Input container for a single SARM scenario."""
    scenario_id: str
    question: str
    verbalized_context: str
    subgraph: object  # HeteroData or dict
    entity_count: int


class SARMOrchestrator:
    """Executes the full SARM benchmark protocol."""

    def __init__(self, config: BenchmarkConfig, client: OpenRouterClient):
        self.config = config
        self.client = client

        self.inquisitor_gen = InquisitorGenerator(
            model=config.sarm_inquisitor_generator_model,
            client=client,
        )
        self.inquisitor_val = InquisitorValidator(
            model=config.sarm_inquisitor_validator_model,
            client=client,
        )
        self.judges = [
            Judge(model=model, client=client)
            for model in config.sarm_judge_models
        ]

    async def _get_base_response(self, scenario: SARMScenario) -> str:
        """Get the benchmarked model's base response R₀."""
        response = await self.client.complete(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": BENCHMARKED_MODEL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"GRAPH CONTEXT:\n{scenario.verbalized_context}\n\n"
                        f"QUERY:\n{scenario.question}"
                    ),
                },
            ],
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        return response.content

    async def _get_probe_response(
        self,
        scenario: SARMScenario,
        base_response: str,
        probe: Probe,
    ) -> str:
        """Get the benchmarked model's response to a probe.

        Includes conversation context: system prompt + base response as
        assistant turn + probe question as user follow-up.
        """
        response = await self.client.complete(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": BENCHMARKED_MODEL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"GRAPH CONTEXT:\n{scenario.verbalized_context}\n\n"
                        f"QUERY:\n{scenario.question}"
                    ),
                },
                {"role": "assistant", "content": base_response},
                {"role": "user", "content": probe.question},
            ],
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        return response.content

    async def _evaluate_with_judges(
        self,
        scenario: SARMScenario,
        base_response: str,
        probe: Probe,
        probe_response: str,
    ) -> list:
        """Run all judges concurrently on a probe response."""
        tasks = [
            judge.evaluate(
                question=scenario.question,
                verbalized_context=scenario.verbalized_context,
                base_response=base_response,
                probe=probe,
                probe_response=probe_response,
            )
            for judge in self.judges
        ]
        return await asyncio.gather(*tasks)

    async def run_scenario(self, scenario: SARMScenario) -> ScenarioResult:
        """Execute the full SARM protocol for one scenario (Section 8.1)."""
        tier = classify_tier(scenario.entity_count)

        # Step 1: Base response
        base_response = await self._get_base_response(scenario)

        # Step 2: Generate probes
        raw_probes = await self.inquisitor_gen.generate(
            question=scenario.question,
            verbalized_context=scenario.verbalized_context,
            subgraph=scenario.subgraph,
            num_probes=self.config.sarm_probes_per_scenario,
        )

        # Step 3: Validate probes
        validated = await self.inquisitor_val.validate(
            probes=raw_probes,
            verbalized_context=scenario.verbalized_context,
            question=scenario.question,
        )
        robust_probes = [v.probe for v in validated]

        # If no probes survived validation, use raw probes as fallback
        if not robust_probes:
            robust_probes = raw_probes[:self.config.sarm_probes_per_scenario]

        # Step 4: For each probe — get response and judge
        probe_results = []
        for probe in robust_probes:
            probe_response = await self._get_probe_response(scenario, base_response, probe)
            verdicts = await self._evaluate_with_judges(
                scenario, base_response, probe, probe_response
            )
            agg = aggregate_judges(
                verdicts,
                threshold=self.config.sarm_judge_disagreement_threshold,
            )
            probe_results.append(ProbeResult(
                probe=probe,
                base_response=base_response,
                probe_response=probe_response,
                verdicts=verdicts,
                aggregated_score=agg,
            ))

        # Step 5: Aggregate across probes
        if probe_results:
            n = len(probe_results)
            scenario_agg = AggregatedScore(
                causal_consistency=sum(pr.aggregated_score.causal_consistency for pr in probe_results) / n,
                spatiotemporal_coherence=sum(pr.aggregated_score.spatiotemporal_coherence for pr in probe_results) / n,
                attribution_accuracy=sum(pr.aggregated_score.attribution_accuracy for pr in probe_results) / n,
                uncertainty_calibration=sum(pr.aggregated_score.uncertainty_calibration for pr in probe_results) / n,
                phi=sum(pr.aggregated_score.phi for pr in probe_results) / n,
                sarm_score=sum(pr.aggregated_score.sarm_score for pr in probe_results) / n,
                hallucination_detected=any(pr.aggregated_score.hallucination_detected for pr in probe_results),
                disagreement_flags={},
            )
        else:
            scenario_agg = AggregatedScore(
                causal_consistency=1.0,
                spatiotemporal_coherence=1.0,
                attribution_accuracy=1.0,
                uncertainty_calibration=1.0,
                phi=0.0,
                sarm_score=0.0,
                hallucination_detected=False,
            )

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            question=scenario.question,
            entity_count=scenario.entity_count,
            tier=tier,
            probe_results=probe_results,
            aggregated_score=scenario_agg,
        )

    async def run_benchmark(
        self,
        scenarios: list[SARMScenario],
        runs_per_scenario: Optional[int] = None,
    ) -> BenchmarkReport:
        """Execute multi-run benchmark across all scenarios (Section 8.2)."""
        runs = runs_per_scenario or self.config.sarm_runs_per_scenario
        all_results = []
        high_variance = []

        for scenario in scenarios:
            run_results = []
            for run_idx in range(runs):
                result = await self.run_scenario(scenario)
                result.run_index = run_idx
                run_results.append(result)

            # Compute mean/std across runs
            sarm_scores = [r.aggregated_score.sarm_score for r in run_results]
            mean_sarm = sum(sarm_scores) / len(sarm_scores)
            if len(sarm_scores) > 1:
                std_sarm = math.sqrt(
                    sum((s - mean_sarm) ** 2 for s in sarm_scores) / len(sarm_scores)
                )
            else:
                std_sarm = 0.0

            if std_sarm > 0.5:
                high_variance.append(scenario.scenario_id)

            # Use mean result as representative
            mean_result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                question=scenario.question,
                entity_count=scenario.entity_count,
                tier=classify_tier(scenario.entity_count),
                probe_results=run_results[0].probe_results,
                aggregated_score=AggregatedScore(
                    causal_consistency=sum(r.aggregated_score.causal_consistency for r in run_results) / runs,
                    spatiotemporal_coherence=sum(r.aggregated_score.spatiotemporal_coherence for r in run_results) / runs,
                    attribution_accuracy=sum(r.aggregated_score.attribution_accuracy for r in run_results) / runs,
                    uncertainty_calibration=sum(r.aggregated_score.uncertainty_calibration for r in run_results) / runs,
                    phi=sum(r.aggregated_score.phi for r in run_results) / runs,
                    sarm_score=mean_sarm,
                    hallucination_detected=any(r.aggregated_score.hallucination_detected for r in run_results),
                ),
            )
            all_results.append(mean_result)

        # Per-tier scoring
        tier_scores = aggregate_scenarios(all_results)

        # Failure mode analysis
        failure_analysis = self._analyze_failure_modes(all_results)

        hallucination_count = sum(
            1 for r in all_results if r.aggregated_score.hallucination_detected
        )

        return BenchmarkReport(
            scenarios=all_results,
            tier_scores=tier_scores,
            failure_mode_analysis=failure_analysis,
            hallucination_count=hallucination_count,
            total_scenarios=len(all_results),
            high_variance_scenarios=high_variance,
        )

    def _analyze_failure_modes(self, results: list[ScenarioResult]) -> list[FailureModeAnalysis]:
        """Compute failure rates per failure mode across all scenarios."""
        mode_counts = {mode: 0 for mode in ALL_FAILURE_MODES}
        mode_failures = {mode: 0 for mode in ALL_FAILURE_MODES}
        mode_examples = {mode: None for mode in ALL_FAILURE_MODES}

        for result in results:
            for pr in result.probe_results:
                mode = pr.probe.probe_type
                if mode in mode_counts:
                    mode_counts[mode] += 1
                    # A "failure" is sarm_score < 3.0 (below conditionally suitable)
                    if pr.aggregated_score.sarm_score < 3.0:
                        mode_failures[mode] += 1
                        if mode_examples[mode] is None:
                            mode_examples[mode] = result.scenario_id

        analysis = []
        for mode in ALL_FAILURE_MODES:
            total = mode_counts[mode]
            rate = mode_failures[mode] / total if total > 0 else 0.0
            analysis.append(FailureModeAnalysis(
                failure_mode=mode,
                failure_rate=rate,
                example_scenario_id=mode_examples[mode],
            ))

        return analysis
