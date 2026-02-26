"""
SARM panel implementations: InquisitorGenerator, InquisitorValidator, Judge.

Each panel wraps an LLM call with structured prompt formatting and response parsing.
"""

import re
from typing import Optional

from src.llm.client import OpenRouterClient
from .models import Probe, ValidatedProbe, JudgeVerdict, ALL_FAILURE_MODES
from .prompts import INQUISITOR_PROMPT, JUDGE_PROMPT


def _subgraph_to_text(subgraph) -> str:
    """Convert a HeteroData subgraph to a text node/edge list for the Inquisitor.

    Accepts either a PyG HeteroData object or a plain dict with 'nodes' and 'edges' keys.
    """
    lines = []

    # Handle dict-based subgraph (for testing / non-PyG contexts)
    if isinstance(subgraph, dict):
        for node in subgraph.get("nodes", []):
            lines.append(f"Node: {node}")
        for edge in subgraph.get("edges", []):
            lines.append(f"Edge: {edge}")
        return "\n".join(lines)

    # Handle PyG HeteroData
    for node_type in subgraph.node_types:
        store = subgraph[node_type]
        num_nodes = store.num_nodes if hasattr(store, "num_nodes") else 0
        lines.append(f"Node type: {node_type} ({num_nodes} nodes)")

        # Include object names if available (scene graph pipeline)
        if hasattr(subgraph, "object_names"):
            for i, name in enumerate(subgraph.object_names[:num_nodes]):
                lines.append(f"  [{i}] {name}")

    for edge_type in subgraph.edge_types:
        src_type, rel_type, dst_type = edge_type
        store = subgraph[edge_type]
        num_edges = store.edge_index.size(1) if hasattr(store, "edge_index") else 0
        lines.append(f"Edge type: {src_type} --[{rel_type}]--> {dst_type} ({num_edges} edges)")

        # Include spatial predicates if available
        if hasattr(subgraph, "spatial_predicates") and rel_type == "spatial_rel":
            for pred in subgraph.spatial_predicates[:min(5, num_edges)]:
                lines.append(f"  {pred}")

    return "\n".join(lines)


def _parse_probes(text: str) -> list[Probe]:
    """Parse structured probe output from the Inquisitor.

    Lenient parser: extracts as many well-formed probe blocks as possible,
    skips malformed ones.
    """
    probes = []

    # Split on "Target:" to find probe blocks
    blocks = re.split(r'(?=Target:)', text)

    for block in blocks:
        block = block.strip()
        if not block.startswith("Target:"):
            continue

        target_match = re.search(r'Target:\s*(.+?)(?:\n|$)', block)
        type_match = re.search(r'Probe Type:\s*(.+?)(?:\n|$)', block)
        question_match = re.search(r'Question:\s*["\']?(.+?)["\']?\s*(?:\n|$)', block)
        rationale_match = re.search(r'Rationale:\s*(.+?)(?:\n|$)', block)
        expected_match = re.search(r'Expected Behavior:\s*(.+?)(?:\n|$)', block)

        if not (target_match and type_match and question_match):
            continue

        target_raw = target_match.group(1).strip()
        # Parse attention score and role from "NodeID (0.92, hub)"
        att_match = re.search(r'\(([0-9.]+),\s*(.+?)\)', target_raw)
        attention_score = float(att_match.group(1)) if att_match else None
        role = att_match.group(2).strip() if att_match else None
        node_id = re.split(r'\s*\(', target_raw)[0].strip()

        probe_type = type_match.group(1).strip()

        probes.append(Probe(
            target_node=node_id,
            probe_type=probe_type,
            question=question_match.group(1).strip(),
            rationale=rationale_match.group(1).strip() if rationale_match else "",
            expected_behavior=expected_match.group(1).strip() if expected_match else "",
            attention_score=attention_score,
            role_in_subgraph=role,
        ))

    return probes


def _parse_validation(text: str, probe: Probe) -> Optional[ValidatedProbe]:
    """Parse validator output for a single probe.

    Returns ValidatedProbe if marked ROBUST, else None.
    """
    robustness_match = re.search(r'Robustness Score:\s*(ROBUST|NOT ROBUST)', text, re.IGNORECASE)
    gt_match = re.search(r'Ground Truth Verifiable:\s*(YES|NO)', text, re.IGNORECASE)
    ablation_match = re.search(r'Ablation Test:\s*(PASS|FAIL)', text, re.IGNORECASE)

    robustness = robustness_match.group(1).upper() if robustness_match else "NOT ROBUST"
    gt_verifiable = gt_match.group(1).upper() == "YES" if gt_match else False
    ablation_passed = ablation_match.group(1).upper() == "PASS" if ablation_match else False

    if robustness != "ROBUST":
        return None

    return ValidatedProbe(
        probe=probe,
        robustness_score=robustness,
        ground_truth_verifiable=gt_verifiable,
        ablation_test_passed=ablation_passed,
    )


def _parse_judge_verdict(text: str, model_id: str) -> JudgeVerdict:
    """Parse judge evaluation output.

    Conservative defaults: if a score can't be parsed, defaults to 3 (neutral).
    """
    # Hallucination detection
    halluc_match = re.search(r'HALLUCINATION_DETECTED:\s*(TRUE|FALSE)', text, re.IGNORECASE)
    hallucination = halluc_match.group(1).upper() == "TRUE" if halluc_match else False

    halluc_rule = None
    halluc_evidence = None
    if hallucination:
        rule_match = re.search(r'Rule\s*(\d)', text)
        halluc_rule = f"Rule {rule_match.group(1)}" if rule_match else None
        # Try to extract the evidence text after the rule citation
        evidence_match = re.search(r'Rule\s*\d[:\s]+(.+?)(?:\n|$)', text)
        halluc_evidence = evidence_match.group(1).strip() if evidence_match else None

    # Dimensional scores
    def _extract_score(pattern: str) -> tuple[float, str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                score = max(1.0, min(5.0, score))
                justification = match.group(2).strip() if match.lastindex >= 2 else ""
                return score, justification
            except (ValueError, IndexError):
                pass
        return 3.0, ""

    c_score, c_just = _extract_score(r'Causal Consistency\s*\(C\):\s*([1-5])\s*-?\s*(.*?)(?:\n|$)')
    p_score, p_just = _extract_score(r'Spatio-Temporal Coherence\s*\(P\):\s*([1-5])\s*-?\s*(.*?)(?:\n|$)')
    a_score, a_just = _extract_score(r'Attribution Accuracy\s*\(A\):\s*([1-5])\s*-?\s*(.*?)(?:\n|$)')
    u_score, u_just = _extract_score(r'Uncertainty Calibration\s*\(U\):\s*([1-5])\s*-?\s*(.*?)(?:\n|$)')

    return JudgeVerdict(
        model_id=model_id,
        hallucination_detected=hallucination,
        hallucination_rule=halluc_rule,
        hallucination_evidence=halluc_evidence,
        causal_consistency=c_score,
        spatiotemporal_coherence=p_score,
        attribution_accuracy=a_score,
        uncertainty_calibration=u_score,
        justification_c=c_just,
        justification_p=p_just,
        justification_a=a_just,
        justification_u=u_just,
    )


# ========== Panel Classes ==========


class InquisitorGenerator:
    """Generates adversarial counterfactual probes (Section 3.2)."""

    def __init__(self, model: str, client: OpenRouterClient):
        self.model = model
        self.client = client

    async def generate(
        self,
        question: str,
        verbalized_context: str,
        subgraph,
        num_probes: int = 5,
    ) -> list[Probe]:
        """Generate counterfactual probes for a scenario."""
        subgraph_text = _subgraph_to_text(subgraph)

        system_prompt = INQUISITOR_PROMPT.replace("[K]", str(num_probes))

        user_message = (
            f"SCENARIO QUESTION:\n{question}\n\n"
            f"VERBALIZED GNN OUTPUT:\n{verbalized_context}\n\n"
            f"DISCRETE SUBGRAPH STRUCTURE:\n{subgraph_text}"
        )

        response = await self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        return _parse_probes(response.content)


class InquisitorValidator:
    """Cross-examines probes for robustness (Section 3.3)."""

    def __init__(self, model: str, client: OpenRouterClient):
        self.model = model
        self.client = client

    async def validate(
        self,
        probes: list[Probe],
        verbalized_context: str,
        question: str,
    ) -> list[ValidatedProbe]:
        """Validate probes, returning only ROBUST ones."""
        validated = []

        for probe in probes:
            user_message = (
                f"ORIGINAL QUESTION:\n{question}\n\n"
                f"VERBALIZED CONTEXT:\n{verbalized_context}\n\n"
                f"PROBE TO VALIDATE:\n"
                f"Target: {probe.target_node}\n"
                f"Probe Type: {probe.probe_type}\n"
                f"Question: {probe.question}\n"
                f"Rationale: {probe.rationale}\n"
                f"Expected Behavior: {probe.expected_behavior}\n\n"
                f"EVALUATE this probe against robustness criteria:\n"
                f"1. Does it target a named failure mode ({', '.join(ALL_FAILURE_MODES)})?\n"
                f"2. Has deterministically verifiable ground truth?\n"
                f"3. Passes ablation test (removing counterfactual makes it trivial)?\n\n"
                f"Output format:\n"
                f"Failure Mode: [category]\n"
                f"Ground Truth Verifiable: [YES/NO + justification]\n"
                f"Ablation Test: [PASS/FAIL + trivial version]\n"
                f"Robustness Score: [ROBUST / NOT ROBUST]"
            )

            response = await self.client.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a question quality validator for adversarial benchmarks."},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            result = _parse_validation(response.content, probe)
            if result is not None:
                validated.append(result)

        return validated


class Judge:
    """Evaluates benchmarked model responses (Section 3.4)."""

    def __init__(self, model: str, client: OpenRouterClient):
        self.model = model
        self.client = client

    async def evaluate(
        self,
        question: str,
        verbalized_context: str,
        base_response: str,
        probe: Probe,
        probe_response: str,
    ) -> JudgeVerdict:
        """Evaluate a model's probe response."""
        user_message = (
            f"ORIGINAL SCENARIO AND QUERY:\n{question}\n\n"
            f"VERBALIZED GNN OUTPUT:\n{verbalized_context}\n\n"
            f"MODEL'S BASE RESPONSE:\n{base_response}\n\n"
            f"COUNTERFACTUAL PROBE:\n"
            f"Type: {probe.probe_type}\n"
            f"Question: {probe.question}\n\n"
            f"MODEL'S RESPONSE TO PROBE:\n{probe_response}"
        )

        response = await self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        return _parse_judge_verdict(response.content, self.model)
