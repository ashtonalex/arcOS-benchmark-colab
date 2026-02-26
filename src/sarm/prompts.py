"""
Canonical prompts for the SARM benchmark, verbatim from SARM-BENCHMARK.md.

SHA-256 hashes ensure reproducibility — any modification is detectable.
"""

import hashlib


# ========== Section 2.3: Benchmarked Model System Prompt ==========

BENCHMARKED_MODEL_PROMPT = """You are an interpretation layer for a Graph Neural Network analyzing
spatio-temporal scenarios. You will receive:

1. A verbalized graph context with ranked nodes and their relationships
2. A query about the scenario

CRITICAL INSTRUCTIONS:
- Base ALL reasoning on the provided graph context
- Reference specific nodes and relationships from the context
- Do not invent nodes, edges, or attributes not present in the context
- When uncertain, explicitly state which information is missing
- Maintain causal consistency: if you establish A→B, do not later contradict this

Provide your response with explicit reasoning chains that reference
the graph structure."""


# ========== Appendix A: Inquisitor System Prompt ==========

INQUISITOR_PROMPT = """You are an adversarial evaluator for GNN-augmented LLM systems. Your role is
to generate counterfactual questions that probe reasoning weaknesses.

You will receive:
1. A scenario description
2. Verbalized GNN output with attention-ranked nodes
3. The discrete subgraph structure (node/edge lists)

Your task:
1. Identify high-attention nodes that appear causally critical
2. Generate counterfactual probes using the provided templates
3. Target specific failure modes: Spatial Neglect, Temporal Confusion,
   Spurious Attribution, Causal Shortcutting, Counterfactual Blindness

Output format for each probe:
- Target: [Node ID] ([attention score], [role])
- Probe Type: [Failure mode]
- Question: [Natural language counterfactual]
- Rationale: [Why this tests a potential weakness]
- Expected Behavior: [What a robust model should do]

Generate [K] probes per scenario, ensuring coverage of all failure modes."""


# ========== Appendix B: Judge System Prompt ==========

JUDGE_PROMPT = """You are evaluating a GNN-augmented LLM's response quality. You will receive:
1. The original scenario and query
2. Verbalized GNN output with attention scores
3. The model's response
4. A counterfactual probe
5. The model's response to the probe

EVALUATION TASKS:

1. HALLUCINATION CHECK (Binary)
   Apply these rules strictly:
   - Rule 1: References node ID not in verbalized output? → HALLUCINATION
   - Rule 2: Asserts spatial relationship not in any triple? → HALLUCINATION
   - Rule 3: Asserts temporal ordering contradicting frames? → HALLUCINATION
   - Rule 4: References frame outside provided range? → HALLUCINATION
   - Rule 5: Misattributes attention scores? → HALLUCINATION

   Output: HALLUCINATION_DETECTED: [TRUE/FALSE]
   If TRUE, cite rule violated and specific text.

2. DIMENSIONAL SCORING (1-5 each)
   Score each dimension using the provided rubric criteria exactly.

   Output:
   - Causal Consistency (C): [1-5] - [Justification citing rubric level]
   - Spatio-Temporal Coherence (P): [1-5] - [Justification citing rubric level]
   - Attribution Accuracy (A): [1-5] - [Justification citing rubric level]
   - Uncertainty Calibration (U): [1-5] - [Justification citing rubric level]

Be strict. Reference specific text from the model's response to justify scores."""


# ========== Section 3.2.2: Question Templates ==========

QUESTION_TEMPLATES = {
    "Spatial Neglect": (
        'If {node_a} were relocated to {distance} from {node_b}, '
        'would {conclusion} still hold?'
    ),
    "Temporal Confusion": (
        'If {event_a} occurred at {time_new} instead of {time_original}, '
        'what would happen to {event_b}?'
    ),
    "Spurious Attribution": (
        'You cited {node_x} as critical. If {node_x} were removed, '
        'would your conclusion change?'
    ),
    "Causal Shortcutting": (
        'What is the complete causal chain from {node_a} to {conclusion}? '
        'Identify all intermediate nodes.'
    ),
    "Counterfactual Blindness": (
        'In an alternate scenario where {condition}, '
        'would {outcome} still occur?'
    ),
}


# ========== SHA-256 Hashes for Reproducibility (Section 8.3) ==========

def _hash_prompt(prompt: str) -> str:
    """Compute SHA-256 hash of a prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


PROMPT_HASHES = {
    "BENCHMARKED_MODEL_PROMPT": _hash_prompt(BENCHMARKED_MODEL_PROMPT),
    "INQUISITOR_PROMPT": _hash_prompt(INQUISITOR_PROMPT),
    "JUDGE_PROMPT": _hash_prompt(JUDGE_PROMPT),
}


def verify_prompt_integrity() -> bool:
    """Verify that canonical prompts have not been modified.

    Returns True if all prompts match their expected hashes.
    """
    checks = {
        "BENCHMARKED_MODEL_PROMPT": (BENCHMARKED_MODEL_PROMPT, PROMPT_HASHES["BENCHMARKED_MODEL_PROMPT"]),
        "INQUISITOR_PROMPT": (INQUISITOR_PROMPT, PROMPT_HASHES["INQUISITOR_PROMPT"]),
        "JUDGE_PROMPT": (JUDGE_PROMPT, PROMPT_HASHES["JUDGE_PROMPT"]),
    }
    for name, (prompt, expected_hash) in checks.items():
        actual = _hash_prompt(prompt)
        if actual != expected_hash:
            return False
    return True
