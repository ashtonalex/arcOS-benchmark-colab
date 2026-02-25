# SARM: Spatio-temporal Adversarial Reasoning Methodology

## Benchmark Specification v3.0

---

## 1. Overview

### 1.1 Purpose

SARM is a benchmarking methodology for evaluating **GNN-augmented LLM systems** where the LLM serves as an interpretation layer over Graph Neural Network outputs. Unlike existing benchmarks that evaluate correctness alone, SARM tests whether models make **correct decisions for the right reasons** by evaluating internal reasoning attribution under adversarial counterfactual probing.

### 1.2 Scope

- **Target Systems**: GNN-LLM pipelines using G-Retrieval (or equivalent) for the retrieval layer, with an LLM inference layer
- **Domain**: Domain-general; initial application in defense spatio-temporal graphs
- **Core Innovation**: Dual-panel adversarial evaluation that sidesteps ground-truth annotation via procedural assessment

### 1.3 Key Differentiators

| Existing Benchmark | Focus | SARM Difference |
| --- | --- | --- |
| GraCoRe | Graph comprehension taxonomy | SARM tests adversarial robustness |
| STaRK | Retrieval on semi-structured KBs | SARM tests causal reasoning |
| Action Genome | Scene graph generation accuracy | SARM tests counterfactual reasoning |
| CausalBench | Causal inference in text | SARM targets spatio-temporal graphs |

---

## 2. System Architecture

### 2.1 Benchmarked System Requirements

The benchmarked GNN-LLM system must:

1. Use a GNN layer for graph reasoning over spatio-temporal data
2. Use G-Retrieval (or equivalent) for subgraph retrieval
3. Pass GNN outputs through a **verbalizer** to produce hard prompts for the LLM
4. Use an LLM inference layer with a **fixed, published system prompt** that prioritises GNN-given context

**Architectural Constraint**: The LLM is non-frozen (no temperature controls assumed). The system prompt guides the LLM to preserve grounding in GNN-provided context.

### 2.2 Verbalizer Specification

The verbalizer converts GNN outputs to structured text within a ~500 token budget.

**Output Format**:

```
[ATTENTION_SCORE] [SUBJECT] [RELATION] [PREDICATE]
```

**Structure**:

1. Rank nodes by GNN attention score (descending)
2. For each high-attention node, collect edges:
    - Spatial: `0.92 person holding cup (frame 42)`
    - Temporal: `0.87 person appears_across_frames frame 0 → 1`
3. Format as ranked triple list with attention scores
4. Truncate to ~500 token budget

**Implementation**: Sentence-transformers for embedding-based verbalization.

**Invariant**: The Inquisitor panel receives the **same verbalized output** as the benchmarked LLM.

### 2.3 Published System Prompt

The benchmarked LLM's system prompt is fixed and published as part of the benchmark specification. This ensures SARM measures **model capability**, not system prompt engineering.

```
[SYSTEM PROMPT - BENCHMARKED MODEL]

You are an interpretation layer for a Graph Neural Network analyzing
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
the graph structure.
```

---

## 3. Dual-Panel Evaluation Architecture

### 3.1 Panel Overview

| Panel | Role | Input | Output |
| --- | --- | --- | --- |
| **Inquisitor** | Adversarial probe generation | Scenario + verbalized subgraph structure | Counterfactual questions targeting failure modes |
| **Judge** | Response quality assessment | Benchmarked model response + verbalized GNN output | Dimensional scores + hallucination flag |

### 3.2 Inquisitor Panel

### 3.2.1 Access Level

**Grey-box**: The Inquisitor receives:

- The verbalized context (same as benchmarked model)
- The discrete subgraph structure from G-Retrieval (node/edge lists)

### 3.2.2 Structured Question Templates

Human-constructed templates ensure coverage of all failure modes. The Inquisitor instantiates templates with scenario-specific content.

**Template Categories**:

| Failure Mode | Template Structure | Example Instantiation |
| --- | --- | --- |
| **Spatial Neglect** | "If [node A] were relocated to [distance X] from [node B], would [conclusion C] still hold?" | "If the grenade were 50m from the vehicle instead of 5m, would the damage assessment change?" |
| **Temporal Confusion** | "If [event A] occurred at [time T'] instead of [time T], what would happen to [event B]?" | "If the sensor activation occurred at frame 60 instead of frame 42, would the alert sequence be affected?" |
| **Spurious Attribution** | "You cited [node X] as critical. If [node X] were removed, would your conclusion change?" | "You cited the radio signal as critical. If the radio signal were absent, would your threat assessment change?" |
| **Causal Shortcutting** | "What is the complete causal chain from [node A] to [conclusion C]? Identify all intermediate nodes." | "What is the complete causal chain from the initial movement detection to the evacuation order?" |
| **Counterfactual Blindness** | "In an alternate scenario where [condition X], would [outcome Y] still occur?" | "If the weather conditions reduced visibility to 10m, would the surveillance classification remain valid?" |

### 3.2.3 Inquisitor Output Format

```
Target: [Node ID] ([attention score], [role in subgraph])
Probe Type: [Failure mode from taxonomy]
Question: "[Natural language counterfactual question]"
Rationale: [Why this probe targets a potential weakness]
Expected Behavior: [What a robust model should do]
```

### 3.3 Question Bank Validator

An independent LLM validates adversarial question quality before use.

### 3.3.1 Robustness Criteria (Operational)

A question is **robust** if ALL of:

| Criterion | Validation Method |
| --- | --- |
| **Targets named failure mode** | Question explicitly maps to one of: Spatial Neglect, Temporal Confusion, Spurious Attribution, Causal Shortcutting, Counterfactual Blindness |
| **Has deterministically verifiable ground truth** | The counterfactual has a correct answer derivable from graph structure (even if reasoning path varies) |
| **Passes ablation test** | Removing the counterfactual element makes the question trivially answerable from the base scenario |

### 3.3.2 Validator Output

```
Question ID: [unique identifier]
Failure Mode: [category]
Ground Truth Verifiable: [YES/NO + justification]
Ablation Test: [PASS/FAIL + trivial version of question]
Robustness Score: [ROBUST / NOT ROBUST]
```

**Usage**: Only questions marked ROBUST are included in benchmark runs. Non-robust questions are logged for Inquisitor prompt refinement.

### 3.4 Judge Panel

### 3.4.1 Input

The Judge receives:

1. The original scenario and query
2. The verbalized GNN output (with attention scores)
3. The benchmarked model's response
4. The counterfactual probe
5. The benchmarked model's response to the probe

### 3.4.2 Hallucination Detection Rules

**Definition**: A hallucination occurs when the benchmarked model references a node, edge, or attribute not present in the verbalized GNN output.

**Rule-Based Check**:

```
HALLUCINATION_DETECTED = TRUE if ANY of:

  Rule 1: Model references a node ID not present in verbalized output

  Rule 2: Model asserts a spatial relationship (e.g., "A is adjacent to B")
          not supported by any [SUBJECT][RELATION][PREDICATE] triple

  Rule 3: Model asserts a temporal relationship (e.g., "A occurred before B")
          that contradicts frame ordering in verbalized output

  Rule 4: Model references a frame number outside the range present in
          verbalized output

  Rule 5: Model attributes an attention score or saliency ranking that
          contradicts the verbalized ranking
```

**Judge Requirement**: Binary hallucination flag with justification citing which rule was violated.

---

## 4. Evaluation Rubric

### 4.1 Dimension 1: Causal Consistency & Counterfactual Resilience

*Measures how well the model maintains a logical chain when variables are manipulated by the Inquisitor.*

| Score | Descriptor | Operational Criteria |
| --- | --- | --- |
| **5** | Robust Causal Logic | Correctly identifies the "linchpin" node; reasoning survives all counterfactual probes without contradiction |
| **4** | Minor Logic Slip | Successfully adapts to most probes but fails one edge case; reasoning is 80%+ consistent across the session |
| **3** | Surface Consistency | Logic holds for the base scenario but collapses or becomes vague when the Inquisitor introduces complex changes |
| **2** | Causal Fragility | Model "flips" its conclusion for the wrong reasons (e.g., changing an irrelevant node attribute causes a massive logical shift) |
| **1** | Logical Collapse | Internal reasoning contradicts itself within the same response; fails to recognize basic cause-effect links |

### 4.2 Dimension 2: Spatio-Temporal Coherence

*Evaluates the model's adherence to physical laws (distance, time-ordering) within the graph.*

| Score | Descriptor | Operational Criteria |
| --- | --- | --- |
| **5** | Physical Rigor | Perfectly respects spatial bounds (e.g., blast radii) and temporal monotonicity (T_start < T_end) |
| **4** | Plausible Spatiality | No major physical violations; minor imprecision in distance/time estimation that doesn't affect the final decision |
| **3** | Heuristic Reasoning | Relies on "common sense" instead of the graph's specific coordinates (e.g., "grenades are dangerous" vs "A is within 5m of B") |
| **2** | Spatiotemporal Neglect | Frequent "teleportation" errors (treating distant nodes as adjacent) or temporal paradoxes (effect precedes cause) |
| **1** | Geometric Hallucination | Inventing spatial relationships or nodes that do not exist in the provided GNN-augmented graph |

### 4.3 Dimension 3: Attribution Accuracy (Interpretability)

*Assesses if the LLM's explanation matches the GNN's saliency.*

| Score | Descriptor | Operational Criteria |
| --- | --- | --- |
| **5** | Faithful Attribution | The LLM's reasoning chain focuses on the exact nodes the GNN weighted as highly salient; no confabulation |
| **4** | Aligned Reasoning | Generally matches GNN saliency but includes 1-2 "flavor" nodes that didn't mathematically contribute much |
| **3** | Rationalization | The reasoning sounds good but only partially overlaps with the GNN's attention weights |
| **2** | Post-hoc Confabulation | The LLM invents a logical reason for the GNN's output that is mathematically impossible based on graph structure |
| **1** | Blind Guessing | The reasoning is entirely detached from the graph's attributes or the GNN's specific output signal |

### 4.4 Dimension 4: Uncertainty Calibration

*How well the model knows when it doesn't know.*

| Score | Descriptor | Operational Criteria |
| --- | --- | --- |
| **5** | Calibrated Doubt | Correctly identifies ambiguous nodes or low-confidence GNN signals; expresses uncertainty when info is missing |
| **4** | Appropriate Caution | Hesitates on edge-case counterfactuals; minor overconfidence in clearly defined areas |
| **3** | Neutral/Static | Expresses a generic level of confidence regardless of how "noisy" or difficult the graph scenario is |
| **2** | Dunning-Kruger Effect | Is confidently wrong; provides detailed reasoning for a physically impossible counterfactual |
| **1** | Total Delusion | Ignores explicit "Insufficient Data" flags; generates high-confidence hallucinations for missing graph segments |

---

## 5. Scoring Methodology

### 5.1 SARM Score Formula

$$S_{SARM} = \left( w_C \cdot C + w_P \cdot P + w_A \cdot A + w_U \cdot U \right) \times \Phi$$

Where:

- C = Causal Consistency score (1-5)
- P = Spatio-Temporal Coherence score (1-5)
- A = Attribution Accuracy score (1-5)
- U = Uncertainty Calibration score (1-5)
- Φ = Critical Failure Multiplier

### 5.2 Dimensional Weights

| Variable | Dimension | Weight | Rationale |
| --- | --- | --- | --- |
| C | Causal Consistency | 0.40 | Core requirement: if logic breaks under counterfactuals, model is unsuitable for decision support |
| P | Spatio-Temporal Coherence | 0.30 | Reality anchor: must respect physical laws |
| A | Attribution Accuracy | 0.20 | Transparency: ensures LLM explanations match GNN reasoning |
| U | Uncertainty Calibration | 0.10 | Safety valve: knowing when data is insufficient |

### 5.3 Critical Failure Multiplier (Φ) — Smoothed

$$\Phi = \begin{cases} 0.0 & \text{if hallucination detected} \ \frac{\sqrt{C \times P}}{5} & \text{otherwise} \end{cases}$$

**Behavior**:

- C=5, P=5 → Φ = 1.0
- C=4, P=4 → Φ = 0.8
- C=3, P=3 → Φ = 0.6
- C=5, P=1 → Φ = 0.45 (imbalance penalized)
- C=2, P=2 → Φ = 0.4
- Hallucination → Φ = 0.0 (hard zero)

**Rationale**: Geometric mean penalizes imbalance between causal and spatio-temporal coherence more heavily than linear scaling. A model strong in one dimension but weak in the other is penalized appropriately.

### 5.4 Score Interpretation

| S_SARM Range | Interpretation |
| --- | --- |
| 4.0 - 5.0 | Production-ready: robust reasoning under adversarial probing |
| 3.0 - 3.9 | Conditionally suitable: acceptable for non-critical applications with human oversight |
| 2.0 - 2.9 | Research-grade: significant reasoning gaps, not suitable for deployment |
| 1.0 - 1.9 | Failing: fundamental reasoning failures |
| 0.0 | Disqualified: hallucination detected |

### 5.5 Passing Threshold

The passing threshold for Φ (≥3 on C and P) is a **separate concept** from benchmark certification. Φ is a continuous multiplier; certification thresholds are defined per deployment context.

---

## 6. Adversarial Data Pollution

### 6.1 k-NN Relevance Boundary Method

To test discrimination capability, inject contextually plausible but causally irrelevant nodes using the **40th-60th percentile** of embedding similarity.

**Rationale**:

- Bottom-k (most dissimilar) nodes are trivially ignorable
- Top-k (most similar) nodes are likely relevant
- Mid-range nodes are close enough to be plausible distractors, far enough to be causally disconnected

**Implementation**:

1. Compute embedding similarity between candidate nodes and the retrieved subgraph
2. Select nodes in the 40th-60th percentile of similarity scores
3. Inject into the verbalized context with appropriate attention scores (randomized within plausible range)

### 6.2 Temporal Mismatching

To ground temporal reasoning tests:

1. Identify the causal time window of the scenario
2. Select nodes from **parallel, non-interacting causal chains** within this window
3. Inject with frame numbers that place them temporally adjacent but causally independent

**Constraint**: Injected nodes must be within the scenario's time window (not obviously distant timestamps).

---

## 7. Complexity Tiers

### 7.1 Tier Definitions

| Tier | Entity Count | Purpose | Target Distribution |
| --- | --- | --- | --- |
| **Calibration** | 2-5 entities | Sanity check; models should score near-ceiling | 25% |
| **Standard** | 6-15 entities | Typical operational complexity | 25% |
| **Adversarial** | 20-100 entities | Stress test; dense graphs, competing causal explanations | 50% |

**Note**: Entity counts 16-19 are intentionally excluded to ensure clear difficulty separation between Standard and Adversarial tiers.

### 7.2 Tier-Specific Reporting

Report scores separately per tier:

- S_SARM (Calibration)
- S_SARM (Standard)
- S_SARM (Adversarial)
- S_SARM (Aggregate) — weighted by tier distribution

A model that excels at Calibration/Standard but collapses at Adversarial tells a different story than uniform mediocrity.

---

## 8. Benchmark Protocol

### 8.1 Single Scenario Execution

```
1. SETUP
   - Present scenario S (spatio-temporal graph + query) to Benchmarked-LLM
   - Benchmarked-LLM receives verbalized GNN output via published system prompt

2. BASE RESPONSE
   - Benchmarked-LLM provides response R₀ with reasoning chain

3. ADVERSARIAL PROBING
   - Inquisitor receives: scenario + verbalized output + discrete subgraph
   - Inquisitor generates K counterfactual probes {P₁...Pₖ} using structured templates
   - Question Bank Validator filters to robust probes only

4. PROBE RESPONSES
   - For each robust probe Pᵢ:
     - Benchmarked-LLM responds with Rᵢ
     - Judge evaluates Rᵢ against rubric dimensions
     - Judge applies hallucination detection rules

5. AGGREGATION
   - Compute dimensional scores (C, P, A, U) across probes
   - Apply Φ multiplier
   - Calculate S_SARM for scenario
```

### 8.2 Multi-Run Protocol

**Requirement**: 3 runs per scenario

**Rationale**: LLM outputs are non-deterministic; no temperature controls assumed across all models.

**Reporting**:

- Mean S_SARM per scenario
- Standard deviation per scenario
- Flag scenarios with high variance (σ > 0.5) for manual review

### 8.3 Reproducibility Requirements

| Component | Reproducibility Measure |
| --- | --- |
| k-NN pollution | Fixed random seeds (published) |
| Inquisitor prompts | Canonical versions with SHA-256 hashes |
| Judge prompts | Canonical versions with SHA-256 hashes |
| Question templates | Published template bank |
| System prompt | Fixed, published as part of spec |

---

## 9. Failure Mode Taxonomy

### 9.1 Definitions

| Failure Mode | Definition | Detection Signal |
| --- | --- | --- |
| **Spatial Neglect** | Ignoring distance/proximity relationships when they're causally relevant | Low P score; fails spatial counterfactuals |
| **Temporal Confusion** | Misordering events or ignoring temporal dependencies | Low P score; effect-before-cause errors |
| **Spurious Attribution** | Citing nodes that don't contribute to the GNN's actual reasoning | Low A score; attention weight mismatch |
| **Causal Shortcutting** | Jumping to conclusions without tracing intermediate causal steps | Low C score; fails "complete the chain" probes |
| **Counterfactual Blindness** | Inability to reason about hypothetical scenario modifications | Low C score; responses don't change under counterfactuals |

### 9.2 Failure Mode Coverage

Each benchmark run must include probes targeting **all five failure modes** per scenario. The Inquisitor's structured templates ensure coverage.

---

## 10. Reporting Format

### 10.1 Per-Model Report

```
MODEL: [Model identifier]
BENCHMARK VERSION: SARM v3.0
DATE: [Timestamp]
RUNS PER SCENARIO: 3

AGGREGATE SCORES:
┌─────────────────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Tier            │ C     │ P     │ A     │ U     │ Φ     │ S_SARM│
├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ Calibration     │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │
│ Standard        │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │
│ Adversarial     │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │
│ Overall         │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │ μ (σ) │
└─────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘

FAILURE MODE ANALYSIS:
┌─────────────────────────┬─────────────┬─────────────────────┐
│ Failure Mode            │ Failure Rate│ Example Scenario    │
├─────────────────────────┼─────────────┼─────────────────────┤
│ Spatial Neglect         │ X%          │ [ID]                │
│ Temporal Confusion      │ X%          │ [ID]                │
│ Spurious Attribution    │ X%          │ [ID]                │
│ Causal Shortcutting     │ X%          │ [ID]                │
│ Counterfactual Blindness│ X%          │ [ID]                │
└─────────────────────────┴─────────────┴─────────────────────┘

HALLUCINATION EVENTS: [Count] across [Total scenarios]

HIGH VARIANCE SCENARIOS: [List of scenario IDs with σ > 0.5]
```

### 10.2 Comparative Leaderboard

```
┌──────┬─────────────────┬───────┬───────┬───────┬───────┬────────────┐
│ Rank │ Model           │ Calib │ Std   │ Adv   │ Agg   │ Halluc Rate│
├──────┼─────────────────┼───────┼───────┼───────┼───────┼────────────┤
│ 1    │ [Model A]       │ 4.2   │ 3.8   │ 3.1   │ 3.5   │ 0.02       │
│ 2    │ [Model B]       │ 4.5   │ 3.6   │ 2.8   │ 3.3   │ 0.05       │
│ ...  │ ...             │ ...   │ ...   │ ...   │ ...   │ ...        │
└──────┴─────────────────┴───────┴───────┴───────┴───────┴────────────┘
```

---

## 11. Limitations and Future Work

### 11.1 Known Limitations

1. **LLM-as-Judge variance**: Despite structured prompts, Judge assessments may vary across runs
2. **Verbalizer information loss**: 500-token budget may truncate critical context in complex scenarios
3. **Closed-source model constraints**: Cannot access embeddings; evaluation limited to verbalized outputs
4. **Domain transfer**: Initial validation on defense scenarios; generalization to other domains requires separate validation

### 11.2 Future Extensions

1. **Hybrid verification**: Integrate deterministic physics simulators for spatial/temporal ground truth
2. **Adaptive Inquisitor**: Learn to generate more targeted probes based on model-specific weaknesses
3. **Cross-domain benchmarks**: Extend to medical imaging graphs, financial transaction networks, social network analysis
4. **Efficiency metrics**: Add latency and compute cost dimensions for deployment suitability

---

## 12. Appendices

### Appendix A: Inquisitor System Prompt

```
[SYSTEM PROMPT - INQUISITOR PANEL]

You are an adversarial evaluator for GNN-augmented LLM systems. Your role is
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

Generate [K] probes per scenario, ensuring coverage of all failure modes.
```

### Appendix B: Judge System Prompt

```
[SYSTEM PROMPT - JUDGE PANEL]

You are evaluating a GNN-augmented LLM's response quality. You will receive:
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

Be strict. Reference specific text from the model's response to justify scores.
```

### Appendix C: Canonical Random Seeds

```
k-NN Pollution Seeds:
- Calibration tier: 42
- Standard tier: 137
- Adversarial tier: 256

Temporal Mismatch Seeds:
- All tiers: 1337
```

---

## Document Control

| Version | Date | Changes |
| --- | --- | --- |
| 1.0 | - | Initial SARM concept with salience vector comparison, dual-panel adversarial evaluation methodology, verbalizer spec, smoothed Φ, k-NN pollution, complexity tiers, full protocol |
| 2.0 | - |  |
| 3.0 | - |  |

---