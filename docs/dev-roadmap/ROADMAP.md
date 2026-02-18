# arcOS Benchmark - Implementation Roadmap

**Version:** 2.1 (Colab Edition)
**Date:** 2026-02-18
**Last Updated:** 2026-02-18

---

## Roadmap Summary

| Phase | Name | Goal | Status | Est. Cells |
|-------|------|------|--------|------------|
| 1 | Environment & Data | Colab setup, dataset loaded, graph built | COMPLETE | 8 |
| 2 | Retrieval Pipeline | Query-to-subgraph extraction working | COMPLETE | 3 (cells 9-11) |
| 3 | GNN Encoder | Trainable GNN producing node embeddings with attention | COMPLETE | 5 (cells 12-16) |
| 4 | Graph Verbalization | Attention-guided subgraph-to-text conversion | TODO | 3-4 |
| 5 | LLM Integration | OpenRouter API calls with hard prompt | TODO | 4-5 |
| 6 | End-to-End Pipeline | Full query-to-answer pipeline assembled | TODO | 3-4 |
| 7 | Evaluation & Benchmarking | Metrics, baselines, reproducibility | TODO | 5-6 |
| 8 | Hardening & Polish | Error handling, checkpointing, documentation | TODO | 3-4 |

---

## Phase 1: Environment & Data Foundation — COMPLETE

**Goal:** Colab notebook boots cleanly, all dependencies install via uv, RoG-webqsp dataset is loaded, and a NetworkX knowledge graph is constructed from the dataset's triples.

### Tasks

1.1 **Colab environment setup cell**
- Clear `UV_CONSTRAINT` and `UV_BUILD_CONSTRAINT` env vars
- Install `uv` via pip
- Install all dependencies via `uv pip install --system`
- Dynamically detect PyTorch/CUDA versions for PyG wheel matching
- Verify GPU availability with `torch.cuda.is_available()`

1.2 **Seed & determinism initialization**
- `src/utils/seeds.py` with `set_seeds()` function
- Sets `torch.manual_seed()`, `numpy.random.seed()`, `random.seed()`, `PYTHONHASHSEED`
- Sets `torch.backends.cudnn.deterministic = True`

1.3 **Configuration dataclasses**
- `src/config.py` with `BenchmarkConfig` dataclass
- All hyperparameters for all 8 phases with sensible defaults
- Validation in `__post_init__()` (seed, dropout range, PCST params, etc.)
- Properties: `checkpoint_dir`, `results_dir`
- Methods: `get_checkpoint_path()`, `get_results_path()`, `print_summary()`

1.4 **Dataset loading**
- `src/data/dataset_loader.py` with `RoGWebQSPLoader` class
- Load `rmanluo/RoG-webqsp` via HuggingFace `datasets` library
- Redirect HF cache to Google Drive for persistence
- Schema validation, statistics, split count verification
- `slice_dataset()` for reduced dev iterations (600/50/1628)

1.5 **NetworkX graph construction**
- `src/data/graph_builder.py` with `GraphBuilder` class
- **Noise filtering** (key implementation detail):
  - CVT node removal: strips opaque Freebase MIDs matching `^[mg]\.[0-9a-z_]+$`
  - Junk relation removal: `freebase.valuenotation.*`, `freebase.type_profile.*`, `type.object.*`, `kg.object_profile.*`, `rdf-schema#*`
  - Drops ~15-20% of triples, improves embedding quality
- Builds per-example subgraphs and unified graph from all training triples
- Comprehensive statistics: nodes, edges, connectivity, relation distribution

1.6 **Google Drive persistence setup**
- `src/utils/checkpoints.py` with `save_checkpoint()` / `load_checkpoint()` / `checkpoint_exists()`
- Supports pickle, JSON, GraphML formats
- `ensure_drive_mounted()` utility
- Automatic directory creation

### Implementation Details

- **Notebook cells:** 1-8
- **Module structure:** `src/config.py`, `src/utils/`, `src/data/`
- **Total LOC:** ~1,200
- **Cold start:** ~5 min | Warm start: ~2.5 min | Re-run: ~15 sec

### Success Criteria — ALL MET

- [x] `torch.cuda.is_available()` returns True
- [x] All imports succeed without errors
- [x] Dataset loads all 3 splits with correct row counts
- [x] Unified graph has >2K nodes and >6K edges (dev config)
- [x] Graph serializes to and deserializes from Drive correctly

---

## Phase 2: Retrieval Pipeline — COMPLETE

**Goal:** Given a natural language question, retrieve a compact, relevant subgraph from the knowledge graph using semantic search and PCST optimization.

### Tasks

2.1 **Text embedding module** — `src/retrieval/embeddings.py`
- `TextEmbedder` class wrapping Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim)
- `embed_texts(texts)` — batch encode to 384-dim vectors
- `embed_graph_entities(G)` — embed all nodes with **relation enrichment** context
  (e.g. "Cleveland" → "Cleveland | containedby, time zone, adjoins")
- `embed_relations(G)` — embed unique relations with cleaned Freebase names
- Checkpoints: `entity_embeddings.pkl`, `relation_embeddings.pkl`

2.2 **FAISS index construction** — `src/retrieval/faiss_index.py`
- `EntityIndex` class with FAISS `IndexFlatIP` (exact inner product / cosine similarity)
- `build(entity_embeddings)` — index from embeddings dict
- `search(query_embedding, k=10)` — k-NN search returning `(entity_name, score)` tuples
- Checkpoint: `faiss_index.bin`

2.3 **PCST subgraph extraction** — `src/retrieval/pcst_solver.py`
- `PCSTSolver` class implementing Prize-Collecting Steiner Tree
- **Localize-then-optimize:** BFS 500-node neighborhood from seeds, then PCST on that subgraph
- Prize structure: raw cosine similarity scores (0-1) from k-NN search
- **Local prize computation:** computes cosine similarity for root's connected component when disconnected
- **Component bridging:** shortest-path bridges between disconnected PCST components (max 6 hops)
- BFS fallback for cases where PCST solver fails
- Parameters: `cost=0.015`, `budget=70`, `local_budget=500`, `pruning="gw"`, `edge_weight_alpha=0.5`

2.4 **Retriever orchestration** — `src/retrieval/retriever.py`
- `Retriever` class coordinating TextEmbedder + EntityIndex + PCSTSolver
- `retrieve(question, q_entity, answer_entities)` → `RetrievedSubgraph`
- `build_from_checkpoint_or_new(config, unified_graph)` — factory pattern
- `RetrievedSubgraph` dataclass: subgraph, metadata, timing, `has_answer` flag

### Implementation Details

- **Notebook cells:** 9 (build), 10 (validate), 11 (criteria)
- **Module structure:** `src/retrieval/` (embeddings, faiss_index, pcst_solver, retriever)
- **Checkpoint size:** ~3 GB total (entity embeddings + FAISS index + relation embeddings)
- **Build time (cold):** ~5-10 min | Query time: 100-500 ms

### Success Criteria — ALL MET

- [x] Retrieval completes in <1 second per query
- [x] Subgraph contains answer entity in >60% of validation examples
- [x] All subgraphs are connected (verified, with component bridging)
- [x] Subgraph size respects budget cap

---

## Phase 3: GNN Encoder — COMPLETE

**Goal:** Train a GATv2-based GNN that produces node embeddings with attention weights, learning to identify query-relevant nodes within retrieved subgraphs.

### Tasks

3.1 **PyG data conversion** — `src/gnn/data_utils.py`
- `SubgraphConverter` class converting NetworkX subgraphs → PyG `Data` objects
- Node features: Sentence-Transformer embeddings (384-dim)
- Edge features: relation embeddings (384-dim)
- Query embedding: question encoding (384-dim)
- Node labels: binary (1 = answer entity, 0 = other)
- Deterministic ordering: sorted node names for reproducibility
- `GNNOutput` dataclass: `node_embeddings`, `attention_scores` (dict), `graph_embedding`

3.2 **GATv2 encoder** — `src/gnn/encoder.py`
- `GATv2Encoder(nn.Module)` with query conditioning:
  - Input projection: 384 → hidden_dim (256)
  - GATv2Conv layers: 3 layers, 4 heads each, with edge feature support
  - Residual connections + LayerNorm between layers
  - Dropout: 0.1
  - Query conditioning: broadcast query embedding to all nodes
  - Output: `(node_embeddings, attention_weights)`

3.3 **Attention pooling** — `src/gnn/pooling.py`
- `AttentionPooling(nn.Module)` using PyG `GlobalAttention`
  - Gate network: `Linear(hidden_dim, 1)` — per-node importance
  - Feature network: `Linear(hidden_dim, hidden_dim)` — pre-pool transform
  - Returns `(graph_embedding, node_attention_scores)`
- `MeanPooling` and `MaxPooling` baselines also implemented

3.4 **GraphSAGE alternative** — `src/gnn/encoder.py`
- `GraphSAGEEncoder(nn.Module)` — drop-in alternative
- SAGEConv layers with mean aggregation
- Separate learned attention head for scoring
- Same interface as GATv2Encoder

3.5 **Training loop** — `src/gnn/trainer.py`
- `GNNTrainer` class for answer node prediction task
- **Loss:** Focal loss (`gamma=2.0`) — handles severe class imbalance (1-3 answer nodes out of ~50)
- **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- **Scheduler:** `ReduceLROnPlateau` (`patience=3`, `factor=0.5`)
- **Early stopping:** patience=5 epochs
- Gradient clipping: `max_norm=1.0`
- Metrics tracked: loss, accuracy, precision, recall, F1
- `FocalLoss` class: `FL(p_t) = -(1 - p_t)^gamma * log(p_t)`

3.6 **High-level API** — `src/gnn/model_wrapper.py`
- `GNNModel` class matching Phase 2 `Retriever` pattern
- `build_from_checkpoint_or_train(config, retriever, train_data, val_data)` — factory
- `encode(retrieved_subgraph, question)` → `GNNOutput`
- `encode_batch(subgraphs, questions)` — batch processing
- Auto-loads from checkpoint or trains from scratch
- Checkpoint: `gnn_model.pt`

### Implementation Details

- **Notebook cells:** 12 (build/train), 13 (inference), 14 (metrics), 15 (visualization), 16 (memory)
- **Module structure:** `src/gnn/` (data_utils, encoder, pooling, trainer, model_wrapper)
- **Checkpoint size:** ~1.7 GB (model + PyG train/val data)
- **Training time:** ~15-20 min cold | Inference: immediate from checkpoint
- **Peak GPU VRAM:** ~800 MB

### Success Criteria — ALL MET

- [x] GNN trains without OOM on T4 GPU
- [x] Validation answer-node prediction F1 > 0.5
- [x] Attention weights visibly concentrate on answer-relevant nodes
- [x] Training completes in < 30 minutes on train split
- [x] Both GATv2 and GraphSAGE encoders produce compatible outputs

---

## Phase 4: Graph Verbalization — TODO

**Goal:** Convert GNN attention-weighted subgraphs into structured natural language text (hard prompts) suitable for LLM consumption.

**Dependencies:** Phase 3 (GNN attention scores) COMPLETE
**Can start now:** Yes

### Tasks

4.1 **Verbalizer module**
- Create `src/verbalization/verbalizer.py`
- `GraphVerbalizer` class with method:
  `verbalize(subgraph: nx.DiGraph, attention_scores: dict[str, float]) -> str`
- Rank triples by attention score of their source/target nodes
- Select top-K triples (configurable, default: top 15 via `config.top_k_triples`)
- Clean Freebase relation strings: `"people.person.sibling_s"` -> `"sibling of"`

4.2 **Prompt formatting**
- Create `src/verbalization/prompt_builder.py`
- `PromptBuilder` class that assembles the full LLM prompt:
  ```
  System: You are a knowledge graph QA assistant. Answer using ONLY the provided graph context.

  Graph context:
  1. (Justin Bieber) --[sibling of]--> (Jaxon Bieber)
  2. (Justin Bieber) --[born in]--> (London, Ontario)
  ...

  Question: {question}
  Answer:
  ```
- Configurable max context tokens (~500)
- Truncation if verbalized graph exceeds token budget
- Supports both `natural` and `structured` formats (via `config.verbalization_format`)

4.3 **Baseline verbalizer (no attention)**
- Implement `UnweightedVerbalizer` that verbalizes all triples equally
- Random ordering (seeded) up to the same token budget
- Used as the no-GNN baseline for comparison

4.4 **Relation cleaning dictionary**
- Build a mapping of common Freebase relation paths to human-readable strings
- Handle unknown relations with heuristic: split on `.`, take last segment, replace `_` with space
- Leverage the noise filtering already in `GraphBuilder` (CVT nodes and junk relations already removed)

### Deliverables
- `GraphVerbalizer` producing attention-ranked triple text
- `PromptBuilder` assembling complete LLM prompts
- `UnweightedVerbalizer` for baseline comparison
- Relation cleaning utility

### Success Criteria
- [ ] Verbalized text is human-readable and grammatically sensible
- [ ] Attention-guided verbalizer prioritizes answer-relevant triples (manual inspection on 5 examples)
- [ ] Output stays within ~500 token budget
- [ ] Baseline verbalizer produces same-length output without attention weighting

---

## Phase 5: LLM Integration — TODO

**Goal:** Call LLMs via OpenRouter API with the hard prompt, parse structured responses, and handle errors/retries gracefully.

**Dependencies:** Phase 2 (Retrieval) COMPLETE
**Can start now:** Yes (parallel with Phase 4)

### Tasks

5.1 **OpenRouter client**
- Create `src/llm/openrouter_client.py`
- Wrap `openai.OpenAI(base_url="https://openrouter.ai/api/v1")`
- API key from Colab secrets (`userdata.get("OPENROUTER_API_KEY")`)
- `generate(prompt: str, model: str, temperature: float, max_tokens: int) -> str`
- Default model: `google/gemini-2.5-flash-preview` (cost-effective)
- Set `temperature=0.0` and `seed=42` for determinism

5.2 **Retry and fallback logic**
- Tenacity decorator: retry on 429 (rate limit) and 5xx errors
- Exponential backoff: 1s, 2s, 4s, 8s, max 5 retries
- Model fallback chain: if primary model fails after retries, try next model
- No retry on 402 (no credits) or 403 (moderation) - raise immediately
- Log all API calls (model, latency, token counts, cost)

5.3 **Response parsing**
- Create `src/llm/response_parser.py`
- Extract answer text from LLM response
- Normalize: strip whitespace, lowercase for comparison
- `CausalExplanation` dataclass:
  - `answer: str` - the predicted answer
  - `reasoning: str` - full LLM response text
  - `model_used: str` - which OpenRouter model served the request
  - `latency_ms: float` - API call duration
  - `prompt_tokens: int` - input token count
  - `completion_tokens: int` - output token count

5.4 **Cost tracking**
- Track cumulative API cost across the benchmark run
- Log per-request cost based on model pricing
- Print running total after each batch

### Deliverables
- `OpenRouterClient` with retry/fallback logic
- Response parser producing `CausalExplanation` objects
- Cost tracking across benchmark run

### Success Criteria
- [ ] Successful API call to OpenRouter with a test prompt
- [ ] Retry logic handles simulated 429 errors correctly
- [ ] Fallback chain switches models on persistent failure
- [ ] Response parser extracts clean answer text
- [ ] Cost tracker accumulates correctly over multiple calls

---

## Phase 6: End-to-End Pipeline Assembly — TODO

**Goal:** Wire all components into a single `BenchmarkPipeline` class that takes a question and returns a grounded answer.

**Dependencies:** Phase 4 AND Phase 5 both complete

### Tasks

6.1 **Pipeline orchestration**
- Create `src/pipeline.py` with `BenchmarkPipeline` class
- Constructor takes `BenchmarkConfig` and initializes all components
- `run(question: str, graph_triples: list) -> BenchmarkResult` method:
  1. Build per-question NetworkX subgraph from triples
  2. Retrieve relevant subgraph via `Retriever`
  3. Encode subgraph via GNN -> attention weights
  4. Verbalize via `GraphVerbalizer` -> hard prompt
  5. Generate answer via `OpenRouterClient`
  6. Return `BenchmarkResult`

6.2 **Result container**
- `BenchmarkResult` dataclass:
  - `question: str`
  - `predicted_answer: str`
  - `gold_answers: list[str]`
  - `retrieved_subgraph_size: int`
  - `verbalized_context: str`
  - `explanation: CausalExplanation`
  - `timings: dict[str, float]` - per-stage latency

6.3 **Batch runner**
- `run_batch(dataset_split, max_samples) -> list[BenchmarkResult]`
- Progress bar with tqdm
- Checkpoint every N examples (configurable, default 50)
- Resume from last checkpoint on restart
- Configurable delay between API calls (rate limit safety)

6.4 **Smoke test**
- Run pipeline on 5 examples from validation split
- Print: question, retrieved subgraph, verbalized prompt, LLM answer, gold answer
- Verify no errors, reasonable outputs

### Deliverables
- `BenchmarkPipeline` class running end-to-end
- Batch runner with checkpointing and resume
- Smoke test passing on 5 validation examples

### Success Criteria
- [ ] Single question runs end-to-end without errors
- [ ] Batch of 5 questions completes with printed results
- [ ] Checkpointing saves and resumes correctly (simulate by interrupting and restarting)
- [ ] Per-stage timings reported for each question

---

## Phase 7: Evaluation & Benchmarking — TODO

**Goal:** Quantitative evaluation with proper metrics, baseline comparison, and reproducibility verification.

**Dependencies:** Phase 6 complete

### Tasks

7.1 **Metrics implementation**
- Create `src/evaluation/metrics.py`
- `exact_match(predicted: str, golds: list[str]) -> bool`
- `f1_score(predicted: str, golds: list[str]) -> float` (token-level, best match)
- `hit_at_1(predicted: str, golds: list[str]) -> bool`
- `graph_grounding_rate(predicted_entities: list, subgraph_entities: list) -> float`
- All metrics handle normalization (lowercase, strip articles, punctuation)

7.2 **Evaluation runner**
- Create `src/evaluation/evaluator.py`
- `Evaluator` class that takes `list[BenchmarkResult]` and computes aggregate metrics
- Report: mean and std for each metric
- Breakdown by question type if identifiable

7.3 **Baseline comparison**
- Run the no-GNN baseline (Phase 4 `UnweightedVerbalizer`) on same test set
- Run direct LLM baseline (no graph context at all) on same test set
- Three-way comparison table:
  | Method | EM | F1 | Hit@1 | Grounding |
  |--------|----|----|-------|-----------|
  | GNN-guided | ? | ? | ? | ? |
  | No-GNN (unweighted) | ? | ? | ? | ? |
  | No-graph (LLM only) | ? | ? | ? | ? |

7.4 **Reproducibility verification**
- Run full pipeline 3 times with seeds 42, 123, 456
- Compute variance across runs for each metric
- Assert variance < 2%

7.5 **Results persistence**
- Save all `BenchmarkResult` objects to Drive (JSON lines format)
- Save aggregate metrics to Drive
- Save comparison table as CSV

### Deliverables
- Metrics module with EM, F1, Hit@1, grounding rate
- Three-way baseline comparison table
- Reproducibility report (3 seeds)
- All results persisted to Google Drive

### Success Criteria
- [ ] Metrics compute correctly on manual examples (unit-tested in notebook)
- [ ] GNN-guided pipeline outperforms no-GNN baseline on F1 by >5%
- [ ] Variance across 3 seeds < 2% for all metrics
- [ ] Full test set (1,630 questions) evaluated or clear subset with justification
- [ ] Results saved to Drive and loadable in a fresh session

---

## Phase 8: Hardening & Polish — TODO

**Goal:** Production-quality error handling, clear documentation within the notebook, and a clean reproducible artifact.

**Dependencies:** Phase 7 complete

### Tasks

8.1 **Error handling audit**
- Add try/except around all API calls with meaningful error messages
- Handle Colab disconnect gracefully (checkpoint before long operations)
- Validate all inputs at module boundaries
- Test edge cases: empty subgraph, no answer entity in graph, API timeout

8.2 **Notebook organization**
- Structure notebook with clear markdown headers per phase
- Table of contents cell at top
- Configuration cell (single place to change all params)
- Each cell is idempotent (safe to re-run)
- Collapse helper code into importable `.py` files in Drive

8.3 **Logging and diagnostics**
- Structured logging with Python `logging` module
- Per-phase timing summary
- GPU memory usage reporting at key checkpoints
- API cost summary at end of run

8.4 **README and usage instructions**
- In-notebook markdown: how to set up OpenRouter API key
- How to adjust config for quick vs. full runs
- How to interpret results
- Known limitations and troubleshooting

### Deliverables
- Robust error handling across all modules
- Clean, well-organized notebook with documentation
- Diagnostic logging and cost reporting

### Success Criteria
- [ ] Notebook runs end-to-end from a fresh Colab runtime (cold start test)
- [ ] Clear error messages for common failures (no API key, OOM, rate limit)
- [ ] All cells are idempotent (running twice produces same result)
- [ ] Total runtime for full benchmark < 4 hours on free-tier Colab

---

## Dependency Graph

```
Phase 1 (Environment & Data) ✅
    |
    v
Phase 2 (Retrieval) ✅
    |
    +--------+
    |        |
    v        v
Phase 3    Phase 4.3 (Baseline Verbalizer - no GNN dependency)
(GNN) ✅     |
    |        |
    v        |
Phase 4    Phase 5 (LLM Integration - independent of GNN)
(Verbalize)  |
    |        |
    +--------+
    |
    v
Phase 6 (End-to-End Pipeline)
    |
    v
Phase 7 (Evaluation)
    |
    v
Phase 8 (Hardening)
```

**Current parallelization opportunities (as of 2026-02-18):**
- Phase 4 (Verbalization) and Phase 5 (LLM Integration) can be developed **in parallel** now
- Phase 4.3 (baseline verbalizer) has no GNN dependency
- Phase 4.1-4.2 (attention-guided verbalizer) requires Phase 3 output (COMPLETE)
- Phase 5 only requires Phase 2 output (COMPLETE)

---

## Risk Mitigations Per Phase

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | PyG wheel mismatch | Dynamic version detection; fallback to CPU-only PyG |
| 2 | PCST solver fails on disconnected graphs | BFS fallback; component bridging via shortest paths |
| 3 | GNN OOM on large subgraphs | Cap subgraph size; focal loss for class imbalance; gradient clipping |
| 4 | Freebase relations unreadable | Relation cleaning dictionary + heuristic fallback; CVT nodes already filtered |
| 5 | OpenRouter rate limits | Exponential backoff; configurable delay; model fallback |
| 6 | Session timeout mid-batch | Checkpoint every 50 examples; resume logic |
| 7 | GNN doesn't outperform baseline | Acceptable result - document finding; tune hyperparameters |
| 8 | Fresh runtime fails | Pin all dependency versions; test cold start explicitly |

---

## Codebase Summary

### Module Structure (as implemented)

```
src/
├── __init__.py              # Package exports (v0.1.0)
├── config.py                # BenchmarkConfig (all phases)
├── utils/
│   ├── seeds.py             # set_seeds() — determinism
│   └── checkpoints.py       # save/load/exists — Drive persistence
├── data/
│   ├── dataset_loader.py    # RoGWebQSPLoader — HuggingFace integration
│   └── graph_builder.py     # GraphBuilder — NetworkX with noise filtering
├── retrieval/
│   ├── embeddings.py        # TextEmbedder — Sentence-Transformers (384-dim)
│   ├── faiss_index.py       # EntityIndex — FAISS IndexFlatIP
│   ├── pcst_solver.py       # PCSTSolver — localize-then-optimize
│   └── retriever.py         # Retriever — orchestration + RetrievedSubgraph
└── gnn/
    ├── data_utils.py        # SubgraphConverter + GNNOutput
    ├── encoder.py           # GATv2Encoder + GraphSAGEEncoder
    ├── pooling.py           # AttentionPooling + MeanPooling + MaxPooling
    ├── trainer.py           # GNNTrainer + FocalLoss
    └── model_wrapper.py     # GNNModel — high-level API

scripts/
├── test_phase1_imports.py   # Local validation (no GPU)
├── test_phase2_imports.py   # Retrieval module validation
├── test_phase3_imports.py   # GNN module validation
├── update_notebook_cell1.py # Notebook cell updater
└── delete_checkpoints.py    # Clear Drive checkpoints

notebooks/
└── arcOS_benchmark.ipynb    # 16 cells (Phases 1-3)
```

### Total Code: ~3,200 lines across 15 Python modules

### Checkpoint Budget on Google Drive: ~4.7 GB

| Component | Size |
|-----------|------|
| Entity embeddings | ~1.5 GB |
| FAISS index | ~1.5 GB |
| Relation embeddings | ~8 MB |
| PyG training data | ~1.5 GB |
| PyG validation data | ~150 MB |
| GNN model weights | ~50 MB |
