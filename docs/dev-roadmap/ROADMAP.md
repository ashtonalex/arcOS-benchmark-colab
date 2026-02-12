# arcOS Benchmark - Implementation Roadmap

**Version:** 2.0 (Colab Edition)
**Date:** 2026-02-09

---

## Roadmap Summary

| Phase | Name | Goal | Est. Cells |
|-------|------|------|------------|
| 1 | Environment & Data | Colab setup, dataset loaded, graph built | 6-8 |
| 2 | Retrieval Pipeline | Query-to-subgraph extraction working | 5-7 |
| 3 | GNN Encoder | Trainable GNN producing node embeddings with attention | 6-8 |
| 4 | Graph Verbalization | Attention-guided subgraph-to-text conversion | 3-4 |
| 5 | LLM Integration | OpenRouter API calls with hard prompt | 4-5 |
| 6 | End-to-End Pipeline | Full query-to-answer pipeline assembled | 3-4 |
| 7 | Evaluation & Benchmarking | Metrics, baselines, reproducibility | 5-6 |
| 8 | Hardening & Polish | Error handling, checkpointing, documentation | 3-4 |

---

## Phase 1: Environment & Data Foundation

**Goal:** Colab notebook boots cleanly, all dependencies install via uv, RoG-webqsp dataset is loaded, and a NetworkX knowledge graph is constructed from the dataset's triples.

### Tasks

1.1 **Colab environment setup cell**
- Clear `UV_CONSTRAINT` and `UV_BUILD_CONSTRAINT` env vars
- Install `uv` via pip
- Install all dependencies via `uv pip install --system`
- Dynamically detect PyTorch/CUDA versions for PyG wheel matching
- Verify GPU availability with `torch.cuda.is_available()`

1.2 **Seed & determinism initialization**
- Create `utils/seeds.py` module with centralized seed management
- Set `torch.manual_seed()`, `numpy.random.seed()`, `random.seed()`, `PYTHONHASHSEED`
- Set `torch.backends.cudnn.deterministic = True`

1.3 **Configuration dataclasses**
- Create `config.py` with `BenchmarkConfig` Pydantic dataclass
- All hyperparameters with sensible defaults (as specified in PRD Section 5)
- Validation rules (e.g., `gnn_layers >= 1`, `temperature >= 0`)

1.4 **Dataset loading**
- Load `rmanluo/RoG-webqsp` via HuggingFace `datasets` library
- Redirect HF cache to Google Drive for persistence
- Inspect and validate schema (question, answer, graph, q_entity, a_entity)
- Print dataset statistics (splits, sizes, avg graph size per example)

1.5 **NetworkX graph construction**
- For each example, convert `graph` field (list of triples) into a NetworkX `DiGraph`
- Build a **unified** graph from all training triples for embedding index
- Store node attributes (entity name) and edge attributes (relation type)
- Print graph statistics (nodes, edges, connected components)

1.6 **Google Drive persistence setup**
- Mount Google Drive
- Create checkpoint directory structure
- Implement `save_checkpoint()` / `load_checkpoint()` utility functions
- Serialize NetworkX graph to Drive (pickle format)

### Deliverables
- Working Colab cell that installs all deps in <60 seconds
- `BenchmarkConfig` dataclass with all parameters
- NetworkX graph loaded from RoG-webqsp with verified statistics
- Checkpoint infrastructure on Google Drive

### Success Criteria
- [ ] `torch.cuda.is_available()` returns True
- [ ] All imports succeed without errors
- [ ] Dataset loads all 3 splits with correct row counts
- [ ] Unified graph has >10K nodes and >30K edges
- [ ] Graph serializes to and deserializes from Drive correctly

---

## Phase 2: Retrieval Pipeline

**Goal:** Given a natural language question, retrieve a compact, relevant subgraph from the knowledge graph using semantic search and PCST optimization.

### Tasks

2.1 **Text embedding module**
- Create `retriever/embeddings.py`
- Load Sentence-Transformers model (`all-MiniLM-L6-v2`)
- Batch-encode all node entity names into embeddings
- Batch-encode all edge relation strings into embeddings
- Cache embeddings to Google Drive

2.2 **FAISS index construction**
- Create `retriever/vector_index.py`
- Build FAISS `IndexFlatIP` (inner product) from node embeddings
- Build separate index for edge embeddings
- Implement `search(query_text, top_k) -> list[node_ids]`
- Serialize/deserialize index to Drive

2.3 **PCST subgraph extraction**
- Create `retriever/pcst_solver.py`
- Implement prize assignment: rank-based prizes for top-k nodes
- Handle virtual nodes for edge prizes
- Call `pcst_fast` solver with configurable cost parameter
- Fallback to BFS-based extraction if PCST fails (e.g., disconnected nodes)
- Return connected subgraph as NetworkX `DiGraph`

2.4 **Retriever orchestration**
- Create `retriever/retriever.py` with `Retriever` class
- `retrieve(question: str) -> RetrievedSubgraph` method
- `RetrievedSubgraph` dataclass: NetworkX subgraph + metadata (scores, timings)
- Cap subgraph at `max_subgraph_nodes` (default 20)

2.5 **Retrieval validation**
- Test retriever on 10 examples from validation split
- Measure: subgraph contains answer entity? (hit rate)
- Measure: avg subgraph size, avg retrieval time
- Visualize one example subgraph with `networkx.draw()`

### Deliverables
- `Retriever` class producing connected subgraphs from queries
- FAISS index with all node/edge embeddings
- Retrieval quality metrics on validation subset

### Success Criteria
- [ ] Retrieval completes in <1 second per query
- [ ] Subgraph contains answer entity in >60% of validation examples
- [ ] All subgraphs are connected (verified)
- [ ] Subgraph size respects `max_subgraph_nodes` cap

---

## Phase 3: GNN Encoder

**Goal:** Train a GATv2-based GNN that produces node embeddings with attention weights, learning to identify query-relevant nodes within retrieved subgraphs.

### Tasks

3.1 **PyG data conversion**
- Create `gnn/data_utils.py`
- Convert NetworkX subgraph + Sentence-Transformer embeddings into PyG `Data` object
- Node features: Sentence-Transformer embeddings of entity names (384-dim)
- Edge features: Sentence-Transformer embeddings of relation strings (384-dim)
- Include query embedding as a global graph attribute

3.2 **GATv2 encoder**
- Create `gnn/encoder.py`
- Implement `GATv2Encoder(nn.Module)`:
  - Configurable layers (default 2), heads (default 4), hidden dim (default 256)
  - Input projection: 384 -> hidden_dim
  - GATv2Conv layers with edge features
  - Dropout between layers
  - Return both node embeddings and attention weights

3.3 **Attention pooling**
- Implement `AttentionPooling(nn.Module)` using PyG `GlobalAttention`
- Gate network: `Linear(hidden_dim, 1)` - learns per-node importance
- Feature network: `Linear(hidden_dim, hidden_dim)` - transforms before pooling
- Returns graph-level embedding + per-node attention scores

3.4 **GraphSAGE alternative encoder**
- Implement `GraphSAGEEncoder(nn.Module)` as drop-in alternative
- Same interface as GATv2Encoder
- SAGEConv layers with mean aggregation
- For attention scores: use a separate learned scoring MLP over node embeddings

3.5 **Training loop**
- Create `gnn/trainer.py`
- Training objective: cross-entropy on answer entity prediction
  - Given query + subgraph, predict which node(s) are answer entities
  - Uses `a_entity` field as supervision signal
- DataLoader: batch of (subgraph, query, answer_entities) tuples
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Early stopping on validation loss
- Save best model checkpoint to Drive

3.6 **Training validation**
- Train on train split, validate on validation split
- Track: loss curve, answer node prediction accuracy
- Visualize: attention weights on example subgraphs (should focus on answer-relevant nodes)

### Deliverables
- `GATv2Encoder` producing node embeddings + attention weights
- `AttentionPooling` producing graph-level representation + node scores
- Trained GNN checkpoint on Google Drive
- Training curves and attention visualizations

### Success Criteria
- [ ] GNN trains without OOM on T4 GPU
- [ ] Validation answer-node prediction accuracy > 50%
- [ ] Attention weights visibly concentrate on answer-relevant nodes
- [ ] Training completes in < 30 minutes on train split
- [ ] Both GATv2 and GraphSAGE encoders produce compatible outputs

---

## Phase 4: Graph Verbalization

**Goal:** Convert GNN attention-weighted subgraphs into structured natural language text (hard prompts) suitable for LLM consumption.

### Tasks

4.1 **Verbalizer module**
- Create `verbalization/verbalizer.py`
- `GraphVerbalizer` class with method:
  `verbalize(subgraph: nx.DiGraph, attention_scores: dict[str, float]) -> str`
- Rank triples by attention score of their source/target nodes
- Select top-K triples (configurable, default: top 10)
- Clean Freebase relation strings: `"people.person.sibling_s"` -> `"sibling of"`

4.2 **Prompt formatting**
- Create `verbalization/prompt_builder.py`
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

4.3 **Baseline verbalizer (no attention)**
- Implement `UnweightedVerbalizer` that verbalizes all triples equally
- Random ordering (seeded) up to the same token budget
- Used as the no-GNN baseline for comparison

4.4 **Relation cleaning dictionary**
- Build a mapping of common Freebase relation paths to human-readable strings
- Handle unknown relations with a simple heuristic (split on `.`, take last segment, replace `_` with space)

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

## Phase 5: LLM Integration

**Goal:** Call LLMs via OpenRouter API with the hard prompt, parse structured responses, and handle errors/retries gracefully.

### Tasks

5.1 **OpenRouter client**
- Create `llm/openrouter_client.py`
- Wrap `openai.OpenAI(base_url="https://openrouter.ai/api/v1")`
- API key from Colab secrets (`userdata.get("OPENROUTER_API_KEY")`)
- `generate(prompt: str, model: str, temperature: float, max_tokens: int) -> str`
- Set `temperature=0.0` and `seed=42` for determinism

5.2 **Retry and fallback logic**
- Tenacity decorator: retry on 429 (rate limit) and 5xx errors
- Exponential backoff: 1s, 2s, 4s, 8s, max 5 retries
- Model fallback chain: if primary model fails after retries, try next model
- No retry on 402 (no credits) or 403 (moderation) - raise immediately
- Log all API calls (model, latency, token counts, cost)

5.3 **Response parsing**
- Create `llm/response_parser.py`
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

## Phase 6: End-to-End Pipeline Assembly

**Goal:** Wire all components into a single `BenchmarkPipeline` class that takes a question and returns a grounded answer.

### Tasks

6.1 **Pipeline orchestration**
- Create `pipeline.py` with `BenchmarkPipeline` class
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

## Phase 7: Evaluation & Benchmarking

**Goal:** Quantitative evaluation with proper metrics, baseline comparison, and reproducibility verification.

### Tasks

7.1 **Metrics implementation**
- Create `evaluation/metrics.py`
- `exact_match(predicted: str, golds: list[str]) -> bool`
- `f1_score(predicted: str, golds: list[str]) -> float` (token-level, best match)
- `hit_at_1(predicted: str, golds: list[str]) -> bool`
- `graph_grounding_rate(predicted_entities: list, subgraph_entities: list) -> float`
- All metrics handle normalization (lowercase, strip articles, punctuation)

7.2 **Evaluation runner**
- Create `evaluation/evaluator.py`
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

## Phase 8: Hardening & Polish

**Goal:** Production-quality error handling, clear documentation within the notebook, and a clean reproducible artifact.

### Tasks

8.1 **Error handling audit**
- Add try/except around all API calls with meaningful error messages
- Handle Colab disconnect gracefully (checkpoint before long operations)
- Validate all inputs at module boundaries (Pydantic)
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
Phase 1 (Environment & Data)
    |
    v
Phase 2 (Retrieval)
    |
    +--------+
    |        |
    v        v
Phase 3    Phase 4.3 (Baseline Verbalizer - no GNN dependency)
(GNN)        |
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

**Parallelization opportunities:**
- Phase 3 (GNN) and Phase 5 (LLM Integration) can be developed in parallel after Phase 2
- Phase 4.3 (baseline verbalizer) can be built alongside Phase 3
- Phase 4.1-4.2 (attention-guided verbalizer) requires Phase 3 completion

---

## Risk Mitigations Per Phase

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | PyG wheel mismatch | Dynamic version detection; fallback to CPU-only PyG |
| 2 | PCST solver fails on disconnected graphs | BFS fallback; filter to largest connected component first |
| 3 | GNN OOM on large subgraphs | Cap subgraph size; reduce batch size; gradient accumulation |
| 4 | Freebase relations unreadable | Relation cleaning dictionary + heuristic fallback |
| 5 | OpenRouter rate limits | Exponential backoff; configurable delay; model fallback |
| 6 | Session timeout mid-batch | Checkpoint every 50 examples; resume logic |
| 7 | GNN doesn't outperform baseline | Acceptable result - document finding; tune hyperparameters |
| 8 | Fresh runtime fails | Pin all dependency versions; test cold start explicitly |
