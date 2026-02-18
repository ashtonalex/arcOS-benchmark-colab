# Hit Rate Mechanism Fixes — Deferred Items

Fixes 1, 3, and 5 are implemented. The two items below are deferred for a follow-up pass.

## Fix 2: Add `has_answer` flag to `RetrievedSubgraph` (Medium effort)

**Problem:** Hit tracking is a notebook-only ad-hoc computation. The retriever has no awareness of whether the answer was captured.

**Proposed change:**
- Add `has_answer: Optional[bool] = None` to the `RetrievedSubgraph` dataclass.
- Add an optional `answer_entities` parameter to `Retriever.retrieve()`.
- When provided, set `has_answer = any(ans in subgraph.nodes() for ans in answer_entities)`.
- Add a `Retriever.evaluate_batch()` method that retrieves over a list of examples and returns aggregate hit statistics.

**Why deferred:** Fix 1 already adds coverage-aware hit rate computation as standalone functions on the Retriever. Baking it into the dataclass is a cleaner API but not urgently needed — the diagnostic and training improvements from Fixes 1/3/5 are the priority.

## Fix 4: Expand validation sample size (Trivial effort)

**Problem:** Cell 10 only evaluates 10 validation examples. At 60% hit rate, the 95% confidence interval is roughly +/-30% — statistically meaningless.

**Proposed change:**
- Change `range(10)` to `range(min(50, len(dataset["validation"])))` in Cell 10.
- At 50 examples, 60% hit rate has a 95% CI of +/-14% — still wide but usable.
- For final evaluation, use the full validation set.

**Why deferred:** Trivial to change but depends on retrieval speed. With CPU-only FAISS and PCST on 50 examples, this may take 2-3 minutes per run. Should be done alongside any notebook cell update pass.
