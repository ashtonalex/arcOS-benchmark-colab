# Benchmark System in Google Colab - Research Notes

## 1. OpenRouter API

### Overview
OpenRouter provides a unified API gateway to 400+ LLMs from dozens of providers (OpenAI, Anthropic, Google, Meta, Mistral, etc.) through a single endpoint and API key. It is OpenAI-compatible, meaning you can use the standard `openai` Python SDK with minimal changes.

### Authentication
- Create an API key at https://openrouter.ai/keys
- Set the `Authorization` header: `Bearer <OPENROUTER_API_KEY>`
- Optional headers for attribution:
  - `HTTP-Referer`: Your site URL (helps with leaderboard ranking)
  - `X-Title`: Your app's display name

### API Endpoint & Format
- **Base URL**: `https://openrouter.ai/api/v1`
- **Chat completions**: `POST /api/v1/chat/completions`
- **List models**: `GET /api/v1/models`
- **Check key/credits**: `GET /api/v1/key`
- **Query generation**: `GET /api/v1/generation?id=$GENERATION_ID`

### Python Code - Basic Usage (OpenAI SDK)

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "arcOS Benchmark"
    }
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Python Code - Streaming

```python
stream = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Explain graph theory"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Python Code - Raw requests (no SDK)

```python
import requests

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "arcOS Benchmark"
    },
    json={
        "model": "anthropic/claude-sonnet-4",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())
```

### Model Naming Convention
Models use `provider/model-name` format:
- `openai/gpt-4o`
- `anthropic/claude-sonnet-4`
- `google/gemini-2.5-pro-preview`
- `meta-llama/llama-3.1-8b-instruct`
- Use `openrouter/auto` for automatic model selection

### Model Fallbacks

```python
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "models": [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "google/gemini-2.5-pro-preview"
        ]
    }
)
# If claude fails, tries gpt-4o, then gemini automatically
# Pricing is based on the model that actually serves the request
```

### Provider Routing Control

```python
response = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "provider": {
            "order": ["together", "fireworks", "perplexity"],
            "quantizations": ["fp8"]  # optional quantization preference
        }
    }
)
```

### Rate Limits
- **Free models**: ~20 requests per minute (RPM)
- **Daily limits (free models)**:
  - Without credits: ~50 requests/day
  - With credits purchased: ~1000 requests/day
- Per-model limits vary; different models have different rate limits
- Cloudflare DDoS protection blocks excessive usage
- Monitor usage via `GET /api/v1/key`

### Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 400  | Bad Request - invalid params | Fix request parameters |
| 401  | Unauthorized - invalid API key | Check/refresh API key |
| 402  | Payment Required - no credits | Add credits to account |
| 403  | Forbidden - content moderation | Modify flagged input |
| 408  | Request Timeout | Retry with backoff |
| 429  | Rate Limited | Exponential backoff + wait |
| 502  | Bad Gateway - provider down | Retry or use fallback |
| 503  | Service Unavailable | Try different model/provider |

### Error Response Format

```json
{
  "error": {
    "code": 429,
    "message": "Rate limit exceeded",
    "metadata": {}
  }
}
```

### Recommended Retry Strategy

```python
import time
import random

def call_with_retry(client, max_retries=5, **kwargs):
    """Call OpenRouter with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            error_code = getattr(e, 'status_code', None)
            if error_code == 402:
                raise  # No credits - don't retry
            if error_code == 403:
                raise  # Moderation - don't retry
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt+1} failed ({e}), retrying in {wait:.1f}s...")
            time.sleep(wait)
```

### Streaming Error Handling
- **Pre-stream errors**: Standard HTTP error codes
- **Mid-stream errors**: Sent as SSE with `finish_reason: "error"` (HTTP status remains 200 since headers already sent)
- Use `debug: {"echo_upstream_body": true}` during development to see how OpenRouter transforms your request for each provider

---

## 2. uv Package Manager in Colab

### Overview
`uv` is an extremely fast Python package manager written in Rust by Astral (creators of Ruff). It is reportedly up to 100x faster than pip.

### Installation in Colab

```python
# Cell 1: Install uv
!pip install uv
```

### Known Issue & Workaround
Google Colab sets environment variables (`UV_CONSTRAINT`, `UV_BUILD_CONSTRAINT`) that point to non-existent constraint files, causing errors when running `uv pip install`.

**Error**: `File not found: /backend-container/containers/requirements.constraints`

**Workaround (set once per session)**:

```python
# Cell 2: Fix Colab environment variables for uv
import os
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""
```

**Alternative per-command workaround**:

```bash
!UV_CONSTRAINT= UV_BUILD_CONSTRAINT= uv pip install <package>
```

**Note**: As of late 2025, the Colab team has addressed the underlying issue, but the environment variable workaround remains a safe defensive pattern.

### Recommended Setup Pattern for Colab

```python
# === Cell 1: Setup uv package manager ===
import os
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""

!pip install uv

# === Cell 2: Install project dependencies (fast) ===
!uv pip install --system openai networkx requests tenacity
```

**Key flag**: `--system` is needed in Colab because there is no virtual environment; uv installs directly into the system Python.

### Benefits in Colab
- Dramatically faster dependency resolution and installation (important since Colab sessions are ephemeral and dependencies must be reinstalled on each new session)
- Supports all pip features (requirements files, extras, etc.)
- Global cache reduces redundant downloads (within a session)

---

## 3. Google Colab Constraints

### GPU Memory Limits

| Tier | GPU Types | VRAM | System RAM | Notes |
|------|-----------|------|------------|-------|
| Free | T4 | ~15 GB | ~13 GB | Not guaranteed; may get older GPUs |
| Pro ($9.99/mo) | T4, L4 | 15-22.5 GB | Up to 32 GB | Priority access; L4 has 22.5 GB |
| Pro+ ($49.99/mo) | T4, L4, A100 | 15-40 GB | Up to 52 GB | A100 has 40 GB VRAM |

**GPU Details**:
- **T4**: 15 GB usable VRAM (16 GB minus 1 GB for ECC)
- **L4**: 22.5 GB VRAM
- **A100**: 40 GB VRAM
- V100 and P100 are no longer available in Colab

**Important**: Google does NOT guarantee specific GPU types or publish exact limits. Resources fluctuate dynamically.

### Session Timeout Limits

| Tier | Max Runtime | Idle Timeout | Notes |
|------|-------------|-------------|-------|
| Free | ~12 hours | ~90 minutes | Dynamic; may be shorter under load |
| Pro | ~24 hours | Longer | Based on compute units |
| Pro+ | ~24 hours | Longest | Continuous execution with sufficient CU |

**Compute unit consumption**:
- T4: ~11.7 CU/hour
- A100: ~62 CU/hour

### File System Persistence

**What is ephemeral (lost on restart)**:
- Everything in `/content/` (the working directory)
- Installed pip/uv packages
- Downloaded datasets
- Trained model weights (if not saved externally)
- Environment variables, running processes

**What persists**:
- Google Drive (when mounted at `/content/drive`)
- Google Cloud Storage (via `gcsfs`)
- External databases/APIs

### Best Practices for Long-Running ML Workloads

1. **Mount Google Drive for persistence**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   SAVE_DIR = '/content/drive/MyDrive/arcOS-benchmark/'
   ```

2. **Save checkpoints frequently**:
   ```python
   # Save results/state periodically to Drive
   import json
   def save_checkpoint(data, path):
       with open(path, 'w') as f:
           json.dump(data, f)
   ```

3. **Use idempotent/resumable workflows**:
   - Track which benchmark tasks have completed
   - On restart, skip already-completed work
   - Store progress in a JSON/CSV file on Drive

4. **Install dependencies at the top of every notebook**:
   ```python
   # This must run every time since packages don't persist
   !pip install uv
   import os
   os.environ["UV_CONSTRAINT"] = ""
   os.environ["UV_BUILD_CONSTRAINT"] = ""
   !uv pip install --system openai networkx requests
   ```

5. **Upload datasets as zip files** for faster transfer, then unzip in Colab.

6. **For very large files**, use Google Cloud Storage instead of Drive.

7. **Avoid remote-control/headless patterns** on free tier - Colab may terminate sessions running without active notebook UI interaction.

8. **Keep the browser tab active** (or use Colab Pro) to avoid idle timeout.

---

## 4. Memgraph Alternatives for Colab

### The Problem
Memgraph requires Docker to run locally, which is not available in Google Colab. Three main alternatives exist:

### Option A: NetworkX (Recommended for Benchmarks)

NetworkX is a pure-Python in-memory graph library. It is the best fit for a Colab-based benchmark system because it requires zero infrastructure.

**Pros**:
- Pure Python, no external services needed
- Pre-installed in Colab (no installation needed)
- Rich algorithm library (PageRank, shortest paths, community detection, etc.)
- Easy serialization (pickle, JSON, GraphML)
- Well-documented, large community
- Sufficient for graphs with up to ~100K-500K nodes in Colab's RAM

**Cons**:
- Pure Python = slower than C++ implementations for large graphs
- No built-in persistence (must serialize/deserialize manually)
- Not a database - no query language, no ACID transactions
- Memory-bound: entire graph must fit in RAM

**Usage Pattern**:

```python
import networkx as nx
import json

# Create and populate graph
G = nx.DiGraph()
G.add_node("concept_1", label="Neural Networks", type="concept")
G.add_node("concept_2", label="Backpropagation", type="concept")
G.add_edge("concept_1", "concept_2", relation="uses")

# Run algorithms
pagerank = nx.pagerank(G)
shortest = nx.shortest_path(G, "concept_1", "concept_2")

# Serialize to Drive for persistence
nx.write_graphml(G, "/content/drive/MyDrive/graph.graphml")
# Or as JSON
from networkx.readwrite import json_graph
data = json_graph.node_link_data(G)
with open("/content/drive/MyDrive/graph.json", "w") as f:
    json.dump(data, f)
```

### Option B: Memgraph Cloud (External Service)

Connect to a managed Memgraph instance from Colab.

**Pros**:
- Full graph database with Cypher query language
- Persistent storage (data survives Colab restarts)
- High performance (C++ engine)
- Built-in NetworkX algorithm wrappers

**Cons**:
- 14-day free trial only; paid after that
- Requires network connectivity from Colab
- Adds latency (network round-trips)
- External dependency for benchmark reproducibility

**Usage Pattern**:

```python
!pip install gqlalchemy

from gqlalchemy import Memgraph

# Connect to Memgraph Cloud instance
mg = Memgraph(
    host="your-instance.memgraph.cloud",
    port=7687,
    username="your-username",
    password="your-password",
    encrypted=True
)

# Execute Cypher queries
mg.execute("CREATE (n:Concept {name: 'Neural Networks'})")
results = mg.execute_and_fetch("MATCH (n) RETURN n")
```

### Option C: Neo4j AuraDB Free Tier

Neo4j offers a permanently free cloud tier (AuraDB Free) that works from Colab.

**Pros**:
- Permanently free tier (1 database, 200K nodes, 400K relationships)
- Full Cypher query support
- Mature ecosystem with good Python drivers
- Persistent storage

**Cons**:
- Limited to 200K nodes / 400K relationships on free tier
- Network latency
- External dependency

### Option D: SQLite + Graph Queries

Use SQLite (built into Python) with adjacency-list tables for basic graph operations.

**Pros**:
- Zero dependencies; built into Python
- Persistent (save .db file to Drive)
- SQL-based queries

**Cons**:
- No native graph traversal algorithms
- Must implement graph algorithms manually
- Not designed for graph workloads

### Recommendation for Benchmark System

**Use NetworkX as the primary graph layer** for the following reasons:

1. Zero infrastructure requirements (works in any Colab session)
2. No external service dependencies (benchmark is self-contained)
3. Sufficient performance for benchmark-scale graphs (typically < 100K nodes)
4. Easy to serialize to Google Drive for persistence across sessions
5. If performance becomes an issue, can later add an adapter to connect to Memgraph Cloud

**Persistence pattern with NetworkX**:

```python
import networkx as nx
import pickle
import os

GRAPH_PATH = "/content/drive/MyDrive/arcOS-benchmark/knowledge_graph.pkl"

def save_graph(G, path=GRAPH_PATH):
    """Save graph to Google Drive."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(G, f)

def load_graph(path=GRAPH_PATH):
    """Load graph from Google Drive, or create new if not found."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return nx.DiGraph()
```

---

## Summary: Recommended Architecture for Colab Benchmark

```
[Google Colab Notebook]
    |
    |-- uv (fast package installs, with env var workaround)
    |-- openai SDK -> OpenRouter API (access 400+ LLMs)
    |       |-- Fallback chains for reliability
    |       |-- Exponential backoff for rate limits
    |-- NetworkX (in-memory graph, replaces Memgraph)
    |       |-- Serialize to Google Drive for persistence
    |-- Google Drive (mounted at /content/drive)
            |-- Checkpoints, results, graph state
```
