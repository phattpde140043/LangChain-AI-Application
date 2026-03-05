# System Design

## Problem Statement

Research-intensive workflows require analysts to query large, heterogeneous document collections and receive accurate, cited answers without hallucination. Traditional keyword search returns raw passages; generative AI alone fabricates information not grounded in documents. The LangChain AI Research Assistant solves this by combining semantic retrieval with grounded language model generation, producing answers that are both natural-language fluent and traceable to specific source documents.

The system must serve multiple concurrent users, enforce safety constraints against adversarial inputs, operate within predictable cost envelopes, and remain maintainable as document collections and models evolve.

## Scale Requirements

- **Throughput**: 60 requests/minute per client (configurable rate limit); horizontally scalable to hundreds of concurrent users behind a load balancer.
- **Latency**: P95 response time under 5 seconds for document Q&A (dominated by LLM inference latency).
- **Document corpus**: Designed for thousands of documents totalling tens of millions of tokens in the vector store.
- **Availability**: Stateless API workers enable rolling deployments with zero downtime.

## System Architecture

The system is structured in three tiers:

**Tier 1 — API Gateway**: FastAPI application servers behind an HTTP load balancer. Stateless workers can scale horizontally. Auth middleware, rate limiting, and request validation are applied at this tier.

**Tier 2 — Processing**: The RAG pipeline and agent framework perform query understanding, retrieval, context assembly, and LLM generation. These components are in-process with the API workers.

**Tier 3 — Data**: The vector store (Chroma or FAISS) persists document embeddings. An optional Redis instance caches computed embeddings and frequent query results. LLM provider APIs (OpenAI) are external dependencies.

```
Clients
   │
   ▼
[Load Balancer]
   │
   ├──▶ [API Worker 1]  ─┐
   ├──▶ [API Worker 2]   ├──▶ [Vector Store (Chroma/FAISS)]
   └──▶ [API Worker N]  ─┘       │
                │                └──▶ [Embedding Files on Disk]
                │
                ├──▶ [Redis Cache (optional)]
                │
                └──▶ [OpenAI API (LLM + Embeddings)]
```

## Component Interaction

**Request flow for `POST /ask`:**

1. The load balancer forwards the request to an available API worker.
2. `APIKeyMiddleware` validates the bearer token (bypassed when `API_KEY` is empty).
3. `RateLimiter` increments the per-IP counter; returns HTTP 429 if the limit is exceeded.
4. The request body is deserialized and validated by Pydantic into `AskRequest`.
5. `GuardrailsManager.check_input` rejects queries with injection patterns or excessive length.
6. `CoordinatorAgent.route` scores the query against intent classifiers and selects an agent.
7. For research queries, `RAGPipeline.run` executes: query expansion → vector retrieval → context selection → LLM generation.
8. The LLM response is validated by `GuardrailsManager.check_output`.
9. Token usage is appended to `./data/costs.jsonl`; latency is appended to `./data/metrics.jsonl`.
10. `AskResponse` is serialized and returned with HTTP 200.

## Scalability Strategy

### Stateless Workers

API workers hold no per-request state. The vector store path is a shared persistent volume (or object storage mount in cloud deployments), accessible by all workers. Scaling out requires only adding worker replicas.

### Embedding Cache

Embedding computation is deterministic and expensive. A Redis cache keyed by `sha256(text + model_name)` stores pre-computed embedding vectors. On a cache hit the embedding API call is skipped entirely, reducing both latency and cost for repeated or similar queries.

### Semantic Response Cache

Completed RAG responses for a query are cached by query hash. If a semantically identical query arrives within the TTL, the cached response is returned in milliseconds. This is particularly effective for high-traffic FAQ-style document corpora.

### Horizontal Pod Autoscaling

In Kubernetes deployments, the Horizontal Pod Autoscaler scales API worker replicas based on CPU utilization and custom request-rate metrics exported via the `/metrics` endpoint (Prometheus format).

## Data Management

**Vector Store**: Document chunks are stored as embedding vectors alongside metadata (filename, chunk index, page number). Chroma persists data to `./data/vector_store` by default. Re-ingestion is idempotent: existing documents are replaced by their new embeddings.

**Feedback Log**: User ratings (`POST /feedback`) are appended to `./data/feedback.jsonl` in newline-delimited JSON format for offline quality analysis.

**Eval Dataset**: `./data/eval_dataset.json` holds Q&A pairs used for offline accuracy benchmarking. The `generate_eval_dataset.py` script seeds this file from curated examples.

**Cost Log**: Each LLM call records model name, prompt tokens, completion tokens, and estimated USD cost to `./data/costs.jsonl`, enabling per-team cost attribution and budget alerting.

## Deployment Considerations

### Docker

A `Dockerfile` builds a minimal Python image. A `docker-compose.yml` orchestrates the API server with optional Redis and a volume mount for the persistent vector store. Environment variables are sourced from `.env` (not committed to source control).

### Kubernetes

For production, each API worker runs as a `Deployment` with a `PodDisruptionBudget` ensuring at least one replica is available during rolling updates. The vector store volume is backed by a `PersistentVolumeClaim`. Secrets (OpenAI API key) are injected via Kubernetes `Secret` objects, not environment variable files.

### Environment Profiles

Three profiles—`development`, `testing`, `production`—are defined in `get_profile_settings`. Development enables debug logging and disables cost tracking. Testing disables safety checks and switches to FAISS for in-memory vector operations. Production enforces safety checks and full cost tracking.

## Security Design

**Authentication**: Optional API key enforcement via bearer token in the `Authorization` header. When `API_KEY` is set, unauthenticated requests receive HTTP 403.

**Rate Limiting**: Per-IP sliding-window rate limiting (default 60 req/min) prevents abuse and protects LLM cost budgets. Limits are configurable via `RATE_LIMIT_PER_MINUTE`.

**Prompt Injection Defense**: The `PromptInjectionDetector` scans inputs for 12 adversarial patterns (e.g., "ignore previous instructions", "jailbreak", "reveal system prompt"). Queries with risk score ≥ 0.6 are blocked before reaching the LLM.

**Secret Management**: The OpenAI API key is read from environment variables at startup and never logged, stored in vector metadata, or returned in API responses. Production deployments use a secrets manager (AWS Secrets Manager or HashiCorp Vault) to inject the key at runtime.

**Output Validation**: `GuardrailsManager.check_output` validates that LLM responses meet minimum length requirements and optionally verifies grounding via word-overlap heuristics against retrieved source chunks before the response is returned to the client.
