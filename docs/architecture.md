# Architecture

## System Overview

The LangChain AI Research Assistant is a production-grade document Q&A platform that combines Retrieval-Augmented Generation (RAG) with a multi-agent framework. Users submit natural-language questions through a REST API; the system retrieves the most relevant passages from an indexed document corpus, assembles a grounded context window, and generates precise, cited answers using a large language model.

The system is designed for horizontal scalability, safety-by-default operation, and observable production deployments.

## Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│   /ask  /summarize  /compare  /ingest  /feedback  /knowledge-base│
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │     Agent Coordinator       │
              │  (intent classification)    │
              └──┬──────────┬──────────────┘
                 │          │
    ┌────────────▼──┐  ┌────▼──────────────┐
    │ Research Agent│  │ Summarization /    │
    │  (RAG Q&A)    │  │ Comparison /       │
    └────────┬──────┘  │ Analysis Agents    │
             │         └────────────────────┘
    ┌────────▼──────────────────────────────┐
    │            RAG Pipeline               │
    │  Query Expansion → Retrieval →        │
    │  Context Manager → LLM Generation     │
    └────────┬──────────────────────────────┘
             │
    ┌────────▼──────────┐   ┌───────────────────┐
    │   Vector Store     │   │  Ingestion Pipeline│
    │ (Chroma / FAISS)   │   │  Loader → Chunker  │
    └────────────────────┘   └───────────────────┘
             │
    ┌────────▼──────────────────────────────┐
    │         Security Layer                │
    │  Injection Detector + Guardrails      │
    └───────────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────┐
    │         Monitoring                    │
    │  Structured Logger + Cost Tracker +   │
    │  Metrics Collector                    │
    └───────────────────────────────────────┘
```

## Component Descriptions

### API Layer

Built with FastAPI, the API layer exposes six REST endpoints. API key authentication (optional) and per-minute rate limiting are applied as ASGI middleware. A global exception handler prevents stack traces from leaking to clients. The `/health` endpoint is used by load balancer health checks. All request/response schemas are validated with Pydantic.

### RAG Pipeline

The RAG pipeline orchestrates four sequential stages for every research query:

1. **Query Expansion** — generates semantically related queries to broaden retrieval coverage, reducing the impact of vocabulary mismatch between the question and indexed documents.
2. **Retrieval** — executes hybrid search (dense vector similarity + optional BM25 keyword match) against the vector store, returning the top-K candidate chunks.
3. **Context Manager** — ranks retrieved chunks by keyword overlap with the query, removes near-duplicates via Jaccard similarity, and greedily selects chunks within the configured token budget.
4. **LLM Generation** — formats the assembled context and question into the active prompt template, invokes the language model, and returns a structured response with answer, sources, and confidence.

### Agent Framework

The `CoordinatorAgent` classifies incoming queries by intent and delegates to the appropriate specialist:

- **ResearchAgent** — handles document Q&A by running the full RAG pipeline.
- **SummarizationAgent** — produces concise summaries of one or more documents.
- **ComparisonAgent** — performs structured analysis of two documents, highlighting similarities, differences, and trade-offs.
- **AnalysisAgent** — extracts key concepts, technologies, and architectural patterns from a document.

Agents share a `BaseAgent` contract and use a common tool registry (`document_search`, `summarization`, `comparison`, `concept_extraction`, `citation_finder`).

### Vector Store

The `VectorStoreManager` wraps Chroma (default) or FAISS (lightweight / testing) behind a uniform interface. Documents are stored as embedding vectors alongside metadata (filename, page number, chunk index). The embedding model (`text-embedding-3-small` by default) is managed by `EmbeddingManager`, which caches model instances to avoid repeated initialization.

### Ingestion Pipeline

The `IngestionPipeline` orchestrates end-to-end document loading: `DocumentLoader` reads Markdown, plain text, and PDF files; `TextChunker` splits documents into overlapping chunks (default 800 chars, 100 overlap); `EmbeddingManager` computes embeddings; `VectorStoreManager` persists the indexed chunks.

### Security Layer

`PromptInjectionDetector` scans every user query for 12 known injection patterns and returns a risk score. `GuardrailsManager` enforces length limits, blocks high-risk queries, and validates LLM outputs before they are returned to clients. Safety checks are configurable via the `ENABLE_SAFETY_CHECKS` feature flag.

### Monitoring

`setup_logging` configures structured JSON logging with configurable level. `CostTracker` records token usage and estimated cost per request to a JSONL file. `MetricsCollector` captures latency, retrieval count, and cache hit rate. All monitoring components write to the `./data/` directory by default.

## Data Flow

1. Client sends `POST /ask { "question": "..." }`.
2. Auth middleware validates the API key (if configured).
3. Rate limiter checks the per-minute quota.
4. `GuardrailsManager.check_input` scans for injection patterns.
5. `CoordinatorAgent.route` classifies intent and selects an agent.
6. `ResearchAgent` calls `RAGPipeline.run`:
   a. `QueryExpansion.expand` generates related queries.
   b. `Retriever.retrieve` fetches top-K chunks from the vector store.
   c. `ContextManager` ranks, deduplicates, and selects chunks.
   d. LLM generates an answer from the assembled context.
7. `GuardrailsManager.check_output` validates the response.
8. `CostTracker` logs token usage; `MetricsCollector` logs latency.
9. Structured `RAGResponse` is serialized and returned to the client.

## Design Decisions

**Pydantic Settings** — all configuration is managed through `pydantic-settings`, enabling type-safe environment variable overrides and profile-specific defaults without code changes.

**Provider Abstraction** — `ModelManager` and `EmbeddingManager` wrap provider clients behind thin abstractions, making it straightforward to swap OpenAI for a local inference server (Ollama) by changing `MODEL_PROVIDER`.

**Feature Flags** — hybrid retrieval, query expansion, context compression, safety checks, cost tracking, and streaming are all individually togglable flags, enabling gradual rollout and A/B testing.

**Pluggable Vector Store** — the `VECTOR_DB_TYPE` setting selects Chroma for persistent production use or FAISS for stateless testing, with no changes to the pipeline code.
