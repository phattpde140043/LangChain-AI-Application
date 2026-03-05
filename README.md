# LangChain AI Research Assistant

Production-grade AI research assistant built with LangChain, FastAPI, and RAG. Supports intelligent document search, summarization, comparison, and question answering.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangChain AI Research Assistant               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Client ──► FastAPI ──► APIKeyAuth ──► RateLimiter              │
│                │                                                   │
│                ▼                                                   │
│         CoordinatorAgent                                          │
│        /       |       \      \                                   │
│  Research  Summarize  Compare  Analyze                           │
│    Agent     Agent     Agent    Agent                            │
│       \        |        |       /                                 │
│        ─────────────────────────                                  │
│                    │                                               │
│              RAG Pipeline                                         │
│         ┌────────────────┐                                        │
│         │ QueryExpander  │                                        │
│         │ HybridRetriever│ ◄── Vector DB (Chroma/FAISS)         │
│         │ ContextManager │     + BM25 Index                      │
│         │ LLM Generation │ ◄── OpenAI / Local LLM               │
│         └────────────────┘                                        │
│                                                                   │
│  Documents ──► Loader ──► Chunker ──► Embeddings ──► Vector DB  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Multi-Agent Architecture**: Coordinator routes to specialized agents (Research, Summarization, Analysis, Comparison)
- **Hybrid RAG**: Combines vector similarity search + BM25 keyword search
- **Query Expansion**: LLM-powered alternative query generation for better recall
- **Context Management**: Smart token budget management with deduplication and ranking
- **Security**: Prompt injection detection, guardrails, API key authentication
- **Monitoring**: Structured logging with trace IDs, cost tracking, performance metrics
- **Multi-Profile Config**: Development, Testing, Production profiles
- **Evaluation Framework**: Automated evaluation with configurable datasets
- **Document Support**: PDF, Markdown, plain text ingestion

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| LLM Framework | LangChain | ≥0.1.0 |
| REST API | FastAPI | ≥0.104.0 |
| Vector Store | ChromaDB / FAISS | ≥0.4.0 / ≥1.7.4 |
| Default LLM | OpenAI GPT-4o-mini | via openai ≥1.0.0 |
| Keyword Search | BM25 (rank-bm25) | ≥0.2.2 |
| Data Validation | Pydantic v2 | ≥2.0.0 |
| ASGI Server | Uvicorn | ≥0.24.0 |
| Embeddings | sentence-transformers | ≥2.2.0 |

## Project Structure

```
LangChain-AI-Application/
├── documents/                     # Sample documents for ingestion
│   ├── sample_ai_architecture.md
│   ├── sample_microservices.md
│   └── sample_system_design.md
├── prompts/                       # Versioned LLM prompt templates
│   ├── rag_prompt_v1.txt
│   ├── rag_prompt_v2.txt
│   ├── summarization_prompt.txt
│   ├── comparison_prompt.txt
│   └── concept_extraction_prompt.txt
├── scripts/
│   ├── ingest_documents.py        # CLI: ingest documents into vector store
│   ├── run_api.py                 # CLI: start the FastAPI server
│   ├── benchmarking/
│   │   └── run_benchmark.py
│   └── dataset_generation/
│       └── generate_eval_dataset.py
├── src/
│   ├── agents/                    # Multi-agent system
│   │   ├── base_agent.py
│   │   ├── coordinator.py         # Routes queries to specialized agents
│   │   ├── research_agent.py
│   │   ├── summarization_agent.py
│   │   ├── comparison_agent.py
│   │   └── analysis_agent.py
│   ├── api/                       # FastAPI application
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── middleware/
│   │   │   ├── auth.py            # API key authentication
│   │   │   └── rate_limit.py
│   │   └── routes/
│   │       ├── ask.py
│   │       ├── summarize.py
│   │       ├── compare.py
│   │       ├── ingest.py
│   │       ├── feedback.py
│   │       └── knowledge_base.py
│   ├── config/
│   │   └── settings.py            # Pydantic Settings with profile support
│   ├── embeddings/
│   │   └── embedding_manager.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── experiment_tracker.py
│   ├── ingestion/                 # Document loading and chunking
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── memory/
│   │   └── conversation_memory.py
│   ├── models/
│   │   └── model_manager.py
│   ├── monitoring/
│   │   ├── logger.py              # Structured logging with trace IDs
│   │   ├── metrics.py
│   │   └── cost_tracker.py
│   ├── rag/                       # RAG pipeline components
│   │   ├── pipeline.py
│   │   ├── retriever.py           # Hybrid (vector + BM25) retrieval
│   │   ├── query_expansion.py
│   │   └── context_manager.py
│   ├── security/
│   │   ├── injection_detector.py  # Prompt injection detection
│   │   └── guardrails.py
│   ├── tools/                     # LangChain tools for agents
│   │   ├── base_tool.py
│   │   ├── document_search.py
│   │   ├── summarization.py
│   │   ├── comparison.py
│   │   ├── citation_finder.py
│   │   └── concept_extraction.py
│   ├── utils/
│   │   └── helpers.py
│   └── vector_store/
│       └── store.py               # Chroma / FAISS abstraction
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_config.py
│   │   ├── test_context_manager.py
│   │   └── test_security.py
│   └── integration/
│       └── test_api.py
├── data/                          # Runtime data (vector store, logs)
├── docs/
│   ├── architecture.md
│   └── system_design.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- OpenAI API key (optional — runs with mock responses without one)

### Installation

```bash
git clone <repo>
cd LangChain-AI-Application
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Ingest Documents

```bash
python scripts/ingest_documents.py
# Or specify a directory:
python scripts/ingest_documents.py --dir ./documents
```

### Run the API

```bash
python scripts/run_api.py
# Or directly:
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key principles of microservices architecture?"}'
```

### Summarize a Document

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"document_name": "sample_microservices", "max_length": 300}'
```

### Compare Documents

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"document1": "sample_microservices", "document2": "sample_system_design"}'
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "response": "RAG is...", "rating": "helpful"}'
```

### Knowledge Base Stats

```bash
curl http://localhost:8000/knowledge-base
```

## Configuration Reference

All settings are read from environment variables or a `.env` file.

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_DB_TYPE` | `chroma` | Vector DB: `chroma` or `faiss` |
| `VECTOR_DB_PATH` | `./data/vector_store` | Path to persist vector store |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `MAX_CONTEXT_TOKENS` | `4000` | Token budget for LLM context |
| `API_KEY` | `""` | API key auth (empty = disabled) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per minute |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ACTIVE_PROFILE` | `development` | Config profile |
| `ACTIVE_RAG_PROMPT` | `rag_prompt_v1` | Prompt template name |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | Enable BM25 + vector search |
| `ENABLE_QUERY_EXPANSION` | `true` | Enable LLM query expansion |
| `ENABLE_CONTEXT_COMPRESSION` | `false` | Enable LLM context compression |
| `ENABLE_SAFETY_CHECKS` | `true` | Enable guardrails |
| `ENABLE_COST_TRACKING` | `true` | Track OpenAI API costs |
| `ENABLE_STREAMING` | `true` | Enable streaming responses |
| `REDIS_URL` | `""` | Redis URL for caching |

### Profiles

| Profile | Changes from defaults |
|---------|-----------------------|
| `development` | `LOG_LEVEL=DEBUG`, `ENABLE_COST_TRACKING=false` |
| `testing` | `ENABLE_SAFETY_CHECKS=false`, `ENABLE_COST_TRACKING=false`, `VECTOR_DB_TYPE=faiss` |
| `production` | `LOG_LEVEL=INFO`, `ENABLE_SAFETY_CHECKS=true`, `ENABLE_COST_TRACKING=true` |

## Testing

```bash
pytest tests/ -v               # All tests
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests only
```

## Docker

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
# Redis at localhost:6379
```

## Security Features

- **API key authentication** via `X-API-Key` header (set `API_KEY` env var to enable)
- **Prompt injection detection** with configurable pattern matching
- **Input/output guardrails**: length limits, empty-query blocking, injection blocking
- **Rate limiting**: configurable per-minute request cap via `slowapi`
- **No sensitive data in logs**: API keys and secrets are never logged

## Future Improvements

- Streaming responses with LangChain streaming
- Fine-tuning pipeline integration
- Multi-modal document support (images, tables)
- Web search tool integration
- LangSmith tracing integration
- GraphRAG for relationship-aware retrieval
