# RAG Agent System

A production-oriented RAG agent scaffold with:

- Semantic + sliding-window chunking (500-1200 token chunks)
- Multi-format parsing (`txt`, `md`, `json`, `pdf`)
- Dual-route retrieval (metadata + semantic) with fusion (normalize + rerank + RRF)
- LangChain tool-calling agent with a Pydantic tool registry
- FastAPI interface with trace and source visibility
- Observability for cost, latency, and groundedness

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
mypy src
uvicorn rag_agent.api.main:app --reload
```

### API behavior

- If `OPENAI_API_KEY` is set: uses LangChain tool-calling agent.
- If `OPENAI_API_KEY` is missing: falls back to deterministic planner (still fully functional for ingest/query/trace).
- Useful endpoints:
  - `POST /ingest`
  - `POST /query`
  - `POST /sources/search`
  - `GET /traces`, `GET /traces/{trace_id}`
  - `GET /metrics`

## Performance goals encoded in code

- Retrieval strategy oversamples candidates and fuses to optimize `Recall@5`
- System prompt enforces citation-first answers to maintain groundedness >95%
- Agent run loop tracks latency and targets end-to-end <8s under normal load

## Interview prep

- See `INTERVIEW_PREP.md` for architecture talk track, tradeoffs, and common Q&A.
