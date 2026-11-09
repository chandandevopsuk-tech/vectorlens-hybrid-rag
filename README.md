# VectorLens — Local Hybrid RAG Retrieval

A privacy-friendly **retrieval backend** for grounded AI applications. Index documents into SQLite FTS5 and deterministic hashed sparse vectors; score using a weighted retrieval blend and return chunk citations. Reproducible and usable without paid model API keys.

## Stack
Python 3.11+, FastAPI, SQLite FTS5, feature hashing, OpenAPI, pytest, Docker, GitHub Actions.

## Quick start
```bash
python -m pip install -e '.[dev]'
pytest
export VECTORLENS_DB=./vectorlens.db
uvicorn vectorlens.api:app --reload
```

Add document: `POST /v1/documents` with `{ "tenant":"team-a", "doc_id":"runbook-1", "text":"Restart the service using systemctl..." }`.
Retrieve: `POST /v1/search` with `{ "tenant":"team-a", "query":"restart service", "top_k":5 }`.
Docs: `http://127.0.0.1:8000/docs`.

## Design decisions & production hardening
- Documents are chunked at 900 characters with 100-character overlap; upserts replace older chunks.
- Returns **retrieval candidates** with citations, not LLM-generated answers. Feature hashing is a lexical projection, **not** an embedding model or guaranteed semantic understanding.
- SQLite is suited to a single-process demo. Multi-replica production should use PostgreSQL/pgvector or a dedicated search service, transaction isolation, and migrations.
- `tenant` is caller-supplied; it is **not an authentication boundary**. Bind it to verified identity in a real service. Add PII redaction, request authentication, content scanning, throttling and metrics.
