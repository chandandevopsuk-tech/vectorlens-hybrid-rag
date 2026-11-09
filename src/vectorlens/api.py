import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .engine import DocumentIndex

app = FastAPI(title="VectorLens Hybrid RAG Retrieval", version="0.1.0")

@lru_cache

def index():
    return DocumentIndex(os.environ.get('VECTORLENS_DB', '/tmp/vectorlens.db'))

class Document(BaseModel):
    tenant: str = Field(min_length=1, max_length=80)
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500000)

class Search(BaseModel):
    tenant: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

@app.get('/healthz')
def health():
    return {'status':'ok'}

@app.post('/v1/documents')
def upsert_document(data: Document):
    return {'doc_id':data.doc_id,'chunks':index().ingest(data.tenant,data.doc_id,data.text)}

@app.post('/v1/search')
def search(data: Search):
    return {'results':index().search(data.tenant,data.query,data.top_k)}
