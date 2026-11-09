from vectorlens.engine import DocumentIndex, chunk_text, vectorize, similarity
from vectorlens.api import app
from fastapi.testclient import TestClient

def test_index_is_tenant_scoped(tmp_path):
    index=DocumentIndex(str(tmp_path/'docs.db'))
    assert index.ingest('acme','policy','Encryption is mandatory for database backups') == 1
    assert index.search('other','encryption backups') == []
    result=index.search('acme','database encryption')
    assert result[0]['doc_id']=='policy'
    assert result[0]['citation']=='policy#chunk-0'

def test_upsert_replaces_old_chunks(tmp_path):
    index=DocumentIndex(str(tmp_path/'docs.db'))
    index.ingest('a','doc','database backups green')
    index.ingest('a','doc','kubernetes scaling blue')
    assert index.search('a','green') == []
    assert index.search('a','kubernetes')[0]['text']=='kubernetes scaling blue'

def test_feature_hashing_is_deterministic():
    assert vectorize('hello hello')==vectorize('hello hello')
    assert similarity(vectorize('hello'),vectorize('hello'))>0.99

def test_api_validates_request():
    client=TestClient(app)
    assert client.get('/healthz').status_code==200
    assert client.post('/v1/search',json={'tenant':'a','query':'hi','top_k':21}).status_code==422
