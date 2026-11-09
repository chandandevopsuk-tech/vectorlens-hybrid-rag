from __future__ import annotations
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]{2,}")
DIMENSIONS = 256

def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())

def vectorize(text: str) -> dict[int, float]:
    counts = Counter(tokens(text))
    buckets = Counter()
    for token, frequency in counts.items():
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % DIMENSIONS
        sign = 1 if digest[4] & 1 else -1
        buckets[bucket] += sign * (1 + math.log(frequency))
    mag = math.sqrt(sum(v*v for v in buckets.values())) or 1.0
    return {key: value / mag for key, value in buckets.items() if value}

def similarity(a: dict[int, float], b: dict[int, float]) -> float:
    return sum(value*b.get(key, 0) for key,value in a.items())

def chunk_text(text: str, size: int = 900, overlap: int = 100) -> list[str]:
    if size < 20 or overlap < 0 or overlap >= size:
        raise ValueError("invalid chunk size or overlap")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError("document cannot be empty")
    parts = []
    for start in range(0, len(cleaned), size-overlap):
        parts.append(cleaned[start:start+size])
        if start + size >= len(cleaned):
            break
    return parts

class DocumentIndex:
    def __init__(self, path: str = ':memory:'):
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, doc_id TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, features TEXT NOT NULL, UNIQUE(tenant,doc_id,ordinal))")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(text, content='chunks', content_rowid='id')")
        self.db.commit()

    def ingest(self, tenant: str, doc_id: str, text: str) -> int:
        parts = chunk_text(text)
        with self.db:
            old = self.db.execute('SELECT id FROM chunks WHERE tenant=? AND doc_id=?', (tenant,doc_id)).fetchall()
            for row in old:
                # FTS5 external-content delete commands require the old text.
                old_text = self.db.execute('SELECT text FROM chunks WHERE id=?',(row['id'],)).fetchone()[0]
                self.db.execute("INSERT INTO chunk_search(chunk_search,rowid,text) VALUES('delete',?,?)",(row['id'],old_text))
            self.db.execute('DELETE FROM chunks WHERE tenant=? AND doc_id=?',(tenant,doc_id))
            for i,chunk in enumerate(parts):
                cursor=self.db.execute('INSERT INTO chunks(tenant,doc_id,ordinal,text,features) VALUES (?,?,?,?,?)',(tenant,doc_id,i,chunk,json.dumps(vectorize(chunk))))
                self.db.execute('INSERT INTO chunk_search(rowid,text) VALUES (?,?)',(cursor.lastrowid,chunk))
        return len(parts)

    def search(self, tenant: str, query: str, k: int = 5) -> list[dict]:
        if not 1 <= k <= 20:
            raise ValueError('k must be between 1 and 20')
        terms = list(dict.fromkeys(tokens(query)))[:12]
        if not terms:
            return []
        phrase = ' OR '.join('"'+term+'"' for term in terms)
        lexical = self.db.execute('SELECT rowid,bm25(chunk_search) AS rank FROM chunk_search WHERE chunk_search MATCH ? LIMIT 100',(phrase,)).fetchall()
        lex_scores = {row['rowid']:1/(1+abs(row['rank'])) for row in lexical}
        qvec=vectorize(query)
        rows=self.db.execute('SELECT id,doc_id,ordinal,text,features FROM chunks WHERE tenant=?',(tenant,)).fetchall()
        scored=[]
        for row in rows:
            cos=max(0.0,similarity(qvec,{int(k):v for k,v in json.loads(row['features']).items()}))
            lex=lex_scores.get(row['id'],0.0)
            score=.65*cos+.35*lex
            if score>0:
                scored.append({'doc_id':row['doc_id'],'chunk':row['ordinal'],'score':round(score,4),'text':row['text'],'citation':f"{row['doc_id']}#chunk-{row['ordinal']}"})
        return sorted(scored,key=lambda p:(-p['score'],p['doc_id'],p['chunk']))[:k]
