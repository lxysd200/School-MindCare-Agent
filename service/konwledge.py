import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session
from service.chroma_store import ChromaManager



logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from Entities.entities import KnowledgeChunk

load_dotenv()

@dataclass
class SearchResult:
    chunk_id: int | None
    source: str
    content: str
    score: float


@dataclass
class RetrievalCandidate:
    result: SearchResult
    vector_score: float = 0.0
    bm25_score: float = 0.0



class KnowledgeService:
    def __init__(self,db: Session,manager: ChromaManager):
        self.db = db
        self.manager = manager
        self._bm25_index: _BM25Index | None = None
        
    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        env_top_k = (os.getenv("TOP_K") or "").strip()
        resolved_top_k = int(env_top_k) if env_top_k else (top_k or 5)
        candidate_key = max(int((os.getenv("CANDIDATE_KEY") or resolved_top_k)), resolved_top_k)
        use_bm25 = (os.getenv("ENABLE_BM25") or "1").strip() not in {"0", "false", "False"}
        if not use_bm25:
            vector_results = self._retrieve_vector(query, candidate_key)
            return vector_results[:resolved_top_k]

        return self._retrieve_hybrid(query=query, top_k=resolved_top_k, candidate_k=candidate_key)

    def _retrieve_hybrid(self, *, query: str, top_k: int, candidate_k: int) -> list[SearchResult]:
        gate_threshold = float((os.getenv("HYBRID_GATE_THRESHOLD") or "0.75").strip())
        vector_results = self._retrieve_vector(query, candidate_k)
        if not vector_results:
            bm25_results = self._retrieve_bm25(query, candidate_k)
            return bm25_results[:top_k]

        if vector_results[0].score >= gate_threshold:
            return vector_results[:top_k]

        bm25_results = self._retrieve_bm25(query, candidate_k)
        merged = self._augment_results(vector_results=vector_results, bm25_results=bm25_results, limit=candidate_k)
        return merged[:top_k]

    def _augment_results(
        self,
        *,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        def key_of(item: SearchResult) -> object:
            if item.chunk_id is not None:
                return item.chunk_id
            return (item.source, (item.content or "")[:64])

        seen: set[object] = set()
        merged: list[SearchResult] = []

        for item in vector_results:
            key = key_of(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged

        for item in bm25_results:
            key = key_of(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged

        return merged

    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        try:
            query_embedding = self.manager.embed_texts([query])[0]
            hits = self.manager.query(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Error retrieving vector results: {e}")
            return []
        results = []
        for hit in hits:
            chunk = self.db.get(KnowledgeChunk, hit.chunk_id) if hit.chunk_id is not None else None
            results.append(
                SearchResult(
                    chunk.id if chunk is not None else hit.chunk_id,
                    chunk.source if chunk is not None else hit.source,
                    chunk.content if chunk is not None else hit.content,
                    hit.score,
                )
            )
        return results

    def _merge_results(
        self,
        *,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:
        alpha = float((os.getenv("HYBRID_ALPHA") or "0.7").strip())
        alpha = min(1.0, max(0.0, alpha))

        by_key: dict[int | None, RetrievalCandidate] = {}

        for item in vector_results:
            key = item.chunk_id
            candidate = by_key.get(key)
            if candidate is None:
                candidate = RetrievalCandidate(result=item, vector_score=item.score, bm25_score=0.0)
                by_key[key] = candidate
            else:
                candidate.vector_score = max(candidate.vector_score, item.score)

        for item in bm25_results:
            key = item.chunk_id
            candidate = by_key.get(key)
            if candidate is None:
                candidate = RetrievalCandidate(result=item, vector_score=0.0, bm25_score=item.score)
                by_key[key] = candidate
            else:
                candidate.bm25_score = max(candidate.bm25_score, item.score)

        max_bm25 = max((c.bm25_score for c in by_key.values()), default=0.0)
        for candidate in by_key.values():
            bm25_norm = candidate.bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            combined = alpha * candidate.vector_score + (1.0 - alpha) * bm25_norm
            candidate.result.score = combined

        merged = sorted((c.result for c in by_key.values()), key=lambda r: r.score, reverse=True)
        return merged

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens: list[str] = []

        for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
            part = match.group(0)
            if re.fullmatch(r"[a-z0-9]+", part):
                if len(part) >= 2:
                    tokens.append(part)
                continue

            if len(part) >= 2:
                for i in range(len(part) - 1):
                    tokens.append(part[i : i + 2])
            else:
                tokens.append(part)

        return tokens

    def _get_bm25_index(self) -> "_BM25Index | None":
        rebuild = (os.getenv("BM25_REBUILD_EVERY_QUERY") or "0").strip() in {"1", "true", "True"}
        if self._bm25_index is not None and not rebuild:
            return self._bm25_index

        rows = self.db.execute(select(KnowledgeChunk)).scalars().all()
        if not rows:
            self._bm25_index = None
            return None

        docs_tf: list[Counter[str]] = []
        doc_lens: list[int] = []
        df: dict[str, int] = defaultdict(int)

        for chunk in rows:
            tokens = self._tokenize(chunk.content or "")
            if not tokens:
                docs_tf.append(Counter())
                doc_lens.append(0)
                continue

            tf = Counter(tokens)
            docs_tf.append(tf)
            doc_lens.append(len(tokens))
            for term in tf.keys():
                df[term] += 1

        N = len(rows)
        avgdl = (sum(doc_lens) / N) if N else 0.0
        if avgdl <= 0:
            self._bm25_index = None
            return None

        self._bm25_index = _BM25Index(
            chunks=rows,
            docs_tf=docs_tf,
            doc_lens=doc_lens,
            df=df,
            N=N,
            avgdl=avgdl,
        )
        return self._bm25_index

    def _retrieve_bm25(self, query: str, top_k: int) -> list[SearchResult]:
        index = self._get_bm25_index()
        if index is None:
            return []

        query_tokens = self._tokenize(query)
        q_terms = [t for t in query_tokens if t in index.df]
        if not q_terms:
            return []

        k1 = float((os.getenv("BM25_K1") or "1.2").strip())
        b = float((os.getenv("BM25_B") or "0.75").strip())

        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(index.docs_tf):
            dl = index.doc_lens[idx] or 0
            if dl <= 0:
                continue
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                term_df = index.df.get(term, 0)
                idf = math.log((index.N - term_df + 0.5) / (term_df + 0.5) + 1.0)
                denom = f + k1 * (1.0 - b + b * (dl / index.avgdl))
                score += idf * (f * (k1 + 1.0)) / denom
            if score > 0:
                scores.append((idx, score))

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        max_score = top[0][1] if top else 1.0
        max_score = max(max_score, 1e-9)

        results: list[SearchResult] = []
        for idx, score in top:
            chunk = index.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    source=chunk.source,
                    content=chunk.content,
                    score=score / max_score,
                )
            )

        return results
        

@dataclass
class _BM25Index:
    chunks: list[KnowledgeChunk]
    docs_tf: list[Counter[str]]
    doc_lens: list[int]
    df: dict[str, int]
    N: int
    avgdl: float
