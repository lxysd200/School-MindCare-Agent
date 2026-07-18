from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any

import chromadb
from sqlalchemy import select

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


load_dotenv()

@dataclass
class VectorSearchHit:
    chunk_id: int | None
    source: str
    source_index: int
    content: str
    score: float

class ChromaManager:
    def __init__(
        self,
        path: str | None = None,
        default_collection: str | None = None,
    ) -> None:
        self._path = (path or os.getenv("CHROMA_PATH") or "./data/chroma").strip()
        self._default_collection = (
            default_collection or os.getenv("CHROMA_COLLECTION") or "knowledge_chunks"
        ).strip()
        self._embedding_model = (
            os.getenv("OPENAI_EMBEDDING_MODEL")
            or os.getenv("CHROMA_EMBEDDING_MODEL")
            or ""
        ).strip()
        self._embedding_dimensions = int((os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or "1024").strip())
        self._chunk_size = int((os.getenv("CHUNK_SIZE") or "512").strip())
        self._chunk_overlap = int((os.getenv("CHUNK_OVERLAP") or "60").strip())
        self._client: chromadb.PersistentClient = chromadb.PersistentClient(path=self._path)
        self.collection = self._create_collection(self._default_collection)

    def get_client(self) -> chromadb.PersistentClient:
        return self._client

    def get_collection(self, name: str | None = None):
        collection_name = (name or self._default_collection).strip()
        if not collection_name:
            raise ValueError("Chroma collection name cannot be empty.")

        if collection_name == self._default_collection:
            return self.collection
        return self._create_collection(collection_name)

    def reset_collection(self, name: str | None = None):
        collection_name = (name or self._default_collection).strip()
        if not collection_name:
            raise ValueError("Chroma collection name cannot be empty.")

        client = self.get_client()
        existing_names = self.list_collection_names()
        if collection_name in existing_names:
            client.delete_collection(name=collection_name)

        collection = self._create_collection(collection_name)
        if collection_name == self._default_collection:
            self.collection = collection
        return collection

    def _create_collection(self, name: str):
        metadata: dict[str, str] = {"hnsw:space": "cosine"}
        if self._embedding_model:
            metadata["embedding_model"] = self._embedding_model

        return self.get_client().get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata=metadata,
        )

    def list_collection_names(self) -> list[str]:
        return [collection.name for collection in self.get_client().list_collections()]

    def get_collection_info(self, name: str | None = None) -> dict[str, Any]:
        collection = self.get_collection(name)
        return {
            "collection": collection.name,
            "count": collection.count(),
        }

    def preview_collection(self, name: str | None = None, limit: int = 5) -> dict[str, Any]:
        collection = self.get_collection(name)
        return {
            "collection": collection.name,
            "count": collection.count(),
            "preview": collection.get(limit=limit, include=["documents", "metadatas"]),
        }

    def normalize_content(self, content: str) -> str:
        return "\n".join(line.rstrip() for line in content.strip().splitlines())

    def make_content_hash(self, content: str) -> str:
        normalized = self.normalize_content(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def build_chunk_id(self, db_id: int) -> str:
        return f"knowledge-chunk-{db_id}"

    def _find_chunk_end(self, text: str, start: int, size: int) -> int:
        text_length = len(text)
        max_end = min(start + size, text_length)
        if max_end >= text_length:
            return text_length

        punctuation = "。！？；.!?;\n"
        boundary = -1
        for index in range(max_end - 1, start, -1):
            if text[index] in punctuation:
                boundary = index + 1
                break

        if boundary == -1:
            for index in range(max_end - 1, start, -1):
                if text[index].isspace() or text[index] in "，,、：:":
                    boundary = index + 1
                    break

        if boundary == -1 or boundary <= start:
            return max_end
        return boundary

    def split_text_into_chunks(
        self,
        content: str,
        *,
        source: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized = self.normalize_content(content)
        if not normalized:
            return []

        size = chunk_size or self._chunk_size
        overlap = self._chunk_overlap if chunk_overlap is None else chunk_overlap
        if size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if overlap >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        chunks: list[dict[str, Any]] = []
        text_length = len(normalized)
        start = 0
        index = 0

        while start < text_length:
            end = self._find_chunk_end(normalized, start, size)
            chunk_content = normalized[start:end].strip()
            if chunk_content:
                chunks.append(
                    {
                        "source": source,
                        "source_index": index,
                        "content": chunk_content,
                        "content_hash": self.make_content_hash(chunk_content),
                        "char_start": start,
                        "char_end": end,
                        "chunk_size": len(chunk_content),
                    }
                )
                index += 1

            if end >= text_length:
                break

            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _get_embedding_client(self):
        from openai import OpenAI

        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if not api_key:
            raise ValueError("Missing embedding API key. Set DASHSCOPE_API_KEY or OPENAI_API_KEY.")
        if not base_url:
            raise ValueError("Missing embedding base URL. Set OPENAI_BASE_URL.")
        if not self._embedding_model:
            raise ValueError("Missing embedding model. Set OPENAI_EMBEDDING_MODEL or CHROMA_EMBEDDING_MODEL.")

        return OpenAI(api_key=api_key, base_url=base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_embedding_client()
        response = client.embeddings.create(
            model=self._embedding_model,
            input=texts,
            encoding_format="float",
            dimensions=self._embedding_dimensions,
        )
        return [item.embedding for item in response.data]

    def sync_knowledge_chunk(
        self,
        *,
        source: str,
        source_index: int,
        content: str,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        from Entities.entities import KnowledgeChunk
        from service.db import get_db_session

        normalized_content = self.normalize_content(content)
        content_hash = self.make_content_hash(normalized_content)
        db = get_db_session()

        try:
            existing = db.execute(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.source == source,
                    KnowledgeChunk.source_index == source_index,
                )
            ).scalar_one_or_none()

            if existing is not None and existing.content_hash == content_hash:
                return {
                    "status": "skipped",
                    "db_id": existing.id,
                    "chunk_id": self.build_chunk_id(existing.id),
                    "source": source,
                    "source_index": source_index,
                    "content_hash": content_hash,
                }

            embedding = self.embed_texts([normalized_content])[0]
            embedding_json = json.dumps(embedding, ensure_ascii=False)

            if existing is None:
                record = KnowledgeChunk(
                    source=source,
                    source_index=source_index,
                    content=normalized_content,
                    embedding_json=embedding_json,
                    content_hash=content_hash,
                )
                db.add(record)
                status = "created"
            else:
                existing.content = normalized_content
                existing.embedding_json = embedding_json
                existing.content_hash = content_hash
                record = existing
                status = "updated"

            db.flush()

            metadata = {
                "db_id": record.id,
                "source": source,
                "source_index": source_index,
                "content_hash": content_hash,
            }
            chunk_id = self.build_chunk_id(record.id)
            collection = self.get_collection(collection_name)
            collection.upsert(
                ids=[chunk_id],
                documents=[normalized_content],
                metadatas=[metadata],
                embeddings=[embedding],
            )

            db.commit()
            return {
                "status": status,
                "db_id": record.id,
                "chunk_id": chunk_id,
                "source": source,
                "source_index": source_index,
                "content_hash": content_hash,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def sync_document_chunks(
        self,
        *,
        source: str,
        content: str,
        collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        chunks = self.split_text_into_chunks(
            content,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        results: list[dict[str, Any]] = []
        for chunk in chunks:
            result = self.sync_knowledge_chunk(
                source=source,
                source_index=chunk["source_index"],
                content=chunk["content"],
                collection_name=collection_name,
            )
            result["char_start"] = chunk["char_start"]
            result["char_end"] = chunk["char_end"]
            result["chunk_size"] = chunk["chunk_size"]
            results.append(result)

        return results

    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        collection = self.get_collection()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits


_chroma_manager: ChromaManager | None = None


def get_chroma_manager() -> ChromaManager:
    global _chroma_manager

    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager


__all__ = ["ChromaManager", "get_chroma_manager"]
