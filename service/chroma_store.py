from __future__ import annotations

import os
from typing import Any

import chromadb


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


_chroma_manager: ChromaManager | None = None


def get_chroma_manager() -> ChromaManager:
    global _chroma_manager

    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager


__all__ = ["ChromaManager", "get_chroma_manager"]
