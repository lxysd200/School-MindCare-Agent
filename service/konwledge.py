from __future__ import annotations

from service.chroma_store import get_chroma_manager


def init_knowledge_collection(collection_name: str | None = None):
    """Initialize or load the Chroma collection used by the knowledge base."""
    return get_chroma_manager().get_collection(collection_name)


__all__ = ["init_knowledge_collection"]
