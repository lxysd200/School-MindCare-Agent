from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from service.chroma_store import get_chroma_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync one knowledge document into MySQL and Chroma.")
    parser.add_argument(
        "--path",
        default="knowledge/risk-policy.md",
        help="Path to the knowledge document to test.",
    )
    parser.add_argument(
        "--collection",
        default="knowledge_chunks_test_1024",
        help="Chroma collection name used for this test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document_path = Path(args.path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    content = document_path.read_text(encoding="utf-8")
    source = document_path.name

    manager = get_chroma_manager()
    chunks = manager.split_text_into_chunks(content=content, source=source)

    print(f"Testing document: {document_path}")
    print(f"Chunk count: {len(chunks)}")
    if chunks:
        first_chunk = chunks[0]
        preview = first_chunk["content"][:120].replace("\n", " ")
        print(
            "First chunk preview:",
            {
                "source_index": first_chunk["source_index"],
                "char_start": first_chunk["char_start"],
                "char_end": first_chunk["char_end"],
                "chunk_size": first_chunk["chunk_size"],
                "content_hash": first_chunk["content_hash"],
                "preview": preview,
            },
        )

    print(f"Target collection: {args.collection}")
    results = manager.sync_document_chunks(
        source=source,
        content=content,
        collection_name=args.collection,
    )
    status_counter = Counter(result["status"] for result in results)

    print("Sync summary:", dict(status_counter))
    for result in results[:5]:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
