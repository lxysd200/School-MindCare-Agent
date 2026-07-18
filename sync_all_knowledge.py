from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from service.chroma_store import get_chroma_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync all knowledge markdown files into MySQL and Chroma.")
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Directory containing knowledge markdown files.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Target Chroma collection. Defaults to CHROMA_COLLECTION from environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    knowledge_dir = Path(args.knowledge_dir)
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    manager = get_chroma_manager()
    all_results: list[dict] = []
    files = sorted(knowledge_dir.glob("*.md"))
    manager.reset_collection(args.collection)
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        results = manager.sync_document_chunks(
            source=file_path.name,
            content=content,
            collection_name=args.collection,
        )
        all_results.extend(results)

    summary = {
        "file_count": len(files),
        "chunk_count": len(all_results),
        "status_summary": dict(Counter(result["status"] for result in all_results)),
        "results": all_results,
    }

    print(f"Knowledge files synced: {summary['file_count']}")
    print(f"Total chunks processed: {summary['chunk_count']}")
    print("Status summary:", summary["status_summary"])

    for result in summary["results"][:10]:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
