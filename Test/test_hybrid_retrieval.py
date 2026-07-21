from __future__ import annotations

import argparse
import os

from service.chroma_store import get_chroma_manager
from service.db import get_db_session
from service.konwledge import KnowledgeService

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke test (Vector + BM25).")
    parser.add_argument(
        "--question",
        default="学校心理咨询中心一般能提供哪些帮助？",
        help="Question to retrieve.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name. Defaults to CHROMA_COLLECTION from environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = get_chroma_manager()
    if args.collection:
        manager.get_collection(args.collection)

    db = get_db_session()
    try:
        service = KnowledgeService(db=db, manager=manager)
        results = service.retrieve(args.question)
    finally:
        db.close()

    print("Question:", args.question)
    print("TOP_K(env):", os.getenv("TOP_K"))
    print("Retrieved:", len(results))
    for item in results:
        print(
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "score": item.score,
                "preview": (item.content or "").replace("\n", " ")[:160],
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
