from __future__ import annotations

import argparse
import json

from service.chroma_store import get_chroma_manager


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_list() -> int:
    manager = get_chroma_manager()
    _print_json({"collections": manager.list_collection_names()})
    return 0


def cmd_info(collection: str | None) -> int:
    manager = get_chroma_manager()
    _print_json(manager.get_collection_info(collection))
    return 0


def cmd_preview(collection: str | None, limit: int) -> int:
    manager = get_chroma_manager()
    _print_json(manager.preview_collection(collection, limit))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")

    p_info = sub.add_parser("info")
    p_info.add_argument("--collection", default=None)

    p_preview = sub.add_parser("preview")
    p_preview.add_argument("--collection", default=None)
    p_preview.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.command == "list":
        return cmd_list()
    if args.command == "info":
        return cmd_info(args.collection)
    if args.command == "preview":
        return cmd_preview(args.collection, args.limit)

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
