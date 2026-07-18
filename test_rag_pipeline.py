from __future__ import annotations

import argparse

from llm_agent import LLMAgent
from service.chroma_store import get_chroma_manager

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG smoke test: embed -> retrieve -> generate answer.")
    parser.add_argument(
        "--question",
        default="学校心理咨询中心一般能提供哪些帮助？",
        help="Question to ask.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name. Defaults to CHROMA_COLLECTION from environment.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only test retrieval, skip LLM generation.",
    )
    return parser.parse_args()


def build_context(results) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        source = item.source or "unknown"
        content = (item.content or "").strip()
        if len(content) > 800:
            content = content[:800]
        lines.append(f"[{index}] source={source} score={item.score:.4f}\n{content}")
    return "\n\n".join(lines).strip()


def main() -> int:
    args = parse_args()
    manager = get_chroma_manager()
    if args.collection:
        manager.get_collection(args.collection)

    query_embedding = manager.embed_texts([args.question])[0]
    results = manager.query(query_embedding=query_embedding, top_k=args.top_k)

    print("Question:", args.question)
    print("Retrieved:", len(results))
    for item in results:
        print(
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "score": item.score,
                "preview": (item.content or "").replace("\n", " ")[:120],
            }
        )

    context = build_context(results)
    if not context:
        print("No context retrieved.")
        return 0

    if args.no_llm:
        print("\nContext:\n")
        print(context)
        return 0

    try:
        agent = LLMAgent.from_env(agent_name="RAG")
    except Exception as exc:
        print("LLM is not configured:", str(exc))
        print("\nContext:\n")
        print(context)
        return 0

    messages = [
        {
            "role": "system",
            "content": "你是一个面向学生的陪伴与校园心理关怀助手。仅根据提供的知识片段回答，不要编造；如果知识片段不够就明确说明。",
        },
        {
            "role": "user",
            "content": f"问题：{args.question}\n\n可用知识片段：\n{context}\n\n请基于上述知识片段回答，并尽量简洁。",
        },
    ]

    answer = agent.chat(messages)
    print("\nAnswer:\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
