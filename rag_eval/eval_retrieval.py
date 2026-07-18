from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from service.chroma_store import get_chroma_manager
from service.db import get_db_session
from service.konwledge import KnowledgeService, SearchResult

load_dotenv()

EVAL_FILE = ROOT_DIR / "rag_eval" / "mindbridge-rag-eval.json"
OUTPUT_DIR = ROOT_DIR / "rag_eval" / "results"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _resolve_top_k() -> int:
    return int((os.getenv("TOP_K") or "5").strip())


def _resolve_candidate_k(top_k: int) -> int:
    candidate = int((os.getenv("CANDIDATE_KEY") or str(top_k)).strip())
    return max(top_k, candidate)


def _infer_category(case_id: str) -> str:
    parts = case_id.split("-")
    if len(parts) >= 2 and parts[0] == "risk":
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else "unknown"


def _infer_accept_sources(case: dict[str, Any]) -> list[str]:
    explicit = [item.strip() for item in case.get("expectedSources", []) if item]
    if explicit:
        base = list(dict.fromkeys(explicit))
    else:
        base = []

    question = _normalize_text(case.get("question", ""))
    case_id = case.get("id", "")

    def add(source: str) -> None:
        if source not in base:
            base.append(source)

    if "boundary" in case_id:
        add("privacy-boundaries-and-ethics.md")
        add("campus-mental-health.md")

    if case_id.startswith("support-"):
        add("campus-mental-health.md")
        if any(keyword in question for keyword in ["anxiety", "焦虑", "panic", "grounding", "breathing", "呼吸"]):
            add("anxiety-panic-grounding.md")
        if any(keyword in question for keyword in ["sleep", "失眠", "作息", "routine", "睡眠"]):
            add("sleep-routine-self-care.md")
        if any(keyword in question for keyword in ["low mood", "低落", "depression", "抑郁", "journaling", "孤独"]):
            add("low-mood-depression-support.md")
        if any(keyword in question for keyword in ["stress", "压力", "burnout", "崩溃"]):
            add("academic-stress-and-burnout.md")
        if any(keyword in question for keyword in ["counselor", "辅导员", "咨询", "心理中心", "trusted human", "求助"]):
            add("counselor-referral-and-resources.md")

    return base


def _run_strategy(
    service: KnowledgeService,
    *,
    question: str,
    strategy: str,
    top_k: int,
    candidate_k: int,
) -> list[SearchResult]:
    if strategy == "vector":
        return service._retrieve_vector(question, candidate_k)[:top_k]
    if strategy == "bm25":
        return service._retrieve_bm25(question, candidate_k)[:top_k]
    if strategy == "hybrid":
        return service._retrieve_hybrid(query=question, top_k=top_k, candidate_k=candidate_k)[:top_k]
    raise ValueError(f"Unsupported strategy: {strategy}")


def _score_case(case: dict[str, Any], results: list[SearchResult], top_k: int) -> dict[str, Any]:
    expected_sources = {item.strip() for item in case.get("expectedSources", []) if item}
    accept_sources = _infer_accept_sources(case)
    accept_source_set = set(accept_sources)
    expected_terms = [item.strip() for item in case.get("expectedTerms", []) if item]

    source_hit = 0.0
    mrr = 0.0
    for rank, result in enumerate(results[:top_k], start=1):
        if result.source in expected_sources:
            source_hit = 1.0
            mrr = 1.0 / rank
            break

    flexible_source_hit = 0.0
    flexible_mrr = 0.0
    for rank, result in enumerate(results[:top_k], start=1):
        if result.source in accept_source_set:
            flexible_source_hit = 1.0
            flexible_mrr = 1.0 / rank
            break

    matched_terms: list[str] = []
    term_first_rank = 0
    for rank, result in enumerate(results[:top_k], start=1):
        content = _normalize_text(result.content or "")
        local_hits = [term for term in expected_terms if _normalize_text(term) in content]
        if local_hits and term_first_rank == 0:
            term_first_rank = rank
        for term in local_hits:
            if term not in matched_terms:
                matched_terms.append(term)

    term_recall = len(matched_terms) / len(expected_terms) if expected_terms else 0.0
    term_mrr = (1.0 / term_first_rank) if term_first_rank else 0.0

    return {
        "source_hit": source_hit,
        "mrr": mrr,
        "flexible_source_hit": flexible_source_hit,
        "flexible_mrr": flexible_mrr,
        "term_recall": term_recall,
        "term_first_rank": term_first_rank,
        "term_mrr": term_mrr,
        "accept_sources": accept_sources,
        "matched_terms": matched_terms,
        "missing_terms": [term for term in expected_terms if term not in matched_terms],
        "retrieved_sources": [item.source for item in results[:top_k]],
        "retrieved_chunk_ids": [item.chunk_id for item in results[:top_k]],
        "results": [
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "score": item.score,
                "preview": (item.content or "").replace("\n", " ")[:200],
            }
            for item in results[:top_k]
        ],
    }


def _aggregate_case_scores(case_scores: list[dict[str, Any]]) -> dict[str, float]:
    total = len(case_scores)
    if total == 0:
        return {
            "count": 0,
            "source_hit_rate": 0.0,
            "mrr": 0.0,
            "flexible_source_hit_rate": 0.0,
            "flexible_mrr": 0.0,
            "term_recall": 0.0,
            "term_mrr": 0.0,
        }

    return {
        "count": total,
        "source_hit_rate": sum(item["source_hit"] for item in case_scores) / total,
        "mrr": sum(item["mrr"] for item in case_scores) / total,
        "flexible_source_hit_rate": sum(item["flexible_source_hit"] for item in case_scores) / total,
        "flexible_mrr": sum(item["flexible_mrr"] for item in case_scores) / total,
        "term_recall": sum(item["term_recall"] for item in case_scores) / total,
        "term_mrr": sum(item["term_mrr"] for item in case_scores) / total,
    }


def main() -> int:
    top_k = _resolve_top_k()
    candidate_k = _resolve_candidate_k(top_k)
    strategies = ["vector", "bm25", "hybrid"]

    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    manager = get_chroma_manager()
    db = get_db_session()
    try:
        service = KnowledgeService(db=db, manager=manager)

        report: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "eval_file": str(EVAL_FILE),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "strategies": {},
        }

        for strategy in strategies:
            case_details: list[dict[str, Any]] = []
            by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for case in cases:
                results = _run_strategy(
                    service,
                    question=case["question"],
                    strategy=strategy,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )
                metrics = _score_case(case, results, top_k)
                detail = {
                    "id": case["id"],
                    "category": _infer_category(case["id"]),
                    "question": case["question"],
                    "expected_sources": case.get("expectedSources", []),
                    "expected_terms": case.get("expectedTerms", []),
                    **metrics,
                }
                case_details.append(detail)
                by_category[detail["category"]].append(detail)

            report["strategies"][strategy] = {
                "summary": _aggregate_case_scores(case_details),
                "by_category": {
                    category: _aggregate_case_scores(items)
                    for category, items in sorted(by_category.items())
                },
                "cases": case_details,
            }
    finally:
        db.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"retrieval_eval_{timestamp}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Eval file: {EVAL_FILE}")
    print(f"TOP_K={top_k}, CANDIDATE_KEY={candidate_k}")
    print(f"Saved report: {output_path}")
    print()
    for strategy, payload in report["strategies"].items():
        summary = payload["summary"]
        print(
            f"[{strategy}] count={summary['count']} "
            f"source_hit_rate={summary['source_hit_rate']:.4f} "
            f"mrr={summary['mrr']:.4f} "
            f"flex_source_hit_rate={summary['flexible_source_hit_rate']:.4f} "
            f"flex_mrr={summary['flexible_mrr']:.4f} "
            f"term_recall={summary['term_recall']:.4f} "
            f"term_mrr={summary['term_mrr']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
