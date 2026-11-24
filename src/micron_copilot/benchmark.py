"""Synthetic benchmark: retrieval + all three agents + evaluation, timed."""
from __future__ import annotations

import time

from .agents import AnomalyDigestAgent, BusinessCaseAgent, RAGAgent
from .corpus import generate_corpus
from .evaluation import BENCH_QUESTIONS, run_evaluation, score_result
from .llm import get_llm
from .telemetry import generate_telemetry
from .vector_store import VectorStore


def run_benchmark(provider: str = "mock") -> dict:
    t0 = time.perf_counter()
    docs = generate_corpus(200)
    store = VectorStore(docs)
    index_s = time.perf_counter() - t0
    llm = get_llm(provider)
    rag, biz = RAGAgent(store, llm), BusinessCaseAgent(store, llm)

    t = time.perf_counter()
    eval_out = run_evaluation(store, rag, biz)
    eval_s = time.perf_counter() - t

    t = time.perf_counter()
    digest = AnomalyDigestAgent().digest(generate_telemetry())
    tel = score_result(digest, store)
    digest_s = time.perf_counter() - t

    # Retrieval sanity: fraction of bench questions returning >=1 hit.
    hit_rate = sum(1 for q in BENCH_QUESTIONS if store.search(q)) / len(BENCH_QUESTIONS)
    total = time.perf_counter() - t0
    return {"documents": len(docs), "index_seconds": round(index_s, 3),
            "provider": llm.provider, "model": llm.model,
            "retrieval_hit_rate": round(hit_rate, 4),
            "evaluation": eval_out, "anomaly_digest": {
                "anomaly_count": digest.meta["anomaly_count"],
                "action_usefulness": tel["action_usefulness"],
                "seconds": round(digest_s, 3)},
            "eval_seconds": round(eval_s, 3), "total_seconds": round(total, 3)}


def format_benchmark(b: dict) -> str:
    e = b["evaluation"]["mean"]
    lines = [
        "micron-copilot-lab synthetic benchmark (offline mock LLM)",
        f"documents={b['documents']} provider={b['provider']} model={b['model']}",
        f"index_seconds={b['index_seconds']} retrieval_hit_rate={b['retrieval_hit_rate']}",
        f"eval_tasks={b['evaluation']['n_tasks']} faithfulness={e['faithfulness']} "
        f"citation_accuracy={e['citation_accuracy']} "
        f"action_usefulness={e['action_usefulness']} overall={e['overall']}",
        f"anomalies_detected={b['anomaly_digest']['anomaly_count']} "
        f"digest_action_usefulness={b['anomaly_digest']['action_usefulness']}",
        f"eval_seconds={b['eval_seconds']} total_seconds={b['total_seconds']}",
    ]
    return "\n".join(lines)
