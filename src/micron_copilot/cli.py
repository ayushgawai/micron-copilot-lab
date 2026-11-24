"""CLI: ask | business-case | digest | eval | bench"""
from __future__ import annotations

import argparse
import json
import sys

from .agents import AnomalyDigestAgent, BusinessCaseAgent, RAGAgent
from .benchmark import format_benchmark, run_benchmark
from .corpus import generate_corpus
from .evaluation import run_evaluation, score_result
from .llm import get_llm
from .telemetry import generate_telemetry
from .vector_store import VectorStore


def _stack(provider: str):
    store = VectorStore(generate_corpus(200))
    llm = get_llm(provider)
    return store, RAGAgent(store, llm), BusinessCaseAgent(store, llm)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="micron-copilot",
                                description="Enterprise Copilot PoC (synthetic data, offline by default)")
    p.add_argument("--provider", default="mock",
                   choices=["mock", "openai", "anthropic", "azure"],
                   help="LLM provider (default: offline mock)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("business-case").add_argument("initiative")
    sub.add_parser("digest")
    sub.add_parser("eval")
    sub.add_parser("bench")
    args = p.parse_args(argv)

    if args.cmd == "bench":
        print(format_benchmark(run_benchmark(args.provider)))
        return 0
    store, rag, biz = _stack(args.provider)
    if args.cmd == "ask":
        r = rag.answer(args.question)
        print(r.text)
        print(f"\nCitations: {', '.join(r.citations) or 'none'}")
        print(f"Scores: {json.dumps(score_result(r, store))}")
    elif args.cmd == "business-case":
        r = biz.draft(args.initiative)
        print(r.text)
        print(f"\nCitations: {', '.join(r.citations) or 'none'}")
    elif args.cmd == "digest":
        r = AnomalyDigestAgent().digest(generate_telemetry())
        print(r.text)
    elif args.cmd == "eval":
        print(json.dumps(run_evaluation(store, rag, biz), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
