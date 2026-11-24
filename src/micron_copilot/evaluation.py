"""Deterministic evaluation harness (no LLM judge, no randomness).

faithfulness      -- share of answer sentences supported by a cited source
citation_accuracy -- share of citations that exist AND support their sentence
action_usefulness -- share of an action checklist present in agent actions
All scores are in [0, 1] and computed with token-overlap heuristics.
"""
from __future__ import annotations

import re

from .agents import AgentResult
from .vector_store import VectorStore, tokenize

_STOP = set("the a an and or of to in for at on is are was were be with per".split())
_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[(DOC-\d+)\]")
_ACTION_SIGNALS = ["owner", "within", "review", "track", "validate", "pilot"]


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def _supported(sentence: str, source_text: str) -> bool:
    s, src = _content_tokens(sentence), _content_tokens(source_text)
    if not s:
        return True
    return len(s & src) / len(s) >= 0.35


def score_result(result: AgentResult, store: VectorStore) -> dict[str, float]:
    sentences = [s for s in _SENT.split(result.text.strip()) if s.strip()]
    factual = [s for s in sentences if not s.startswith(("#", "-", "1.", "2.", "3."))]
    if not factual:
        factual = sentences
    supported = cited_ok = citation_total = 0
    for sent in factual:
        cites = _CITE.findall(sent)
        docs = [store.get(c) for c in cites]
        if cites:
            citation_total += len(cites)
            cited_ok += sum(1 for d in docs if d is not None)
        if docs and any(d is not None and _supported(sent, f"{d.title} {d.text}")
                        for d in docs):
            supported += 1
        elif not cites and result.kind != "rag":
            supported += 1  # template scaffolding, not a factual claim
    faithfulness = supported / len(factual) if factual else 1.0
    if result.kind == "rag" and citation_total == 0:
        citation_accuracy = 0.0
    elif citation_total == 0:
        citation_accuracy = 1.0  # e.g. anomaly digest cites data, not docs
    else:
        # A citation counts only if it also supports its sentence.
        ok = 0
        for sent in factual:
            for c in _CITE.findall(sent):
                d = store.get(c)
                if d is not None and _supported(sent, f"{d.title} {d.text}"):
                    ok += 1
        citation_accuracy = ok / citation_total
    blob = " ".join(result.actions).lower()
    action_usefulness = (sum(1 for s in _ACTION_SIGNALS if s in blob)
                         / len(_ACTION_SIGNALS)) if result.actions else 0.0
    overall = round((faithfulness + citation_accuracy + action_usefulness) / 3, 4) \
        if result.actions else round((faithfulness + citation_accuracy) / 2, 4)
    return {"faithfulness": round(faithfulness, 4),
            "citation_accuracy": round(citation_accuracy, 4),
            "action_usefulness": round(action_usefulness, 4),
            "overall": overall}


BENCH_QUESTIONS = [
    "What is the preventive maintenance interval for equipment?",
    "How is defect density handled in quality reviews?",
    "What happens when material supply falls below the reorder point?",
    "What are the safety steps before working on equipment?",
    "How are shift operations and escalations handled?",
    "What is the target cycle time and yield for a process step?",
]


def run_evaluation(store: VectorStore, rag, business) -> dict:
    rows = []
    for q in BENCH_QUESTIONS:
        r = rag.answer(q)
        rows.append({"task": f"rag: {q[:38]}...", **score_result(r, store)})
    bc = business.draft("reduce defect density and unplanned downtime")
    rows.append({"task": "business-case: defect/downtime", **score_result(bc, store)})
    keys = ["faithfulness", "citation_accuracy", "action_usefulness", "overall"]
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    return {"rows": rows, "mean": mean, "n_tasks": len(rows)}
