"""Three agent flows: grounded RAG QA, business-case drafting, anomaly digest."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient, MockLLM
from .telemetry import Anomaly, TelemetryPoint, detect_anomalies
from .vector_store import VectorStore

_CITE = re.compile(r"\[(DOC-\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+")

ACTION_BY_METRIC = {
    "yield_pct": "Hold affected lots, review the last recipe change, and re-run qualification wafers",
    "defect_density": "Increase inspection sampling, clean the suspect tool, and inspect for particle contamination",
    "cycle_time_h": "Rebalance the line, check queue times at the bottleneck tool, and review staffing",
    "downtime_h": "Dispatch maintenance, check spare-parts cover, and review the preventive-maintenance schedule",
}


@dataclass
class AgentResult:
    kind: str
    text: str
    citations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text,
                "citations": self.citations, "sources": self.sources,
                "actions": self.actions, "meta": self.meta}


def _evidence_prompt(question: str, hits) -> str:
    lines = [f"Question: {question}", "EVIDENCE:"]
    for doc, score in hits:
        lines.append(f"[{doc.doc_id}] {doc.title}: {doc.text} [{doc.doc_id}]")
    lines.append("Answer using only the evidence and cite [DOC-nnn] per claim.")
    return "\n".join(lines)


class RAGAgent:
    """Grounded question answering: retrieve -> cite -> verify citations."""

    def __init__(self, store: VectorStore, llm: LLMClient | None = None):
        self.store = store
        self.llm = llm or MockLLM()

    def answer(self, question: str, k: int = 4) -> AgentResult:
        hits = self.store.search(question, k=k)
        if not hits:
            return AgentResult("rag", "No supporting evidence was retrieved "
                               "for this question.", meta={"grounded": False})
        resp = self.llm.complete(
            "You are a grounded operations assistant. Use only supplied "
            "evidence; cite every claim as [DOC-nnn]; never invent facts.",
            _evidence_prompt(question, hits))
        retrieved = [d.doc_id for d, _ in hits]
        citations = [c for c in dict.fromkeys(_CITE.findall(resp.text))
                     if c in retrieved]
        # Drop sentences whose citation was not retrieved (hallucination guard).
        kept = [s for s in _SENT.split(resp.text.strip())
                if not _CITE.search(s) or any(c in retrieved
                                              for c in _CITE.findall(s))]
        text = " ".join(kept).strip() or resp.text
        return AgentResult("rag", text, citations=citations, sources=retrieved,
                           meta={"grounded": bool(citations),
                                 "model": resp.model, "provider": resp.provider,
                                 "scores": [s for _, s in hits]})


class BusinessCaseAgent:
    """Drafts a structured, cited business case from retrieved evidence."""

    def __init__(self, store: VectorStore, llm: LLMClient | None = None):
        self.store = store
        self.llm = llm or MockLLM()

    def draft(self, initiative: str, k: int = 4) -> AgentResult:
        hits = self.store.search(initiative, k=k)
        resp = self.llm.complete(
            "Draft from evidence only; cite claims as [DOC-nnn].",
            _evidence_prompt(initiative, hits))
        retrieved = [d.doc_id for d, _ in hits]
        citations = [c for c in dict.fromkeys(_CITE.findall(resp.text))
                     if c in retrieved]
        evidence = resp.text if citations else "Limited retrieved evidence; validate assumptions before approval."
        actions = [
            "Owner: operations lead — validate the evidence and baseline metric within 5 business days",
            "Owner: finance partner — size cost, benefit and payback with stated assumptions within 10 business days",
            "Owner: area manager — run a 2-week pilot, track the target metric daily, and report variance",
            "Owner: programme lead — hold a go/no-go review with risks and mitigations at day 15",
        ]
        text = "\n".join([
            f"# Business case: {initiative}", "",
            "## Problem & evidence", evidence, "",
            "## Options",
            "1. Do nothing — no cost, problem and its operational risk persist.",
            "2. Pilot the change on one line — bounded cost, measurable result.",
            "3. Roll out everywhere — fastest benefit, highest execution risk.", "",
            "## Recommendation",
            "Option 2 (pilot). It tests the claimed benefit against the cited "
            "evidence before wider spend. All figures from the evidence above "
            "must be re-validated in the pilot; no external figures are assumed.", "",
            "## Risks & mitigations",
            "- Evidence is synthetic/thin -> pilot before committing budget.",
            "- Line disruption during pilot -> schedule in a planned maintenance window.", "",
            "## Next steps (owner + timeline)",
            *[f"- {a}" for a in actions],
        ])
        return AgentResult("business_case", text, citations=citations,
                           sources=retrieved, actions=actions,
                           meta={"grounded": bool(citations),
                                 "model": resp.model, "provider": resp.provider})


class AnomalyDigestAgent:
    """Summarises detected telemetry anomalies into actions with owners."""

    def digest(self, points: list[TelemetryPoint],
               threshold: float = 2.5) -> AgentResult:
        anomalies = detect_anomalies(points, threshold=threshold)
        if not anomalies:
            text = ("# Anomaly digest\n\nNo anomalies above the "
                    f"z>={threshold} threshold in {len(points)} points. "
                    "Continue routine monitoring.")
            return AgentResult("anomaly_digest", text,
                               meta={"anomaly_count": 0})
        lines = ["# Anomaly digest",
                 f"{len(anomalies)} anomalies (z>={threshold}) in "
                 f"{len(points)} telemetry points.", ""]
        actions: list[str] = []
        for a in anomalies:
            act = ACTION_BY_METRIC.get(a.metric, "Investigate the affected area")
            action = (f"Day {a.day} {a.site} {a.metric}: {act} "
                      f"(owner: area supervisor, within 1 shift)")
            actions.append(action)
            lines.append(f"- Day {a.day} | {a.site} | {a.metric} | "
                         f"value={a.value:.2f} | z={a.z_score:+.2f} ({a.direction})")
        lines += ["", "## Recommended actions",
                  *[f"- {a}" for a in actions]]
        return AgentResult("anomaly_digest", "\n".join(lines), actions=actions,
                           meta={"anomaly_count": len(anomalies),
                                 "anomalies": [a.to_dict() for a in anomalies]})
