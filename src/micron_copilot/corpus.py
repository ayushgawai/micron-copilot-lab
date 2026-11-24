"""Deterministic synthetic operations corpus (~200 documents).

Fictional fabs, tools and lines only. Every fact is generated from a
fixed seed so retrieval, tests and benchmarks are reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    category: str
    site: str
    text: str

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title,
                "category": self.category, "site": self.site, "text": self.text}


SITES = ["Fab-A", "Fab-B", "Fab-C", "Fab-D"]
TOOLS = ["etcher", "deposition chamber", "lithography scanner",
         "inspection station", "wafer handler", "thermal processor"]
PROCESSES = ["etch", "deposition", "lithography", "inspection",
             "cleaning", "thermal anneal", "metrology", "packaging"]

_TEMPLATES: list[tuple[str, str, str]] = [
    ("process", "{proc} process control at {site}",
     "The {proc} step at {site} uses {tool} units with a target cycle time of "
     "{ct} hours and a yield target of {yld} percent. Operators review control "
     "charts every shift and escalate when defect density exceeds {dd} defects "
     "per wafer. Preventive maintenance for the {tool} runs every {pm} days and "
     "reduces unplanned downtime. Documented corrective action: recalibrate the "
     "{tool}, verify the {proc} recipe, and re-run qualification wafers before "
     "releasing the line."),
    ("quality", "{proc} defect reduction note at {site}",
     "Defect density for {proc} at {site} averaged {dd} defects per wafer last "
     "quarter against a limit of {lim} defects per wafer. The dominant defect "
     "mode was particle contamination near the {tool}. Containment action was "
     "to increase inspection sampling, clean the {tool}, and hold affected lots "
     "for review. Yield improved to {yld} percent after the corrective action."),
    ("equipment", "{tool} maintenance procedure at {site}",
     "The {tool} at {site} requires preventive maintenance every {pm} days. "
     "Mean time between failures is {mtbf} hours and mean repair time is "
     "{mttr} hours. Technicians lock out the {tool}, replace wear parts, run "
     "diagnostics, and record downtime in the maintenance log. Spare parts for "
     "the {tool} are stocked for {spares} days of cover."),
    ("supply", "Material supply note for {proc} at {site}",
     "Material supply for {proc} at {site} keeps {spares} days of safety stock. "
     "Lead time is {lead} days and the reorder point triggers a purchase review. "
     "If supply falls below the reorder point, planners expedite delivery, "
     "qualify an alternate lot, and notify production scheduling. Consumption "
     "is reviewed weekly against the {proc} production plan."),
    ("operations", "Shift operations briefing at {site}",
     "Shift operations at {site} prioritise safety checks, line balance, and "
     "on-time starts for {proc}. Cycle time averaged {ct} hours and on-time "
     "delivery was {yld} percent. Supervisors assign owners for each open "
     "action, review blockers at the daily standup, and escalate tool downtime "
     "over {mttr} hours to the area manager within one shift."),
    ("safety", "Safety procedure for {proc} at {site}",
     "Safety procedure for {proc} at {site}: wear required protective "
     "equipment, follow lockout steps for the {tool}, and report incidents "
     "within one shift. Audits occur every {pm} days. Any chemical spill, "
     "alarm, or injury stops work in the area until the supervisor clears "
     "the {proc} area and records the corrective action."),
]


def generate_corpus(n: int = 200, seed: int = 42) -> list[Document]:
    rng = random.Random(seed)
    docs: list[Document] = []
    for i in range(n):
        category, title_t, body_t = _TEMPLATES[i % len(_TEMPLATES)]
        site = SITES[rng.randrange(len(SITES))]
        proc = PROCESSES[rng.randrange(len(PROCESSES))]
        tool = TOOLS[rng.randrange(len(TOOLS))]
        vals = {
            "site": site, "proc": proc, "tool": tool,
            "ct": f"{rng.uniform(2, 18):.1f}", "yld": f"{rng.uniform(88, 99):.1f}",
            "dd": f"{rng.uniform(0.1, 2.5):.2f}", "lim": f"{rng.uniform(2.6, 4.0):.2f}",
            "pm": rng.randint(7, 90), "mtbf": rng.randint(120, 900),
            "mttr": rng.randint(1, 12), "spares": rng.randint(3, 30),
            "lead": rng.randint(2, 21),
        }
        docs.append(Document(
            doc_id=f"DOC-{i + 1:03d}",
            title=title_t.format(**vals).title(),
            category=category, site=site,
            text=body_t.format(**vals),
        ))
    return docs
