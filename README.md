# micron-copilot-lab

Enterprise Copilot / digital-assistant proof of concept: a **grounded RAG
answer assistant**, an **agentic business-case generator**, and a **telemetry
anomaly digest** over a synthetic semiconductor-style operations corpus
(~200 documents, fictional fabs/tools only — no real operational data).

Everything runs **offline by default** with a deterministic mock LLM, zero
runtime dependencies, and a deterministic evaluation harness. Hosted LLM
providers are pluggable adapters, never required.

> Status: working PoC. All three agent flows, the evaluation harness and
> the benchmark below run end-to-end offline. Hosted-provider adapters are
> thin and untested in CI (no keys in CI); see *Limitations*.

## Design

```
                        +----------------------+
   user / CLI --------> |  Agents              |
                        |  - RAG QA            |
                        |  - Business-case     |
                        |  - Anomaly digest    |
                        +---+-----------+------+
                            |           |
              grounded      |           |  telemetry points
              prompt +      v           v
              evidence  +--------+  +-----------+
                        | LLM    |  | Anomaly   |
                        | client |  | detector  |
                        +--------+  | (z-score) |
                            ^       +-----------+
                            |           |
   +-------------------+    |           v
   | Vector store      |----+     digest + owner /
   | TF-IDF + cosine   |          timeline actions
   | 200 synthetic docs|
   +-------------------+
                            |
                        +---v-----------+
                        | Evaluation    |
                        | faithfulness  |
                        | citation acc. |
                        | action useful.|
                        +---------------+
```

| Component | Implementation | Why |
|---|---|---|
| Corpus (`corpus.py`) | 200 deterministic synthetic docs (seed 42): process, quality, equipment, supply, operations, safety | Reproducible retrieval/eval; no real data |
| Retrieval (`vector_store.py`) | Dependency-free TF-IDF + cosine, in-memory | Transparent, offline, easy to swap for a hosted search/vector service |
| LLM client (`llm.py`) | Provider interface with `mock` (default), `openai`, `anthropic`, `azure` adapters over stdlib `urllib` | Model-agnostic; tests/bench need no key or network |
| RAG QA (`agents.py`) | Retrieve k=4 → grounded prompt → drop sentences citing non-retrieved docs | Citations `[DOC-nnn]` on every claim; hallucination guard |
| Business-case agent | Same evidence → Problem / Options / Recommendation / Risks / Next steps with owner + timeline | Structured, actionable draft, not free-form prose |
| Anomaly digest | Synthetic telemetry (4 metrics × 2 sites × 28 days), z-score ≥ 2.5, per-metric playbook actions | Turns spikes into assigned actions within 1 shift |
| Evaluation (`evaluation.py`) | Token-overlap faithfulness, citation existence+support, action-signal checklist | Deterministic — no LLM judge, no flaky scores |
| Benchmark (`benchmark.py`) | Corpus build + 7 eval tasks + anomaly digest, timed | One-command sanity check (output below) |

## Run

```bash
pip install -e ".[dev]"        # or just: pip install pytest  (runtime deps: none)

export PYTHONPATH=src          # if not installed
python -m micron_copilot.cli ask "What is the preventive maintenance interval for equipment?"
python -m micron_copilot.cli business-case "reduce defect density and unplanned downtime"
python -m micron_copilot.cli digest
python -m micron_copilot.cli eval

# hosted provider (optional, needs your own key/endpoint in env):
python -m micron_copilot.cli --provider azure ask "..."
```

Sample `ask` output (offline mock):

```
The wafer handler at Fab-C requires preventive maintenance every 9 days [DOC-171].
The inspection station at Fab-C requires preventive maintenance every 13 days [DOC-111].
...
Citations: DOC-171, DOC-111, DOC-087, DOC-165
```

## Test

```bash
pytest -q
# 12 passed in 0.07s
```

12 tests covering: corpus determinism/uniqueness, retrieval relevance and
no-match behaviour, mock-LLM determinism, provider factory, grounded/cited
RAG, unanswerable-question handling, business-case structure, injected-anomaly
detection, quiet flat series, bounded evaluation scores, and the benchmark.

## Benchmark (synthetic sanity run — scaled down, honestly)

One run, copied verbatim (offline mock, this repo's machine; timings vary):

```bash
python -m micron_copilot.cli bench
```

```
micron-copilot-lab synthetic benchmark (offline mock LLM)
documents=200 provider=mock model=mock-extractive-v1
index_seconds=0.007 retrieval_hit_rate=1.0
eval_tasks=7 faithfulness=1.0 citation_accuracy=1.0 action_usefulness=0.1429 overall=1.0
anomalies_detected=10 digest_action_usefulness=0.5
eval_seconds=0.004 total_seconds=0.014
```

How to read this (do not over-claim):

- This is a **200-document, 7-task synthetic sanity run**, not a production
  evaluation. Scale, corpus realism and task difficulty are all toy-level.
- `faithfulness=1.0 / citation_accuracy=1.0` are expected here: the mock LLM
  is *extractive by construction* (it quotes retrieved evidence), so this
  validates the grounding/citation plumbing — it says nothing about how a
  real generative model would score. Run `--provider ... eval` to measure that.
- `action_usefulness=0.1429` is the mean across all 7 tasks; 6 are Q&A tasks
  with no actions (scored 0). The business-case task itself scores 1.0 and
  the anomaly digest 0.5 on the action-signal checklist.
- `anomalies_detected=10` matches the injected ground truth (5 per site:
  defect/downtime spikes on days 10/21, yield dip on day 15, × 2 sites).

## Limitations (honest)

- **Synthetic everything**: corpus, telemetry, injected anomalies and the
  business-case figures are generated. No conclusion transfers to real fabs.
- **Mock LLM is extractive**, not generative; hosted adapters are thin
  stdlib HTTP wrappers, untested against live endpoints, with no retries,
  streaming, tool-calling, or token accounting.
- **Retrieval is TF-IDF**, not embeddings: no synonym/semantic recall,
  no chunking strategy, no reranking, no access control on documents.
- **Evaluation is heuristic** token overlap (≥0.35 support threshold) plus a
  keyword action checklist — a smoke signal, not human/LLM-judged quality.
  Near-paraphrase answers can be under- or over-scored.
- **Anomaly detection is a global z-score**: no seasonality, drift, rolling
  windows, or multivariate correlation; injected spikes also shift the mean.
- No auth, multi-tenancy, conversation memory, UI, persistence, PII handling,
  prompt-injection defence, or cost/latency controls — all required before
  any real enterprise deployment.
- Business-case "figures" come only from retrieved synthetic evidence and
  must be re-validated; the agent deliberately assumes no external numbers.

## License

MIT — see `LICENSE`.
