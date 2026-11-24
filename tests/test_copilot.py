import pytest

from micron_copilot.agents import AnomalyDigestAgent, BusinessCaseAgent, RAGAgent
from micron_copilot.benchmark import run_benchmark
from micron_copilot.corpus import generate_corpus
from micron_copilot.evaluation import run_evaluation, score_result
from micron_copilot.llm import MockLLM, get_llm
from micron_copilot.telemetry import detect_anomalies, generate_telemetry
from micron_copilot.vector_store import VectorStore


@pytest.fixture(scope="module")
def store():
    return VectorStore(generate_corpus(200))


def test_corpus_size_and_determinism():
    a, b = generate_corpus(200), generate_corpus(200)
    assert len(a) == 200 and [d.text for d in a] == [d.text for d in b]


def test_corpus_ids_unique_and_categories():
    docs = generate_corpus(200)
    assert len({d.doc_id for d in docs}) == 200
    assert {d.category for d in docs} >= {"process", "quality", "equipment"}


def test_vector_store_retrieves_relevant_doc(store):
    hits = store.search("preventive maintenance equipment failure repair", k=3)
    assert hits and hits[0][0].category in ("equipment", "process", "operations")
    assert hits[0][1] > 0


def test_vector_store_no_match_returns_empty(store):
    assert store.search("zzzzqqqq nonexistenttoken", k=3) == []


def test_mock_llm_is_deterministic():
    llm = MockLLM()
    prompt = "EVIDENCE:\n[DOC-001] T: Alpha beta gamma. Delta. [DOC-001]"
    assert llm.complete("s", prompt).text == llm.complete("s", prompt).text
    assert "[DOC-001]" in llm.complete("s", prompt).text


def test_llm_factory_defaults_to_mock_and_rejects_unknown():
    assert get_llm().provider == "mock"
    assert get_llm("azure").provider == "azure-openai"
    with pytest.raises(ValueError):
        get_llm("no-such-provider")


def test_rag_answer_is_grounded_and_cited(store):
    r = RAGAgent(store, MockLLM()).answer(
        "What is the preventive maintenance interval for equipment?")
    assert r.citations and all(c in r.sources for c in r.citations)
    assert r.meta["grounded"] is True


def test_rag_unanswerable_question_reports_no_evidence(store):
    empty = VectorStore([])
    r = RAGAgent(empty, MockLLM()).answer("anything at all")
    assert "No supporting evidence" in r.text and r.citations == []


def test_business_case_has_sections_actions_and_citations(store):
    r = BusinessCaseAgent(store, MockLLM()).draft(
        "reduce defect density and unplanned downtime")
    for section in ("Problem", "Options", "Recommendation", "Next steps"):
        assert section in r.text
    assert len(r.actions) >= 3 and all("Owner:" in a for a in r.actions)


def test_anomaly_digest_detects_injected_anomalies():
    points = generate_telemetry(inject=True)
    anomalies = detect_anomalies(points)
    assert anomalies and any(a.day in (10, 15, 21) for a in anomalies)
    digest = AnomalyDigestAgent().digest(points)
    assert digest.meta["anomaly_count"] >= 1 and len(digest.actions) >= 1


def test_anomaly_detector_quiet_on_flat_series():
    from micron_copilot.telemetry import TelemetryPoint
    flat = [TelemetryPoint(d, "Fab-A", "yield_pct", 95.0) for d in range(1, 29)]
    assert detect_anomalies(flat) == []


def test_evaluation_scores_bounded_and_benchmark_runs(store):
    out = run_evaluation(store, RAGAgent(store, MockLLM()),
                         BusinessCaseAgent(store, MockLLM()))
    for row in out["rows"]:
        for k in ("faithfulness", "citation_accuracy", "action_usefulness", "overall"):
            assert 0.0 <= row[k] <= 1.0
    assert out["mean"]["citation_accuracy"] >= 0.8
    bench = run_benchmark()
    assert bench["documents"] == 200 and bench["retrieval_hit_rate"] == 1.0
    assert bench["anomaly_digest"]["anomaly_count"] >= 1
    from micron_copilot.agents import AgentResult
    bad = AgentResult("rag", "The yield is exactly 42 percent with no citation.")
    bad_scores = score_result(bad, store)
    assert bad_scores["citation_accuracy"] == 0.0 and bad_scores["faithfulness"] == 0.0
