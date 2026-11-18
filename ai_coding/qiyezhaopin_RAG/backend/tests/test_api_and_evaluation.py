import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from evaluation.metrics import score_ids
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.query import JobFilters


@pytest.mark.parametrize("payload", [{"query":" "}, {"query":"test", "top_k":0},
    {"query":"test", "top_k":21}, {"query":"test", "session_id":"x"},
    {"query":"test", "filters":{"location":"北京", "sql":"bad"}}])
def test_api_rejects_bad_chat_inputs(payload):
    from main import app
    assert TestClient(app).post("/api/chat", json=payload).status_code == 422


def test_unready_health_does_not_connect_to_db():
    from main import app
    client = TestClient(app)
    assert client.get("/api/health").json()["ready"] is False
    assert client.post("/api/chat", json={"query":"test"}).status_code == 503


def test_graph_input_is_bounded():
    from main import app
    assert TestClient(app).get("/api/graph/subgraph", params={"node_id":"q", "max_depth":100}).status_code == 422


def test_metrics_handle_duplicates_and_unanswerable():
    assert score_ids(["x","x","a"], ["a","b"], 3)["recall"] == .5
    assert score_ids(["x","x","a"], ["a","b"], 3)["mrr"] == .5
    assert score_ids([], [], 5)["abstention"] == 1
    assert score_ids(["a"], [], 5)["abstention"] == 0
    assert score_ids([], [], 5)["recall"] is None


def test_gold_labels_match_seed_constraints(settings):
    prep = GraphDataPreparationModule(settings.seed_path, "", "", "")
    docs = prep.build_job_documents()
    gold = json.loads((Path(__file__).parents[1]/"evaluation/gold.json").read_text(encoding="utf8"))
    assert len(gold["cases"]) >= 30
    for case in gold["cases"]:
        expected = JobFilters(**case["filters"])
        matching = {d.metadata["node_id"] for d in docs if expected.matches(d)}
        assert set(case["relevant_ids"]) == matching, case["id"]


def test_gap_distinguishes_required_bonus_and_new_opportunities(settings):
    from rag_modules.gap_analysis import GapAnalysisModule
    data = {"candidates":[{"name":"person", "nodeId":"c", "skills":[{"name":"Python"}]}],
            "jobs":[{"name":"job", "nodeId":"j", "skills":[{"name":"Python"}, {"name":"Go"},
                    {"name":"Redis", "isRequired":False}]}]}
    tool = GapAnalysisModule(settings, data=data)
    assert tool.analyze_gap("person","j")["match_rate"] == .5
    result = tool.what_if("person", ["golang"])
    assert not result["new_opportunities"]
    assert result["improved_matches"][0]["delta"] == .5
    assert "error" in tool.analyze_gap("missing","j")
