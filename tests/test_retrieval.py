from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.dataset import Scenario
from backend.main import app
from backend.retrieval import LocalScenarioRetriever, build_index, retrieval_query


def scenario(identifier: str, text: str, category: str, need: str, condition: str) -> Scenario:
    return Scenario({
        "scenario_id": identifier,
        "scenario_text": text,
        "product_category": category,
        "need_classification": need,
        "product_condition": condition,
        "rationale": "test rationale",
        "record_origin": "synthetic",
    })


def test_local_index_round_trips_and_retrieves_relevant_perspective(tmp_path):
    records = [
        scenario("a", "My headphones work and I want better sound quality.", "audio", "Optional or preference-driven upgrade", "works"),
        scenario("b", "My laptop is broken and I need a replacement for work tomorrow.", "computers", "Essential replacement", "failed"),
        scenario("c", "My headphones are broken and I need a reliable replacement.", "audio", "Essential replacement", "failed"),
    ]
    index_path = tmp_path / "index.json"
    build_index(records, index_path)
    loaded = json.loads(index_path.read_text())
    assert loaded["record_count"] == 3
    results = LocalScenarioRetriever(index_path, threshold=0.05).retrieve(
        "headphones broken replacement needed",
        top_k=2,
    )
    assert results
    assert results[0]["scenario_id"] == "c"
    assert results[0]["record_origin"] == "synthetic"


def test_missing_or_corrupt_index_is_non_blocking(tmp_path):
    missing = LocalScenarioRetriever(tmp_path / "missing.json")
    assert missing.retrieve("a purchase decision") == []
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("not json")
    assert LocalScenarioRetriever(corrupt_path).retrieve("a purchase decision") == []


def test_retrieval_query_preserves_explicit_contrasts():
    query = retrieval_query({
        "name": "gaming laptop",
        "current": "It works, but has limitations",
        "need_classification": "Functional upgrade",
        "user_affordability": "reported_uncertain",
        "waiting_period": "2 months",
        "external_pressure": True,
    }, "The sale ends tonight but I can wait two months")
    assert "gaming laptop" in query
    assert "reported_uncertain" in query
    assert "2 months" in query
    assert "sale ends tonight" in query


def test_api_remains_deterministic_when_retrieval_is_disabled(monkeypatch):
    monkeypatch.setenv("SEMANTIC_RETRIEVAL_ENABLED", "0")
    response = TestClient(app).post("/api/chat", json={
        "message": "My current headphones work, but I want better sound quality and can wait a month.",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["decision_state"]["recommendation"] in {"BUY", "WAIT", "DON'T BUY"}
    assert body["decision_state"]["retrieved_cases"] == []
    assert "semantic_retrieval" not in body["tools_used"]


def test_unresearched_models_are_explicitly_preserved(monkeypatch):
    monkeypatch.setenv("SEMANTIC_RETRIEVAL_ENABLED", "0")
    response = TestClient(app).post("/api/chat", json={
        "message": "I am considering a gaming laptop for 95000. My current laptop works but struggles with games. I have not compared refurbished laptops or cheaper models.",
    })
    assert response.status_code == 200
    assert response.json()["session_facts"]["alternatives_checked"] is False
