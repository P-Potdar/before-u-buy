from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_short_lifespan_followup_updates_state_without_repeating_question(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    session = "followup-lifespan-regression"
    first = client.post("/api/chat", json={
        "session_id": session,
        "message": "I want Sony headphones for ₹8000. My current headphones work and I use them daily for classes.",
    }).json()
    second = client.post("/api/chat", json={"session_id": session, "message": "4 years"}).json()
    assert first["session_id"] == second["session_id"] == session
    assert second["session_facts"]["lifespan_years"] == 4
    assert "for how many years" not in second["reply"].casefold()
    assert "4 years" in second["reply"]


def test_short_motivation_followup_updates_pressure_in_context(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    session = "followup-motivation-regression"
    client.post("/api/chat", json={
        "session_id": session,
        "message": "I am considering headphones for ₹8000 and already have a working pair.",
    })
    result = client.post("/api/chat", json={"session_id": session, "message": "social pressure"}).json()
    assert result["session_facts"]["external_pressure"] is True
    assert result["session_facts"]["urgency"] == "Social pressure or FOMO"
    assert "social pressure" in result["reply"].casefold()
    assert result["decision_state"]["recommendation"] == "WAIT"


def test_answer_plus_new_recommendation_is_not_replaced_by_acknowledgement(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    session = "followup-new-question-regression"
    initial = client.post("/api/chat", json={
        "session_id": session,
        "message": "I want Sony headphones for ₹8000. My current headphones work and I use them daily for classes.",
    }).json()
    assert "recorded that motivation" not in initial["reply"].casefold()
    client.post("/api/chat", json={"session_id": session, "message": "4 years"})
    result = client.post("/api/chat", json={
        "session_id": session,
        "message": "Social pressure is a major reason I am considering buying them. Given that I cannot comfortably afford them and my current headphones work, what is your recommendation now?",
    }).json()
    assert "recorded" not in result["reply"].casefold()
    assert "recommendation" in result["reply"].casefold()
    assert result["decision_state"]["recommendation"] == "WAIT"
    assert "social pressure" in result["session_facts"]["motivation"].casefold()
    assert "_decision_state" not in result["session_facts"]


def test_cost_per_use_discloses_inferred_daily_usage_and_arithmetic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    session = "cost-basis-regression"
    client.post("/api/chat", json={
        "session_id": session,
        "message": "I want headphones for ₹8000. My current headphones work and I use them daily for classes.",
    })
    result = client.post("/api/chat", json={"session_id": session, "message": "4 years"}).json()
    basis = result["decision_state"]["cost_per_use_basis"]
    assert result["decision_state"]["estimated_total_uses"] == 1456
    assert result["decision_state"]["estimated_cost_per_use"] == 5.4945
    assert basis["uses_per_week_source"] == "inferred from user wording"
    assert basis["weeks_per_year"] == 52
    assert basis["lifespan_source"] == "user reported"
    assert result["session_facts"]["fact_evidence"]["uses_per_week"]["source"] == "inferred from user wording"


def test_pending_answer_and_new_question_are_processed_in_one_turn(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    session = "followup-answer-and-question"
    client.post("/api/chat", json={
        "session_id": session,
        "message": "I want headphones for ₹8000. My current headphones work and I use them daily for classes.",
    })
    result = client.post("/api/chat", json={
        "session_id": session,
        "message": "4 years. Given that I cannot afford them, what is your recommendation?",
    }).json()
    assert result["session_facts"]["lifespan_years"] == 4
    assert "recommendation" in result["reply"].casefold()
    assert result["decision_state"]["recommendation"] == "WAIT"


def test_model_cannot_reask_a_resolved_question_on_a_later_turn():
    text = "The recommendation remains WAIT. For how many years do you expect to use it?"
    facts = {
        "uses_per_week": 7,
        "lifespan_years": 4,
        "pending_question": {
            "topic": "usage",
            "field": "lifespan_years",
            "expected_type": "number_with_years",
            "status": "resolved",
        },
    }
    result = __import__("backend.main", fromlist=["_remove_redundant_question"])._remove_redundant_question(text, facts)
    assert "for how many years" not in result.casefold()
    assert "WAIT" in result

