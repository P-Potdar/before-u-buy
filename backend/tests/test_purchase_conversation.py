"""Regression coverage for the purchase-reflection conversation contract.

These tests deliberately use the deterministic provider.  They verify that an
LLM can add wording to a response, but cannot become the source of facts,
scores, or the decision signal.
"""

from fastapi.testclient import TestClient

from backend import main
from backend.main import app


client = TestClient(app)


def chat(message: str, session_id: str) -> dict:
    response = client.post("/api/chat", json={"message": message, "session_id": session_id})
    assert response.status_code == 200
    return response.json()


def test_purchase_can_start_with_only_item_and_price(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat("I am considering Sony headphones for ₹18,000.", "purchase-minimal")

    assert result["session_facts"]["name"] == "Headphones"
    assert result["session_facts"]["price"] == 18000
    assert result["assessment_update"]["price"] == 18000
    assert result["provider"] == "deterministic_fallback"
    assert result["decision_state"]["signal"] != "BUY" or result["decision_state"]["provisional"]


def test_purchase_intent_is_contextual_not_generic_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat("I want to buy Sony headphones for ₹18,000.", "purchase-intent")

    assert result["session_facts"]["name"] == "Headphones"
    assert "headphone" in result["reply"].lower()
    assert "what purchase" not in result["reply"].lower()


def test_full_message_extracts_multiple_facts_and_keeps_alternatives(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I already have headphones that still work, but the battery is weak. "
        "I use them for laptop meetings and commute almost every day. "
        "I want better noise cancellation and comfort, though friends' premium "
        "headphones influence me. I can wait a month and haven't checked repair yet.",
        "purchase-facts",
    )
    facts = result["session_facts"]

    assert facts["existing_ownership"] is True
    assert facts["current_condition"] == "limitations"
    assert facts["current"] == "It works, but has limitations"
    assert facts["use_case"] == "calls or meetings; commuting or travel; work"
    assert facts["uses_per_week"] == 7
    assert "better noise cancellation" in facts["motivation"]
    assert "comfort" in facts["motivation"]
    assert facts["urgency"] == "Social pressure or FOMO"
    assert facts["waiting_period"] == "one month"
    assert facts["repair_investigated"] is False
    assert result["decision_state"]["missing_inputs"]


def test_followup_facts_are_retained_and_questions_adapt(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    first = chat("I want headphones for ₹18,000 and already have a working pair.", "purchase-adaptive")
    second = chat("I use them for meetings and commuting every day.", "purchase-adaptive")

    assert first["session_facts"]["price"] == second["session_facts"]["price"] == 18000
    assert second["session_facts"]["existing_ownership"] is True
    assert second["session_facts"]["uses_per_week"] == 7
    assert second["reply"].count("?") <= 1
    assert "what price" not in second["reply"].lower()


def test_correction_replaces_outdated_fact(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    chat("I am considering headphones for ₹18,000.", "purchase-correction")
    result = chat("Actually, the price is ₹15,000.", "purchase-correction")

    assert result["session_facts"]["price"] == 15000
    assert main.chat_facts["purchase-correction"]["corrections"]
    assert "price" not in main.chat_facts["purchase-correction"].get("conflicts", {})


def test_fresh_session_does_not_inherit_private_context(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    chat("I want Sony headphones for ₹18,000 and use them for meetings.", "purchase-private-a")
    fresh = chat("What price did I mention?", "purchase-private-b")

    assert "price" not in fresh["session_facts"]
    assert "earlier purchase details" in fresh["reply"].lower()


def test_missing_financial_data_stays_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat("I want headphones for ₹18,000. I use them every day.", "purchase-finance-unknown")

    facts = result["session_facts"]
    state = result["decision_state"]
    assert "budget" not in facts
    assert "monthly_surplus" not in facts
    assert any("affordability" in item.lower() for item in state["missing_inputs"])
    assert "unafford" not in result["reply"].lower()


def test_ownership_is_evidence_not_an_automatic_no(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I already own headphones, but the battery is weak and I need better noise cancellation for daily meetings.",
        "purchase-ownership",
    )

    assert result["session_facts"]["existing_ownership"] is True
    assert result["session_facts"]["current_condition"] == "limitations"
    assert "working alternative" not in result["decision_state"]["signal_explanation"].lower()


def test_scores_recalculate_for_relevant_evidence_and_stay_stable_on_irrelevant_turn(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    first = chat("I want headphones for ₹18,000.", "purchase-score-changes")
    relevant = chat("My current pair works fine and friends are influencing me.", "purchase-score-changes")
    stable = chat("The room is quiet today.", "purchase-score-changes")

    assert relevant["decision_state"]["regret_risk"]["score"] != first["decision_state"]["regret_risk"]["score"]
    assert stable["decision_state"]["regret_risk"]["score"] == relevant["decision_state"]["regret_risk"]["score"]
    assert stable["decision_state"]["purchase_pressure"]["score"] == relevant["decision_state"]["purchase_pressure"]["score"]


def test_scores_expose_deterministic_explanations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I want headphones for ₹18,000 because social pressure is influencing me, but my current pair works.",
        "purchase-explanations",
    )

    for score_name in ("purchase_pressure", "regret_risk"):
        score = result["decision_state"][score_name]
        assert isinstance(score["score"], int)
        assert 0 <= score["score"] <= 100
        assert score["factors"]
        assert score["explanation"]
    assert "not a probability" in result["decision_state"]["purchase_pressure"]["scale"]


def test_model_cannot_override_deterministic_scores_or_verdict(monkeypatch):
    async def discovered():
        return ["test-model"]

    async def model_reply(*_args, **_kwargs):
        return "You should buy it now."

    monkeypatch.setattr(main, "_available_ollama_models", discovered)
    monkeypatch.setattr(main, "_ollama_reply", model_reply)
    result = client.post(
        "/api/chat",
        json={
            "message": "I want headphones for ₹18,000, already have a working pair, and social pressure is influencing me.",
            "session_id": "purchase-model-boundary",
            "model": "test-model",
        },
    ).json()

    assert result["provider"] == "deterministic_fallback"
    assert "should buy" not in result["reply"].lower()
    for score_name in ("regret_risk", "purchase_pressure"):
        score = result["decision_state"][score_name]
        assert score["score"] == min(100, sum(item["points"] for item in score["factors"]))
    assert result["decision_state"]["signal"] in {"WAIT", "DON'T BUY YET", "INSUFFICIENT INFORMATION"}


def test_invalid_llm_output_and_timeout_have_useful_fallbacks(monkeypatch):
    async def discovered():
        return ["test-model"]

    async def unavailable(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_available_ollama_models", discovered)
    monkeypatch.setattr(main, "_ollama_reply", unavailable)
    result = client.post(
        "/api/chat",
        json={"message": "I want headphones for ₹18,000.", "session_id": "purchase-timeout", "model": "test-model"},
    ).json()

    assert result["provider"] == "deterministic_fallback"
    assert result["reply"]
    assert result["fallback_reason"]
    assert "headphone" in result["reply"].lower()


def test_final_state_has_evidence_counterarguments_and_uncertainty(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I want headphones for ₹18,000. I use them for work meetings daily, "
        "already have a pair with a weak battery, and friends influenced me. "
        "I can wait a month and have not checked repair.",
        "purchase-final-state",
    )
    state = result["decision_state"]

    assert state["supporting_factors"]
    assert state["missing_inputs"]
    assert state["signal_explanation"]
    assert "unknown" in state["limitations"].lower()
