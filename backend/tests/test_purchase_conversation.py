"""Regression coverage for the purchase-reflection conversation contract.

These tests deliberately use the deterministic provider.  They verify that an
LLM can add wording to a response, but cannot become the source of facts,
scores, or the decision signal.
"""

import asyncio
import json
from pathlib import Path

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


def test_synthetic_baseline_scenarios_preserve_evidence_and_factor_directions(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "purchase_scenarios.jsonl"
    scenarios = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    results = {}

    for scenario in scenarios:
        result = chat(scenario["user_message"], f"dataset-{scenario['scenario_id']}")
        facts = result["session_facts"]
        results[scenario["scenario_id"]] = result

        for field, expected in scenario["expected_facts"].items():
            assert facts[field] == expected, (scenario["scenario_id"], field, facts)
        for field in scenario["expected_unknowns"]:
            assert facts.get(field) is None, (scenario["scenario_id"], field, facts)
        assert result["decision_state"]["recommendation"] in {"BUY", "WAIT", "DON'T BUY"}
        assert result["decision_state"]["signal"] == result["decision_state"]["recommendation"]
        if result["assessment_update"] is None:
            assert facts.get("price") is None
        else:
            assert result["assessment_update"]["decision"] == result["decision_state"]["recommendation"]
            assert result["assessment_update"]["confidence"] == result["decision_state"]["confidence"]
        assert result["decision_state"]["provisional"] is True
        assert result["decision_state"]["confidence"]["level"] == "low"
        assert result["reply"].count("?") <= 1
        assert result["session_facts"]["fact_evidence"]

    for scenario in scenarios:
        for factor, direction in scenario["expected_factor_directions"].items():
            comparison = direction.split(" than ")
            actual_score = results[scenario["scenario_id"]]["decision_state"][factor]["score"]
            comparison_score = results[comparison[1]]["decision_state"][factor]["score"]
            assert actual_score > comparison_score, scenario["scenario_id"]


def test_missing_price_asks_contextual_question_and_unknown_affordability_is_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    missing_price = chat("I am considering an ergonomic chair.", "purchase-missing-price")
    assert "price" in missing_price["reply"].lower()

    unknown = chat(
        "I want headphones for ₹18,000. I am unsure whether I can comfortably afford them.",
        "purchase-affordability-unknown",
    )
    assert unknown["session_facts"]["budget_status"] == "unknown"
    assert "budget" not in unknown["session_facts"]
    assert "monthly_surplus" not in unknown["session_facts"]
    assert any("affordability is unknown" in item.lower() for item in unknown["decision_state"]["missing_inputs"])


def test_exact_optional_upgrade_message_preserves_savings_social_pressure_and_wait_window(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I want to buy premium headphones for ₹18,000. My current headphones work perfectly well for everyday use, "
        "but I want better noise cancellation and improved sound quality. Honestly, I started considering this "
        "purchase after seeing my friends upgrade theirs. I have enough money saved, but buying them is not urgent, "
        "and I can comfortably wait another month. I haven't compared cheaper alternatives yet. "
        "Help me decide whether I should buy them now.",
        "purchase-exact-optional-upgrade",
    )
    facts = result["session_facts"]
    state = result["decision_state"]

    assert facts["price"] == 18000
    assert "budget" not in facts
    assert facts["current"] == "I have something that works"
    assert facts["need"] == "Useful improvement"
    assert facts["savings_adequacy"] == "reported_sufficient"
    assert facts.get("user_affordability") is None
    assert facts["urgency"] == "Social pressure or FOMO"
    assert facts["purchase_timing"] == "not_urgent"
    assert facts["waiting_period"] == "one month"
    assert facts["alternatives_checked"] is False
    assert "improved sound quality" in facts["motivation"]
    assert state["recommendation"] == "WAIT"
    assert state["signal"] == state["recommendation"]
    assert "wait one month" in state["recommendation_explanation"]
    assert "current item still works" in state["recommendation_explanation"]
    assert state["financial_context"]["product_price"] == 18000
    assert state["financial_context"]["stated_budget"] is None
    assert state["financial_context"]["savings_adequacy"] == "reported_sufficient"
    assert any("self-reported" in item.lower() for item in result["assessment_update"]["missing_inputs"])


def test_exact_urgent_replacement_message_separates_budget_from_product_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I need to replace my headphones, which stopped working completely this morning. I have daily work meetings "
        "starting tomorrow, and I have no spare headphones or other usable alternative. My budget is ₹18,000, and "
        "I can afford to spend that amount without affecting rent, bills, or savings. I need a reliable replacement "
        "urgently, but I haven't compared models or checked whether a cheaper option would meet my needs. Should I "
        "buy headphones now, and what should I consider before choosing a specific model?",
        "purchase-exact-urgent-replacement",
    )
    facts = result["session_facts"]
    state = result["decision_state"]

    assert facts["budget"] == 18000
    assert "price" not in facts
    assert facts["current"] == "Mine is broken or unusable"
    assert facts["need"] == "Solves a real problem"
    assert facts["urgency"] == "A genuine deadline"
    assert facts["purchase_timing"] == "urgent"
    assert facts["deadline_consequences"] is True
    assert facts["alternative_available"] is False
    assert facts["alternatives_checked"] is False
    assert facts["user_affordability"] == "reported_affordable"
    assert facts["essential_expenses_affected"] is False
    assert facts["savings_affected"] is False
    assert result["assessment_update"] is None
    assert state["recommendation"] == "WAIT"
    assert state["signal"] == state["recommendation"]
    assert "time-sensitive" in state["recommendation_explanation"]
    assert "no specific model or price" in state["recommendation_explanation"]
    assert state["financial_context"]["product_price"] is None
    assert state["financial_context"]["stated_budget"] == 18000
    assert state["financial_context"]["user_affordability"] == "reported_affordable"
    assert state["replacement_context"]["usable_alternative_available"] is False
    assert not any("affordability is unknown" in item.lower() for item in state["missing_inputs"])
    assert not any("timing and consequences" in item.lower() for item in state["missing_inputs"])
    assert not any("availability of another usable alternative" in item.lower() for item in state["missing_inputs"])


def test_real_world_regression_preserves_optional_upgrade_and_useful_followup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "I want to buy premium headphones for ₹18,000. My current headphones work perfectly well, and I use them every day. "
        "I want better noise cancellation and sound quality, partly because several of my friends recently upgraded theirs. "
        "I have enough money saved, this purchase will not affect my essential expenses, and I can comfortably wait another month. "
        "I have not compared cheaper alternatives yet. Should I buy them now? Explain the main reasons behind your recommendation and identify any important information you still need.",
        "real-world-optional-upgrade",
    )
    facts = result["session_facts"]
    assert facts["need"] == "Useful improvement"
    assert facts["need_classification"] == "Optional or preference-driven upgrade"
    assert "stated practical need" not in facts["motivation"]
    assert "sound quality" in facts["motivation"]
    assert facts["waiting_period"] == "one month"
    assert result["decision_state"]["recommendation"] == "WAIT"
    assert "cheaper model" in result["reply"]
    assert result["session_facts"]["fact_evidence"]["waiting_period"]["source"] == "user message"


def test_real_world_regression_distinguishes_budget_from_product_price(monkeypatch):
    result = chat(
        "My headphones stopped working completely this morning. I have important work meetings starting tomorrow, and I have no spare headphones or other usable alternative. "
        "My budget is ₹18,000, and I can afford this amount without affecting rent, bills, or savings. I need a reliable replacement urgently, but I have not compared models or checked whether a cheaper option would meet my needs. "
        "Should I buy a replacement now? Distinguish the urgency of replacing my headphones from whether a particular model is worth ₹18,000.",
        "real-world-urgent-replacement",
    )
    facts = result["session_facts"]
    assert facts["budget"] == 18000
    assert "price" not in facts
    assert facts["need_classification"] == "Essential replacement"
    assert facts["purchase_timing"] == "urgent"
    assert result["decision_state"]["recommendation"] == "WAIT"
    assert "specific model or price" in result["decision_state"]["recommendation_explanation"]


def test_real_world_regression_preserves_gaming_goal_wait_window_and_expenses(monkeypatch):
    result = chat(
        "I am considering buying a gaming laptop for ₹95,000. My current laptop still works, although it struggles with newer games. "
        "I want better performance, but I am not sure whether I can comfortably afford this purchase because I have upcoming expenses. "
        "I could wait two months, and I have not compared refurbished laptops or cheaper models. A limited-time sale ends tonight, which is making me feel pressured to buy. "
        "Help me decide whether to buy now, wait, or avoid this purchase.",
        "real-world-gaming-laptop",
    )
    facts = result["session_facts"]
    state = result["decision_state"]
    assert facts["use_case"] == "gaming"
    assert "better gaming performance" in facts["motivation"]
    assert facts["need_classification"] == "Functional upgrade"
    assert facts["upcoming_expenses"] is True
    assert facts["waiting_period"] == "two months"
    assert facts["budget_status"] == "unknown"
    assert state["recommendation"] == "WAIT"
    assert "sale deadline" in state["recommendation_explanation"]


def test_social_pressure_and_replacement_deadline_remain_separate_facts(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "My headphones stopped working. I need them for work meetings starting tomorrow, and I have no usable "
        "alternative. My friends are pushing me to buy a premium model.",
        "purchase-urgent-and-social",
    )

    facts = result["session_facts"]
    state = result["decision_state"]
    assert facts["purchase_timing"] == "urgent"
    assert facts["urgency"] == "Social pressure or FOMO"
    assert facts["external_pressure"] is True
    assert not main.chat_facts["purchase-urgent-and-social"].get("conflicts")
    assert "time-sensitive" in state["recommendation_explanation"]
    factors = state["purchase_pressure"]["factors"]
    assert any("social pressure" in item["factor"].lower() for item in factors)
    assert any("deadline" in item["factor"].lower() for item in factors)


def test_financial_claim_contradictions_and_corrections_are_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    first = chat("I can afford the headphones and have enough savings.", "purchase-finance-correction")
    contradicted = chat("I cannot afford them.", "purchase-finance-correction")

    assert first["session_facts"]["user_affordability"] == "reported_affordable"
    assert first["session_facts"]["savings_adequacy"] == "reported_sufficient"
    assert main.chat_facts["purchase-finance-correction"]["user_affordability"] is None
    assert "user_affordability" in main.chat_facts["purchase-finance-correction"]["conflicts"]
    corrected = chat("Actually, I cannot afford them.", "purchase-finance-correction")

    assert corrected["session_facts"]["user_affordability"] == "reported_not_affordable"
    assert "user_affordability" not in main.chat_facts["purchase-finance-correction"]["conflicts"]
    assert main.chat_facts["purchase-finance-correction"]["corrections"][-1]["field"] == "user_affordability"


def test_chat_response_always_has_canonical_recommendation_without_inventing_price(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = chat(
        "My budget is ₹18,000, I can afford it, and I have no usable alternative.",
        "purchase-budget-is-not-price",
    )
    state = result["decision_state"]

    assert "price" not in result["session_facts"]
    assert result["assessment_update"] is None
    assert state["recommendation"] in {"BUY", "WAIT", "DON'T BUY"}
    assert state["signal"] == state["recommendation"]
    assert state["financial_context"]["product_price"] is None
    assert state["financial_context"]["stated_budget"] == 18000


def test_repeated_identical_decision_state_is_deterministic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    message = "I want headphones for ₹18,000. My current pair works fine, and I have not checked alternatives."
    first = chat(message, "purchase-deterministic-a")
    second = chat(message, "purchase-deterministic-b")
    for key in ("regret_risk", "purchase_pressure"):
        assert first["decision_state"][key]["score"] == second["decision_state"][key]["score"]
    assert first["decision_state"]["recommendation"] == second["decision_state"]["recommendation"]


def test_concurrent_turns_for_one_session_are_serialized(monkeypatch):
    async def discovered():
        await asyncio.sleep(0.01)
        return ["test-model"]

    async def model_reply(*_args, **_kwargs):
        await asyncio.sleep(0.01)
        return "I have updated the reflection using the details you provided."

    monkeypatch.setattr(main, "_available_ollama_models", discovered)
    monkeypatch.setattr(main, "_ollama_reply", model_reply)

    async def send_both():
        return await asyncio.gather(
            main.chat(main.ChatRequest(
                message="I want headphones for ₹18,000.",
                session_id="purchase-concurrent-session",
                model="test-model",
            )),
            main.chat(main.ChatRequest(
                message="My current pair is broken and I need them for work tomorrow.",
                session_id="purchase-concurrent-session",
                model="test-model",
            )),
        )

    results = asyncio.run(send_both())
    facts = main.chat_facts["purchase-concurrent-session"]
    assert facts["price"] == 18000
    assert facts["current"] == "Mine is broken or unusable"
    assert facts["urgency"] == "A genuine deadline"
    assert all(result["decision_state"]["signal"] == result["decision_state"]["recommendation"] for result in results)
