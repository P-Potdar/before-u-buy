import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import (
    AssessmentInput,
    ScoringContext,
    calculate_purchase_pressure,
    calculate_regret_risk,
    classify_sentiment,
    evaluate,
    recommend_purchase,
)
import pytest


def test_budget_overrun_flags_review():
    result = evaluate(AssessmentInput(name="Headphones", price=20000, budget=15000, recurring_monthly=500))
    assert result.recommendation == "Review the entered budget figures"
    assert result.decision == "DON'T BUY"
    assert result.budget_exceeded_by == 11000


def test_recommendation_rules_distinguish_necessary_replacement_and_optional_upgrade():
    replacement = evaluate(AssessmentInput(
        name="Headphones",
        price=8000,
        budget=10000,
        need="Essential need",
        current="Mine is broken or unusable",
        urgency="A genuine deadline",
        deadline_consequences=True,
        alternative_available=False,
        alternatives_checked=True,
    ))
    upgrade = evaluate(AssessmentInput(
        name="Headphones",
        price=18000,
        need="Useful improvement",
        current="I have something that works",
        urgency="Social pressure or FOMO",
    ))

    assert replacement.decision == "BUY"
    assert replacement.provisional is True
    assert upgrade.decision == "WAIT"
    assert upgrade.provisional is True


def test_zero_recurring_cost_and_valid_cost_per_use():
    result = evaluate(AssessmentInput(name="Lamp", price=1000, frequency="Daily", lifespan_years=2))
    assert result.recurring_12_month_total == 0
    assert result.estimated_cost_per_use == pytest.approx(1000 / 730, abs=0.0001)


def test_blank_name_is_rejected():
    with pytest.raises(ValueError):
        AssessmentInput(name="   ", price=1)


def test_negative_price_is_rejected():
    with pytest.raises(ValueError):
        AssessmentInput(name="Item", price=-10)


def test_working_alternative_and_promo_urgency_add_caution():
    result = evaluate(AssessmentInput(name="Upgrade", price=1000, current="I have something that works", urgency="A sale or countdown"))
    assert result.recommendation == "Some factors may merit reflection"
    assert any("urgency" in f.lower() for f in result.factors)


def test_wait_explanation_mentions_only_pressure_that_was_stated():
    no_pressure = evaluate(AssessmentInput(
        name="Laptop",
        price=95000,
        current="It works, but has limitations",
        purchase_timing="not_urgent",
        waiting_period="two months",
    ))
    sale_pressure = evaluate(AssessmentInput(
        name="Laptop",
        price=95000,
        current="It works, but has limitations",
        urgency="A sale or countdown",
        purchase_timing="not_urgent",
        waiting_period="two months",
    ))

    assert "external pressure" not in no_pressure.decision_explanation.lower()
    assert "sale deadline" in sale_pressure.decision_explanation.lower()


def test_cost_per_use_is_missing_until_usage_is_supplied():
    result = evaluate(AssessmentInput(name="Item", price=8000))
    assert result.estimated_cost_per_use is None
    assert any("frequency" in item.lower() for item in result.missing_inputs)


def test_zero_usage_or_lifespan_is_rejected():
    with pytest.raises(ValueError):
        AssessmentInput(name="Item", price=10, uses_per_week=0, lifespan_years=2)
    with pytest.raises(ValueError):
        AssessmentInput(name="Item", price=10, uses_per_week=2, lifespan_years=0)


def test_omitted_currency_is_reported_as_a_default():
    result = evaluate(AssessmentInput(name="Item", price=10))
    assert result.defaults_applied == ["currency"]


def test_user_reported_savings_are_not_a_budget_or_verified_affordability():
    result = evaluate(AssessmentInput(
        name="Headphones",
        price=18000,
        savings_adequacy="reported_sufficient",
    ))

    assert result.affordability["budget_provided"] is False
    assert result.affordability["user_affordability"] is None
    assert result.affordability["savings_adequacy"] == "reported_sufficient"
    assert any("self-reported" in item.lower() for item in result.missing_inputs)
    assert result.decision == "WAIT"


def test_urgent_replacement_with_no_selected_model_is_not_model_approval():
    context = ScoringContext(
        name="Headphones",
        need="Solves a real problem",
        current="Mine is broken or unusable",
        urgency="A genuine deadline",
        deadline_consequences=True,
        alternative_available=False,
        purchase_timing="urgent",
        user_affordability="reported_affordable",
    )

    decision, explanation = recommend_purchase(context)
    assert decision == "WAIT"
    assert "time-sensitive" in explanation
    assert "no specific model or price" in explanation


def test_five_uses_weekly_for_two_years_calculates_520_uses():
    result = evaluate(AssessmentInput(
        name="Headphones", price=8000, uses_per_week=5, lifespan_years=2
    ))
    assert result.estimated_total_uses == 520
    assert result.estimated_cost_per_use == pytest.approx(8000 / 520, abs=0.0001)


def test_explicit_stress_and_social_pressure_are_described_without_diagnosis():
    result = evaluate(AssessmentInput(
        name="Headphones",
        price=8000,
        scenario="They are trending and I had a stressful week",
    ))
    assert "social pressure" in result.sentiment["emotional_cues"]
    assert "stress" in result.sentiment["emotional_cues"]
    assert "wellbeing" in result.sentiment["limitation"]


def test_risk_score_is_bounded_and_explained():
    result = evaluate(AssessmentInput(
        name="Headphones",
        price=8000,
        scenario="I feel stressed because they are trending",
        current="I have something that works",
        need="Mostly a want",
        urgency="Social pressure or FOMO",
        budget=100,
    ))
    assert 0 <= result.regret_risk["score"] <= 100
    assert result.regret_risk["factors"]
    assert "not a probability" in result.regret_risk["explanation"]


def test_pressure_and_regret_scores_have_explicit_boundary_behavior():
    neutral = AssessmentInput(name="Item", price=100)
    assert calculate_purchase_pressure(neutral, classify_sentiment(""))["score"] == 0
    assert calculate_purchase_pressure(neutral, classify_sentiment(""))["band"] == "low"

    moderate_pressure = AssessmentInput(name="Item", price=100, urgency="A sale or countdown")
    pressure = calculate_purchase_pressure(moderate_pressure, classify_sentiment(""))
    assert pressure["score"] == 30
    assert pressure["band"] == "moderate"

    high_pressure = AssessmentInput(
        name="Item",
        price=100,
        urgency="A sale or countdown",
        current="Mine is broken or unusable",
        scenario="I feel stressed about the sale",
    )
    pressure = calculate_purchase_pressure(high_pressure, classify_sentiment(high_pressure.scenario))
    assert pressure["score"] == 65
    assert pressure["band"] == "high"

    regret_boundary = AssessmentInput(
        name="Item",
        price=100,
        urgency="Social pressure or FOMO",
        current="I have something that works",
        need="Mostly a want",
    )
    regret = calculate_regret_risk(regret_boundary, classify_sentiment(""))
    assert regret["score"] == 50
    assert regret["band"] == "higher reflection signal"

    capped = AssessmentInput(
        name="Item",
        price=200,
        budget=100,
        urgency="Social pressure or FOMO",
        current="It works, but has limitations",
        need="Mostly a want",
        alternatives_checked=False,
        repair_investigated=False,
        scenario="I feel stressed and emotional",
    )
    regret = calculate_regret_risk(capped, classify_sentiment(capped.scenario))
    assert sum(item["points"] for item in regret["factors"]) == 115
    assert regret["score"] == 100
