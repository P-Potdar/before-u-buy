"""Validated, transparent purchase-reflection calculations and decision rules.

Heuristic point rules and recommendation rules are design choices, not a
validated model or a prediction. Regret-risk weights are centralized in
``REGRET_RISK_WEIGHTS`` and shown with their factors in assessment output.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.knowledge import retrieve_relevant_notes

Need = Literal["Essential need", "Solves a real problem", "Useful improvement", "Mostly a want"]
Current = Literal["Mine is broken or unusable", "It works, but has limitations", "I have something that works"]
Urgency = Literal["No deadline", "A genuine deadline", "A sale or countdown", "Social pressure or FOMO"]
Frequency = Literal["Daily", "Several times a week", "Weekly", "Monthly", "A few times a year"]
Currency = Literal["INR", "USD", "GBP", "EUR"]
UserAffordability = Literal["reported_affordable", "reported_not_affordable"]
SavingsAdequacy = Literal["reported_sufficient", "reported_insufficient"]

USES_PER_YEAR: dict[str, int] = {
    "Daily": 365,
    "Several times a week": 156,
    "Weekly": 52,
    "Monthly": 12,
    "A few times a year": 3,
}

ASSUMPTIONS: list[str] = [
    "Cost per use divides the upfront price by the estimated number of uses over the lifespan you entered. When uses per week is supplied, the method assumes 52 weeks per year. Recurring costs are not included in that figure.",
    "Expected uses per year are fixed estimates: Daily 365, Several times a week 156, Weekly 52, Monthly 12, and A few times a year 3.",
    "Recurring cost is a monthly amount multiplied by 12. It is added once when comparing against a budget and is not added again to cost per use.",
    "A budget comparison uses only the budget figure you entered. It does not infer income, debt, bills, or whether savings are adequate.",
    "When recurring cost is left blank, the total treats it as zero because no recurring amount was provided. That is not a measurement of your subscriptions.",
    "The entered budget is compared with the upfront price plus 12 months of recurring cost. It is not a judgment of overall affordability.",
    "Allocated savings are mentioned only when they are below the upfront price. That note has a weight of zero, so it does not change the point score or the recommendation.",
    "Regret-risk reflection weights are source-configured: sale/social urgency 25, working alternative 15, mostly discretionary purpose 10, budget overrun 25, explicit emotional pressure 20, unchecked alternatives 10, and unchecked repair for a limited current item 10. The score is capped at 100.",
    "The same point rules are used for every guided workflow. The workflow name records the question you started with and does not change the score.",
    "The reflective risk indicator is a design heuristic, not a validated prediction of regret.",
]

LIMITATIONS: list[str] = [
    "This is a transparent point heuristic, not a validated financial or psychological model, and it is not financial advice.",
    "It does not predict whether you will regret the purchase.",
    "Actual use, prices, product quality, and future costs may differ from the answers entered here.",
    "The assessment does not research products, reviews, current prices, availability, or citations.",
]

DISCLAIMER = "Decision support based on entered information; not financial advice or a prediction of regret."
RISK_LIMITATION = "Reflective heuristic only; it is not scientifically validated and does not predict actual regret."
REGRET_RISK_WEIGHTS = {
    "sale_social_urgency": 25,
    "working_alternative": 15,
    "mostly_want": 10,
    "budget_overrun": 25,
    "explicit_emotional_pressure": 20,
    "unchecked_alternatives": 10,
    "unchecked_repair": 10,
}

PURCHASE_PRESSURE_WEIGHTS = {
    "urgency_or_fomo": 30,
    "social_pressure": 25,
    "emotional_context": 20,
    "practical_deadline": 10,
    "no_usable_alternative": 15,
}

NEED_RULES: dict[str, tuple[int, str]] = {
    "Essential need": (2, "Addresses an essential need"),
    "Solves a real problem": (1, "Solves a stated problem"),
    "Useful improvement": (0, "Benefit may be mostly discretionary"),
    "Mostly a want": (0, "Purpose categorized as mostly discretionary"),
}
CURRENT_RULES: dict[str, tuple[int, str]] = {
    "Mine is broken or unusable": (2, "Current option is no longer usable"),
    "It works, but has limitations": (0, "Upgrade should justify its incremental benefit"),
    "I have something that works": (-1, "A working alternative already exists"),
}
URGENCY_RULES: dict[str, Optional[tuple[int, str]]] = {
    "No deadline": None,
    "A genuine deadline": (1, "User reported a genuine deadline"),
    "A sale or countdown": (-1, "Promotional or social urgency may be worth questioning"),
    "Social pressure or FOMO": (-1, "Promotional or social urgency may be worth questioning"),
}

DEFAULT_FIELDS = (
    "currency",
    "need",
    "current",
    "urgency",
    "frequency",
    "uses_per_week",
    "lifespan_years",
    "recurring_monthly",
)


class AssessmentError(ValueError):
    """A user-input problem discovered after schema parsing."""


def round_half_up(value: float, places: int) -> float:
    """Round half away from zero using the number's shortest decimal string.

    Python ``str(number)`` and JavaScript ``String(number)`` match for the
    values this app calculates. Both sides quantize that text with half-up.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssessmentError("Cannot round a non-finite number")
    if places < 0:
        raise AssessmentError("Cannot round to a negative number of places")
    quantized = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return float(quantized)


def format_amount(value: float, places: int = 2) -> str:
    quantized = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{quantized:.{places}f}"


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


class AssessmentInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0, le=100_000_000)
    scenario: str = Field(default="", max_length=2000)
    currency: Currency = "INR"
    need: Optional[Need] = None
    current: Optional[Current] = None
    urgency: Optional[Urgency] = None
    frequency: Optional[Frequency] = None
    uses_per_week: Optional[float] = Field(default=None, gt=0, le=500)
    lifespan_years: Optional[float] = Field(default=None, gt=0, le=100)
    budget: Optional[float] = Field(default=None, ge=0, le=100_000_000)
    savings_allocated: Optional[float] = Field(default=None, ge=0, le=100_000_000)
    recurring_monthly: Optional[float] = Field(default=None, ge=0, le=10_000_000)
    monthly_surplus: Optional[float] = Field(default=None, ge=0, le=100_000_000)
    essential_expenses: Optional[float] = Field(default=None, ge=0, le=100_000_000)
    user_affordability: Optional[UserAffordability] = None
    savings_adequacy: Optional[SavingsAdequacy] = None
    essential_expenses_affected: Optional[bool] = None
    savings_affected: Optional[bool] = None
    purchase_timing: Optional[Literal["urgent", "not_urgent"]] = None
    waiting_period: Optional[str] = Field(default=None, max_length=80)
    external_pressure: Optional[bool] = None
    alternatives_checked: Optional[bool] = None
    repair_investigated: Optional[bool] = None
    alternative_available: Optional[bool] = None
    deadline_consequences: Optional[bool] = None
    upcoming_expenses: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Name must be text")
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank")
        return value

    @field_validator("price", "lifespan_years", mode="before")
    @classmethod
    def required_finite(cls, value: object, info) -> Optional[float]:
        if value is None and info.field_name == "lifespan_years":
            return None
        label = "Price" if info.field_name == "price" else "Lifespan"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        return _finite(value, label)

    @field_validator("uses_per_week", "budget", "savings_allocated", "recurring_monthly", mode="before")
    @classmethod
    def optional_finite(cls, value: object, info) -> Optional[float]:
        if value is None:
            return None
        label = str(info.field_name).replace("_", " ")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        return _finite(value, label)

    @field_validator("monthly_surplus", "essential_expenses", mode="before")
    @classmethod
    def optional_financial_finite(cls, value: object, info) -> Optional[float]:
        if value is None:
            return None
        label = str(info.field_name).replace("_", " ")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        return _finite(value, label)


class ScoringContext(BaseModel):
    """Partial, explicitly supplied facts for provisional chat-only scores."""

    name: str | None = None
    scenario: str = ""
    price: float | None = None
    budget: float | None = None
    monthly_surplus: float | None = None
    recurring_monthly: float | None = None
    need: str | None = None
    current: str | None = None
    urgency: str | None = None
    alternatives_checked: bool | None = None
    repair_investigated: bool | None = None
    alternative_available: bool | None = None
    deadline_consequences: bool | None = None
    purchase_timing: str | None = None
    waiting_period: str | None = None
    external_pressure: bool | None = None
    user_affordability: UserAffordability | None = None
    savings_adequacy: SavingsAdequacy | None = None
    essential_expenses_affected: bool | None = None
    savings_affected: bool | None = None
    upcoming_expenses: bool | None = None


class AssessmentOutput(BaseModel):
    name: str
    currency: str
    price: float
    recommendation: str
    decision: Literal["BUY", "WAIT", "DON'T BUY"]
    decision_explanation: str
    provisional: bool
    confidence: dict[str, str]
    summary: str
    factors: list[str]
    factor_weights: list[int]
    next_step: str
    estimated_cost_per_use: Optional[float]
    estimated_total_uses: Optional[float]
    recurring_12_month_total: float
    recurring_monthly_provided: bool
    budget_exceeded_by: Optional[float]
    affordability: dict[str, object]
    sentiment: dict[str, object]
    regret_risk: dict[str, object]
    purchase_pressure: dict[str, object]
    missing_inputs: list[str]
    knowledge_sources: list[dict[str, object]]
    tools_used: list[str]
    processing_ms: float = 0.0
    heuristic_points: int
    defaults_applied: list[str]
    assumptions: list[str]
    limitations: list[str]
    disclaimer: str = DISCLAIMER


def recommend_purchase(
    data: AssessmentInput | ScoringContext,
    *,
    has_conflicts: bool = False,
) -> tuple[Literal["BUY", "WAIT", "DON'T BUY"], str]:
    """Return a cautious rule-based recommendation from explicit decision facts."""
    price = data.price
    budget = data.budget
    recurring_total = (data.recurring_monthly or 0) * 12
    over_budget = (
        isinstance(price, (int, float))
        and isinstance(budget, (int, float))
        and price + recurring_total > budget
    )
    affordability_unknown = (
        budget is None
        and getattr(data, "monthly_surplus", None) is None
    )
    urgent_replacement = (
        data.need in {"Essential need", "Solves a real problem"}
        and data.current == "Mine is broken or unusable"
        and (data.urgency == "A genuine deadline" or getattr(data, "purchase_timing", None) == "urgent")
        and data.deadline_consequences is True
        and data.alternative_available is False
    )

    if has_conflicts:
        return "WAIT", "Clarify the conflicting details before choosing this specific purchase."
    if price is None:
        if urgent_replacement:
            return "WAIT", "The replacement need is time-sensitive, so identify a suitable option promptly; no specific model or price was provided to assess."
        return "WAIT", "Clarify the purchase price before deciding whether this specific option fits."
    if over_budget:
        return "DON'T BUY", "The supplied total exceeds the budget figure you entered."
    if (
        getattr(data, "purchase_timing", None) == "not_urgent"
        and data.current in {"I have something that works", "It works, but has limitations"}
    ):
        waiting_period = getattr(data, "waiting_period", None)
        wait_detail = f" You said you can wait {waiting_period}." if isinstance(waiting_period, str) else " You said the purchase is not urgent."
        if data.urgency == "A sale or countdown":
            pressure_detail = " The sale deadline should not determine the decision."
            if getattr(data, "upcoming_expenses", None) is True and affordability_unknown:
                pressure_detail = " The sale deadline should not determine the decision, especially with upcoming expenses and uncertain affordability."
        elif data.urgency == "Social pressure or FOMO":
            pressure_detail = " External pressure alone does not establish a need to buy now."
        else:
            pressure_detail = ""
        return "WAIT", "Your current item still works." + wait_detail + pressure_detail + " Compare the specific improvement and suitable alternatives before choosing."
    if urgent_replacement and (data.alternatives_checked is not True or affordability_unknown):
        return "WAIT", "The replacement need is time-sensitive; compare suitable models and the stated financial details promptly before choosing this specific option."
    if data.urgency in {"A sale or countdown", "Social pressure or FOMO"}:
        if getattr(data, "upcoming_expenses", None) is True and affordability_unknown:
            return "WAIT", "Upcoming expenses and uncertain affordability make it important to separate the sale deadline from the product's practical value before deciding."
        return "WAIT", "Separate external pressure from the product's practical value before deciding."
    if (
        data.need == "Mostly a want"
        and data.current in {"I have something that works", "It works, but has limitations"}
    ):
        return "WAIT", "An optional upgrade with a current alternative has no established need to buy now."
    if (
        data.repair_investigated is not True
        and data.current == "It works, but has limitations"
    ):
        return "WAIT", "Check repair feasibility and cost before deciding whether replacement is useful."
    if (
        data.alternatives_checked is not True
        or affordability_unknown
    ):
        unresolved = []
        if data.alternatives_checked is not True:
            unresolved.append("suitable alternatives")
        if affordability_unknown:
            unresolved.append("whether the price fits a budget you choose")
        return "WAIT", "Check " + " and ".join(unresolved) + " before deciding on this specific product."
    if (
        data.need in {"Essential need", "Solves a real problem"}
        and data.current == "Mine is broken or unusable"
        and data.urgency == "A genuine deadline"
        and data.deadline_consequences is True
        and data.alternative_available is False
    ):
        return "BUY", "A stated practical need, real deadline, unusable current item, and no usable alternative are supported by the supplied facts."
    return "WAIT", "The available facts do not yet establish that this specific purchase is suitable now."


def decision_confidence(
    data: AssessmentInput | ScoringContext,
    *,
    has_conflicts: bool = False,
) -> dict[str, str]:
    """Describe decision-relevant evidence completeness, not predictive certainty."""
    unresolved = []
    if has_conflicts:
        unresolved.append("conflicting facts")
    if data.price is None:
        unresolved.append("purchase price")
    if data.need is None:
        unresolved.append("purchase purpose")
    if data.current is None:
        unresolved.append("current-item condition")
    if data.budget is None and getattr(data, "monthly_surplus", None) is None:
        if getattr(data, "user_affordability", None) is not None or getattr(data, "savings_adequacy", None) is not None:
            unresolved.append("quantitative financial comparison")
        else:
            unresolved.append("financial comparison")
    if data.alternatives_checked is not True:
        unresolved.append("alternatives")
    if data.urgency is None and getattr(data, "purchase_timing", None) is None:
        unresolved.append("timing")
    if data.current == "It works, but has limitations" and data.repair_investigated is not True:
        unresolved.append("repair feasibility")
    if unresolved:
        return {
            "level": "low",
            "basis": "Decision-relevant information remains unresolved: "
            + ", ".join(unresolved)
            + ". This is evidence completeness, not probability or validated confidence.",
        }
    return {
        "level": "moderate",
        "basis": "The key decision facts were provided; this heuristic is still not scientifically validated or predictive.",
    }


def evaluate(data: AssessmentInput) -> AssessmentOutput:
    points = 0
    factors: list[str] = []
    weights: list[int] = []

    def add(weight: int, label: str) -> None:
        nonlocal points
        points += weight
        factors.append(label)
        weights.append(weight)

    if data.need is not None:
        need_weight, need_label = NEED_RULES[data.need]
        add(need_weight, need_label)
    if data.current is not None:
        current_weight, current_label = CURRENT_RULES[data.current]
        add(current_weight, current_label)
    if data.urgency is not None:
        urgency_rule = URGENCY_RULES[data.urgency]
        if urgency_rule is not None:
            add(urgency_rule[0], urgency_rule[1])

    recurring_provided = "recurring_monthly" in data.model_fields_set and data.recurring_monthly is not None
    recurring = 0.0 if data.recurring_monthly is None else data.recurring_monthly
    recurring_total = round_half_up(recurring * 12, 2)
    upfront = round_half_up(data.price, 2)
    compared_total = round_half_up(upfront + recurring_total, 2)

    exceeded: Optional[float]
    if data.budget is None:
        exceeded = None
    else:
        gap = round_half_up(compared_total - round_half_up(data.budget, 2), 2)
        exceeded = gap if gap > 0 else 0.0
        if exceeded == 0:
            add(1, "Estimated cost fits the entered budget")
        else:
            add(-2, "Upfront price plus 12 months of recurring costs exceeds the entered budget")

    if data.savings_allocated is not None and data.savings_allocated < data.price:
        difference = data.price - data.savings_allocated
        shown = round_half_up(difference, 2)
        if shown == 0:
            label = "Allocated savings are below the upfront price by less than 0.01"
        else:
            label = f"Allocated savings are {format_amount(difference)} below the upfront price"
        add(0, label)

    total_uses: Optional[float] = None
    cost_per_use: Optional[float] = None
    if (data.uses_per_week is not None or data.frequency is not None) and data.lifespan_years is not None:
        uses_per_year = data.uses_per_week * 52 if data.uses_per_week is not None else USES_PER_YEAR[data.frequency]
        total_uses = uses_per_year * data.lifespan_years
        if not math.isfinite(total_uses) or total_uses <= 0:
            raise AssessmentError("Lifespan and frequency do not produce a usable number of uses")
        cost_per_use = round_half_up(data.price / total_uses, 4)

    statement = f"{data.scenario} {data.name}".strip()
    sentiment = classify_sentiment(statement)
    regret_risk = calculate_regret_risk(data, sentiment)
    purchase_pressure = calculate_purchase_pressure(data, sentiment)
    affordability: dict[str, object] = {
        "budget_provided": data.budget is not None,
        "budget_comparison_total": compared_total,
        "budget_exceeded_by": exceeded,
        "monthly_surplus_provided": data.monthly_surplus is not None,
        "months_of_stated_surplus": (
            round_half_up(data.price / data.monthly_surplus, 2)
            if data.monthly_surplus is not None and data.monthly_surplus > 0
            else None
        ),
        "savings_provided": data.savings_allocated is not None,
        "essential_expenses_provided": data.essential_expenses is not None,
        "user_affordability": data.user_affordability,
        "savings_adequacy": data.savings_adequacy,
        "essential_expenses_affected": data.essential_expenses_affected,
        "savings_affected": data.savings_affected,
        "upcoming_expenses": data.upcoming_expenses,
        "note": "Affordability and savings claims are user-reported, not independently verified; this is not a full affordability assessment.",
    }
    missing_inputs = []
    if data.budget is None and data.monthly_surplus is None:
        if data.user_affordability is None and data.savings_adequacy is None:
            missing_inputs.append("Affordability is unknown: a budget or monthly surplus was not provided, so no affordability conclusion is available.")
        else:
            missing_inputs.append("Only self-reported affordability or savings information is available; no numeric budget comparison was provided.")
    if (data.frequency is None and data.uses_per_week is None) or data.lifespan_years is None:
        missing_inputs.append("Usage frequency and expected lifespan are needed for cost per use.")
    if data.need is None:
        missing_inputs.append("The purpose or need for the purchase was not provided.")
    if data.current is None:
        missing_inputs.append("Information about the current alternative was not provided.")
    if data.urgency is None and data.purchase_timing is None:
        missing_inputs.append("Purchase timing and consequences of waiting were not provided.")
    if data.alternatives_checked is not True:
        missing_inputs.append("Suitable or lower-cost alternatives have not been confirmed.")
    if data.repair_investigated is not True and data.current == "It works, but has limitations":
        missing_inputs.append("Repair feasibility and cost have not been checked.")
    if data.current == "Mine is broken or unusable" and data.alternative_available is not False:
        missing_inputs.append("Availability of another usable alternative was not confirmed.")
    decision, decision_explanation = recommend_purchase(data)
    confidence = decision_confidence(data)

    retrieval_parts = [statement, data.need or "", data.urgency or ""]
    if cost_per_use is not None:
        retrieval_parts.append("cost per use usage frequency lifespan")
    if data.budget is not None or data.monthly_surplus is not None:
        retrieval_parts.append("budget simple budgeting affordability")
    retrieval_query = " ".join(part for part in retrieval_parts if part)
    knowledge_sources = retrieve_relevant_notes(retrieval_query)[:3]

    if exceeded is not None and exceeded > 0:
        recommendation = "Review the entered budget figures"
        summary = "The upfront price plus the recurring costs you entered are above the budget figure you supplied. This comparison does not establish what is affordable for you."
        next_step = "Review the total and the budget figure you chose to compare."
    elif points >= 4:
        recommendation = "Several factors align with the stated need"
        summary = "Several answers align with the purpose you described. The heuristic does not make the decision for you; compare the full cost and alternatives."
        next_step = "Compare the stated benefit, total cost, and available alternatives."
    elif points >= 2:
        recommendation = "Compare the stated benefit and alternatives"
        summary = "Some answers support the stated purpose, while other details may be worth comparing. This is a reflection aid, not a decision instruction."
        next_step = "Name the specific benefit you expect and compare it with alternatives."
    elif points <= -1:
        recommendation = "Some factors may merit reflection"
        summary = "Some answers suggest reviewing the timing or existing alternatives. This does not predict regret or tell you what to decide."
        next_step = "Review the reasons and timing in your own words."
    else:
        recommendation = "More information needed"
        summary = "Some details remain unspecified. You can add context or continue with the information currently available."
        next_step = "Add any missing details that matter to your decision."

    default_fields = [
        name for name in DEFAULT_FIELDS
        if name not in data.model_fields_set and getattr(data, name) is not None
    ]
    tools_used = ["assessment", "sentiment_tone"]
    if knowledge_sources:
        tools_used.append("knowledge_retrieval")
    if total_uses is not None:
        tools_used.append("cost_per_use")
    if data.budget is not None or data.monthly_surplus is not None:
        tools_used.append("affordability_comparison")
    return AssessmentOutput(
        name=data.name,
        currency=data.currency,
        price=data.price,
        recommendation=recommendation,
        decision=decision,
        decision_explanation=decision_explanation,
        provisional=bool(missing_inputs),
        confidence=confidence,
        summary=summary,
        factors=factors,
        factor_weights=weights,
        next_step=next_step,
        estimated_cost_per_use=cost_per_use,
        estimated_total_uses=total_uses,
        recurring_12_month_total=recurring_total,
        recurring_monthly_provided=recurring_provided,
        budget_exceeded_by=exceeded,
        affordability=affordability,
        sentiment=sentiment,
        regret_risk=regret_risk,
        purchase_pressure=purchase_pressure,
        missing_inputs=missing_inputs,
        knowledge_sources=knowledge_sources,
        tools_used=tools_used,
        heuristic_points=points,
        defaults_applied=default_fields,
        assumptions=list(ASSUMPTIONS),
        limitations=list(LIMITATIONS),
    )


def classify_sentiment(text: str) -> dict[str, object]:
    """Return a small lexicon-based signal; it is not a psychological assessment."""
    normalized = re.sub(r"[^a-z0-9\s'-]", " ", text.lower())
    positive = {"excited", "love", "happy", "like", "want", "looking", "enjoy"}
    negative = {"stress", "stressed", "anxious", "worried", "pressure", "regret", "sad", "emotional"}
    positive_hits = sorted(set(normalized.split()) & positive)
    negative_hits = sorted(set(normalized.split()) & negative)
    if positive_hits and negative_hits:
        label = "mixed"
    elif len(positive_hits) > len(negative_hits):
        label = "positive"
    elif len(negative_hits) > len(positive_hits):
        label = "negative"
    else:
        label = "neutral"
    cues = []
    for cue, terms in {
        "excitement": {"excited", "love", "can't wait"},
        "stress": {"stress", "stressed", "anxious", "worried"},
        "emotional context": {"emotional", "hard week", "rough week", "terrible week", "cheer me up"},
        "urgency": {"urgent", "countdown", "limited time", "today only"},
        "social pressure": {"trending", "everyone", "fomo", "influencer", "pressure"},
        "uncertainty": {"unsure", "uncertain", "not sure", "maybe", "still want"},
    }.items():
        if any(term in normalized for term in terms):
            cues.append(cue)
    total_hits = len(positive_hits) + len(negative_hits)
    confidence = "low" if total_hits == 0 else "moderate"
    return {
        "label": label,
        "confidence": confidence,
        "emotional_cues": cues,
        "evidence": sorted(set(positive_hits + negative_hits)),
        "method": "transparent keyword indicators; ambiguous text is neutral",
        "limitation": "This does not infer emotional wellbeing or diagnose a condition.",
    }


def calculate_purchase_pressure(data: AssessmentInput | ScoringContext, sentiment: dict[str, object]) -> dict[str, object]:
    """Separate emotional-pressure reflection from financial regret risk."""
    factors: list[dict[str, object]] = []
    if data.urgency == "A sale or countdown":
        factors.append({"factor": "A sale or countdown creates external timing pressure", "points": PURCHASE_PRESSURE_WEIGHTS["urgency_or_fomo"]})
    elif data.urgency == "Social pressure or FOMO":
        factors.append({"factor": "Social pressure or FOMO was explicitly mentioned", "points": PURCHASE_PRESSURE_WEIGHTS["social_pressure"]})
    elif data.urgency == "A genuine deadline":
        factors.append({"factor": "A practical deadline was explicitly mentioned", "points": PURCHASE_PRESSURE_WEIGHTS["practical_deadline"]})
    if getattr(data, "purchase_timing", None) == "urgent" and data.urgency in {
        "A sale or countdown",
        "Social pressure or FOMO",
    }:
        factors.append({"factor": "A practical replacement deadline was also explicitly mentioned", "points": PURCHASE_PRESSURE_WEIGHTS["practical_deadline"]})
    if getattr(data, "external_pressure", None) is True and data.urgency not in {
        "A sale or countdown",
        "Social pressure or FOMO",
    }:
        factors.append({"factor": "Social or promotional pressure was explicitly mentioned", "points": PURCHASE_PRESSURE_WEIGHTS["social_pressure"]})
    cues = sentiment.get("emotional_cues", [])
    if isinstance(cues, list) and any(cue in {"social pressure", "urgency"} for cue in cues) and data.urgency not in {"A sale or countdown", "Social pressure or FOMO", "A genuine deadline"}:
        factors.append({"factor": "Social or timing pressure appeared in the supplied wording", "points": PURCHASE_PRESSURE_WEIGHTS["social_pressure"]})
    if isinstance(cues, list) and any(cue in {"stress", "emotional context"} for cue in cues):
        factors.append({"factor": "Stress or emotional context was explicitly mentioned", "points": PURCHASE_PRESSURE_WEIGHTS["emotional_context"]})
    if data.alternative_available is False or data.current == "Mine is broken or unusable":
        factors.append({"factor": "No usable alternative was reported", "points": PURCHASE_PRESSURE_WEIGHTS["no_usable_alternative"]})
    score = min(100, sum(int(item["points"]) for item in factors))
    return {
        "score": score,
        "scale": "0–100 reflection scale; not a probability or diagnosis",
        "band": "high" if score >= 60 else "moderate" if score >= 30 else "low",
        "factors": factors,
        "explanation": "This score reflects explicit practical urgency, social or promotional pressure, and stated stress. Missing information is listed separately and does not itself count as purchase pressure.",
        "limitations": "Pressure can be present in a healthy purchase, and a low score does not prove a purchase is suitable.",
    }


def calculate_regret_risk(data: AssessmentInput | ScoringContext, sentiment: dict[str, object]) -> dict[str, object]:
    """Score explicit reflective risk factors on a 0-100 design scale."""
    score = 0
    factors: list[dict[str, object]] = []
    if data.urgency in {"A sale or countdown", "Social pressure or FOMO"} or (
        getattr(data, "external_pressure", None) is True
    ):
        factors.append({"factor": "Sale or social urgency was selected or matched from an explicit cue", "points": REGRET_RISK_WEIGHTS["sale_social_urgency"]})
    if data.current in {"I have something that works", "It works, but has limitations"}:
        factors.append({"factor": "A working current alternative was described", "points": REGRET_RISK_WEIGHTS["working_alternative"]})
    if data.need == "Mostly a want":
        factors.append({"factor": "Purpose was categorized as mostly discretionary", "points": REGRET_RISK_WEIGHTS["mostly_want"]})
    if data.alternatives_checked is False:
        factors.append({"factor": "Suitable or lower-cost alternatives have not been checked", "points": REGRET_RISK_WEIGHTS["unchecked_alternatives"]})
    if data.repair_investigated is False and data.current == "It works, but has limitations":
        factors.append({"factor": "Repair feasibility or cost remains unknown for the limited current item", "points": REGRET_RISK_WEIGHTS["unchecked_repair"]})
    if (
        data.price is not None
        and data.budget is not None
        and data.price + (data.recurring_monthly or 0) * 12 > data.budget
    ):
        factors.append({"factor": "Entered total exceeds the entered budget", "points": REGRET_RISK_WEIGHTS["budget_overrun"]})
    cues = sentiment.get("emotional_cues", [])
    if isinstance(cues, list) and any(cue in {"stress", "emotional context"} for cue in cues):
        factors.append({"factor": "Stress or emotional context was explicitly mentioned", "points": REGRET_RISK_WEIGHTS["explicit_emotional_pressure"]})
    score = min(100, sum(int(item["points"]) for item in factors))
    return {
        "score": score,
        "scale": "0-100 reflective heuristic",
        "band": "higher reflection signal" if score >= 50 else "some reflection signals" if score >= 20 else "few detected signals",
        "factors": factors,
        "limitations": RISK_LIMITATION,
        "explanation": "Points come only from selected answers and explicit text cues; the score is not a probability or prediction.",
    }
