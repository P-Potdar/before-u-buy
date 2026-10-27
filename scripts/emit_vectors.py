#!/usr/bin/env python3
"""Write shared/assessment_vectors.json from the canonical Python engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.assessment import (  # noqa: E402
    ASSUMPTIONS,
    DISCLAIMER,
    LIMITATIONS,
    AssessmentInput,
    evaluate,
)

CASES: list[dict] = [
    {"id": "budget-overrun", "input": {"name": "Headphones", "price": 20000, "budget": 15000, "recurring_monthly": 500}},
    {"id": "budget-equal", "input": {"name": "Lamp", "price": 100, "budget": 100, "recurring_monthly": 0}},
    {"id": "budget-under-by-recurring", "input": {"name": "Membership", "price": 100, "budget": 220, "recurring_monthly": 10}},
    {"id": "budget-just-over", "input": {"name": "Membership", "price": 100, "budget": 219.99, "recurring_monthly": 10}},
    {"id": "zero-budget", "input": {"name": "Snack", "price": 5, "budget": 0, "recurring_monthly": 0}},
    {"id": "omitted-recurring", "input": {"name": "Lamp", "price": 1000, "frequency": "Daily", "lifespan_years": 2}},
    {"id": "explicit-zero-recurring", "input": {"name": "Lamp", "price": 1000, "frequency": "Daily", "lifespan_years": 2, "recurring_monthly": 0}},
    {"id": "strong-case", "input": {"name": "Washer", "price": 1000, "need": "Essential need", "current": "Mine is broken or unusable", "urgency": "A genuine deadline", "budget": 5000, "recurring_monthly": 0}},
    {"id": "points-four-boundary", "input": {"name": "Tool", "price": 80, "need": "Essential need", "current": "Mine is broken or unusable"}},
    {"id": "worth-comparing", "input": {"name": "Desk", "price": 200, "need": "Essential need", "current": "It works, but has limitations"}},
    {"id": "points-three", "input": {"name": "Chair", "price": 150, "need": "Solves a real problem", "current": "Mine is broken or unusable"}},
    {"id": "more-information", "input": {"name": "Notebook", "price": 12, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "points-one", "input": {"name": "Adapter", "price": 15, "need": "Solves a real problem", "current": "It works, but has limitations"}},
    {"id": "consider-waiting", "input": {"name": "Upgrade", "price": 1000, "current": "I have something that works", "urgency": "A sale or countdown"}},
    {"id": "social-pressure", "input": {"name": "Sneakers", "price": 90, "need": "Mostly a want", "current": "I have something that works", "urgency": "Social pressure or FOMO"}},
    {"id": "no-deadline", "input": {"name": "Kettle", "price": 40, "urgency": "No deadline", "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "genuine-deadline", "input": {"name": "Passport photo", "price": 20, "need": "Essential need", "current": "It works, but has limitations", "urgency": "A genuine deadline"}},
    {"id": "savings-below", "input": {"name": "Bike", "price": 500, "savings_allocated": 125, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "savings-equal", "input": {"name": "Bike", "price": 500, "savings_allocated": 500, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "savings-above", "input": {"name": "Bike", "price": 500, "savings_allocated": 800, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "savings-zero", "input": {"name": "Bike", "price": 100, "savings_allocated": 0, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "savings-tiny-gap", "input": {"name": "Bike", "price": 100, "savings_allocated": 99.996, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "frequency-daily", "input": {"name": "Item", "price": 120, "frequency": "Daily", "lifespan_years": 1, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "frequency-several", "input": {"name": "Item", "price": 120, "frequency": "Several times a week", "lifespan_years": 1, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "frequency-weekly", "input": {"name": "Item", "price": 120, "frequency": "Weekly", "lifespan_years": 1, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "frequency-monthly", "input": {"name": "Item", "price": 120, "frequency": "Monthly", "lifespan_years": 1, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "frequency-rare", "input": {"name": "Item", "price": 120, "frequency": "A few times a year", "lifespan_years": 1, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "lifespan-max", "input": {"name": "Item", "price": 365, "frequency": "Daily", "lifespan_years": 100, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "lifespan-half", "input": {"name": "Item", "price": 3, "frequency": "A few times a year", "lifespan_years": 0.5, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "currency-inr", "input": {"name": "Item", "price": 10, "currency": "INR", "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "currency-usd", "input": {"name": "Item", "price": 10, "currency": "USD", "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "currency-gbp", "input": {"name": "Item", "price": 10, "currency": "GBP", "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "currency-eur", "input": {"name": "Item", "price": 10, "currency": "EUR", "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "want-and-working", "input": {"name": "Gadget", "price": 75, "need": "Mostly a want", "current": "I have something that works"}},
    {"id": "budget-overrides-strong-points", "input": {"name": "Washer", "price": 900, "need": "Essential need", "current": "Mine is broken or unusable", "budget": 100, "recurring_monthly": 0}},
    {"id": "small-per-use", "input": {"name": "Pen", "price": 1, "frequency": "Daily", "lifespan_years": 100, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "rounding-price", "input": {"name": "Cable", "price": 10.005, "budget": 10, "recurring_monthly": 0, "need": "Useful improvement", "current": "It works, but has limitations"}},
    {"id": "explicit-enums-no-defaults", "input": {"name": "Camera", "price": 300, "currency": "USD", "need": "Solves a real problem", "current": "It works, but has limitations", "urgency": "No deadline", "frequency": "Monthly", "lifespan_years": 4, "recurring_monthly": 2}},
]


def main() -> None:
    cases = []
    for case in CASES:
        result = evaluate(AssessmentInput(**case["input"]))
        payload = result.model_dump()
        expect = {
            key: payload[key]
            for key in (
                "name",
                "currency",
                "price",
                "recommendation",
                "summary",
                "factors",
                "factor_weights",
                "next_step",
                "estimated_cost_per_use",
                "recurring_12_month_total",
                "recurring_monthly_provided",
                "budget_exceeded_by",
                "heuristic_points",
                "defaults_applied",
            )
        }
        cases.append({"id": case["id"], "input": case["input"], "expect": expect})
    document = {
        "version": 1,
        "assumptions": ASSUMPTIONS,
        "limitations": LIMITATIONS,
        "disclaimer": DISCLAIMER,
        "cases": cases,
    }
    destination = ROOT / "shared" / "assessment_vectors.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {destination}")


if __name__ == "__main__":
    main()
