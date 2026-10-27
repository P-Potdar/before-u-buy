from __future__ import annotations
from collections import defaultdict
from datetime import datetime

def summarize(decisions: list[dict], watchlist: list[dict]) -> dict:
    planned = sum(float(d.get("price") or 0) for d in decisions if d.get("status") in {"awaiting", "planned", "postponed"})
    actual = sum(float(d.get("price") or 0) for d in decisions if d.get("status") == "completed")
    saved = sum(float(d.get("confirmed_savings") or 0) for d in decisions if d.get("status") in {"skipped", "cancelled"})
    monthly = defaultdict(float); categories = defaultdict(float); outcomes = defaultdict(int); pressure=[]; regret=[]
    monthly_by_currency = defaultdict(lambda: defaultdict(float))
    categories_by_currency = defaultdict(lambda: defaultdict(float))
    totals_by_currency = defaultdict(lambda: {"planned_spending": 0.0, "confirmed_spending": 0.0, "confirmed_savings": 0.0})
    for d in decisions:
        status = d.get("status") or "awaiting"; outcomes[status] += 1
        currency = str(d.get("currency") or "INR").upper()
        if status in {"awaiting", "planned", "postponed"}:
            totals_by_currency[currency]["planned_spending"] += float(d.get("price") or 0)
        elif status == "completed":
            totals_by_currency[currency]["confirmed_spending"] += float(d.get("price") or 0)
        if status in {"skipped", "cancelled"}:
            totals_by_currency[currency]["confirmed_savings"] += float(d.get("confirmed_savings") or 0)
        if status == "completed":
            key = str(d.get("created_at", ""))[:7] or "unknown"
            category = d.get("category") or "Uncategorised"
            amount = float(d.get("price") or 0)
            monthly[key] += amount
            categories[category] += amount
            monthly_by_currency[currency][key] += amount
            categories_by_currency[currency][category] += amount
        payload = d.get("data") or {}
        for target, bucket in (("purchase_pressure", pressure), ("regret_risk", regret)):
            score = payload.get(target, {}).get("score") if isinstance(payload.get(target), dict) else None
            if isinstance(score, (int, float)): bucket.append(score)
    currency_safe = len(totals_by_currency) <= 1
    return {
        "planned_spending": round(planned, 2) if currency_safe else None,
        "confirmed_spending": round(actual, 2) if currency_safe else None,
        "decision_count": len(decisions),
        "outcomes": dict(outcomes),
        "confirmed_savings": round(saved, 2) if currency_safe else None,
        "potential_savings": None,
        "totals_by_currency": {
            code: {key: round(value, 2) for key, value in totals.items()}
            for code, totals in totals_by_currency.items()
        },
        "tracked_products": len(watchlist),
        "awaiting_decision": outcomes.get("awaiting", 0),
        "spending_by_month": dict(sorted(monthly.items())) if currency_safe else {},
        "spending_by_category": dict(categories) if currency_safe else {},
        "spending_by_month_currency": {
            code: dict(sorted(values.items())) for code, values in monthly_by_currency.items()
        },
        "spending_by_category_currency": {
            code: dict(values) for code, values in categories_by_currency.items()
        },
        "average_purchase_pressure": round(sum(pressure) / len(pressure), 1) if pressure else None,
        "average_regret_risk": round(sum(regret) / len(regret), 1) if regret else None,
        "assumptions": [
            "Only completed records count as confirmed spending.",
            "Totals are grouped by currency; unlike currencies are not added together.",
            "Legacy aggregate totals and ungrouped trends are null or empty if decisions use multiple currencies.",
            "Postponed records are not savings.",
            "Savings are user-confirmed amounts on skipped or cancelled records.",
            "Potential savings are not estimated because the journal has no verified counterfactual price.",
        ],
    }
