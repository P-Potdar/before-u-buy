#!/usr/bin/env python3
"""Generate the reproducible synthetic purchase-decision corpus.

The generator deliberately varies decision-relevant dimensions rather than
copying a small prompt list. Run it from the repository root.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.dataset import DATASET_PATH, DATASET_VERSION

CATEGORIES = [
    ("audio", ["noise-cancelling headphones", "wireless speaker"]),
    ("phones", ["smartphone", "phone with a larger battery"]),
    ("computers", ["work laptop", "desktop computer"]),
    ("gaming", ["gaming laptop", "gaming monitor"]),
    ("tablets", ["tablet", "e-reader"]),
    ("peripherals", ["mechanical keyboard", "ergonomic mouse"]),
    ("appliances", ["washing machine", "vacuum cleaner"]),
    ("kitchen", ["air fryer", "espresso machine"]),
    ("furniture", ["ergonomic chair", "standing desk"]),
    ("photography", ["mirrorless camera", "camera lens"]),
    ("fitness", ["treadmill", "fitness tracker"]),
    ("education", ["online course", "scientific calculator"]),
    ("productivity", ["project-management subscription", "office software subscription"]),
    ("subscriptions", ["music subscription", "cloud-storage plan"]),
    ("travel", ["carry-on suitcase", "travel backpack"]),
    ("work-equipment", ["portable monitor", "label printer"]),
]

CONDITIONS = [
    ("working", "My current item works reliably.", "Optional or preference-driven upgrade"),
    ("limited", "My current item still works, but it has a limitation that affects some tasks.", "Functional upgrade"),
    ("failed", "My current item stopped working completely.", "Essential replacement"),
    ("new_use", "I do not have an existing item that can handle this new use case.", "New use case"),
]
MOTIVATIONS = [
    ("practical", "I want a concrete improvement in reliability or day-to-day usefulness.", ["genuine practical benefit"]),
    ("performance", "I want better performance for a hobby or demanding task.", ["performance improvement", "hobby or entertainment"]),
    ("social", "Friends and online trends have influenced me to consider it.", ["social influence"]),
    ("convenience", "I mainly want greater convenience and comfort.", ["convenience"]),
]
FINANCES = [
    ("sufficient", "I report having enough savings, but I have not provided a numeric budget.", "reported_sufficient", False),
    ("uncertain", "I am unsure whether I can comfortably afford it.", "unknown", None),
    ("commitments", "I have upcoming expenses and this purchase may affect essential commitments.", "uncertain_due_to_commitments", True),
    ("budget_only", "My budget is stated, but I have not confirmed the price of a particular product.", "reported_budget_only", None),
]
TIMING = [
    ("pause", "I can wait two months and there is no practical deadline.", "not_urgent", "two months", None),
    ("short_pause", "I can wait two weeks.", "not_urgent", "two weeks", None),
    ("deadline", "I need a usable replacement for an important commitment tomorrow.", "urgent", None, True),
    ("sale", "A limited-time sale ends tonight, which makes me feel pressured.", "not_urgent", None, None),
]
ALTERNATIVES = [
    ("unchecked", "I have not compared cheaper, refurbished, or repair alternatives.", False, None),
    ("cheaper", "I compared cheaper models, but have not confirmed whether they meet my needs.", True, None),
    ("refurbished", "I found a refurbished option that may meet the same need.", True, None),
    ("repair", "Repair may be possible, but I have not checked its cost or reliability.", False, "possible"),
]


def recommendation(condition: str, finance: str, timing: str, alternatives: str, price: int) -> tuple[str, str, list[str]]:
    if finance == "commitments":
        return "WAIT", "Upcoming commitments make the purchase timing and affordability unresolved.", ["Whether essential commitments can be covered", "Whether waiting changes the need"]
    if finance == "uncertain":
        return "WAIT", "Affordability is explicitly uncertain, so the purchase should be evaluated after the financial position is clearer.", ["A comfortable spending amount", "Upcoming obligations"]
    if condition == "failed" and timing == "deadline" and alternatives == "unchecked":
        return "WAIT", "Replacement is practically urgent, but urgency does not establish that a particular model or full budget is good value.", ["A reliable suitable replacement", "Whether a cheaper model meets the need"]
    if condition in {"working", "limited"} and timing in {"pause", "short_pause", "sale"}:
        return "WAIT", "A working option and time to compare reduce the case for making a pressure-driven purchase now.", ["The measurable improvement", "Cheaper, refurbished, or repair alternatives"]
    if condition == "failed" and timing == "deadline" and alternatives in {"cheaper", "refurbished"}:
        return "BUY", "A replacement is time-sensitive, provided the selected option is reliable and suitable.", ["Compatibility and total cost"]
    if price > 80000 and finance == "budget_only":
        return "DON'T BUY", "The product price is not established against the stated budget, so this specific option is not supported by the evidence.", ["Confirmed product price", "Total cost"]
    return "WAIT", "The available facts support further comparison before committing to this specific purchase.", ["Alternatives and total cost"]


def make_record(number: int, category_index: int, product_index: int, condition_index: int, motivation_index: int, finance_index: int, timing_index: int, alternatives_index: int) -> dict:
    category, products = CATEGORIES[category_index]
    product = products[product_index]
    condition_key, condition_text, need_class = CONDITIONS[condition_index]
    motivation_key, motivation_text, motivations = MOTIVATIONS[motivation_index]
    finance_key, finance_text, savings, expenses_affected = FINANCES[finance_index]
    timing_key, timing_text, purchase_timing, waiting_period, deadline = TIMING[timing_index]
    alternatives_key, alternatives_text, alternatives_checked, repair = ALTERNATIVES[alternatives_index]
    price = 18000 + category_index * 4100 + product_index * 2300 + condition_index * 1700 + motivation_index * 900
    action, rationale, unknown = recommendation(condition_key, finance_key, timing_key, alternatives_key, price)
    facts = {
        "product": product,
        "product_category": category,
        "condition": condition_key,
        "motivation_profile": motivation_key,
        "finance_profile": finance_key,
        "timing_profile": timing_key,
        "alternatives_profile": alternatives_key,
        "product_price": price if finance_key != "budget_only" else None,
        "user_budget": price if finance_key == "budget_only" else None,
        "affordability_claim": "reported_affordable" if finance_key == "sufficient" else None,
        "savings_adequacy": savings,
        "essential_expenses_affected": expenses_affected,
        "purchase_timing": purchase_timing,
        "waiting_period": waiting_period,
        "external_pressure": timing_key == "sale" or motivation_key == "social",
        "alternatives_checked": alternatives_checked,
        "repair_feasibility": repair,
        "use_case": "daily work, study, or recreation depending on the product",
    }
    provenance = {
        key: "explicitly_stated_in_synthetic_scenario" if value is not None else "unknown_not_provided"
        for key, value in facts.items()
    }
    completeness = "high" if finance_key in {"sufficient", "budget_only"} and alternatives_checked else "medium" if condition_key != "new_use" else "low"
    if finance_key in {"uncertain", "commitments"} or not alternatives_checked:
        completeness = "low"
    text = (
        f"I am considering a {product} in the {category} category for about ₹{price:,}. "
        f"{condition_text} {motivation_text} {finance_text} {timing_text} {alternatives_text} "
        "Help me decide whether to buy now, wait, or choose another way to meet the need."
    )
    return {
        "scenario_id": f"synthetic-v2-{number:04d}",
        "dataset_version": DATASET_VERSION,
        "scenario_text": text,
        "product_category": category,
        "product_name_or_type": product,
        "facts": facts,
        "fact_provenance": provenance,
        "need_classification": need_class,
        "motivations": motivations,
        "product_condition": condition_key,
        "purchase_price": facts["product_price"],
        "user_budget": facts["user_budget"],
        "affordability_status": "reported_sufficient" if finance_key == "sufficient" else "unknown" if finance_key == "uncertain" else "uncertain_due_to_commitments" if finance_key == "commitments" else "budget_only",
        "savings_adequacy": savings,
        "essential_expenses_affected": expenses_affected,
        "purchase_urgency": "practical_deadline" if timing_key == "deadline" else "external_pressure" if timing_key == "sale" else "none",
        "waiting_period": waiting_period,
        "external_pressure": timing_key == "sale" or motivation_key == "social",
        "alternatives_status": alternatives_key,
        "repair_feasibility": repair,
        "use_case": facts["use_case"],
        "evidence_completeness": completeness,
        "recommended_action": action,
        "rationale": rationale,
        "counterarguments": ["The product's actual quality and compatibility have not been independently verified."],
        "unknown_information": unknown,
        "suggested_follow_up_questions": ["What specific result must the purchase deliver, and which lower-cost or repair option can meet it?"],
        "record_origin": "synthetic",
        "validation_status": "generated_consistency_checked",
    }


def main() -> None:
    records = []
    number = 1
    for category_index, product_index, condition_index, motivation_index, finance_index in product(
        range(len(CATEGORIES)), range(2), range(4), range(4), range(4)
    ):
        timing_index = (category_index + product_index + condition_index + motivation_index) % len(TIMING)
        alternatives_index = (category_index + product_index + condition_index + motivation_index + finance_index) % len(ALTERNATIVES)
        records.append(make_record(number, category_index, product_index, condition_index, motivation_index, finance_index, timing_index, alternatives_index))
        number += 1
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} records to {DATASET_PATH}")


if __name__ == "__main__":
    main()
