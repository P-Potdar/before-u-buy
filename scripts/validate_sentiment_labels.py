#!/usr/bin/env python3
"""Validate sentiment label schemas without claiming label correctness."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLARITIES = {"positive", "negative", "neutral", "mixed", "uncertain"}
SIGNALS = {
    "purchase_intent", "purchase_hesitation", "fomo", "social_pressure", "sales_pressure",
    "urgency", "financial_stress", "affordability_concern", "need_urgency",
    "impulse_buying_risk", "regret_concern", "product_dissatisfaction", "replacement_need",
    "upgrade_desire", "price_sensitivity", "comparison_intent",
}


def main() -> None:
    csv_path = ROOT / "sentiment_labels.csv"
    context_path = ROOT / "sentiment_context_labels.jsonl"
    failures: list[str] = []
    original_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    if not {"case_id", "text", "label"}.issubset(original_rows[0]):
        failures.append("original CSV is missing required columns")
    original_labels = Counter(row.get("label", "") for row in original_rows)
    context_rows = [json.loads(line) for line in context_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids: set[str] = set()
    context_labels = Counter()
    for row in context_rows:
        identifier = row.get("case_id")
        if not isinstance(identifier, str) or identifier in ids:
            failures.append("duplicate or missing context case_id")
        ids.add(str(identifier))
        if row.get("polarity") not in POLARITIES:
            failures.append(f"unsupported polarity for {identifier}")
        if row.get("verification_status") not in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}:
            failures.append(f"missing verification status for {identifier}")
        unknown = set(row.get("signals", [])) - SIGNALS
        if unknown:
            failures.append(f"unsupported signals for {identifier}: {sorted(unknown)}")
        context_labels[str(row.get("polarity"))] += 1
    report = {
        "original_file": {"path": str(csv_path), "records": len(original_rows), "polarity_counts": dict(original_labels), "status": "PARTIALLY_VERIFIED", "reason": "schema is valid but no independent annotation audit is stored"},
        "context_file": {"path": str(context_path), "records": len(context_rows), "polarity_counts": dict(context_labels), "status": "UNVERIFIED", "reason": "project-authored synthetic examples"},
        "failures": failures,
        "valid_schema": not failures,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
