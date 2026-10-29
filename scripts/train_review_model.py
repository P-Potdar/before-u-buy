#!/usr/bin/env python3
"""Fit and evaluate a reproducible review-sentiment QA model."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.public_data import MODEL_PATH, RAW_REVIEWS_PATH, load_jsonl  # noqa: E402
from backend.review_model import fit, predict, product_split  # noqa: E402


def main() -> None:
    rows = load_jsonl(RAW_REVIEWS_PATH)
    train = [row for row in rows if product_split(row) == "train"]
    test = [row for row in rows if product_split(row) == "test"]
    model = fit(train)
    majority_label = Counter(str(row.get("source_sentiment", "Unknown")) for row in train).most_common(1)[0][0]
    majority_accuracy = sum(str(row.get("source_sentiment", "Unknown")) == majority_label for row in test) / max(1, len(test))
    confusion: dict[str, Counter[str]] = {}
    correct = 0
    for row in test:
        actual = str(row.get("source_sentiment", "Unknown"))
        predicted, _ = predict(model, f"{row.get('title', '')} {row.get('text', '')}")
        confusion.setdefault(actual, Counter())[predicted] += 1
        correct += predicted == actual
    labels = model["labels"]
    metrics = {}
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(row.get(label, 0) for actual, row in confusion.items() if actual != label)
        fn = sum(value for predicted, value in confusion.get(label, {}).items() if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    report = {
        "model_type": model["model_type"],
        "seed": "sha1(parent_asin) modulo 5; deterministic grouped split",
        "train_records": len(train),
        "test_records": len(test),
        "train_products": len({row.get("parent_asin") for row in train}),
        "test_products": len({row.get("parent_asin") for row in test}),
        "accuracy": correct / max(1, len(test)),
        "majority_baseline": {"label": majority_label, "accuracy": majority_accuracy},
        "per_class": metrics,
        "macro_f1": sum(item["f1"] for item in metrics.values()) / max(1, len(metrics)),
        "confusion_matrix": {actual: dict(values) for actual, values in confusion.items()},
        "limitations": [
            "The source sentiment labels are dataset annotations classified with Mistral, not independent human labels.",
            "This model is QA/supporting classification only; it does not predict purchase suitability or BUY/WAIT/DON'T BUY.",
        ],
    }
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    report_path = MODEL_PATH.with_name("review_model_evaluation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
