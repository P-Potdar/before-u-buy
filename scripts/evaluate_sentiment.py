#!/usr/bin/env python3
"""Evaluate the transparent sentiment keyword classifier against labeled examples."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.assessment import classify_sentiment  # noqa: E402

LABELS = ("positive", "negative", "neutral")


def evaluate(path: Path) -> dict[str, object]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain text and label columns")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text:
                continue
            if label not in LABELS:
                raise ValueError(f"Unsupported label: {label}")
            rows.append((text, label))
    if not rows:
        raise ValueError("No labeled examples found")

    matrix = {expected: Counter() for expected in LABELS}
    for text, expected in rows:
        predicted = str(classify_sentiment(text)["label"])
        matrix[expected][predicted] += 1
    correct = sum(matrix[label][label] for label in LABELS)
    metrics: dict[str, dict[str, float]] = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[other][label] for other in LABELS if other != label)
        false_negative = sum(matrix[label][other] for other in LABELS if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "count": len(rows),
        "distribution": dict(Counter(label for _, label in rows)),
        "accuracy": correct / len(rows),
        "per_class": metrics,
        "confusion_matrix": {label: {pred: matrix[label][pred] for pred in LABELS} for label in LABELS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", type=Path, default=ROOT / "sentiment_labels.csv")
    args = parser.parse_args()
    result = evaluate(args.csv_path)
    print(f"Examples: {result['count']}")
    print(f"Label distribution: {result['distribution']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print("Per-class precision / recall / F1:")
    for label, scores in result["per_class"].items():
        print(f"  {label}: {scores['precision']:.3f} / {scores['recall']:.3f} / {scores['f1']:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("          " + " ".join(f"{label:>9}" for label in LABELS))
    for label, row in result["confusion_matrix"].items():
        print(f"{label:>9} " + " ".join(f"{row[pred]:9d}" for pred in LABELS))
    print("Limitations: a small manually labeled sample and a lexicon heuristic; metrics do not establish generalization.")


if __name__ == "__main__":
    main()
