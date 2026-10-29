#!/usr/bin/env python3
"""Evaluate the transparent sentiment keyword classifier against labeled examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.assessment import classify_sentiment  # noqa: E402
from backend.sentiment import analyze_message  # noqa: E402

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


def evaluate_context(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No context examples found")
    polarity_labels = ("positive", "negative", "neutral", "mixed", "uncertain")
    polarity_matrix = {label: Counter() for label in polarity_labels}
    signal_counts: Counter[str] = Counter()
    signal_tp: Counter[str] = Counter()
    signal_fp: Counter[str] = Counter()
    signal_fn: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    emotion_tp: Counter[str] = Counter()
    emotion_fp: Counter[str] = Counter()
    emotion_fn: Counter[str] = Counter()
    for row in rows:
        result = analyze_message(str(row["text"]))
        expected_polarity = str(row["polarity"])
        polarity_matrix[expected_polarity][result["polarity"]["label"]] += 1
        expected_signals = set(row.get("signals", []))
        predicted_signals = {key for key, value in result["purchase_signals"].items() if value["value"] is True}
        for label in expected_signals | predicted_signals:
            signal_counts[label] += 1
            signal_tp[label] += int(label in expected_signals and label in predicted_signals)
            signal_fp[label] += int(label not in expected_signals and label in predicted_signals)
            signal_fn[label] += int(label in expected_signals and label not in predicted_signals)
        expected_emotions = set(row.get("emotions", []))
        predicted_emotions = {item["label"] for item in result["emotions"]}
        for label in expected_emotions | predicted_emotions:
            emotion_counts[label] += 1
            emotion_tp[label] += int(label in expected_emotions and label in predicted_emotions)
            emotion_fp[label] += int(label not in expected_emotions and label in predicted_emotions)
            emotion_fn[label] += int(label in expected_emotions and label not in predicted_emotions)

    def multilabel(counter: Counter[str], tp: Counter[str], fp: Counter[str], fn: Counter[str]) -> dict[str, object]:
        per_label = {}
        for label in sorted(counter):
            precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
            recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
            per_label[label] = {
                "examples_with_label": counter[label], "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        f1_values = [item["f1"] for item in per_label.values()]
        return {"per_label": per_label, "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0}

    return {
        "examples": len(rows),
        "provenance": "project-authored synthetic examples; UNVERIFIED; not human validation",
        "polarity": {
            "accuracy": sum(polarity_matrix[label][label] for label in polarity_labels) / len(rows),
            "per_label": {
                label: {
                    "support": sum(polarity_matrix[label].values()),
                    "precision": polarity_matrix[label][label] / max(1, sum(row[label] for row in polarity_matrix.values())),
                    "recall": polarity_matrix[label][label] / max(1, sum(polarity_matrix[label].values())),
                    "f1": (
                        2 * (polarity_matrix[label][label] / max(1, sum(row[label] for row in polarity_matrix.values())))
                        * (polarity_matrix[label][label] / max(1, sum(polarity_matrix[label].values())))
                        / ((polarity_matrix[label][label] / max(1, sum(row[label] for row in polarity_matrix.values())))
                           + (polarity_matrix[label][label] / max(1, sum(polarity_matrix[label].values()))))
                        if polarity_matrix[label][label] else 0.0
                    ),
                }
                for label in polarity_labels
            },
            "confusion_matrix": {label: dict(polarity_matrix[label]) for label in polarity_labels},
        },
        "purchase_signals": multilabel(signal_counts, signal_tp, signal_fp, signal_fn),
        "emotions": multilabel(emotion_counts, emotion_tp, emotion_fp, emotion_fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", type=Path, default=ROOT / "sentiment_labels.csv")
    parser.add_argument("--context-path", type=Path, default=ROOT / "sentiment_context_labels.jsonl")
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
    context = evaluate_context(args.context_path)
    print("\nStructured context evaluation (synthetic, UNVERIFIED):")
    print(json.dumps(context, indent=2, ensure_ascii=False))
    print("Limitations: original labels have no independent annotation audit; context labels are project-authored synthetic examples; metrics do not establish generalization.")


if __name__ == "__main__":
    main()
