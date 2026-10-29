"""Small local multinomial Naive Bayes model for source-review sentiment QA."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.casefold())


def product_split(row: dict[str, Any]) -> str:
    key = str(row.get("parent_asin") or row.get("asin") or row.get("source_record_id"))
    return "test" if int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


def fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(row.get("source_sentiment", "Unknown")) for row in rows})
    vocabulary: set[str] = set()
    class_counts: Counter[str] = Counter()
    word_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
    token_totals: Counter[str] = Counter()
    for row in rows:
        label = str(row.get("source_sentiment", "Unknown"))
        words = tokens(f"{row.get('title', '')} {row.get('text', '')}")
        class_counts[label] += 1
        word_counts[label].update(words)
        token_totals[label] += len(words)
        vocabulary.update(words)
    return {
        "model_type": "multinomial_naive_bayes",
        "version": "1",
        "alpha": 1.0,
        "labels": labels,
        "vocabulary": sorted(vocabulary),
        "class_counts": dict(class_counts),
        "token_totals": dict(token_totals),
        "word_counts": {label: dict(counts) for label, counts in word_counts.items()},
        "training_records": len(rows),
    }


def predict(model: dict[str, Any], text: str) -> tuple[str, dict[str, float]]:
    words = tokens(text)
    vocabulary = set(model["vocabulary"])
    total_records = sum(model["class_counts"].values())
    scores: dict[str, float] = {}
    for label in model["labels"]:
        prior = (model["class_counts"][label] + 1) / (total_records + len(model["labels"]))
        denominator = model["token_totals"][label] + model["alpha"] * max(1, len(vocabulary))
        score = math.log(prior)
        for word in words:
            if word in vocabulary:
                score += math.log((model["word_counts"][label].get(word, 0) + model["alpha"]) / denominator)
        scores[label] = score
    maximum = max(scores.values())
    probabilities = {label: math.exp(score - maximum) for label, score in scores.items()}
    normalizer = sum(probabilities.values()) or 1.0
    probabilities = {label: value / normalizer for label, value in probabilities.items()}
    return max(probabilities, key=probabilities.get), probabilities
