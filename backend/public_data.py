"""Schemas and provenance helpers for acquired public review data.

Public reviews are intentionally kept separate from synthetic decision scenarios.
The source labels are source-provided annotations, not ground truth about a
product or a user's purchase suitability.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA_DIR = ROOT / "backend" / "data" / "public_reviews"
RAW_REVIEWS_PATH = PUBLIC_DATA_DIR / "raw_electronics_reviews.jsonl"
EVIDENCE_PATH = PUBLIC_DATA_DIR / "curated_evidence.jsonl"
MANIFEST_PATH = PUBLIC_DATA_DIR / "source_manifest.json"
MODEL_PATH = PUBLIC_DATA_DIR / "review_sentiment_model.json"

SENTIMENTS = {"Positive", "Negative", "Neutral", "Unknown"}
ASPECTS = {
    "battery_life", "comfort", "durability", "performance", "compatibility",
    "price_value", "reliability", "ease_of_use", "customer_support", "other",
}


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def review_fingerprint(title: str, text: str) -> str:
    return hashlib.sha256(normalize_text(f"{title} {text}").encode()).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {source}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {source}:{number}")
        rows.append(row)
    return rows


def validate_reviews(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    sentiment_counts: dict[str, int] = {}
    aspect_counts: dict[str, int] = {}
    records = list(rows)
    for number, row in enumerate(records, 1):
        identifier = row.get("source_record_id")
        if not isinstance(identifier, str) or not identifier:
            failures.append({"record": number, "error": "missing source_record_id"})
        elif identifier in ids:
            failures.append({"record": number, "error": "duplicate source_record_id"})
        ids.add(str(identifier))
        title = row.get("title", "")
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            failures.append({"record": number, "error": "missing review text"})
        fingerprint = review_fingerprint(str(title), str(text))
        if fingerprint in fingerprints:
            failures.append({"record": number, "error": "duplicate review text"})
        fingerprints.add(fingerprint)
        sentiment = row.get("source_sentiment", "Unknown")
        if sentiment not in SENTIMENTS:
            failures.append({"record": number, "error": "invalid source sentiment"})
        sentiment_counts[str(sentiment)] = sentiment_counts.get(str(sentiment), 0) + 1
        if row.get("source_url") is None or row.get("license") is None:
            failures.append({"record": number, "error": "missing provenance"})
        for aspect in row.get("aspects", []):
            if aspect not in ASPECTS:
                failures.append({"record": number, "error": "invalid aspect", "aspect": aspect})
            aspect_counts[str(aspect)] = aspect_counts.get(str(aspect), 0) + 1
    return {
        "record_count": len(records),
        "unique_id_count": len(ids),
        "unique_text_count": len(fingerprints),
        "sentiments": dict(sorted(sentiment_counts.items())),
        "aspects": dict(sorted(aspect_counts.items())),
        "validation_failures": failures,
        "valid": not failures,
    }
