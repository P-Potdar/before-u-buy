#!/usr/bin/env python3
"""Create auditable aspect evidence from acquired reviews without inventing claims."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.public_data import EVIDENCE_PATH, RAW_REVIEWS_PATH, load_jsonl, normalize_text  # noqa: E402

ASPECT_PATTERNS = {
    "battery_life": r"\b(?:battery|charge|charging|last(?:s|ed|ing)?|hours?|capacity)\b",
    "comfort": r"\b(?:comfortable|comfort|fit|fits|heavy|lightweight|ergonomic|ear)\b",
    "durability": r"\b(?:durable|durability|broke|broken|lasted|sturdy|fragile|wear)\b",
    "performance": r"\b(?:performance|speed|fast|slow|works? well|quality|sound|clarity|accurate)\b",
    "compatibility": r"\b(?:compatible|compatibility|fit|fits|connect|connection|port|laptop|iphone|ipad|android)\b",
    "price_value": r"\b(?:price|priced|money|value|worth|cheap|expensive|cost)\b",
    "reliability": r"\b(?:reliable|reliability|worked|work(?:s|ing)?|failed|fail|defective|issue|problem)\b",
    "ease_of_use": r"\b(?:easy|hard|difficult|simple|install|setup|use|used)\b",
    "customer_support": r"\b(?:return|refund|warranty|support|company|seller|replacement)\b",
}


def sentences(text: str) -> list[str]:
    clean = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", clean) if part.strip()]


def extract(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        text = str(row.get("text", ""))
        for sentence in sentences(text):
            aspects = [name for name, pattern in ASPECT_PATTERNS.items() if re.search(pattern, sentence, re.I)]
            if not aspects:
                continue
            uncertain = bool(re.search(r"\b(?:maybe|perhaps|not sure|might|could be|however|but)\b", sentence, re.I))
            for aspect in aspects:
                key = (str(row.get("source_record_id")), aspect, normalize_text(sentence))
                if key in seen:
                    continue
                seen.add(key)
                evidence.append({
                    "evidence_id": f"ev-{len(evidence) + 1:07d}",
                    "source_record_id": row.get("source_record_id"),
                    "product_category": "electronics",
                    "product_identity": row.get("parent_asin") or row.get("asin"),
                    "review_excerpt": sentence[:500],
                    "rating": row.get("rating"),
                    "review_timestamp": row.get("timestamp"),
                    "sentiment": row.get("source_sentiment", "Unknown"),
                    "aspect": aspect,
                    "claim": sentence[:500],
                    "extraction_method": "deterministic aspect lexicon with source sentence span",
                    "confidence": "low" if uncertain else "medium",
                    "validation_status": "source_span_checked",
                    "source_url": row.get("source_url"),
                    "license": row.get("license"),
                    "provenance": "source review text; no generated product claim",
                })
    return evidence


def main() -> None:
    rows = load_jsonl(RAW_REVIEWS_PATH)
    evidence = extract(rows)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evidence), encoding="utf-8")
    aspects: dict[str, int] = {}
    for row in evidence:
        aspects[str(row["aspect"])] = aspects.get(str(row["aspect"]), 0) + 1
    report = {
        "source_records": len(rows),
        "evidence_records": len(evidence),
        "aspects": dict(sorted(aspects.items())),
        "method": "deterministic sentence-level aspect matching; source sentiment retained without relabeling",
        "limitations": ["Aspect lexicon misses implicit aspects and does not establish causality or population rates."],
    }
    report_path = EVIDENCE_PATH.with_name("evidence_quality_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
