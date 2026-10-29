#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.public_data import EVIDENCE_PATH, RAW_REVIEWS_PATH, PUBLIC_DATA_DIR, validate_reviews  # noqa: E402

from backend.public_data import load_jsonl  # noqa: E402


def main() -> None:
    rows = load_jsonl(RAW_REVIEWS_PATH)
    report = validate_reviews(rows)
    evidence = load_jsonl(EVIDENCE_PATH)
    evidence_failures = []
    for number, row in enumerate(evidence, 1):
        for field in ("evidence_id", "source_record_id", "review_excerpt", "claim", "aspect", "source_url", "license"):
            if not row.get(field):
                evidence_failures.append({"record": number, "error": f"evidence missing {field}"})
    report["evidence_record_count"] = len(evidence)
    report["evidence_validation_failures"] = evidence_failures
    report["valid"] = report["valid"] and not evidence_failures
    destination = PUBLIC_DATA_DIR / "raw_quality_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
