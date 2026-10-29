#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.dense_retrieval import REVIEW_INDEX_PATH, SCENARIO_INDEX_PATH, build_dense_index, review_rows, scenario_rows  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    scenario_records = scenario_rows()
    review_records = review_rows()
    scenario_info = build_dense_index(scenario_records, SCENARIO_INDEX_PATH, text_key="scenario_text")
    review_records = [
        {**row, "retrieval_text": f"{row.get('title', '')}. {row.get('text', '')}"}
        for row in review_records
    ]
    review_info = build_dense_index(review_records, REVIEW_INDEX_PATH, text_key="retrieval_text")
    report = {
        "scenario_index": scenario_info,
        "review_index": review_info,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    destination = REVIEW_INDEX_PATH.with_name("dense_index_build_report.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
