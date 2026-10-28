#!/usr/bin/env python3
"""Validate the generated purchase corpus and write a quality report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.dataset import DATASET_PATH, load_scenarios, normalized_text, validate_scenarios


def shingles(text: str, size: int = 3) -> set[str]:
    words = normalized_text(text).split()
    return {" ".join(words[index:index + size]) for index in range(max(0, len(words) - size + 1))}


def near_duplicate_pairs(scenarios, threshold: float = 0.92, limit: int = 100) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    buckets: dict[str, list[int]] = {}
    for index, scenario in enumerate(scenarios):
        key = "|".join(str(scenario.payload.get(field, "")) for field in ("product_category", "product_name_or_type", "product_condition"))
        buckets.setdefault(key, []).append(index)
    for indexes in buckets.values():
        for left_position, left_index in enumerate(indexes):
            left = shingles(scenarios[left_index].text)
            if not left:
                continue
            for right_index in indexes[left_position + 1:]:
                right = shingles(scenarios[right_index].text)
                score = len(left & right) / max(1, len(left | right))
                if score >= threshold:
                    pairs.append({"left": scenarios[left_index].scenario_id, "right": scenarios[right_index].scenario_id, "jaccard": round(score, 4)})
                    if len(pairs) >= limit:
                        return pairs
    return pairs


def main() -> int:
    scenarios = load_scenarios(DATASET_PATH)
    report = validate_scenarios(scenarios)
    pairs = near_duplicate_pairs(scenarios)
    report["near_duplicate_threshold"] = 0.92
    report["near_duplicate_pair_count_capped_at"] = 100
    report["near_duplicate_pairs"] = pairs
    report["near_duplicate_observed_count"] = len(pairs)
    report_path = DATASET_PATH.with_name("purchase_decision_dataset_quality.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "near_duplicate_pairs"}, indent=2))
    print(f"Quality report: {report_path}")
    return 0 if report["valid"] and not pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
