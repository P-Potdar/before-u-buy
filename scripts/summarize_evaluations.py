#!/usr/bin/env python3
"""Summarize a human-scored response evaluation CSV without scoring replies."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", type=Path, default=ROOT / "response_evaluation.csv")
    args = parser.parse_args()
    reviewed = 0
    passed = 0
    failed = 0
    uncertain = 0
    categories: Counter[str] = Counter()
    with args.csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            verdict = (row.get("pass") or "").strip().casefold()
            status = (row.get("review_status") or "").strip().casefold()
            if verdict == "1" or status == "pass":
                passed += 1
                reviewed += 1
            elif verdict == "0" or status == "fail":
                failed += 1
                reviewed += 1
                for category in (row.get("note") or "").split(";"):
                    if category.strip():
                        categories[category.strip()] += 1
            elif status in {"uncertain", "review", "pending"}:
                uncertain += 1
    denominator = passed + failed
    rate = passed / denominator if denominator else None
    print(f"Reviewed: {reviewed}; pass: {passed}; fail: {failed}; uncertain/pending: {uncertain}")
    print(f"Pass rate: {rate:.3%}" if rate is not None else "Pass rate: not available (no binary-scored replies)")
    print(f"Failure notes/categories: {dict(categories) if categories else 'none recorded'}")


if __name__ == "__main__":
    main()
