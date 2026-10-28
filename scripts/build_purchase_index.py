#!/usr/bin/env python3
"""Build the local scenario index from the versioned JSONL corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.dataset import DATASET_PATH, load_scenarios
from backend.retrieval import INDEX_PATH, build_index


def main() -> None:
    scenarios = load_scenarios(DATASET_PATH)
    if not scenarios:
        raise SystemExit("Dataset is missing or empty; refusing to build an empty index")
    result = build_index(scenarios, INDEX_PATH)
    print(result)
    print(f"Index: {INDEX_PATH}")


if __name__ == "__main__":
    main()
