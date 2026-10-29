#!/usr/bin/env python3
"""Compare dense MiniLM retrieval with lexical overlap on a grouped holdout."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.dataset import DATASET_PATH, load_scenarios, normalized_text  # noqa: E402
from backend.dense_retrieval import DenseScenarioRetriever, build_dense_index  # noqa: E402


def lexical(query: str, candidates, k: int = 5):
    query_terms = set(normalized_text(query).split())
    scored = []
    for scenario in candidates:
        terms = set(normalized_text(scenario.text).split())
        scored.append((len(query_terms & terms) / max(1, len(query_terms | terms)), scenario.scenario_id))
    return [sid for _, sid in sorted(scored, key=lambda item: (-item[0], item[1]))[:k]]


def relevant(query, candidate) -> bool:
    return query.payload.get("product_category") == candidate.payload.get("product_category") and (
        query.payload.get("need_classification") == candidate.payload.get("need_classification")
        or query.payload.get("product_condition") == candidate.payload.get("product_condition")
    )


def score(retrieved_ids, query, train_by_id):
    positions = [position for position, identifier in enumerate(retrieved_ids, 1) if relevant(query, train_by_id[identifier])]
    return bool(positions), len(positions) / 5, 1 / positions[0] if positions else 0.0


def main() -> None:
    scenarios = load_scenarios(DATASET_PATH)
    held_out = {
        (row.payload.get("product_category"), row.payload.get("product_name_or_type"))
        for row in scenarios
        if sum(ord(char) for char in str(row.payload.get("product_name_or_type"))) % 2 == 0
    }
    train = [row for row in scenarios if (row.payload.get("product_category"), row.payload.get("product_name_or_type")) not in held_out]
    test = [row for row in scenarios if (row.payload.get("product_category"), row.payload.get("product_name_or_type")) in held_out]
    max_queries = max(1, int(os.getenv("RETRIEVAL_EVAL_MAX_QUERIES", "256")))
    queries = test[::max(1, len(test) // max_queries)]
    train_by_id = {row.scenario_id: row for row in train}
    with tempfile.TemporaryDirectory(prefix="beforeyoubuy-dense-") as directory:
        index_path = Path(directory) / "index.npz"
        build_dense_index([row.payload for row in train], index_path, text_key="scenario_text")
        retriever = DenseScenarioRetriever(index_path, threshold=0.30)
        metrics = {name: [0, 0.0, 0.0] for name in ("dense", "lexical")}
        for query in queries:
            dense_ids = [row["scenario_id"] for row in retriever.retrieve(query.text, top_k=5)]
            lexical_ids = lexical(query.text, train)
            for name, identifiers in (("dense", dense_ids), ("lexical", lexical_ids)):
                hit, precision, reciprocal_rank = score(identifiers, query, train_by_id)
                metrics[name][0] += hit
                metrics[name][1] += precision
                metrics[name][2] += reciprocal_rank
    report = {
        "dataset_records": len(scenarios),
        "train_records": len(train),
        "held_out_records": len(test),
        "evaluated_query_count": len(queries),
        "split": "stable product-name parity; one product per category held out",
        "relevance": "same product category and matching need class or product condition",
        "metrics": {
            name: {
                "recall_at_5": values[0] / len(queries),
                "precision_at_5": values[1] / len(queries),
                "mrr_at_5": values[2] / len(queries),
            }
            for name, values in metrics.items()
        },
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "limitations": ["Synthetic relevance labels are consistency labels, not human judgments."],
    }
    destination = ROOT / "backend" / "data" / "purchase_dense_retrieval_evaluation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
