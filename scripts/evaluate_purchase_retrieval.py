#!/usr/bin/env python3
"""Evaluate local retrieval against a lexical overlap baseline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.dataset import DATASET_PATH, load_scenarios, normalized_text
from backend.retrieval import LocalScenarioRetriever, build_index


def lexical(query: str, candidates, k: int = 5):
    query_terms = set(normalized_text(query).split())
    scored = []
    for scenario in candidates:
        terms = set(normalized_text(scenario.text).split())
        score = len(query_terms & terms) / max(1, len(query_terms | terms))
        scored.append((score, scenario.scenario_id))
    return [sid for _, sid in sorted(scored, key=lambda x: (-x[0], x[1]))[:k]]


def relevant(query, candidate) -> bool:
    return query.payload.get("product_category") == candidate.payload.get("product_category") and (
        query.payload.get("need_classification") == candidate.payload.get("need_classification")
        or query.payload.get("product_condition") == candidate.payload.get("product_condition")
    )


def main() -> None:
    all_scenarios = load_scenarios(DATASET_PATH)
    # Use a stable split; Python's hash is intentionally randomized per process.
    held_out_products = {
        (scenario.payload.get("product_category"), scenario.payload.get("product_name_or_type"))
        for scenario in all_scenarios
        if sum(ord(char) for char in str(scenario.payload.get("product_name_or_type"))) % 2 == 0
    }
    train = [scenario for scenario in all_scenarios if (scenario.payload.get("product_category"), scenario.payload.get("product_name_or_type")) not in held_out_products]
    test = [scenario for scenario in all_scenarios if (scenario.payload.get("product_category"), scenario.payload.get("product_name_or_type")) in held_out_products]
    max_queries = max(1, int(os.getenv("RETRIEVAL_EVAL_MAX_QUERIES", "256")))
    evaluation_queries = test[::max(1, len(test) // max_queries)]
    with tempfile.TemporaryDirectory(prefix="beforeyoubuy-index-") as directory:
        index_path = Path(directory) / "index.json"
        build_index(train, index_path)
        retriever = LocalScenarioRetriever(index_path)
        train_by_id = {scenario.scenario_id: scenario for scenario in train}
        metrics = {"semantic": {"hits": 0, "precision_sum": 0.0, "mrr_sum": 0.0}, "lexical": {"hits": 0, "precision_sum": 0.0, "mrr_sum": 0.0}}
        for query in evaluation_queries:
            semantic_ids = [item["scenario_id"] for item in retriever.retrieve(query.text, top_k=5)]
            lexical_ids = lexical(query.text, train)
            for name, ids in (("semantic", semantic_ids), ("lexical", lexical_ids)):
                relevant_positions = [
                    position for position, sid in enumerate(ids, 1)
                    if relevant(query, train_by_id[sid])
                ]
                metrics[name]["hits"] += bool(relevant_positions)
                metrics[name]["precision_sum"] += len(relevant_positions) / 5
                metrics[name]["mrr_sum"] += 1 / relevant_positions[0] if relevant_positions else 0
        for values in metrics.values():
            values["recall_at_5"] = values.pop("hits") / max(1, len(evaluation_queries))
            values["precision_at_5"] = values.pop("precision_sum") / max(1, len(evaluation_queries))
            values["mrr_at_5"] = values.pop("mrr_sum") / max(1, len(evaluation_queries))
    report = {
        "dataset_records": len(all_scenarios),
        "train_records": len(train),
        "held_out_records": len(test),
        "evaluated_query_count": len(evaluation_queries),
        "split": "stable product-name parity; one product per category held out",
        "relevance": "same product category and matching need class or product condition",
        "metrics": metrics,
        "limitations": ["Synthetic relevance labels are consistency labels, not human judgments.", "The held-out products remain in familiar categories; this is not an unseen-category test."],
    }
    destination = ROOT / "backend" / "data" / "purchase_retrieval_evaluation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
