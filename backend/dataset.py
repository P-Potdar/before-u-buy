"""Versioned purchase-scenario dataset loading and quality validation.

The dataset is synthetic, local, and intended for retrieval/evaluation. It is
not customer data and its labels are generated expectations, not outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "backend" / "data" / "purchase_decision_dataset.jsonl"
DATASET_VERSION = "2.0.0"

NEED_CLASSES = {
    "Essential replacement",
    "Functional upgrade",
    "Optional or preference-driven upgrade",
    "New use case",
    "Unsuitable or incompatible",
}
DECISIONS = {"BUY", "WAIT", "DON'T BUY"}
COMPLETENESS = {"high", "medium", "low"}


@dataclass(frozen=True)
class Scenario:
    payload: dict[str, Any]

    @property
    def scenario_id(self) -> str:
        return str(self.payload["scenario_id"])

    @property
    def text(self) -> str:
        return str(self.payload["scenario_text"])


def load_scenarios(path: str | Path = DATASET_PATH) -> list[Scenario]:
    source = Path(path)
    if not source.exists():
        return []
    scenarios: list[Scenario] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Dataset record at {source}:{line_number} is not an object")
        scenarios.append(Scenario(payload))
    return scenarios


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def validate_scenarios(scenarios: list[Scenario]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    fingerprints: dict[str, int] = {}
    categories: dict[str, int] = {}
    decisions: dict[str, int] = {}
    origins: dict[str, int] = {}
    for index, scenario in enumerate(scenarios, 1):
        p = scenario.payload
        sid = p.get("scenario_id")
        if not isinstance(sid, str) or not sid.strip():
            failures.append({"record": index, "error": "missing scenario_id"})
        elif sid in ids:
            failures.append({"record": index, "error": "duplicate scenario_id", "scenario_id": sid})
        else:
            ids[sid] = index
        text = p.get("scenario_text")
        if not isinstance(text, str) or len(text.strip()) < 80:
            failures.append({"record": index, "error": "scenario_text is too short", "scenario_id": sid})
        else:
            fingerprint = text_fingerprint(text)
            if fingerprint in fingerprints:
                failures.append({"record": index, "error": "exact duplicate scenario_text", "scenario_id": sid})
            fingerprints[fingerprint] = index
        for field in ("dataset_version", "product_category", "record_origin", "validation_status"):
            if not isinstance(p.get(field), str) or not p[field].strip():
                failures.append({"record": index, "error": f"missing {field}", "scenario_id": sid})
        if p.get("need_classification") not in NEED_CLASSES:
            failures.append({"record": index, "error": "invalid need_classification", "scenario_id": sid})
        if p.get("recommended_action") not in DECISIONS:
            failures.append({"record": index, "error": "invalid recommended_action", "scenario_id": sid})
        if p.get("evidence_completeness") not in COMPLETENESS:
            failures.append({"record": index, "error": "invalid evidence_completeness", "scenario_id": sid})
        facts = p.get("facts")
        provenance = p.get("fact_provenance")
        if not isinstance(facts, dict) or not isinstance(provenance, dict):
            failures.append({"record": index, "error": "facts and fact_provenance must be objects", "scenario_id": sid})
        elif set(facts) != set(provenance):
            failures.append({"record": index, "error": "fact/provenance keys differ", "scenario_id": sid})
        if p.get("record_origin") == "synthetic" and p.get("validation_status") not in {"generated_consistency_checked", "manually_reviewed"}:
            failures.append({"record": index, "error": "synthetic record lacks validation status", "scenario_id": sid})
        category = str(p.get("product_category", "unknown"))
        categories[category] = categories.get(category, 0) + 1
        decision = str(p.get("recommended_action", "unknown"))
        decisions[decision] = decisions.get(decision, 0) + 1
        origin = str(p.get("record_origin", "unknown"))
        origins[origin] = origins.get(origin, 0) + 1

        if p.get("user_budget") is not None and p.get("purchase_price") is None:
            if p.get("facts", {}).get("product_price") is not None:
                failures.append({"record": index, "error": "budget/price contradiction", "scenario_id": sid})
        if p.get("affordability_status") == "unknown" and p.get("facts", {}).get("affordability_claim") in {"reported_affordable", "reported_not_affordable"}:
            failures.append({"record": index, "error": "affordability label contradicts claim", "scenario_id": sid})

    return {
        "dataset_version": DATASET_VERSION,
        "record_count": len(scenarios),
        "unique_id_count": len(ids),
        "unique_text_count": len(fingerprints),
        "categories": dict(sorted(categories.items())),
        "recommended_actions": dict(sorted(decisions.items())),
        "record_origins": dict(sorted(origins.items())),
        "validation_failures": failures,
        "valid": not failures and len(ids) == len(scenarios),
    }
