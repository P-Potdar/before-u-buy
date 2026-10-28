"""Local, dependency-free semantic retrieval for purchase scenarios.

This is a reproducible hashed word/bigram embedding index. It is intentionally
small and offline: no text or vector is sent to a hosted service. It provides
semantic-ish matching for the decision flow and has a lexical fallback when
the index is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from backend.dataset import DATASET_PATH, Scenario, load_scenarios, normalized_text

INDEX_PATH = DATASET_PATH.with_name("purchase_decision_index.json")
EMBEDDING_MODEL = "local-hashing-word-bigram-v1"
EMBEDDING_DIMENSION = 256
INDEX_VERSION = "1"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _features(text: str) -> list[str]:
    tokens = _tokens(text)
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def _bucket(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % EMBEDDING_DIMENSION, 1.0 if number & 1 else -1.0


def embed(text: str, idf: dict[str, float] | None = None) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    counts = Counter(_features(text))
    for feature, count in counts.items():
        index, sign = _bucket(feature)
        weight = float(count) * (idf.get(feature, 1.0) if idf else 1.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def build_index(scenarios: Iterable[Scenario], destination: str | Path = INDEX_PATH) -> dict[str, Any]:
    entries = list(scenarios)
    document_frequency: Counter[str] = Counter()
    for scenario in entries:
        document_frequency.update(set(_features(scenario.text)))
    count = max(1, len(entries))
    idf = {feature: math.log((1 + count) / (1 + frequency)) + 1.0 for feature, frequency in document_frequency.items()}
    payload = {
        "index_version": INDEX_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "dataset_path": str(DATASET_PATH.relative_to(DATASET_PATH.parents[2])) if DATASET_PATH.is_absolute() else str(DATASET_PATH),
        "record_count": len(entries),
        "idf": idf,
        "entries": [
            {
                "scenario_id": scenario.scenario_id,
                "category": scenario.payload.get("product_category"),
                "need_classification": scenario.payload.get("need_classification"),
                "product_condition": scenario.payload.get("product_condition"),
                "text": scenario.text,
                "vector": embed(scenario.text, idf),
                "rationale": scenario.payload.get("rationale"),
                "record_origin": scenario.payload.get("record_origin"),
            }
            for scenario in entries
        ],
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return {key: payload[key] for key in ("index_version", "embedding_model", "dimension", "record_count")}


class LocalScenarioRetriever:
    def __init__(self, index_path: str | Path = INDEX_PATH, *, threshold: float = 0.18) -> None:
        self.index_path = Path(index_path)
        self.threshold = threshold
        self._payload: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any] | None:
        if self._payload is not None:
            return self._payload
        with self._lock:
            if self._payload is not None:
                return self._payload
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                if payload.get("index_version") != INDEX_VERSION or payload.get("embedding_model") != EMBEDDING_MODEL:
                    return None
                if not isinstance(payload.get("entries"), list) or payload.get("dimension") != EMBEDDING_DIMENSION:
                    return None
            except (OSError, json.JSONDecodeError, TypeError):
                return None
            self._payload = payload
            return payload

    def retrieve(self, query: str, *, top_k: int = 3, exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
        payload = self._load()
        if not payload or not query.strip():
            return []
        query_vector = embed(query, payload.get("idf", {}))
        excluded = exclude_ids or set()
        scored = []
        for entry in payload["entries"]:
            if entry.get("scenario_id") in excluded:
                continue
            score = cosine(query_vector, entry["vector"])
            if score >= self.threshold:
                scored.append((score, str(entry.get("scenario_id")), entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        seen: set[tuple[Any, Any, Any]] = set()
        for score, _, entry in scored:
            perspective = (entry.get("category"), entry.get("need_classification"), entry.get("product_condition"))
            if perspective in seen:
                continue
            seen.add(perspective)
            results.append({
                "scenario_id": entry.get("scenario_id"),
                "similarity": round(float(score), 4),
                "category": entry.get("category"),
                "need_classification": entry.get("need_classification"),
                "product_condition": entry.get("product_condition"),
                "rationale": entry.get("rationale"),
                "record_origin": entry.get("record_origin"),
            })
            if len(results) >= top_k:
                break
        return results


def retrieval_query(facts: dict[str, Any], message: str = "") -> str:
    fields = [
        message,
        str(facts.get("name", "")),
        str(facts.get("use_case", "")),
        str(facts.get("current", "")),
        str(facts.get("current_condition", "")),
        str(facts.get("need", "")),
        str(facts.get("need_classification", "")),
        str(facts.get("motivation", "")),
        str(facts.get("urgency", "")),
        str(facts.get("purchase_timing", "")),
        str(facts.get("waiting_period", "")),
        str(facts.get("user_affordability", "")),
        str(facts.get("savings_adequacy", "")),
        str(facts.get("alternatives_checked", "")),
        str(facts.get("repair_investigated", "")),
    ]
    return " ".join(field for field in fields if field and field != "None")
