"""Optional local dense retrieval using a pinned Sentence Transformers model."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.dataset import DATASET_PATH, Scenario, load_scenarios
from backend.public_data import RAW_REVIEWS_PATH, load_jsonl

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "main"
SCENARIO_INDEX_PATH = DATASET_PATH.with_name("purchase_decision_dense_index.npz")
REVIEW_INDEX_PATH = RAW_REVIEWS_PATH.with_name("review_dense_index.npz")


class DenseEncoder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    # Retrieval must never download or phone home at request time.
                    # If the model is not already cached, callers fall back to
                    # the dependency-free lexical index.
                    self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._load()
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


_ENCODER = DenseEncoder()


def build_dense_index(records: Iterable[dict[str, Any]], destination: str | Path, *, text_key: str) -> dict[str, Any]:
    rows = list(records)
    texts = [str(row.get(text_key, "")) for row in rows]
    vectors = _ENCODER.encode(texts)
    destination = Path(destination)
    metadata_path = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    np.savez_compressed(temporary, vectors=vectors)
    # numpy appends .npz when given a suffix it does not recognize; normalize it.
    generated = Path(str(temporary) + ".npz") if not temporary.exists() else temporary
    generated.replace(destination)
    metadata = {
        "index_version": "dense-1",
        "embedding_model": MODEL_NAME,
        "embedding_revision": MODEL_REVISION,
        "dimension": int(vectors.shape[1]),
        "record_count": len(rows),
        "entries": rows,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {key: metadata[key] for key in ("index_version", "embedding_model", "dimension", "record_count")}


class DenseRetriever:
    def __init__(self, index_path: str | Path, *, threshold: float = 0.30) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix(".json")
        self.threshold = threshold
        self._vectors: np.ndarray | None = None
        self._metadata: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        if self._vectors is not None and self._metadata is not None:
            return self._vectors, self._metadata
        with self._lock:
            if self._vectors is not None and self._metadata is not None:
                return self._vectors, self._metadata
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                vectors = np.load(self.index_path, allow_pickle=False)["vectors"]
                if metadata.get("index_version") != "dense-1" or metadata.get("embedding_model") != MODEL_NAME:
                    return None
                if vectors.ndim != 2 or vectors.shape[0] != metadata.get("record_count"):
                    return None
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                return None
            self._vectors, self._metadata = vectors, metadata
            return vectors, metadata

    def retrieve(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        loaded = self._load()
        if loaded is None or not query.strip():
            return []
        vectors, metadata = loaded
        query_vector = _ENCODER.encode([query])[0]
        scores = vectors @ query_vector
        order = np.argsort(-scores)
        results = []
        for index in order[: max(top_k * 4, top_k)]:
            score = float(scores[index])
            if score < self.threshold:
                continue
            entry = metadata["entries"][int(index)]
            results.append({**entry, "similarity": round(score, 4)})
            if len(results) >= top_k:
                break
        return results


class DenseScenarioRetriever(DenseRetriever):
    def __init__(self, index_path: str | Path = SCENARIO_INDEX_PATH, *, threshold: float = 0.30) -> None:
        super().__init__(index_path, threshold=threshold)


class DenseReviewRetriever(DenseRetriever):
    def __init__(self, index_path: str | Path = REVIEW_INDEX_PATH, *, threshold: float = 0.35) -> None:
        super().__init__(index_path, threshold=threshold)


def scenario_rows() -> list[dict[str, Any]]:
    return [scenario.payload for scenario in load_scenarios(DATASET_PATH)]


def review_rows() -> list[dict[str, Any]]:
    return load_jsonl(RAW_REVIEWS_PATH)
