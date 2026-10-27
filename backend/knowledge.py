from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

STOP_WORDS = {
    "about", "after", "also", "and", "are", "can", "could", "does", "from",
    "have", "help", "into", "just", "like", "more", "much", "need", "only",
    "that", "the", "their", "there", "this", "what", "when", "with", "would",
}
TOPIC_SYNONYMS = {
    "30-day rule": {"month", "wait", "waiting", "still", "pause", "delay"},
    "cost per use": {"uses", "weekly", "frequency", "lifespan", "usage"},
    "emotional spending": {"stress", "stressed", "emotional", "anxious", "mood"},
    "emi and credit-card risks": {"emi", "credit", "card", "interest", "repayment"},
    "need vs. want": {"need", "want", "alternative", "purpose"},
    "sales and fomo pressure": {"sale", "fomo", "trending", "trend", "countdown", "social"},
    "simple budgeting": {"budget", "surplus", "expenses", "affordability", "savings"},
}


@dataclass
class KnowledgeNote:
    topic: str
    source: str
    last_reviewed: str
    points: list[str] = field(default_factory=list)
    source_title: str = ""

    @property
    def url(self) -> str:
        return self.source

    @property
    def title(self) -> str:
        return self.topic


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def load_notes(directory: str | Path = "data/common") -> list[KnowledgeNote]:
    base_path = Path(directory)
    if not base_path.exists():
        return []

    notes: list[KnowledgeNote] = []
    for file_path in sorted(base_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        lines = [line.rstrip() for line in text.splitlines()]
        source = ""
        source_title = ""
        last_reviewed = ""
        topic = ""
        points: list[str] = []

        for line in lines:
            if line.startswith("source:"):
                source = line.split(":", 1)[1].strip()
            elif line.startswith("source_title:"):
                source_title = line.split(":", 1)[1].strip()
            elif line.startswith("last_reviewed:"):
                last_reviewed = line.split(":", 1)[1].strip()
            elif line.startswith("Topic:"):
                topic = line.split(":", 1)[1].strip()
            elif line.startswith("- "):
                points.append(line[1:].strip())

        if not source or not topic:
            continue
        notes.append(
            KnowledgeNote(
                topic=topic,
                source=source,
                last_reviewed=last_reviewed or "unknown",
                points=points,
                source_title=source_title or topic,
            )
        )
    return notes


def retrieve_relevant_notes(query: str, notes: Iterable[KnowledgeNote] | None = None) -> list[dict[str, object]]:
    if not query or not query.strip():
        return []

    note_list = list(notes) if notes is not None else load_notes()
    if not note_list:
        return []

    normalized_query = _normalize(query)
    query_terms = {
        term for term in normalized_query.split()
        if len(term) > 2 and term not in STOP_WORDS
    }
    if not query_terms:
        return []

    scored: list[tuple[float, str, dict[str, object]]] = []
    for note in note_list:
        title_terms = set(_normalize(note.topic).split())
        point_terms = set(_normalize(" ".join(note.points)).split())
        matched_title = query_terms & title_terms
        matched_points = query_terms & point_terms
        synonym_terms = next(
            (terms for topic, terms in TOPIC_SYNONYMS.items() if topic.casefold() == note.topic.casefold()),
            set(),
        )
        matched_synonyms = query_terms & synonym_terms
        if not matched_title and not matched_points and not matched_synonyms:
            continue
        score = len(matched_title) * 4.0 + len(matched_points) + len(matched_synonyms) * 3.0
        if _normalize(note.topic) in normalized_query:
            score += 2.0
        scored.append((score, note.topic.casefold(), {
            "topic": note.topic,
            "title": note.source_title,
            "source": note.source,
            "url": note.source,
            "last_reviewed": note.last_reviewed,
            "points": note.points,
            "match_count": len(matched_title | matched_points | matched_synonyms),
            "relevance_score": score,
        }))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] < 1:
        return []
    minimum_relevance = max(2.0, scored[0][0] * 0.25)
    return [payload for score, _, payload in scored if score >= minimum_relevance][:3]
