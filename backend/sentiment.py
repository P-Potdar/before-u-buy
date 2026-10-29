"""Transparent, evidence-preserving sentiment and purchase-signal analysis.

This is a deterministic weak-label baseline, not a validated psychological or
financial model. Every non-unknown signal carries the matching user-text span.
"""

from __future__ import annotations

import re
from typing import Any

POLARITIES = ("positive", "negative", "neutral", "mixed", "uncertain")
EMOTIONS = (
    "excitement", "joy", "satisfaction", "curiosity", "hope", "trust",
    "anxiety", "fear", "frustration", "disappointment", "sadness", "anger",
    "regret", "relief", "confusion", "envy", "embarrassment",
)
PURCHASE_SIGNALS = (
    "purchase_intent", "purchase_hesitation", "fomo", "social_pressure",
    "sales_pressure", "urgency", "financial_stress", "affordability_concern",
    "need_urgency", "impulse_buying_risk", "regret_concern",
    "product_dissatisfaction", "replacement_need", "upgrade_desire",
    "price_sensitivity", "comparison_intent",
)

_PATTERNS: dict[str, tuple[str, ...]] = {
    "excitement": (r"\bexcited\b", r"\bcan't wait\b", r"\benthusiastic\b"),
    "joy": (r"\bjoyful\b", r"\bhappy\b", r"\blove\b"),
    "satisfaction": (r"\b(?:satisfied|happy|pleased)\b", r"\bworks? perfectly\b", r"\bworks? fine\b"),
    "curiosity": (r"\bcurious\b", r"\bwondering\b", r"\bhow does\b", r"\bwhat should\b"),
    "hope": (r"\bhope\b", r"\bwould like\b", r"\blooking forward\b"),
    "trust": (r"\breliable\b", r"\btrust\b", r"\bconfidence\b"),
    "anxiety": (r"\banxious\b", r"\b(?:worried|worries)\b", r"\bnervous\b", r"\b(?:pressure|pressured)\b"),
    "fear": (r"\bafraid\b", r"\bscared\b", r"\bfear\b"),
    "frustration": (r"\bfrustrat\w*\b", r"\bannoyed\b", r"\bcan't decide\b"),
    "disappointment": (r"\bdisappoint\w*\b", r"\blet down\b"),
    "sadness": (r"\bsad\b", r"\bunhappy\b"),
    "anger": (r"\bangry\b", r"\bmad\b", r"\bfurious\b"),
    "regret": (r"\bregret\w*\b", r"\bmiss(?:ing)? out\b"),
    "relief": (r"\brelieved\b", r"\bglad\b"),
    "confusion": (r"\bconfus\w*\b", r"\buncertain\b", r"\b(?:unsure|not sure)\b", r"\bdon't know\b"),
    "envy": (r"\benvious\b", r"\bjealous\b", r"\bmy friend(?:s)? .* bought\b"),
    "embarrassment": (r"\bembarrass\w*\b", r"\bashamed\b"),
}

_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "purchase_intent": (r"\b(?:want|plan|planning|considering|buy|buying|purchase|purchasing|looking at)\b",),
    "purchase_hesitation": (r"\b(?:unsure|uncertain|not sure|hesitat\w*|should i)\b",),
    "fomo": (r"\bfomo\b", r"\bmiss out\b", r"\bregret missing\b"),
    "social_pressure": (r"\b(?:friend|friends|everyone|colleagues?)\b.{0,50}\b(?:buy|upgrade|say|says|pressure)\b",),
    "sales_pressure": (r"\b(?:sale|offer|discount|deal|limited[- ]time|ends? tonight|today only|pressured)\b",),
    "urgency": (r"\b(?:urgent(?:ly)?|tomorrow|starting tomorrow|ends? tonight|today only|deadline)\b",),
    "financial_stress": (r"\b(?:financial stress|struggling|can't afford|cannot afford|affordability)\b",),
    "affordability_concern": (r"\b(?:can't afford|cannot afford|cannot comfortably afford|not comfortably afford|unsure .* afford|budget|emi)\b",),
    "need_urgency": (r"\b(?:need|must have|required)\b.{0,60}\b(?:tomorrow|urgent|class|meeting|work)\b",),
    "impulse_buying_risk": (r"\b(?:impulse|spur of the moment|just because|emotional purchase)\b",),
    "regret_concern": (r"\b(?:regret|worried .* regret|miss out)\b",),
    "product_dissatisfaction": (r"\b(?:disappoint\w*|unhappy|poor|bad|struggles with|doesn't work)\b",),
    "replacement_need": (r"\b(?:broken|broke|stopped working|replace|replacement|no spare|unusable)\b",),
    "upgrade_desire": (r"\b(?:upgrade|better performance|better sound|newer model|improvement)\b",),
    "price_sensitivity": (r"\b(?:expensive|cheap|cheaper|price|cost|budget|afford)\b",),
    "comparison_intent": (r"\b(?:compare|compared|comparison|evidence|alternatives?|options?|reviews?)\b",),
}


def _span(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    start = max(0, text.rfind(".", 0, match.start()) + 1)
    end = text.find(".", match.end())
    return text[start: (len(text) if end < 0 else end + 1)].strip()[:240]


def _evidence(text: str, patterns: tuple[str, ...]) -> list[str]:
    spans: list[str] = []
    for pattern in patterns:
        value = _span(text, pattern)
        if value and value not in spans:
            spans.append(value)
    return spans[:4]


def analyze_message(text: str) -> dict[str, Any]:
    """Return separate polarity, emotion, and purchase-context signals."""
    positive = _evidence(text, (r"\b(?:love|like|happy|excited|enjoy|pleased)\b",))
    negative = _evidence(text, (r"\b(?:stress\w*|anxious|worr(?:y|ied|ies)|regret|sad|angry|frustrat\w*|disappoint\w*)\b", r"\b(?:do not|don't) want\b", r"\b(?:cannot|can't)(?: comfortably)? afford\b"))
    if positive and negative:
        polarity = "mixed"
    elif positive:
        polarity = "positive"
    elif negative:
        polarity = "negative"
    elif text.strip().endswith("?") or re.search(r"\b(?:unsure|not sure|uncertain|maybe|don't know)\b", text, re.I):
        polarity = "uncertain"
    else:
        polarity = "neutral"

    emotions: list[dict[str, Any]] = []
    for label, patterns in _PATTERNS.items():
        spans = _evidence(text, patterns)
        if spans:
            emotions.append({"label": label, "evidence": spans, "method": "deterministic_lexicon", "confidence": "heuristic"})

    signals: dict[str, Any] = {}
    for label, patterns in _SIGNAL_PATTERNS.items():
        spans = _evidence(text, patterns)
        signals[label] = {
            "value": bool(spans) if spans else None,
            "evidence": spans,
            "status": "explicit" if spans else "unknown",
            "method": "deterministic_lexicon",
            "confidence": "heuristic" if spans else None,
        }
    contradictory = []
    if signals["purchase_intent"]["value"] and signals["affordability_concern"]["value"]:
        contradictory.append("purchase intent co-occurs with affordability concern")
    if signals["replacement_need"]["value"] and signals["upgrade_desire"]["value"]:
        contradictory.append("replacement need co-occurs with upgrade desire")
    polarity_evidence = list(dict.fromkeys(positive + negative))
    return {
        "polarity": {"label": polarity, "evidence": polarity_evidence, "method": "deterministic_lexicon", "confidence": "heuristic"},
        "emotions": emotions,
        "purchase_signals": signals,
        "contradictory_evidence": contradictory,
        "insufficient_evidence": not positive and not negative and not emotions and not any(item["value"] for item in signals.values()),
        "provenance": "weak deterministic labels; not human-validated and not a psychological or financial assessment",
    }
