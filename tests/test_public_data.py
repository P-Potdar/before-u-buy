from __future__ import annotations

from backend.public_data import validate_reviews
from backend.review_model import fit, predict, product_split


def review(identifier: str, text: str, sentiment: str = "Positive"):
    return {
        "source_record_id": identifier,
        "title": "review",
        "text": text,
        "source_sentiment": sentiment,
        "source_url": "https://example.test/source",
        "license": "test",
        "aspects": ["performance"],
        "parent_asin": identifier,
    }


def test_public_review_schema_rejects_duplicate_and_missing_provenance():
    rows = [review("a", "Works well"), review("b", "Works well")]
    rows[1].pop("license")
    result = validate_reviews(rows)
    assert not result["valid"]
    assert any(item["error"] == "duplicate review text" for item in result["validation_failures"])
    assert any(item["error"] == "missing provenance" for item in result["validation_failures"])


def test_sentiment_model_is_group_split_and_predictable():
    rows = [
        review("a", "excellent fast reliable product", "Positive"),
        review("b", "terrible failed broken product", "Negative"),
        review("c", "average product works", "Neutral"),
    ]
    model = fit(rows)
    prediction, probabilities = predict(model, "excellent reliable product")
    assert prediction in {"Positive", "Negative", "Neutral"}
    assert abs(sum(probabilities.values()) - 1) < 1e-6
    assert product_split(rows[0]) in {"train", "test"}
