from backend.sentiment import analyze_message


def test_mixed_sentiment_keeps_affordability_separate_from_emotion():
    result = analyze_message("I love these headphones, but I cannot comfortably afford them.")
    assert result["polarity"]["label"] == "mixed"
    assert any(item["label"] in {"joy", "satisfaction"} for item in result["emotions"])
    assert result["purchase_signals"]["affordability_concern"]["value"] is True
    assert result["purchase_signals"]["affordability_concern"]["evidence"]
    assert "purchase intent" not in result["contradictory_evidence"]


def test_sale_pressure_does_not_become_need_urgency():
    result = analyze_message("The offer ends tonight, so I feel pressured to buy.")
    assert result["purchase_signals"]["sales_pressure"]["value"] is True
    assert result["purchase_signals"]["urgency"]["value"] is True
    assert result["purchase_signals"]["need_urgency"]["value"] is None


def test_negation_and_neutral_question_remain_uncertain_or_neutral():
    result = analyze_message("I do not want this product despite the discount.")
    assert result["purchase_signals"]["sales_pressure"]["value"] is True
    assert result["polarity"]["label"] in {"negative", "uncertain"}

    neutral = analyze_message("Which evidence supports this comparison?")
    assert neutral["polarity"]["label"] == "uncertain"
    assert neutral["purchase_signals"]["comparison_intent"]["value"] is True


def test_replacement_and_upgrade_are_separate_signals():
    result = analyze_message("My headphones broke, but I also want a newer model.")
    assert result["purchase_signals"]["replacement_need"]["value"] is True
    assert result["purchase_signals"]["upgrade_desire"]["value"] is True
    assert result["contradictory_evidence"]
