from pathlib import Path

from backend.knowledge import load_notes, retrieve_relevant_notes


def test_load_notes_reads_real_common_files():
    notes = load_notes(Path(__file__).resolve().parents[2] / "data" / "common")
    assert len(notes) >= 6
    topics = {note.topic for note in notes}
    assert "Need vs. want" in topics
    assert "Cost per use" in topics
    assert "Sales and FOMO pressure" in topics


def test_retrieve_notes_matches_relevant_keywords():
    notes = load_notes(Path(__file__).resolve().parents[2] / "data" / "common")
    results = retrieve_relevant_notes("I am tempted by a sale and want to avoid FOMO", notes)
    assert results
    assert any("Sales and FOMO pressure" == result["topic"] for result in results)
    assert all(result["title"] and result["url"].startswith("https://") for result in results)


def test_empty_knowledge_base_returns_empty_retrieval():
    empty_notes = []
    assert retrieve_relevant_notes("budgeting help", empty_notes) == []


def test_irrelevant_query_returns_no_notes():
    notes = load_notes(Path(__file__).resolve().parents[2] / "data" / "common")
    assert retrieve_relevant_notes("blue whale migration", notes) == []


def test_month_waiting_query_finds_optional_30_day_note():
    notes = load_notes(Path(__file__).resolve().parents[2] / "data" / "common")
    results = retrieve_relevant_notes("I would still want it in a month", notes)
    assert any(result["topic"] == "30-day rule" for result in results)


def test_missing_knowledge_directory_is_graceful(tmp_path):
    assert load_notes(tmp_path / "missing") == []


def test_notes_have_review_date_source_title_and_four_to_six_points():
    root = Path(__file__).resolve().parents[2] / "data" / "common"
    notes = load_notes(root)
    assert len(notes) >= 7
    for note in notes:
        assert note.source.startswith("https://")
        assert note.source_title
        assert note.last_reviewed == "2026-10-25"
        assert 4 <= len(note.points) <= 6
