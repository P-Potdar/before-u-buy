import pytest
from fastapi.testclient import TestClient

from backend import main


client = TestClient(main.app)


def _request(session: str):
    response = client.post("/api/chat", json={"message": "I want headphones for ₹8000.", "session_id": session})
    assert response.status_code == 200
    return response.json()


def test_ollama_success_does_not_call_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    calls = {"ollama": 0, "gemini": 0}

    async def ollama(*_args):
        calls["ollama"] += 1
        return "Ollama answered the latest message."

    async def gemini(*_args):
        calls["gemini"] += 1
        return "Gemini should not run."

    monkeypatch.setattr(main, "_ollama_reply", ollama)
    monkeypatch.setattr(main, "_gemini_reply", gemini)
    result = _request("provider-ollama-success")
    assert result["provider"] == "ollama"
    assert calls == {"ollama": 1, "gemini": 0}


def test_ollama_failure_uses_gemini_once(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("GEMINI_MODEL", "configured-gemini")
    calls = {"ollama": 0, "gemini": 0}

    async def ollama(*_args):
        calls["ollama"] += 1
        return None

    async def gemini(*_args):
        calls["gemini"] += 1
        return "Gemini answered the latest message."

    monkeypatch.setattr(main, "_ollama_reply", ollama)
    monkeypatch.setattr(main, "_gemini_reply", gemini)
    result = _request("provider-gemini-fallback")
    assert result["provider"] == "gemini"
    assert result["model"] == "configured-gemini"
    assert result["fallback_occurred"] is True
    assert calls == {"ollama": 1, "gemini": 1}


def test_missing_gemini_key_stays_deterministic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def ollama(*_args):
        return None

    monkeypatch.setattr(main, "_ollama_reply", ollama)
    result = _request("provider-no-key")
    assert result["provider"] == "deterministic_fallback"
    assert result["model"] is None
    assert result["fallback_occurred"] is True
    assert "Gemini is not configured" in result["fallback_reason"]
    assert "GEMINI_API_KEY" not in str(result)


@pytest.mark.parametrize("reason", ["authentication", "quota", "timeout", "malformed"])
def test_gemini_failure_keeps_safe_fallback(monkeypatch, reason):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    async def ollama(*_args):
        return None

    async def gemini(*_args):
        return None

    monkeypatch.setattr(main, "_ollama_reply", ollama)
    monkeypatch.setattr(main, "_gemini_reply", gemini)
    result = _request(f"provider-gemini-{reason}")
    assert result["provider"] == "deterministic_fallback"
    assert result["reply"].strip()
    assert result["decision_state"]["recommendation"] in {"BUY", "WAIT", "DON'T BUY"}

