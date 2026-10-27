from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend.assessment import AssessmentInput, calculate_purchase_pressure, classify_sentiment, evaluate
from backend.market import SerpApiProvider, _CACHE, normalize_offer, shortlist
from backend.analytics import summarize


def test_pressure_is_separate_and_bounded():
    calm = AssessmentInput(name="Book", price=1000, need="Solves a real problem", urgency="No deadline")
    pressured = AssessmentInput(name="Phone", price=12000, scenario="Everyone has it and I feel pressure", current="I have something that works", urgency="Social pressure or FOMO")
    assert calculate_purchase_pressure(calm, classify_sentiment(calm.scenario))['score'] < calculate_purchase_pressure(pressured, classify_sentiment(pressured.scenario))['score']
    assert 0 <= evaluate(pressured).purchase_pressure['score'] <= 100
    assert evaluate(pressured).regret_risk['score'] != evaluate(pressured).purchase_pressure['score']


def test_analytics_keeps_currencies_separate_and_does_not_invent_potential_savings():
    result = summarize([
        {"name": "Headphones", "price": 8000, "currency": "INR", "status": "planned"},
        {"name": "Book", "price": 50, "currency": "USD", "status": "completed"},
        {"name": "Camera", "price": 300, "currency": "USD", "status": "skipped", "confirmed_savings": 25},
    ], [])
    assert result["totals_by_currency"] == {
        "INR": {"planned_spending": 8000.0, "confirmed_spending": 0.0, "confirmed_savings": 0.0},
        "USD": {"planned_spending": 0.0, "confirmed_spending": 50.0, "confirmed_savings": 25.0},
    }
    assert result["planned_spending"] is None
    assert result["confirmed_spending"] is None
    assert result["confirmed_savings"] is None
    assert result["potential_savings"] is None


def test_browser_cors_allows_watchlist_and_data_mutations():
    response = TestClient(app).options(
        "/api/watchlist/example",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert "PATCH" in response.headers["access-control-allow-methods"]


def test_workspace_stores_decision_and_analytics():
    client = TestClient(app)
    created = client.post('/api/decisions', json={'name': 'Test item', 'price': 1000, 'status': 'completed', 'data': {}})
    assert created.status_code == 200
    analytics = client.get('/api/analytics').json()
    assert analytics['decision_count'] >= 1
    assert analytics['confirmed_spending'] >= 1000


def test_journal_lists_updates_and_deletes_one_decision():
    client = TestClient(app)
    saved = client.post('/api/decisions', json={
        'name': 'Journal headphones',
        'price': 8000,
        'currency': 'INR',
        'status': 'awaiting',
        'data': {'purchase_pressure': {'score': 30}},
    })
    assert saved.status_code == 200
    decision_id = saved.json()['id']
    created_at = saved.json()['created_at']

    listed = client.get('/api/decisions')
    assert any(item['id'] == decision_id for item in listed.json()['items'])

    updated = client.post('/api/decisions', json={
        'id': decision_id,
        'name': 'Journal headphones',
        'price': 8000,
        'currency': 'INR',
        'status': 'completed',
        'final_decision': 'completed',
        'data': {'purchase_pressure': {'score': 30}},
    })
    assert updated.status_code == 200
    persisted = next(item for item in client.get('/api/decisions').json()['items'] if item['id'] == decision_id)
    assert persisted['created_at'] == created_at
    assert persisted['status'] == 'completed'
    assert persisted['data']['purchase_pressure']['score'] == 30

    assert client.delete(f'/api/decisions/{decision_id}').json() == {'deleted': True}
    assert client.delete(f'/api/decisions/{decision_id}').json() == {'deleted': False}


def test_watchlist_deduplicates_same_variant_and_provider_is_explicit():
    client = TestClient(app)
    body = {'name': 'Manual headphones', 'variant': 'Black / 1TB', 'retailer': 'Manual', 'price': 8000}
    first = client.post('/api/watchlist', json=body)
    second = client.post('/api/watchlist', json=body)
    assert first.status_code == second.status_code == 200
    items = client.get('/api/watchlist').json()
    assert sum(item['name'] == 'Manual headphones' for item in items['items']) == 1
    assert items['provider']['mode'] in {'manual_only', 'live'}


def test_watchlist_keeps_variants_separate_and_supports_target_edits():
    client = TestClient(app)
    first = client.post('/api/watchlist', json={
        'name': 'Variant product', 'variant': 'Black', 'retailer': 'Store', 'price': 1000,
    })
    second = client.post('/api/watchlist', json={
        'name': 'Variant product', 'variant': 'White', 'retailer': 'Store', 'price': 1100,
    })
    assert first.status_code == second.status_code == 200
    item_id = first.json()['id']
    updated = client.patch(f'/api/watchlist/{item_id}', json={'target_price': 900})
    assert updated.status_code == 200
    assert updated.json()['target_price'] == 900
    assert len([item for item in client.get('/api/watchlist').json()['items'] if item['name'] == 'Variant product']) == 2
    assert client.patch('/api/watchlist/missing-item', json={'target_price': 900}).status_code == 404


def test_watchlist_refresh_does_not_replace_a_variant_with_a_different_one(monkeypatch):
    client = TestClient(app)
    created = client.post('/api/watchlist', json={
        'name': 'Variant refresh item',
        'variant': 'Black',
        'retailer': 'Store',
        'price': 1200,
    })
    item_id = created.json()['id']
    monkeypatch.setattr('backend.main.SerpApiProvider.search', lambda *args, **kwargs: {
        'status': 'live',
        'offers': [{
            'name': 'Variant refresh item',
            'variant': 'Silver',
            'retailer': 'Store',
            'price': 500,
            'currency': 'INR',
            'url': 'https://store.example/silver',
            'source': 'Store',
            'status': 'live',
        }],
    })

    response = client.post(f'/api/watchlist/{item_id}/refresh')
    assert response.status_code == 200
    assert 'No matching product variant' in response.json()['notification']
    current = next(item for item in client.get('/api/watchlist').json()['items'] if item['id'] == item_id)
    assert current['price'] == 1200
    assert client.get(f'/api/watchlist/{item_id}/history').json()['observations'] == []


def test_offer_normalization_rejects_invalid_price():
    response = TestClient(app).post('/api/products/normalize', json={'name': 'bad', 'price': -1})
    assert response.status_code == 422


@pytest.mark.parametrize(('price', 'url'), [
    (float('nan'), 'https://shop.example/item'),
    (100, 'javascript:alert(1)'),
    (100, 'https://user:password@shop.example/item'),
    (100, 'https://['),
])
def test_offer_normalization_rejects_nonfinite_price_and_unsafe_urls(price, url):
    with pytest.raises(ValueError):
        normalize_offer({'name': 'invalid offer', 'price': price, 'url': url}, require_url=True)


def test_recommendation_ranking_penalizes_stale_offers():
    response = shortlist({
        'category': 'headphones',
        'budget': 1000,
        'options': [
            {'name': 'Stale offer', 'price': 800, 'status': 'stale'},
            {'name': 'Live offer', 'price': 900, 'status': 'live'},
        ],
    })
    offers = response['recommendations']
    assert offers[0]['offer']['name'] == 'Live offer'
    assert offers[1]['match_score'] < offers[0]['match_score']
    assert 'stale' in offers[1]['why']


def test_watchlist_rejects_invalid_price_observation_and_unsafe_urls():
    client = TestClient(app)
    invalid_price = client.post('/api/watchlist', json={
        'name': 'Invalid observation',
        'observation': {'price': -1, 'currency': 'INR'},
    })
    invalid_url = client.post('/api/watchlist', json={
        'name': 'Unsafe watch URL',
        'url': 'javascript:alert(1)',
    })
    assert invalid_price.status_code == 422
    assert invalid_url.status_code == 422


def test_provider_missing_credentials_is_explicit(monkeypatch):
    result = SerpApiProvider(api_key="").search("headphones")
    assert result["status"] == "unavailable"
    assert result["offers"] == []


def test_provider_normalizes_deduplicates_and_caches(monkeypatch):
    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"shopping_results": [
                {"title": "Headphones 1TB", "extracted_price": 8000, "link": "https://shop.example/item", "source": "Shop", "product_id": "p1"},
                {"title": "Headphones 1TB", "extracted_price": 8000, "link": "https://shop.example/item", "source": "Shop", "product_id": "p1"},
                {"title": "Headphones 2TB", "extracted_price": 9000, "link": "https://shop.example/item2", "source": "Shop", "product_id": "p2"},
                {"title": "Malformed", "extracted_price": None, "link": "not-a-url", "source": "Shop"},
            ]}
    _CACHE.clear()
    monkeypatch.setattr("backend.market.httpx.get", lambda *args, **kwargs: FakeResponse())
    provider = SerpApiProvider(api_key="test", cache_seconds=60)
    first = provider.search("headphones")
    second = provider.search("headphones")
    assert first["status"] == "live" and len(first["offers"]) == 2
    assert second["status"] == "cached"
    assert all(item["currency"] == "INR" for item in first["offers"])


def test_provider_handles_timeout_and_rate_limit(monkeypatch):
    monkeypatch.setattr("backend.market.httpx.get", lambda *args, **kwargs: (_ for _ in ()).throw(__import__("httpx").TimeoutException("slow")))
    assert SerpApiProvider(api_key="x").search("phone")["status"] == "timeout"
    class RateLimit:
        status_code = 429
    monkeypatch.setattr("backend.market.httpx.get", lambda *args, **kwargs: RateLimit())
    assert SerpApiProvider(api_key="x").search("phone")["status"] == "rate_limited"


@pytest.mark.parametrize(('status', 'expected'), [(401, 'auth_error'), (403, 'auth_error'), (500, 'provider_error')])
def test_provider_handles_auth_and_server_errors(monkeypatch, status, expected):
    class ErrorResponse:
        status_code = status
        def raise_for_status(self): raise __import__('httpx').HTTPStatusError('error', request=None, response=None)
    monkeypatch.setattr("backend.market.httpx.get", lambda *args, **kwargs: ErrorResponse())
    assert SerpApiProvider(api_key="x").search(f"error-{status}")["status"] == expected


def test_provider_handles_connection_and_invalid_json(monkeypatch):
    monkeypatch.setattr("backend.market.httpx.get", lambda *args, **kwargs: (_ for _ in ()).throw(__import__('httpx').ConnectError('down')))
    assert SerpApiProvider(api_key="x").search("connection-error")["status"] == "error"
    class BadJson:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): raise ValueError('bad json')
    monkeypatch.setattr("backend.market.httpx.get", lambda *args, **kwargs: BadJson())
    assert SerpApiProvider(api_key="x").search("bad-json")["status"] == "error"


def test_expired_cache_becomes_stale_on_provider_failure(monkeypatch):
    class GoodResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"shopping_results": [{"title": "Cached phone", "extracted_price": 1000, "link": "https://example.com/phone", "source": "Shop", "product_id": "cache-1"}]}
    _CACHE.clear()
    calls = iter([GoodResponse(), __import__("httpx").TimeoutException("expired")])
    def fake_get(*args, **kwargs):
        value = next(calls)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr("backend.market.httpx.get", fake_get)
    provider = SerpApiProvider(api_key="x", cache_seconds=0)
    assert provider.search("cached phone")["status"] == "live"
    stale = provider.search("cached phone")
    assert stale["status"] == "stale" and stale["offers"][0]["status"] == "stale"


def test_price_history_export_and_confirmed_delete():
    client = TestClient(app)
    created = client.post('/api/watchlist', json={'name': 'History item', 'variant': 'A', 'retailer': 'R', 'price': 500}).json()
    item_id = created['id']
    client.post('/api/watchlist', json={'id': item_id, 'name': 'History item', 'variant': 'A', 'retailer': 'R', 'price': 450, 'observation': {'price': 450, 'source': 'manual', 'status': 'manual'}})
    history = client.get(f'/api/watchlist/{item_id}/history').json()
    assert len(history['observations']) == 1
    assert history['sufficient_for_trend'] is False
    exported = client.get('/api/data/export')
    assert exported.status_code == 200 and 'decisions' in exported.json()
    assert client.delete('/api/data').status_code == 400
    assert client.delete('/api/data?confirm=true').json() == {'deleted': True}
