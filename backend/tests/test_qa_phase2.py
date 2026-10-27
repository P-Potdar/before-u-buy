import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.storage import list_watchlist


client = TestClient(app)


def test_api_smoke_matrix_and_error_safety(monkeypatch):
    assert client.get('/health').status_code == 200
    assert client.get('/api/providers/status').status_code == 200
    assert client.get('/api/analytics').status_code == 200
    assert client.get('/api/decisions').status_code == 200
    assert client.get('/api/watchlist').status_code == 200
    assert client.get('/api/data/export').status_code == 200
    assert client.get('/api/models').status_code == 200

    assert client.post('/api/assess', json={'name': 'Headphones', 'price': 8000}).status_code == 200
    assert client.post('/api/assess', json={'name': ' ', 'price': 8000}).status_code == 422
    assert client.post('/api/assess', json={'name': 'Headphones', 'price': 'not-a-number'}).status_code == 422
    assert client.post('/api/assess', json={'name': 'Headphones', 'price': 8000, 'scenario': 'x' * 2001}).status_code == 422

    chat = client.post('/api/chat', json={'message': '  hi  ', 'session_id': 'qa-chat'})
    assert chat.status_code == 200 and chat.json()['reply'].strip()
    assert client.post('/api/chat', json={'message': 'x' * 2001}).status_code == 422
    assert client.post('/api/chat', content='{invalid-json', headers={'content-type': 'application/json'}).status_code == 422

    pdf = client.post('/api/report/pdf', json={'name': 'Book', 'price': 500})
    assert pdf.status_code == 200 and pdf.headers['content-type'].startswith('application/pdf')
    assert client.post('/api/recommendations', json={'category': 'Headphones', 'options': []}).status_code == 200
    assert client.post('/api/products/normalize', json={'name': 'Valid', 'price': 100, 'url': 'https://example.com/p'}).status_code == 200
    assert client.post('/api/products/normalize', json={'name': 'Bad', 'price': 100, 'url': 'javascript:alert(1)'}).status_code == 422

    assert client.get('/api/watchlist/missing/history').status_code == 404
    assert client.post('/api/watchlist/missing/refresh').status_code == 404
    assert client.patch('/api/watchlist/missing', json={'target_price': 1}).status_code == 404
    assert client.delete('/api/data').status_code == 400
    for response in (chat, pdf):
        assert 'Traceback' not in response.text
        assert 'SERPAPI_KEY' not in response.text


def test_watchlist_workflow_target_removal_history_and_delete():
    created = client.post('/api/watchlist', json={'name': 'QA keyboard', 'variant': '96-key', 'retailer': 'QA shop', 'price': 1000, 'target_price': 900, 'url': 'https://example.com/keyboard', 'status': 'manual'}).json()
    item_id = created['id']
    duplicate = client.post('/api/watchlist', json={'name': 'QA keyboard', 'variant': '96-key', 'retailer': 'QA shop', 'price': 950}).json()
    assert duplicate['id'] == item_id
    edited = client.patch(f'/api/watchlist/{item_id}', json={'target_price': 850}).json()
    assert edited['target_price'] == 850
    removed_target = client.patch(f'/api/watchlist/{item_id}', json={'target_price': None}).json()
    assert removed_target['target_price'] is None
    observed = client.post('/api/watchlist', json={'id': item_id, 'name': 'QA keyboard', 'variant': '96-key', 'retailer': 'QA shop', 'price': 950, 'observation': {'price': 950, 'currency': 'INR', 'source': 'manual', 'status': 'manual'}})
    assert observed.status_code == 200
    history = client.get(f'/api/watchlist/{item_id}/history').json()
    assert len(history['observations']) == 1 and history['sufficient_for_trend'] is False
    assert client.delete(f'/api/watchlist/{item_id}').json() == {'deleted': True}
    assert client.get(f'/api/watchlist/{item_id}/history').status_code == 404


def test_search_without_credentials_is_not_presented_as_live(monkeypatch):
    monkeypatch.delenv('SERPAPI_KEY', raising=False)
    response = client.post('/api/products/search', json={'query': 'laptop', 'category': 'electronics'})
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'unavailable'
    assert body['offers'] == []
    assert 'configured' in client.get('/api/providers/status').json()


def test_empty_export_is_valid_json_and_confirmed_delete_is_scoped():
    client.delete('/api/data?confirm=true')
    exported = client.get('/api/data/export')
    assert exported.status_code == 200
    body = exported.json()
    assert body['decisions'] == [] and body['watchlist'] == [] and body['price_observations'] == []
    json.dumps(body, ensure_ascii=False).encode('utf-8')


def test_backpack_scenario_does_not_invent_pain_or_frequency():
    first = client.post('/api/chat', json={'message': 'I am considering a backpack costing ₹4,500.', 'session_id': 'qa-backpack'}).json()
    assert 'you said your current backpack hurts' not in first['reply'].lower()
    assert 'daily travel' not in first['reply'].lower()
    followup = client.post('/api/chat', json={'message': 'What if I only commute to the office three days a week?', 'session_id': 'qa-backpack'}).json()
    assert '3 days a week' in followup['reply']


def test_long_multiparagraph_text_with_unqualified_price_word_does_not_500():
    message = "I am comparing a purchase calmly.\n\nThe product may be useful, but I am still checking the need, price, timing, and alternatives.\n\nThis harmless multi-paragraph input tests response stability and remains within the input limit."
    response = client.post('/api/chat', json={'message': message, 'session_id': 'qa-long-multiparagraph'})
    assert response.status_code == 200
    assert response.json()['reply'].strip()
