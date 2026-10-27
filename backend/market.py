"""Provider-neutral shopping offers and an optional SerpApi adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import re
import math
import time
from typing import Any
from urllib.parse import urlparse

import httpx


@dataclass
class ProductOffer:
    name: str
    brand: str = ""
    variant: str = ""
    retailer: str = ""
    price: float = 0
    currency: str = "INR"
    url: str = ""
    source: str = "manual"
    availability: str = "unknown"
    observed_at: str = ""
    status: str = "manual"
    offer_id: str = ""
    condition: str = "unknown"
    shipping: str = "unknown"


def _valid_url(value: str, *, required: bool = False) -> bool:
    if not value:
        return not required
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
    )


def normalize_offer(raw: dict[str, Any], *, require_url: bool = False) -> ProductOffer:
    name = str(raw.get("name") or raw.get("title") or "").strip()
    if not name:
        raise ValueError("Offer name is required")
    price = raw.get("price", raw.get("extracted_price"))
    if (
        not isinstance(price, (int, float))
        or isinstance(price, bool)
        or not math.isfinite(price)
        or price < 0
    ):
        raise ValueError("Offer price must be a non-negative number")
    currency = str(raw.get("currency") or "INR").upper()
    if currency not in {"INR", "USD", "GBP", "EUR"}:
        raise ValueError("Unsupported offer currency")
    url = str(raw.get("url") or raw.get("link") or raw.get("product_link") or "").strip()
    if not _valid_url(url, required=require_url):
        raise ValueError("Offer URL must be an absolute http(s) URL")
    return ProductOffer(
        name=name, brand=str(raw.get("brand") or "").strip(), variant=str(raw.get("variant") or "").strip(),
        retailer=str(raw.get("retailer") or raw.get("source") or "").strip(), price=float(price), currency=currency, url=url,
        source=str(raw.get("source") or "manual"), availability=str(raw.get("availability") or "unknown"),
        observed_at=str(raw.get("observed_at") or datetime.now(timezone.utc).isoformat()), status=str(raw.get("status") or "manual"),
        offer_id=str(raw.get("offer_id") or raw.get("product_id") or ""), condition=str(raw.get("condition") or raw.get("second_hand_condition") or "unknown"),
        shipping=str(raw.get("shipping") or "unknown"),
    )


def _dedupe_key(offer: ProductOffer) -> tuple[str, str, str, float]:
    identity = offer.offer_id or re.sub(r"\W+", " ", offer.name.casefold()).strip()
    return identity, offer.retailer.casefold(), offer.variant.casefold(), offer.price


def deduplicate_offers(offers: list[ProductOffer]) -> list[ProductOffer]:
    seen: set[tuple[str, str, str, float]] = set(); result = []
    for offer in offers:
        if _dedupe_key(offer) not in seen:
            seen.add(_dedupe_key(offer)); result.append(offer)
    return result


_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class SerpApiProvider:
    name = "serpapi_google_shopping"

    def __init__(self, api_key: str | None = None, timeout: float | None = None, cache_seconds: int | None = None):
        self.api_key = (api_key if api_key is not None else os.getenv("SERPAPI_KEY", "")).strip()
        try: self.timeout = max(1.0, min(float(timeout if timeout is not None else os.getenv("PRODUCT_PROVIDER_TIMEOUT_SECONDS", "12")), 30.0))
        except ValueError: self.timeout = 12.0
        try: self.cache_seconds = max(0, min(int(cache_seconds if cache_seconds is not None else os.getenv("PRODUCT_CACHE_SECONDS", "900")), 86400))
        except ValueError: self.cache_seconds = 900

    @property
    def configured(self) -> bool: return bool(self.api_key)

    def search(self, query: str, *, category: str = "", brand: str = "", specifications: str = "") -> dict[str, Any]:
        query = " ".join(part.strip() for part in (query, category, brand, specifications) if part.strip())
        if not query: return {"status": "invalid", "offers": [], "message": "A product search term is required."}
        if not self.configured: return {"status": "unavailable", "offers": [], "message": "Live product search is not configured. Add SERPAPI_KEY on the backend."}
        cache_key = query.casefold(); cached = _CACHE.get(cache_key)
        if cached and time.monotonic() - cached[0] <= self.cache_seconds:
            body = dict(cached[1]); body["status"] = "cached"; body["cache_age_seconds"] = round(time.monotonic() - cached[0], 1)
            body["offers"] = [dict(offer, status="cached") for offer in body.get("offers", [])]
            return body
        expired = cached
        params = {"engine": "google_shopping", "q": query, "gl": "in", "hl": "en", "google_domain": "google.co.in", "api_key": self.api_key}
        try:
            response = httpx.get("https://serpapi.com/search.json", params=params, timeout=self.timeout)
            if response.status_code == 429:
                return self._stale_or_error(expired, "rate_limited", "The product provider rate limit was reached. Try again later.")
            if response.status_code in {401, 403}:
                return self._stale_or_error(expired, "auth_error", "The provider rejected the configured credentials.")
            if response.status_code >= 500:
                return self._stale_or_error(expired, "provider_error", "The product provider is currently unavailable.")
            response.raise_for_status(); payload = response.json()
        except httpx.TimeoutException: return self._stale_or_error(expired, "timeout", "The product provider timed out. No live result is being shown.")
        except (httpx.HTTPError, ValueError): return self._stale_or_error(expired, "error", "The product provider returned an unusable response.")
        if not isinstance(payload, dict): return self._stale_or_error(expired, "malformed", "The product provider returned malformed data.")
        offers: list[ProductOffer] = []
        for item in payload.get("shopping_results", []) if isinstance(payload.get("shopping_results"), list) else []:
            if not isinstance(item, dict): continue
            try:
                offers.append(normalize_offer({"name": item.get("title"), "price": item.get("extracted_price"), "url": item.get("link") or item.get("product_link"), "source": item.get("source"), "retailer": item.get("source"), "offer_id": item.get("product_id"), "condition": item.get("second_hand_condition"), "shipping": item.get("delivery"), "observed_at": datetime.now(timezone.utc).isoformat(), "status": "live"}, require_url=True))
            except ValueError: continue
        body = {"status": "live", "provider": self.name, "observed_at": datetime.now(timezone.utc).isoformat(), "offers": [asdict(item) for item in deduplicate_offers(offers)], "message": "" if offers else "No valid India shopping results were returned."}
        _CACHE[cache_key] = (time.monotonic(), body)
        return body

    @staticmethod
    def _stale_or_error(cached: tuple[float, dict[str, Any]] | None, status: str, message: str) -> dict[str, Any]:
        if cached:
            body = dict(cached[1]); body["status"] = "stale"; body["provider_status"] = status; body["message"] = message
            body["offers"] = [dict(offer, status="stale") for offer in body.get("offers", [])]
            body["stale_age_seconds"] = round(time.monotonic() - cached[0], 1)
            return body
        return {"status": status, "offers": [], "message": message}


def provider_status() -> dict[str, Any]:
    provider = SerpApiProvider()
    return {"provider": provider.name, "configured": provider.configured, "mode": "live" if provider.configured else "manual_only", "country": "IN", "currency": "INR", "message": "Live India shopping search is enabled." if provider.configured else "No provider credentials configured; manual entry remains available."}


def shortlist(request: dict[str, Any]) -> dict[str, Any]:
    budget = request.get("budget"); name = request.get("category") or request.get("use_case") or "Your shortlist"; ranked = []
    for raw in request.get("options") or []:
        offer = normalize_offer(raw); score = 50; reasons = ["Compared using the supplied offer facts."]
        if isinstance(budget, (int, float)) and offer.price <= budget: score += 25; reasons.append("Within the stated budget.")
        elif isinstance(budget, (int, float)): reasons.append("Above the stated budget.")
        if request.get("preferred_brand") and str(request["preferred_brand"]).casefold() in offer.brand.casefold(): score += 10; reasons.append("Matches the preferred brand.")
        freshness_penalty = {"cached": 5, "stale": 15}.get(offer.status, 0)
        if freshness_penalty:
            reasons.append(f"Price is {offer.status}; verify it before relying on this comparison.")
        ranked.append({
            "offer": asdict(offer),
            "match_score": max(0, min(score - freshness_penalty, 100)),
            "why": " ".join(reasons),
            "tradeoffs": "Verify the exact variant, warranty, delivery, and current price before deciding.",
        })
    ranked.sort(key=lambda item: (-item["match_score"], item["offer"]["price"]))
    return {"query": name, "provider": provider_status(), "recommendations": ranked, "alternatives": ["repair or reuse the current product", "wait for more evidence", "skip if no clear need remains"], "limitations": "Recommendations are ranked only from supplied or provider-returned offers; no product facts are invented."}
