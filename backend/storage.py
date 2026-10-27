"""Small, parameterized SQLite store for decisions, offers, and observations."""
from __future__ import annotations
import json, os, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("BEFOREYOUBUY_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "beforeyoubuy.sqlite3"))

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db

def _now() -> str: return datetime.now(timezone.utc).isoformat()

def init_db() -> None:
    with _connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, name TEXT NOT NULL, category TEXT NOT NULL DEFAULT 'Uncategorised', price REAL NOT NULL, currency TEXT NOT NULL DEFAULT 'INR', status TEXT NOT NULL DEFAULT 'awaiting', final_decision TEXT, confirmed_savings REAL NOT NULL DEFAULT 0, data_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS watchlist (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, name TEXT NOT NULL, variant TEXT NOT NULL DEFAULT '', retailer TEXT NOT NULL DEFAULT '', target_price REAL, price REAL, currency TEXT NOT NULL DEFAULT 'INR', url TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT 'manual', status TEXT NOT NULL DEFAULT 'manual', observations_json TEXT NOT NULL DEFAULT '[]');
        CREATE UNIQUE INDEX IF NOT EXISTS watchlist_identity ON watchlist(name, variant, retailer);
        CREATE TABLE IF NOT EXISTS price_observations (id TEXT PRIMARY KEY, watchlist_id TEXT NOT NULL REFERENCES watchlist(id) ON DELETE CASCADE, offer_id TEXT NOT NULL DEFAULT '', retailer TEXT NOT NULL DEFAULT '', variant TEXT NOT NULL DEFAULT '', price REAL NOT NULL, currency TEXT NOT NULL, url TEXT NOT NULL DEFAULT '', source TEXT NOT NULL, observed_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'live');
        CREATE INDEX IF NOT EXISTS observations_watchlist_time ON price_observations(watchlist_id, observed_at DESC);
        """)

def _row(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    for key in ("data_json", "observations_json"):
        if key in value: value[key.removesuffix("_json")] = json.loads(value.pop(key))
    return value

def list_decisions() -> list[dict[str, Any]]:
    init_db()
    with _connect() as db: rows = db.execute("SELECT * FROM decisions ORDER BY created_at DESC").fetchall()
    return [_row(row) for row in rows]

def save_decision(data: dict[str, Any]) -> dict[str, Any]:
    init_db(); now = _now()
    item = {"id": data.get("id") or uuid.uuid4().hex, "created_at": data.get("created_at") or now, "updated_at": now, "name": data["name"], "category": data.get("category") or "Uncategorised", "price": data["price"], "currency": data.get("currency") or "INR", "status": data.get("status") or "awaiting", "final_decision": data.get("final_decision"), "confirmed_savings": data.get("confirmed_savings") or 0, "data_json": json.dumps(data.get("data") or {}, ensure_ascii=False)}
    with _connect() as db:
        db.execute("""INSERT INTO decisions VALUES (:id,:created_at,:updated_at,:name,:category,:price,:currency,:status,:final_decision,:confirmed_savings,:data_json) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,name=excluded.name,category=excluded.category,price=excluded.price,currency=excluded.currency,status=excluded.status,final_decision=excluded.final_decision,confirmed_savings=excluded.confirmed_savings,data_json=excluded.data_json""", item)
    return item | {"data": data.get("data") or {}}

def delete_decision(decision_id: str) -> bool:
    init_db()
    with _connect() as db: return db.execute("DELETE FROM decisions WHERE id=?", (decision_id,)).rowcount > 0

def list_watchlist() -> list[dict[str, Any]]:
    init_db()
    with _connect() as db: rows = db.execute("SELECT * FROM watchlist ORDER BY updated_at DESC").fetchall()
    return [_row(row) for row in rows]

def save_watchlist(data: dict[str, Any]) -> dict[str, Any]:
    init_db(); now = _now(); item = {"id": data.get("id") or uuid.uuid4().hex, "created_at": data.get("created_at") or now, "updated_at": now, "name": data["name"], "variant": data.get("variant") or "", "retailer": data.get("retailer") or "", "target_price": data.get("target_price"), "price": data.get("price"), "currency": data.get("currency") or "INR", "url": data.get("url") or "", "source": data.get("source") or "manual", "status": data.get("status") or "manual", "observations_json": json.dumps(data.get("observations") or [])}
    with _connect() as db:
        existing = db.execute("SELECT id FROM watchlist WHERE name=? AND variant=? AND retailer=?", (item["name"], item["variant"], item["retailer"])).fetchone()
        if existing and not data.get("id"): item["id"] = existing["id"]
        db.execute("""INSERT INTO watchlist VALUES (:id,:created_at,:updated_at,:name,:variant,:retailer,:target_price,:price,:currency,:url,:source,:status,:observations_json) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,target_price=excluded.target_price,price=excluded.price,currency=excluded.currency,url=excluded.url,source=excluded.source,status=excluded.status,observations_json=excluded.observations_json""", item)
    if data.get("observation"): record_observation(item["id"], data["observation"])
    return item

def update_watchlist(item_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
    current = next((item for item in list_watchlist() if item["id"] == item_id), None)
    if not current: return None
    current.update({key: value for key, value in changes.items() if value is not None or key == "target_price"}); current["id"] = item_id
    return save_watchlist(current)

def delete_watchlist(item_id: str) -> bool:
    init_db()
    with _connect() as db: return db.execute("DELETE FROM watchlist WHERE id=?", (item_id,)).rowcount > 0

def record_observation(watchlist_id: str, data: dict[str, Any]) -> dict[str, Any]:
    init_db(); item = {"id": uuid.uuid4().hex, "watchlist_id": watchlist_id, "offer_id": data.get("offer_id") or "", "retailer": data.get("retailer") or "", "variant": data.get("variant") or "", "price": data["price"], "currency": data.get("currency") or "INR", "url": data.get("url") or "", "source": data.get("source") or "manual", "observed_at": data.get("observed_at") or _now(), "status": data.get("status") or "live"}
    with _connect() as db: db.execute("INSERT INTO price_observations VALUES (:id,:watchlist_id,:offer_id,:retailer,:variant,:price,:currency,:url,:source,:observed_at,:status)", item)
    return item

def list_observations(watchlist_id: str) -> list[dict[str, Any]]:
    init_db()
    with _connect() as db: rows = db.execute("SELECT * FROM price_observations WHERE watchlist_id=? ORDER BY observed_at ASC", (watchlist_id,)).fetchall()
    return [dict(row) for row in rows]

def export_data() -> dict[str, Any]: return {"exported_at": _now(), "decisions": list_decisions(), "watchlist": list_watchlist(), "price_observations": [observation for item in list_watchlist() for observation in list_observations(item["id"])]}

def delete_all_data() -> None:
    init_db()
    with _connect() as db: db.execute("DELETE FROM decisions"); db.execute("DELETE FROM watchlist")
