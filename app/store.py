"""SQLite persistence: runs, share links, and a TTL cache for search/fetch results."""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from app.config import settings

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    depth TEXT NOT NULL,
    status TEXT NOT NULL,            -- running | done | error
    created_at REAL NOT NULL,
    report_json TEXT,
    trace_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS shares (
    slug TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS shares_run ON shares(run_id);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(SCHEMA)
    return _conn


def _exec(sql: str, args: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        c = conn()
        cur = c.execute(sql, args)
        c.commit()
        return cur


# ---------------------------------------------------------------- runs

def create_run(run_id: str, question: str, depth: str) -> None:
    _exec("INSERT INTO runs(id, question, depth, status, created_at) VALUES (?,?,?,?,?)",
          (run_id, question, depth, "running", time.time()))


def finish_run(run_id: str, report: dict, trace: list[dict]) -> None:
    _exec("UPDATE runs SET status='done', report_json=?, trace_json=? WHERE id=?",
          (json.dumps(report, default=str), json.dumps(trace, default=str), run_id))


def fail_run(run_id: str, error: str, trace: list[dict]) -> None:
    _exec("UPDATE runs SET status='error', error=?, trace_json=? WHERE id=?",
          (error, json.dumps(trace, default=str), run_id))


def get_run(run_id: str) -> Optional[dict]:
    row = _exec("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["report"] = json.loads(d.pop("report_json") or "null")
    d["trace"] = json.loads(d.pop("trace_json") or "[]")
    share = _exec("SELECT slug FROM shares WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
    d["share_slug"] = share["slug"] if share else None
    return d


def list_runs(limit: int = 30) -> list[dict]:
    rows = _exec("SELECT id, question, depth, status, created_at FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows.fetchall()]


def delete_run(run_id: str) -> None:
    _exec("DELETE FROM runs WHERE id=?", (run_id,))


def mark_interrupted_runs() -> None:
    _exec("UPDATE runs SET status='error', error='Server restarted during run' WHERE status='running'")


# ---------------------------------------------------------------- shares

def create_share(run_id: str) -> str:
    existing = _exec("SELECT slug FROM shares WHERE run_id=?", (run_id,)).fetchone()
    if existing:
        return existing["slug"]
    slug = secrets.token_urlsafe(9)
    _exec("INSERT INTO shares(slug, run_id, created_at) VALUES (?,?,?)", (slug, run_id, time.time()))
    return slug


def revoke_share(run_id: str) -> None:
    _exec("DELETE FROM shares WHERE run_id=?", (run_id,))


def run_for_share(slug: str) -> Optional[str]:
    row = _exec("SELECT run_id FROM shares WHERE slug=?", (slug,)).fetchone()
    return row["run_id"] if row else None


# ---------------------------------------------------------------- cache

LONG_TTL_PREFIX = "laya:"  # deterministic model outputs: keep 30 days
LONG_TTL_HOURS = 24 * 30


def _ttl_hours(key: str) -> float:
    return LONG_TTL_HOURS if key.startswith(LONG_TTL_PREFIX) else settings.CACHE_TTL_HOURS


def cache_get(key: str) -> Any:
    row = _exec("SELECT value, created_at FROM cache WHERE key=?", (key,)).fetchone()
    if not row or time.time() - row["created_at"] > _ttl_hours(key) * 3600:
        return None
    return json.loads(row["value"])


def cache_get_many(keys: list[str]) -> dict[str, Any]:
    if not keys:
        return {}
    out: dict[str, Any] = {}
    now = time.time()
    for i in range(0, len(keys), 500):
        chunk = keys[i: i + 500]
        rows = _exec(f"SELECT key, value, created_at FROM cache WHERE key IN ({','.join('?' * len(chunk))})", tuple(chunk))
        for r in rows.fetchall():
            if now - r["created_at"] <= _ttl_hours(r["key"]) * 3600:
                out[r["key"]] = json.loads(r["value"])
    return out


def cache_put(key: str, value: Any) -> None:
    _exec("INSERT OR REPLACE INTO cache(key, value, created_at) VALUES (?,?,?)",
          (key, json.dumps(value, default=str), time.time()))


def cache_prune() -> None:
    now = time.time()
    _exec("DELETE FROM cache WHERE key NOT LIKE ? AND created_at < ?",
          (LONG_TTL_PREFIX + "%", now - settings.CACHE_TTL_HOURS * 3600))
    _exec("DELETE FROM cache WHERE key LIKE ? AND created_at < ?", (LONG_TTL_PREFIX + "%", now - LONG_TTL_HOURS * 3600))
