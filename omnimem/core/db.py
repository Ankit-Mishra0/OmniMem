"""
SQLite access: cached connections, the write transaction, and the schema.

The one non-obvious rule here is `BEGIN IMMEDIATE` on every write.

Every OmniMem write path is read-then-write: dedup reads candidate facts,
then inserts. Under the default *deferred* transaction, SQLite takes a read
snapshot on the first SELECT, and if another connection commits before the
first write, the upgrade fails with SQLITE_BUSY_SNAPSHOT -- which is returned
*immediately, without invoking the busy handler*. So `busy_timeout` does not
help, and a blind retry re-runs the write against a stale read. Taking the
write lock up front makes `busy_timeout` and retry both meaningful.
"""

import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_local = threading.local()
_depth: Dict[int, int] = {}

SCHEMA_VERSION = 2


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    # We manage transactions explicitly; autocommit mode is required for that.
    conn.isolation_level = None
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_size_limit=16777216")
    conn.execute("PRAGMA mmap_size=134217728")
    # Per-connection, not persisted in the file -- must be re-asserted every time.
    conn.execute("PRAGMA foreign_keys=ON")


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    """
    Thread-local cached connection.

    Caching is the fix for the old per-call `sqlite3.connect()`: one save used
    to open ~23 connections, re-running `PRAGMA journal_mode=WAL` (the most
    expensive pragma) each time, and never closing any of them. The leaked read
    snapshots blocked WAL checkpointing, so the -wal file grew without bound.
    """
    from omnimem.core import store

    resolved = os.path.abspath(str(path or store.db_path()))
    cache = getattr(_local, "conns", None)
    if cache is None:
        cache = _local.conns = {}
    conn = cache.get(resolved)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            cache.pop(resolved, None)

    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=15.0)
    _configure(conn)
    cache[resolved] = conn
    return conn


def close_all() -> None:
    """Drop this thread's cached connections. Short-lived processes call this."""
    cache = getattr(_local, "conns", None) or {}
    for conn in list(cache.values()):
        _depth.pop(id(conn), None)
        try:
            conn.close()
        except sqlite3.Error:
            pass
    cache.clear()


@contextmanager
def session(path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """
    Open, use, close. For hooks and CLI one-shots, where leaving a connection
    open would pin a WAL read snapshot for the life of the process.
    """
    conn = connect(path)
    try:
        yield conn
    finally:
        close_all()


@contextmanager
def write_txn(conn: sqlite3.Connection, retries: int = 8) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE with jittered backoff. Re-entrant (inner calls no-op)."""
    key = id(conn)
    if _depth.get(key, 0) > 0:
        _depth[key] += 1
        try:
            yield conn
        finally:
            _depth[key] -= 1
        return

    delay = 0.02
    for attempt in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 1.0)

    _depth[key] = 1
    try:
        yield conn
    except BaseException:
        _depth[key] = 0
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        _depth[key] = 0
        conn.execute("COMMIT")


# --------------------------------------------------------------------------- schema

_SCHEMA = """
-- The facts store: source of truth. Markdown files are renders of this.
CREATE TABLE IF NOT EXISTS facts (
    id            TEXT PRIMARY KEY,   -- blake2b(kind|normalised text)
    project       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    text          TEXT NOT NULL,      -- canonical wording, frozen at admission
    norm          TEXT NOT NULL,      -- normalised token string, for dedup
    status        TEXT NOT NULL DEFAULT 'active',   -- active|superseded|archived
    superseded_by TEXT,               -- reserved: supersession rules are deferred
    pinned        INTEGER NOT NULL DEFAULT 0,
    first_seen    REAL NOT NULL,
    in_brief      INTEGER NOT NULL DEFAULT 0,
    meta          TEXT                -- json blob: source path, session, etc.
);
CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project, status, kind);

-- Append-only. Counters are DERIVED from this, never incremented in place,
-- which is what makes re-ingesting the same session a true no-op instead of
-- inflating every score it touches.
CREATE TABLE IF NOT EXISTS fact_sightings (
    fact_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool       TEXT NOT NULL,
    ts         REAL NOT NULL,
    authority  REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (fact_id, session_id, tool)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_sightings_fact ON fact_sightings(fact_id);

CREATE TABLE IF NOT EXISTS project_state (
    project       TEXT PRIMARY KEY,
    title         TEXT,
    root          TEXT,
    created       REAL NOT NULL,
    updated       REAL NOT NULL,
    digest_dirty  INTEGER NOT NULL DEFAULT 1,
    body_hash     TEXT,
    handoff_hash  TEXT,
    tasks_hash    TEXT,
    next_step     TEXT,
    last_tool     TEXT,
    last_session  REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    id      TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    tool    TEXT NOT NULL,
    started REAL NOT NULL,
    ended   REAL,
    turns   INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    path    TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project, started DESC);

CREATE TABLE IF NOT EXISTS tasks (
    id        TEXT PRIMARY KEY,
    project   TEXT NOT NULL,
    text      TEXT NOT NULL,
    state     TEXT NOT NULL DEFAULT 'open',   -- open|done
    created   REAL NOT NULL,
    updated   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project, state);

-- Turn-level raw text, kept for full-text recall ("what did we say about X").
CREATE TABLE IF NOT EXISTS notes (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    content_summary TEXT NOT NULL,
    tags            TEXT NOT NULL,
    platform        TEXT NOT NULL,
    timestamp       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_project ON notes(project_id);
CREATE INDEX IF NOT EXISTS idx_notes_timestamp ON notes(timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    id UNINDEXED, project_id, content_summary, tags, platform
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(id, project_id, content_summary, tags, platform)
    VALUES (new.id, new.project_id, new.content_summary, new.tags, new.platform);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    DELETE FROM notes_fts WHERE id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    UPDATE notes_fts SET
        project_id = new.project_id,
        content_summary = new.content_summary,
        tags = new.tags,
        platform = new.platform
    WHERE id = new.id;
END;

CREATE TABLE IF NOT EXISTS links (
    source_id     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight        REAL NOT NULL,     -- undecayed base score; decay is read-time
    updated_at    REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, relation_type),
    FOREIGN KEY (source_id) REFERENCES notes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES notes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_id);
"""


def init_schema(path: Optional[str] = None) -> sqlite3.Connection:
    """
    Create the schema if it isn't there. Safe to call concurrently.

    Deliberately *not* wrapped in write_txn: executescript() issues an implicit
    COMMIT before it runs, which would leave write_txn's own COMMIT with no
    transaction to close. It doesn't need one anyway -- every statement is
    CREATE ... IF NOT EXISTS, so a second process running this at the same time
    either serialises behind the first or finds the work already done.
    """
    conn = connect(path)
    delay = 0.02
    for attempt in range(8):
        try:
            conn.executescript(_SCHEMA)
            break
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if ("locked" not in msg and "busy" not in msg) or attempt == 7:
                raise
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 1.0)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Additive migrations only. Existing `notes` rows are preserved; older
    databases predate the facts tables, which CREATE TABLE IF NOT EXISTS adds.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(facts)")}
    if cols and "pinned" not in cols:
        with write_txn(conn):
            conn.execute("ALTER TABLE facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")


def maintenance(path: Optional[str] = None) -> Dict[str, object]:
    """
    Checkpoint the WAL and compact the FTS index.

    Only ever called from the long-lived server's background pass -- never from
    a hook, where a TRUNCATE checkpoint could block on another writer and eat
    the hook's timeout budget.
    """
    conn = connect(path)
    result: Dict[str, object] = {}
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        result["checkpoint"] = tuple(row) if row else None
    except sqlite3.Error as exc:
        result["checkpoint_error"] = str(exc)
    try:
        with write_txn(conn):
            conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('optimize')")
        result["fts_optimize"] = True
    except sqlite3.Error as exc:
        result["fts_optimize_error"] = str(exc)
    return result
