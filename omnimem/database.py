"""
Backwards-compatible notes API.

The implementation moved to `omnimem.core.notes` (storage) and
`omnimem.core.db` (connections and transactions). This module keeps the old
free-function signatures -- including the `db_path=` keyword -- so existing
callers and tests are unaffected.

The behavioural differences are all bug fixes, documented at their new sites:
connections are cached per thread instead of leaked per call, notes upsert with
ON CONFLICT so the FTS index cannot accumulate ghost rows, and link weights are
stored undecayed and decayed when read.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from omnimem.core import db as _db
from omnimem.core import notes as _notes
from omnimem.core import store as _store
from omnimem.core.notes import sanitize_fts_query  # re-exported

DEFAULT_DB_PATH = _store.db_path()


def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    return _db.connect(db_path)


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    return _conn(db_path)


def init_db(db_path: Optional[str] = None) -> None:
    # Drop cached handles first: a caller that deleted the database file would
    # otherwise get a connection still bound to the old, unlinked inode.
    _db.close_all()
    _db.init_schema(db_path)


def close_connections() -> None:
    _db.close_all()


def save_note(
    project_id: str,
    content_summary: str,
    tags: Sequence[str],
    platform: str = "unknown",
    note_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    return _notes.save_note(
        _conn(db_path), project_id, content_summary, tags, platform, note_id, timestamp
    )


def search_notes(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return _notes.search_notes(_conn(db_path), query, project_id, limit)


def get_notes_by_project(
    project_id: str, limit: int = 10, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    return _notes.get_notes_by_project(_conn(db_path), project_id, limit)


def upsert_link(
    source_id: str,
    target_id: str,
    relation_type: str,
    weight: float,
    updated_at: Optional[float] = None,
    db_path: Optional[str] = None,
) -> None:
    _notes.upsert_link(_conn(db_path), source_id, target_id, relation_type, weight, updated_at)


def get_connected_notes(
    note_id: str, limit: int = 5, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    return _notes.get_connected_notes(_conn(db_path), note_id, limit)


def list_projects(db_path: Optional[str] = None) -> List[str]:
    return _notes.list_projects(_conn(db_path))
