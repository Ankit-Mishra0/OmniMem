"""
The notes layer: raw turn text plus FTS5/BM25 recall.

Facts answer "what is true about this project". Notes answer "what did we
actually say about X" -- the long tail that is too specific to earn brief space
but is exactly what you want when you go looking for it.

Three bugs from the previous implementation are fixed here, and each is called
out at its site: the FTS ghost row, write-time decay, and the always-true
project condition that made the link graph meaningless.
"""

import math
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from omnimem.core import db

# Half-life for note-to-note association strength. Applied at read time.
LINK_HALF_LIFE_DAYS = 30.0
LINK_MIN_SIMILARITY = 0.20


def sanitize_fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression."""
    clean = re.sub(r"[^\w\s]", " ", query or "")
    tokens = [t for t in (part.strip() for part in clean.split()) if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"*' for t in tokens)


def _row_to_note(r: sqlite3.Row, score: Optional[float] = None) -> Dict[str, Any]:
    note = {
        "id": r["id"],
        "project_id": r["project_id"],
        "content_summary": r["content_summary"],
        "tags": r["tags"].split(",") if r["tags"] else [],
        "platform": r["platform"],
        "timestamp": r["timestamp"],
    }
    if score is not None:
        note["score"] = score
    return note


def save_note(
    conn: sqlite3.Connection,
    project_id: str,
    content_summary: str,
    tags: Sequence[str],
    platform: str = "unknown",
    note_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    note_id = note_id or str(uuid.uuid4())
    timestamp = time.time() if timestamp is None else timestamp
    tags_str = ",".join(tags) if isinstance(tags, (list, tuple)) else str(tags)

    with db.write_txn(conn):
        # ON CONFLICT DO UPDATE, never INSERT OR REPLACE.
        #
        # REPLACE deletes the old row and inserts a new one, but the AFTER
        # DELETE trigger that would clean up notes_fts does NOT fire, because
        # SQLite's recursive_triggers pragma is off by default. Every re-save
        # therefore left a ghost row in the FTS index, and since search joins
        # FTS -> notes, that note came back as duplicate hits forever and
        # skewed BM25's document statistics. DO UPDATE fires notes_au instead,
        # which keeps the index exactly in step.
        conn.execute(
            """
            INSERT INTO notes (id, project_id, content_summary, tags, platform, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                project_id      = excluded.project_id,
                content_summary = excluded.content_summary,
                tags            = excluded.tags,
                platform        = excluded.platform,
                timestamp       = excluded.timestamp
            """,
            (note_id, project_id, content_summary, tags_str, platform, timestamp),
        )
    return {
        "id": note_id,
        "project_id": project_id,
        "content_summary": content_summary,
        "tags": list(tags) if isinstance(tags, (list, tuple)) else [str(tags)],
        "platform": platform,
        "timestamp": timestamp,
    }


def search_notes(
    conn: sqlite3.Connection,
    query: str,
    project_id: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    fts_query = sanitize_fts_query(query)
    results: List[Dict[str, Any]] = []
    seen: set = set()

    if fts_query:
        where = "notes_fts MATCH ?" + (" AND n.project_id = ?" if project_id else "")
        params: List[Any] = [fts_query] + ([project_id] if project_id else []) + [limit]
        try:
            rows = conn.execute(
                f"""
                SELECT n.id, n.project_id, n.content_summary, n.tags, n.platform,
                       n.timestamp, bm25(notes_fts) AS rank
                FROM notes_fts f
                JOIN notes n ON n.id = f.id
                WHERE {where}
                ORDER BY rank ASC, n.timestamp DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            for r in rows:
                # Dedup even on the FTS path. The old code only deduped the LIKE
                # fallback, so ghost rows surfaced as repeats.
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                results.append(_row_to_note(r, abs(float(r["rank"]))))
        except sqlite3.OperationalError:
            pass  # malformed MATCH -> fall through to LIKE

    if len(results) < limit:
        remaining = limit - len(results)
        like = f"%{query}%"
        where = "(content_summary LIKE ? OR tags LIKE ?)" + (
            " AND project_id = ?" if project_id else ""
        )
        params = [like, like] + ([project_id] if project_id else []) + [remaining + len(seen)]
        rows = conn.execute(
            f"""
            SELECT id, project_id, content_summary, tags, platform, timestamp
            FROM notes WHERE {where}
            ORDER BY timestamp DESC LIMIT ?
            """,
            params,
        ).fetchall()
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            results.append(_row_to_note(r, 0.5))
            if len(results) >= limit:
                break

    return results[:limit]


def get_notes_by_project(
    conn: sqlite3.Connection, project_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, project_id, content_summary, tags, platform, timestamp
        FROM notes WHERE project_id = ?
        ORDER BY timestamp DESC LIMIT ?
        """,
        (project_id, limit),
    ).fetchall()
    return [_row_to_note(r) for r in rows]


def upsert_link(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relation_type: str,
    weight: float,
    updated_at: Optional[float] = None,
) -> None:
    """
    Store the *undecayed* similarity. Decay is applied when the edge is read.

    Storing a decayed value was the bug: the multiplier was computed once at
    write time and frozen into the column, so the graph never actually aged.
    """
    updated_at = time.time() if updated_at is None else updated_at
    with db.write_txn(conn):
        conn.execute(
            """
            INSERT INTO links (source_id, target_id, relation_type, weight, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                weight     = MAX(links.weight, excluded.weight),
                updated_at = excluded.updated_at
            """,
            (source_id, target_id, relation_type, weight, updated_at),
        )


def decayed(weight: float, updated_at: float, now: Optional[float] = None) -> float:
    now = time.time() if now is None else now
    age_days = max(0.0, (now - updated_at) / 86400.0)
    return weight * (0.5 ** (age_days / LINK_HALF_LIFE_DAYS))


def get_connected_notes(
    conn: sqlite3.Connection,
    note_id: str,
    limit: int = 5,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT DISTINCT n.id, n.project_id, n.content_summary, n.tags, n.platform,
               n.timestamp, l.weight, l.updated_at, l.relation_type
        FROM links l
        JOIN notes n
          ON n.id = CASE WHEN l.source_id = ? THEN l.target_id ELSE l.source_id END
        WHERE (l.source_id = ? OR l.target_id = ?) AND n.id != ?
        """,
        (note_id, note_id, note_id, note_id),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        note = _row_to_note(r)
        note["link_weight"] = round(decayed(r["weight"], r["updated_at"], now), 4)
        note["relation_type"] = r["relation_type"]
        out.append(note)
    # Ranked here, after decay -- ORDER BY on the stored column would rank by a
    # stale snapshot.
    out.sort(key=lambda n: (-n["link_weight"], n["id"]))
    return out[:limit]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def link_related_notes(
    conn: sqlite3.Connection,
    note: Dict[str, Any],
    scan: int = 20,
) -> int:
    """
    Link a new note to genuinely similar recent notes in the same project.

    No project-membership floor. The old code did `max(sim, 0.3 if
    note.project_id == project_id else 0.0)` over a list that had *already been
    filtered by project*, so the condition was always true, the floor always
    cleared the 0.2 gate, and every note linked to the last 20 -- about 10,000
    edges at 500 notes, all of them encoding only what the project_id column
    already said.
    """
    created = 0
    now = time.time()
    for other in get_notes_by_project(conn, note["project_id"], limit=scan):
        if other["id"] == note["id"]:
            continue
        sim = jaccard(note.get("tags", []), other.get("tags", []))
        if sim >= LINK_MIN_SIMILARITY:
            upsert_link(conn, note["id"], other["id"], "tag_overlap", sim, now)
            created += 1
    return created


def list_projects(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT DISTINCT project_id FROM notes ORDER BY project_id ASC"
    ).fetchall()
    return [r["project_id"] for r in rows]


def fts_row_count(conn: sqlite3.Connection) -> int:
    """Exposed for the ghost-row regression test."""
    return int(conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0])
