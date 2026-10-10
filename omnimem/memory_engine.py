"""
Backwards-compatible note-processing pipeline.

The real logic now lives in `omnimem.core` -- `extract` for gating and tagging,
`facts` for similarity, `notes` for storage and the link graph. This module
keeps the original function names and signatures.

Two behaviours changed here, both deliberately:

* `calculate_decayed_weight` is no longer used to *store* edge weights. Weights
  go into the database undecayed and `core.notes.get_connected_notes` decays
  them on read, so the graph actually ages. The function stays, because it is
  public API and still the right formula.
* `update_zettelkasten_links` no longer applies a 0.3 floor to notes in the
  same project. That floor was applied to a list already filtered by project,
  so it fired on every pair and every note linked to the last 20 regardless of
  content. Links now mean tag overlap and nothing else.
"""

import math
import re
import time
from typing import Any, Dict, List, Optional

from omnimem.core import extract as _extract
from omnimem.core import facts as _facts
from omnimem.core import notes as _notes
from omnimem.core import store as _store
from omnimem.core.extract import FLUFF_PATTERNS, TECHNICAL_KEYWORDS, is_important
from omnimem.database import (
    DEFAULT_DB_PATH,
    get_connected_notes,
    get_db_connection,
    get_notes_by_project,
    save_note,
    search_notes,
    upsert_link,
    _conn,
)

# Kept for API compatibility. ~4 h half-life, which is why per-kind half-lives
# in core.facts replaced it for fact scoring; it is still fine for note links,
# where core.notes.LINK_HALF_LIFE_DAYS (30 d) is the value actually used.
DECAY_LAMBDA = 0.00005

MIN_LINK_WEIGHT = _notes.LINK_MIN_SIMILARITY


def extract_tags(text: str, limit: int = 8) -> List[str]:
    """Keyword tags, technical terms first so they survive the cap."""
    seen = set()
    ranked: List[str] = []
    stemmed = _facts.tokens(text)
    raw = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", (text or "").lower())]
    tech = [w for w in raw if w in TECHNICAL_KEYWORDS]
    for tag in tech + stemmed:
        if tag in seen:
            continue
        seen.add(tag)
        ranked.append(tag)
        if len(ranked) >= limit:
            break
    return ranked or ["general"]


def extract_structured_summary(text: str) -> Dict[str, Any]:
    """
    Structured view of one chunk of text: {topic, decisions, stack, current_task}.

    Unlike the original, the typed claims behind this are the same ones the
    facts store ingests -- so what you see here is what actually gets persisted,
    rather than a parallel extraction that was computed and discarded.
    """
    claims = _extract.extract_claims(text, role="user") + _extract.extract_claims(
        text, role="assistant"
    )
    by_kind: Dict[str, List[str]] = {}
    for kind, body in claims:
        by_kind.setdefault(kind, []).append(body)

    topic = _extract.guess_title(text, fallback="General Note")[:80]
    tags = extract_tags(text)
    decisions = by_kind.get("decision", []) or [topic]
    task = (by_kind.get("task") or by_kind.get("goal") or [topic])[0]
    stack = [w for w in tags if w in TECHNICAL_KEYWORDS] or tags[:4]

    return {
        "topic": topic,
        "decisions": decisions,
        "stack": stack,
        "current_task": task,
        "summary": text[:500],
        "tags": tags,
        "claims": claims,
    }


def calculate_jaccard_similarity(tags1: List[str], tags2: List[str]) -> float:
    return _notes.jaccard(tags1, tags2)


def calculate_decayed_weight(
    initial_score: float, elapsed_seconds: float, decay_lambda: float = DECAY_LAMBDA
) -> float:
    """W = S * e^(-lambda * t)."""
    return initial_score * math.exp(-decay_lambda * max(0.0, elapsed_seconds))


def update_zettelkasten_links(
    new_note: Dict[str, Any], db_path: Optional[str] = None
) -> int:
    return _notes.link_related_notes(_conn(db_path), new_note, scan=20)


def process_and_store_memory(
    project_id: str,
    content: str,
    platform: str = "unknown",
    tags: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Gate -> tag -> store -> link. Returns None if the content was gated out."""
    if not is_important(content):
        return None

    conn = _conn(db_path)
    final_tags = list(tags) if tags else extract_tags(content)
    note = _notes.save_note(
        conn,
        project_id=project_id,
        content_summary=content.strip(),
        tags=final_tags,
        platform=platform,
    )
    _notes.link_related_notes(conn, note, scan=20)
    return note


def build_active_context(
    project_id: str, limit: int = 5, db_path: Optional[str] = None
) -> str:
    """
    The block a tool injects to get up to speed.

    Prefers the rendered brief at ~/.omnimem/projects/<slug>/PROJECT.md, which is
    the curated fact-based view, and falls back to recent raw notes for projects
    that have only ever been written through the legacy notes API.
    """
    conn = _conn(db_path)
    header = f"### ⚡ OmniMem Synced Context [{project_id}]\n"

    brief = ""
    try:
        path = _store.project_dir(_store.slugify(project_id)) + "/PROJECT.md"
        with open(path, "r", encoding="utf-8") as fh:
            brief = fh.read().strip()
    except OSError:
        brief = ""

    notes = _notes.get_notes_by_project(conn, project_id, limit=limit)
    if not brief and not notes:
        return f"<!-- OMNIMEM CONTEXT: Project '{project_id}' (No memory stored yet) -->"

    lines = [header]
    if brief:
        lines.append(brief)
        lines.append("")

    if notes:
        lines.append("#### Recent activity")
        for i, note in enumerate(notes, 1):
            tags_fmt = f"[{', '.join(note['tags'])}]" if note["tags"] else ""
            lines.append(f"**Memory #{i}** ({note['platform']} | {tags_fmt}):")
            lines.append(f"> {note['content_summary']}\n")
            if i == 1:
                connected = _notes.get_connected_notes(conn, note["id"], limit=3)
                if connected:
                    lines.append("*Related Context Graph:*")
                    for c in connected:
                        lines.append(
                            f"- Relative Note ({c['project_id']}): "
                            f"{c['content_summary'][:100]}..."
                        )
                    lines.append("")

    return "\n".join(lines)
