"""
Ingest and digest: everything that turns conversation into files on disk.

Write path, end to end:

    hook/extension/agent
        -> enqueue()            (hooks only: one file into inbox/, ~5ms, exit)
        -> ingest_turn()        redact -> claims -> facts + note + session log
        -> mark_dirty()
        -> digest()             render -> hash gate -> atomic write under flock

Hooks deliberately stop at `enqueue`. They run on every prompt submit and
every assistant turn, inside the editor's latency budget, and they must never
be the reason a coding session stalls. Ingest happens later, in whichever
process next holds the lock.
"""

import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omnimem.core import db, extract, facts as facts_mod, locks, notes, redact, render, store

EVENT_VERSION = 1

# Shipped inside the omnimem_resume tool result rather than in a rules file.
# An instruction in a tool result is read by the model every single session at
# zero token cost to us, which is the cheapest possible way to get the agent to
# volunteer high-quality structured facts on the way out.
RESUME_FOOTER = (
    "\n---\n"
    "You are resuming this project. Continue from the state above without asking "
    "the user to re-explain it.\n"
    "When you finish a meaningful chunk of work, call `omnimem_save_session` with:\n"
    "  facts: [{kind, text}]  — kind is one of "
    "goal|decision|constraint|convention|stack|entity|gotcha|task|open_question; "
    "one atomic claim per entry, under 180 chars, no pleasantries\n"
    "  next_step: the single sentence the next session should start from\n"
)


def _now(ts: Optional[float] = None) -> float:
    return time.time() if ts is None else ts


# --------------------------------------------------------------------------- state

def ensure_state(
    conn: sqlite3.Connection,
    project: str,
    root: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    ts = _now()
    with db.write_txn(conn):
        conn.execute(
            "INSERT INTO project_state(project, title, root, created, updated, digest_dirty)"
            " VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT(project) DO NOTHING",
            (project, title or project, root, ts, ts),
        )
        if root:
            conn.execute(
                "UPDATE project_state SET root = ? WHERE project = ? AND"
                " (root IS NULL OR root = '')",
                (root, project),
            )
        if title:
            conn.execute(
                "UPDATE project_state SET title = ? WHERE project = ?", (title, project)
            )
    store.ensure_project(project)
    row = conn.execute(
        "SELECT * FROM project_state WHERE project = ?", (project,)
    ).fetchone()
    return dict(row) if row else {}


def mark_dirty(conn: sqlite3.Connection, project: str, ts: Optional[float] = None) -> None:
    with db.write_txn(conn):
        conn.execute(
            "UPDATE project_state SET digest_dirty = 1, updated = ? WHERE project = ?",
            (_now(ts), project),
        )


# --------------------------------------------------------------------------- inbox

def enqueue(event: Dict[str, Any]) -> str:
    """
    Drop one event into the inbox. Called by hooks; must stay this cheap.

    One file per event, not a shared append-only log: O_APPEND guarantees
    single-write atomicity only up to PIPE_BUF (512-4096 bytes on macOS/Linux),
    and a captured assistant turn is comfortably larger than that, so
    concurrent hooks writing to one log *will* interleave and corrupt it.
    """
    event.setdefault("v", EVENT_VERSION)
    event.setdefault("ts", time.time())
    inbox = store.inbox_dir()
    os.makedirs(inbox, exist_ok=True)
    path = os.path.join(inbox, f"{event['ts']:.6f}-{uuid.uuid4().hex[:8]}.json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(event, fh)
    return path


def drain_inbox(conn: sqlite3.Connection, limit: int = 500) -> Dict[str, int]:
    """
    Process queued events. Safe to call from anywhere, any number of times.

    Idempotent by construction -- fact ids are content addresses and sightings
    have a composite primary key -- so a double-drain changes nothing.
    """
    inbox = store.inbox_dir()
    if not os.path.isdir(inbox):
        return {"processed": 0, "failed": 0}

    stats = {"processed": 0, "failed": 0}
    lock = os.path.join(inbox, ".drain.lock")
    try:
        with locks.file_lock(lock, timeout=0.05):
            names = sorted(n for n in os.listdir(inbox) if n.endswith(".json"))
            touched: set = set()
            for name in names[:limit]:
                path = os.path.join(inbox, name)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        event = json.load(fh)
                except FileNotFoundError:
                    continue
                except (OSError, ValueError):
                    os.rename(path, path + ".bad")
                    stats["failed"] += 1
                    continue
                try:
                    project = _apply_event(conn, event)
                    if project:
                        touched.add(project)
                    stats["processed"] += 1
                except Exception:
                    os.rename(path, path + ".bad")
                    stats["failed"] += 1
                    continue
                try:
                    os.unlink(path)
                except OSError:
                    pass
            for project in touched:
                mark_dirty(conn, project)
    except locks.LockTimeout:
        return stats  # another process is draining; nothing to do
    return stats


def _apply_event(conn: sqlite3.Connection, event: Dict[str, Any]) -> Optional[str]:
    kind = event.get("kind")
    project = event.get("project") or "default"
    if kind == "turn":
        ingest_turn(
            conn,
            project=project,
            text=event.get("text") or "",
            role=event.get("role") or "user",
            tool=event.get("tool") or "unknown",
            session_id=event.get("session_id") or "adhoc",
            root=event.get("root"),
            ts=event.get("ts"),
            digest_now=False,
        )
        return project
    if kind == "session_end":
        ingest_session(
            conn,
            project=project,
            tool=event.get("tool") or "unknown",
            claims=event.get("facts") or [],
            summary=event.get("summary"),
            next_step=event.get("next_step"),
            session_id=event.get("session_id") or "adhoc",
            root=event.get("root"),
            ts=event.get("ts"),
            digest_now=False,
        )
        return project
    return None


# --------------------------------------------------------------------------- sessions

def _session_path(project: str, session_id: str, tool: str, ts: float) -> str:
    day = time.strftime("%Y-%m-%d", time.localtime(ts))
    safe_tool = store.slugify(tool, "tool")
    name = f"{day}-{safe_tool}-{session_id[:8]}.md"
    return os.path.join(store.sessions_dir(project), name)


def _ensure_session(
    conn: sqlite3.Connection,
    project: str,
    session_id: str,
    tool: str,
    ts: float,
) -> str:
    path = _session_path(project, session_id, tool, ts)
    with db.write_txn(conn):
        conn.execute(
            "INSERT INTO sessions(id, project, tool, started, turns, path)"
            " VALUES (?, ?, ?, ?, 0, ?) ON CONFLICT(id) DO NOTHING",
            (session_id, project, tool, ts, path),
        )
    if not os.path.exists(path):
        # Front matter written once; turns and their extracted claims are
        # appended below it. That keeps sessions/*.md replayable, which is what
        # makes index.db a disposable cache rather than a single point of loss.
        header = (
            "---\n"
            f"session: {session_id}\n"
            f"project: {project}\n"
            f"tool: {tool}\n"
            f"started: {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(ts))}\n"
            "---\n"
        )
        locks.append_line(path, header)
    return path


def _append_turn(
    path: str,
    role: str,
    text: str,
    claims: Sequence[Tuple[str, str]],
    ts: float,
) -> None:
    when = time.strftime("%H:%M:%S", time.localtime(ts))
    block = [f"\n## {when} · {role}", "", text.strip(), ""]
    if claims:
        block.append("<!-- omnimem-facts")
        for kind, claim in claims:
            block.append(f"{kind}: {claim}")
        block.append("-->")
    locks.append_line(path, "\n".join(block) + "\n")


def replay_session_file(path: str) -> List[Tuple[str, str]]:
    """Read claims back out of a session file. Used to rebuild index.db."""
    out: List[Tuple[str, str]] = []
    inside = False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("<!-- omnimem-facts"):
                    inside = True
                    continue
                if inside:
                    if line.startswith("-->"):
                        inside = False
                        continue
                    if ":" in line:
                        kind, claim = line.split(":", 1)
                        out.append((kind.strip(), claim.strip()))
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------- ingest

def ingest_claims(
    conn: sqlite3.Connection,
    project: str,
    claims: Iterable[Tuple[str, str]],
    session_id: str,
    tool: str,
    source: str = "heuristic",
    ts: Optional[float] = None,
) -> Dict[str, int]:
    counts = {"new": 0, "seen": 0, "duplicate": 0, "subsumed": 0, "rejected": 0}
    ts = _now(ts)
    for kind, text in claims:
        result = facts_mod.upsert_fact(
            conn,
            project=project,
            kind=kind,
            text=text,
            session_id=session_id,
            tool=tool,
            source=source,
            ts=ts,
        )
        counts[result["action"]] = counts.get(result["action"], 0) + 1
    return counts


def ingest_turn(
    conn: sqlite3.Connection,
    project: str,
    text: str,
    role: str = "user",
    tool: str = "unknown",
    session_id: str = "adhoc",
    root: Optional[str] = None,
    ts: Optional[float] = None,
    digest_now: bool = True,
) -> Dict[str, Any]:
    """One conversational turn: redact, extract, store, log."""
    ts = _now(ts)
    cfg = store.load_config()
    text = (text or "").strip()
    if cfg.get("redact", True):
        text = redact.redact(text)
    if not text or not extract.is_important(text):
        return {"status": "skipped", "reason": "importance gating"}

    state = ensure_state(conn, project, root=root)
    root = root or state.get("root")

    claims = extract.extract_claims(text, role=role)
    if role == "assistant":
        claims += extract.extract_entities(text, root)

    counts = ingest_claims(
        conn, project, claims, session_id=session_id, tool=tool, source="heuristic", ts=ts
    )

    note = notes.save_note(
        conn,
        project_id=project,
        content_summary=text[:4000],
        tags=extract.turn_tags(text),
        platform=tool,
        timestamp=ts,
    )
    notes.link_related_notes(conn, note)

    path = _ensure_session(conn, project, session_id, tool, ts)
    _append_turn(path, role, text, claims, ts)
    with db.write_txn(conn):
        conn.execute(
            "UPDATE sessions SET turns = turns + 1, ended = ? WHERE id = ?", (ts, session_id)
        )
        conn.execute(
            "UPDATE project_state SET digest_dirty = 1, updated = ?, last_tool = ?,"
            " last_session = ? WHERE project = ?",
            (ts, tool, ts, project),
        )

    result = {"status": "saved", "project": project, "claims": counts, "note_id": note["id"]}
    if digest_now:
        result["digest"] = digest(conn, project, root=root)
    return result


def ingest_session(
    conn: sqlite3.Connection,
    project: str,
    tool: str,
    claims: Optional[Sequence[Any]] = None,
    summary: Optional[str] = None,
    next_step: Optional[str] = None,
    transcript: Optional[str] = None,
    session_id: Optional[str] = None,
    root: Optional[str] = None,
    ts: Optional[float] = None,
    digest_now: bool = True,
) -> Dict[str, Any]:
    """
    The high-authority path: an agent handing over what it learned.

    Structured `claims` outrank a written `summary`, which outranks anything the
    regex tier guessed -- see facts.AUTHORITY for why agent authority is 1.7
    rather than 3.0.
    """
    ts = _now(ts)
    session_id = session_id or uuid.uuid4().hex
    state = ensure_state(conn, project, root=root)
    root = root or state.get("root")
    cfg = store.load_config()
    scrub = (lambda s: redact.redact(s) if s and cfg.get("redact", True) else s)

    normalised: List[Tuple[str, str]] = []
    for item in claims or []:
        if isinstance(item, dict):
            kind = str(item.get("kind") or "decision")
            text = scrub(str(item.get("text") or ""))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            kind, text = str(item[0]), scrub(str(item[1]))
        else:
            kind, text = "decision", scrub(str(item))
        if text:
            normalised.append((kind, text))

    counts = ingest_claims(
        conn, project, normalised, session_id=session_id, tool=tool,
        source="agent_structured", ts=ts,
    )

    if summary:
        summary = scrub(summary)
        prose_claims = extract.extract_claims(summary, role="assistant")
        prose_counts = ingest_claims(
            conn, project, prose_claims, session_id=session_id, tool=tool,
            source="agent_prose", ts=ts,
        )
        for key, value in prose_counts.items():
            counts[key] = counts.get(key, 0) + value

    if root:
        stack_counts = ingest_claims(
            conn, project, extract.scan_stack(root), session_id=session_id,
            tool=tool, source="heuristic", ts=ts,
        )
        for key, value in stack_counts.items():
            counts[key] = counts.get(key, 0) + value

    path = _ensure_session(conn, project, session_id, tool, ts)
    body_parts = []
    if summary:
        body_parts.append(summary.strip())
    if transcript:
        body_parts.append(scrub(transcript)[:20000])
    if body_parts:
        _append_turn(path, "summary", "\n\n".join(body_parts), normalised, ts)

    if summary:
        note = notes.save_note(
            conn,
            project_id=project,
            content_summary=summary[:4000],
            tags=extract.turn_tags(summary),
            platform=tool,
            timestamp=ts,
        )
        notes.link_related_notes(conn, note)

    with db.write_txn(conn):
        conn.execute(
            "UPDATE sessions SET ended = ?, summary = COALESCE(?, summary) WHERE id = ?",
            (ts, summary, session_id),
        )
        conn.execute(
            "UPDATE project_state SET digest_dirty = 1, updated = ?, last_tool = ?,"
            " last_session = ?, next_step = COALESCE(?, next_step) WHERE project = ?",
            (ts, tool, session_id, scrub(next_step) if next_step else None, project),
        )

    result = {
        "status": "saved",
        "project": project,
        "session_id": session_id,
        "claims": counts,
    }
    if digest_now:
        result["digest"] = digest(conn, project, root=root)
    return result


def add_decision(
    conn: sqlite3.Connection,
    project: str,
    text: str,
    tool: str = "cli",
    session_id: str = "manual",
    rationale: Optional[str] = None,
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Record a decision as a fact *and* as an immutable DECISIONS.md block."""
    ts = _now(ts)
    ensure_state(conn, project)
    text = redact.redact(text.strip())
    result = facts_mod.upsert_fact(
        conn, project=project, kind="decision", text=text,
        session_id=session_id, tool=tool, source="human", ts=ts,
    )
    block = render.decision_block(
        text if not rationale else f"{text}\n\n{redact.redact(rationale)}",
        tool, ts, session_id,
    )
    store.ensure_project(project)
    with locks.file_lock(store.lock_path(project)):
        locks.append_line(os.path.join(store.project_dir(project), "DECISIONS.md"), block)
    mark_dirty(conn, project, ts)
    return result


def set_task(
    conn: sqlite3.Connection,
    project: str,
    text: str,
    state: str = "open",
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    ts = _now(ts)
    ensure_state(conn, project)
    text = redact.redact(text.strip())
    key = facts_mod.canonical(text)
    task_id = facts_mod.fingerprint("task", text)
    if not key:
        return {"status": "rejected"}
    with db.write_txn(conn):
        conn.execute(
            "INSERT INTO tasks(id, project, text, state, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET state = excluded.state, updated = excluded.updated",
            (task_id, project, text, state, ts, ts),
        )
    if state == "open":
        facts_mod.upsert_fact(
            conn, project=project, kind="task", text=text,
            session_id="tasks", tool="cli", source="human", ts=ts,
        )
    else:
        # A completed task should stop competing for brief space immediately;
        # waiting for the 3-day half-life would keep stale work visible.
        with db.write_txn(conn):
            conn.execute(
                "UPDATE facts SET status = 'archived' WHERE id = ? AND kind = 'task'",
                (task_id,),
            )
    mark_dirty(conn, project, ts)
    return {"status": "ok", "id": task_id, "state": state}


def _sync_tasks_from_disk(conn: sqlite3.Connection, project: str) -> int:
    """Pull human checkbox edits out of TASKS.md. The one inbound file."""
    path = os.path.join(store.project_dir(project), "TASKS.md")
    if not os.path.exists(path):
        return 0
    disk = render.parse_tasks_md(path)
    if not disk:
        return 0
    changed = 0
    rows = conn.execute(
        "SELECT id, text, state FROM tasks WHERE project = ?", (project,)
    ).fetchall()
    ts = _now()
    for row in rows:
        key = facts_mod.canonical(row["text"])
        if key not in disk:
            continue
        wanted = "done" if disk[key] else "open"
        if wanted != row["state"]:
            with db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET state = ?, updated = ? WHERE id = ?",
                    (wanted, ts, row["id"]),
                )
                conn.execute(
                    "UPDATE facts SET status = ? WHERE id = ? AND kind = 'task'",
                    ("archived" if wanted == "done" else "active", row["id"]),
                )
            changed += 1
    return changed


# --------------------------------------------------------------------------- digest

def digest(
    conn: sqlite3.Connection,
    project: str,
    root: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Render every file for a project and write only what changed.

    Everything here is inside the project's flock, and every write is an atomic
    same-directory temp-swap, so a reader never sees a partial brief.
    """
    row = conn.execute(
        "SELECT * FROM project_state WHERE project = ?", (project,)
    ).fetchone()
    state = dict(row) if row else ensure_state(conn, project, root=root)
    if not force and not state.get("digest_dirty"):
        return {"status": "clean", "project": project}

    store.ensure_project(project)
    cfg = store.load_config()
    project_path = store.project_dir(project)
    written: List[str] = []

    with locks.file_lock(store.lock_path(project), timeout=15.0):
        _sync_tasks_from_disk(conn, project)

        # Manifest scan. Ground truth, so it is worth doing here rather than
        # only on the agent-driven save path -- a hooks-only setup never calls
        # ingest_session and would otherwise never learn the stack. Skipped once
        # stack facts exist, since re-reading five manifests on every render buys
        # nothing; `omnimem digest --force` refreshes it after a dependency
        # change.
        scan_root = root or state.get("root")
        if scan_root:
            has_stack = conn.execute(
                "SELECT 1 FROM facts WHERE project = ? AND kind = 'stack'"
                " AND status = 'active' LIMIT 1",
                (project,),
            ).fetchone()
            if force or not has_stack:
                ingest_claims(
                    conn, project, extract.scan_stack(scan_root),
                    session_id="manifest-scan", tool="omnimem",
                    source="heuristic",
                )

        all_facts = facts_mod.load_facts(conn, project)
        session_rows = conn.execute(
            "SELECT id, tool, started, ended, summary FROM sessions"
            " WHERE project = ? ORDER BY started DESC",
            (project,),
        ).fetchall()
        session_count = len(session_rows)
        last_session = dict(session_rows[0]) if session_rows else None
        task_rows = conn.execute(
            "SELECT id, text, state, updated FROM tasks WHERE project = ?"
            " ORDER BY updated DESC",
            (project,),
        ).fetchall()
        open_tasks = [dict(r) for r in task_rows if r["state"] == "open"]
        done_tasks = [dict(r) for r in task_rows if r["state"] == "done"]

        brief = render.render_project_md(
            slug=project,
            title=state.get("title") or project,
            all_facts=all_facts,
            next_step=state.get("next_step"),
            pinned_block=render.read_pinned(project_path),
            budget=int(cfg.get("brief_chars", 2400)),
            session_count=session_count,
            last_tool=state.get("last_tool"),
        )
        # The gate compares the *body* hash only, so the dated trailer cannot
        # force a rewrite of an otherwise unchanged brief -- and an unchanged
        # brief is not touched at all, leaving mtime alone for file watchers.
        body_only = locks.body_hash(brief.body)
        if state.get("body_hash") != body_only:
            locks.atomic_write(os.path.join(project_path, "PROJECT.md"), brief.text)
            written.append("PROJECT.md")

        handoff = render.render_handoff_md(
            slug=project,
            title=state.get("title") or project,
            last_session=last_session,
            next_step=state.get("next_step"),
            recent_facts=sorted(all_facts, key=lambda f: -f["last_seen"])[:10],
            open_tasks=[t["text"] for t in open_tasks],
        )
        handoff_body = locks.body_hash(handoff.body)
        if state.get("handoff_hash") != handoff_body:
            locks.atomic_write(os.path.join(project_path, "HANDOFF.md"), handoff.text)
            written.append("HANDOFF.md")

        tasks_md = render.render_tasks_md(
            state.get("title") or project, open_tasks, done_tasks
        )
        tasks_body = locks.body_hash(tasks_md.body)
        if state.get("tasks_hash") != tasks_body:
            locks.atomic_write(os.path.join(project_path, "TASKS.md"), tasks_md.text)
            written.append("TASKS.md")

        counts_by_kind: Dict[str, int] = {}
        for item in all_facts:
            counts_by_kind[item["kind"]] = counts_by_kind.get(item["kind"], 0) + 1
        meta = render.render_meta_json(
            slug=project,
            title=state.get("title") or project,
            root=state.get("root"),
            counts=counts_by_kind,
            session_count=session_count,
            last_tool=state.get("last_tool"),
            next_step=state.get("next_step"),
            updated=state.get("updated") or _now(),
        )
        locks.write_if_changed(os.path.join(project_path, "meta.json"), meta.text)

        in_brief = set(brief.fact_ids)
        with db.write_txn(conn):
            conn.execute(
                "UPDATE facts SET in_brief = 0 WHERE project = ? AND in_brief = 1", (project,)
            )
            for fid in in_brief:
                conn.execute("UPDATE facts SET in_brief = 1 WHERE id = ?", (fid,))
            conn.execute(
                "UPDATE project_state SET digest_dirty = 0, body_hash = ?, handoff_hash = ?,"
                " tasks_hash = ? WHERE project = ?",
                (body_only, handoff_body, tasks_body, project),
            )

    return {
        "status": "rendered",
        "project": project,
        "written": written,
        "facts": len(all_facts),
        "in_brief": len(in_brief),
        "brief_chars": len(brief.text),
    }


def digest_all(conn: sqlite3.Connection, force: bool = False) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT project FROM project_state").fetchall()
    return [digest(conn, r["project"], force=force) for r in rows]


# --------------------------------------------------------------------------- read

def resume(
    conn: sqlite3.Connection,
    project: str,
    root: Optional[str] = None,
    with_footer: bool = True,
) -> str:
    """The brief, ready to inject. Drains and re-renders first if needed."""
    ensure_state(conn, project, root=root)
    drain_inbox(conn)
    digest(conn, project, root=root)
    path = os.path.join(store.project_dir(project), "PROJECT.md")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
    except OSError:
        body = ""
    if not body.strip():
        body = (
            f"# {project}\n\nNo memory recorded for this project yet. "
            "Work normally; OmniMem will start building the brief.\n"
        )
    return body + (RESUME_FOOTER if with_footer else "")


def overview(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT p.project, p.title, p.root, p.updated, p.last_tool, p.next_step,
               (SELECT COUNT(*) FROM facts f WHERE f.project = p.project
                  AND f.status = 'active') AS facts,
               (SELECT COUNT(*) FROM sessions s WHERE s.project = p.project) AS sessions
        FROM project_state p ORDER BY p.updated DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def search(
    conn: sqlite3.Connection,
    query: str,
    project: Optional[str] = None,
    limit: int = 8,
) -> Dict[str, Any]:
    """Facts first (they are the distilled answer), then raw turns."""
    matched_facts: List[Dict[str, Any]] = []
    if project:
        wanted = facts_mod.token_set(query)
        for item in facts_mod.load_facts(conn, project):
            overlap = facts_mod.containment(wanted, facts_mod.token_set(item["text"]))
            if overlap >= 0.34:
                matched_facts.append({**item, "match": round(overlap, 2)})
        matched_facts.sort(key=lambda f: (-f["match"], -f["score"]))
        matched_facts = matched_facts[:limit]
    return {
        "query": query,
        "project": project,
        "facts": matched_facts,
        "notes": notes.search_notes(conn, query, project_id=project, limit=limit),
    }
