"""
Facts: normalisation, fingerprinting, dedup, and read-time scoring.

A *fact* is one atomic claim of at most 180 characters. Its identity is
content-derived, so the same claim arriving from Claude Code and from ChatGPT
lands on the same row instead of becoming two near-duplicates in the brief.
"""

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from omnimem.core import db

MAX_FACT_CHARS = 180

KINDS = (
    "goal",
    "decision",
    "constraint",
    "convention",
    "stack",
    "entity",
    "gotcha",
    "task",
    "open_question",
)

# How much a kind is worth in the brief at equal evidence. Goals and decisions
# are why a new session can pick up; entities are cheap to re-derive by looking.
KIND_PRIOR = {
    "goal": 1.60,
    "decision": 1.40,
    "constraint": 1.35,
    "gotcha": 1.30,
    "stack": 1.20,
    "convention": 1.15,
    "task": 1.10,
    "entity": 0.90,
    "open_question": 0.85,
}

# Per-kind half-lives in days. A single global lambda is simply wrong: a task
# is stale in a week, the project's goal is not. (The value this replaces,
# DECAY_LAMBDA = 5e-5/s, was a ~4 hour half-life applied to everything.)
HALF_LIFE_DAYS = {
    "goal": 240.0,
    "decision": 180.0,
    "constraint": 180.0,
    "convention": 180.0,
    "gotcha": 120.0,
    "stack": 120.0,
    "entity": 90.0,
    "open_question": 21.0,
    "task": 3.0,
}

# Who said it. Not 3.0 for agents on purpose: one confident hallucination
# should beat one heuristic guess, but should not outrank a claim that three
# separate sessions across two tools kept confirming.
AUTHORITY = {
    "heuristic": 1.0,
    "agent_prose": 1.3,
    "agent_structured": 1.7,
    "human": 4.0,
}

# An old but corroborated constraint must never be pushed out of the brief by
# today's chatter, so recency is a modest multiplier with a hard floor.
RECENCY_FLOOR = 0.15

_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have
he her him his how i if in into is it its me my no not of on or our out own she so
than that the their them then there these they this those to too us was we were what
when where which who why will with would you your also just very really should may
anything something nothing everything anyone someone everyone thing things
""".split())

# Folded before the token set is built, so "chose SQLite over Postgres" and
# "Postgres was rejected in favour of SQLite" collapse to the same fingerprint.
_SYNONYMS = {
    "db": "database", "dbs": "database", "postgresql": "postgres", "pg": "postgres",
    "sqlite3": "sqlite", "js": "javascript", "ts": "typescript", "py": "python",
    "config": "configuration", "cfg": "configuration", "env": "environment",
    "auth": "authentication", "repo": "repository", "dir": "directory",
    "docs": "documentation", "doc": "documentation", "deps": "dependency",
    "dependencies": "dependency", "lib": "library", "libs": "library",
    "pkg": "package", "impl": "implementation", "perf": "performance",
    "k8s": "kubernetes", "ci": "continuousintegration", "e2e": "endtoend",
    "chose": "choose", "chosen": "choose", "picked": "choose", "pick": "choose",
    "selected": "choose", "select": "choose", "decided": "choose",
    "decide": "choose", "decision": "choose", "going": "choose",
    "using": "use", "used": "use", "uses": "use", "utilise": "use",
    "utilize": "use", "adopt": "use", "adopted": "use",
    "rejected": "reject", "reject": "reject", "dropped": "reject",
    "avoid": "reject", "avoided": "reject", "instead": "reject",
    "must": "require", "required": "require", "requires": "require",
    "need": "require", "needs": "require", "needed": "require",
    "cannot": "reject", "cant": "reject", "wont": "reject",
    "bug": "bug", "bugs": "bug", "broken": "bug", "breaks": "bug",
    "fixed": "fix", "fixes": "fix", "fixing": "fix",
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_./\-]*")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- text

def clean_text(text: str) -> str:
    """Collapse whitespace, strip markdown noise, clamp length."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("`", "").replace("**", "").replace("__", "")
    t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", t)
    t = _WS_RE.sub(" ", t).strip(" \t-–—:;")
    if len(t) > MAX_FACT_CHARS:
        cut = t[:MAX_FACT_CHARS]
        # Prefer a word boundary so the fact does not end mid-token.
        space = cut.rfind(" ")
        t = (cut[:space] if space > MAX_FACT_CHARS * 0.6 else cut).rstrip() + "…"
    return t


def _stem(word: str) -> str:
    if len(word) > 5:
        for suffix in ("ing", "ies", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
    return word


def tokens(text: str) -> List[str]:
    """Content tokens: lowercased, de-stopworded, synonym-folded, stemmed."""
    out: List[str] = []
    for raw in _WORD_RE.findall((text or "").lower()):
        word = raw.strip("._-/")
        if not word or word in _STOPWORDS:
            continue
        word = _SYNONYMS.get(word, word)
        if word in _STOPWORDS:
            continue
        out.append(_stem(word))
    return out


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def canonical(text: str) -> str:
    """Order-independent normalised form. This is the dedup key material."""
    return " ".join(sorted(token_set(text)))


def fingerprint(kind: str, text: str) -> str:
    """
    Content address of a fact.

    blake2b, never the builtin hash(): hash() is salted per process by
    PYTHONHASHSEED, so a hook subprocess and the server would compute different
    ids for the same claim and duplicate every single fact.
    """
    material = f"{kind}|{canonical(text)}".encode("utf-8")
    return hashlib.blake2b(material, digest_size=16).hexdigest()


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(small: Set[str], large: Set[str]) -> float:
    """Fraction of `small` present in `large`."""
    if not small:
        return 0.0
    return len(small & large) / len(small)


_SPECIFIC_RE = re.compile(r"(\d|/|\.py|\.js|\.ts|\.json|\.md|::|\(\)|_)")


def specificity(text: str) -> float:
    """
    Concrete claims earn a small bonus. "use SQLite WAL with busy_timeout=15000"
    is worth more brief space than "improve the database".
    """
    score = 1.0
    if _SPECIFIC_RE.search(text):
        score += 0.15
    n_tokens = len(token_set(text))
    if n_tokens >= 6:
        score += 0.10
    elif n_tokens <= 2:
        score -= 0.10
    return max(0.85, min(1.25, score))


def recency(kind: str, last_seen: float, now: Optional[float] = None) -> float:
    now = now if now is not None else time.time()
    age_days = max(0.0, (now - last_seen) / 86400.0)
    half_life = HALF_LIFE_DAYS.get(kind, 90.0)
    return max(RECENCY_FLOOR, 0.5 ** (age_days / half_life))


def score_fact(
    kind: str,
    sightings: int,
    tools: int,
    last_seen: float,
    authority: float,
    text: str,
    pinned: bool = False,
    now: Optional[float] = None,
) -> float:
    """
    Decay is evaluated *here*, at read time, from the stored last_seen -- not
    frozen into a column at write time, which is why the previous decay
    implementation never actually decayed anything.
    """
    base = (
        KIND_PRIOR.get(kind, 1.0)
        * max(1.0, authority)
        * (1.0 + math.log2(1.0 + max(0, sightings)))
        * (1.0 + 0.35 * max(0, tools - 1))
        * recency(kind, last_seen, now)
        * specificity(text)
    )
    if pinned:
        base *= 2.0
    # Quantised so tiny float drift cannot reorder the brief between renders.
    return round(base, 3)


# --------------------------------------------------------------------------- writes

def _record_sighting(
    conn: sqlite3.Connection,
    fact_id: str,
    session_id: str,
    tool: str,
    ts: float,
    authority: float,
) -> None:
    # INSERT OR IGNORE + a composite PK is the whole idempotency story: replaying
    # the same session file cannot inflate a fact's evidence.
    conn.execute(
        "INSERT OR IGNORE INTO fact_sightings(fact_id, session_id, tool, ts, authority)"
        " VALUES (?, ?, ?, ?, ?)",
        (fact_id, session_id, tool, ts, authority),
    )


def _candidates(conn: sqlite3.Connection, project: str, kind: str) -> List[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT id, text, norm FROM facts"
            " WHERE project = ? AND kind = ? AND status = 'active'",
            (project, kind),
        )
    )


def upsert_fact(
    conn: sqlite3.Connection,
    project: str,
    kind: str,
    text: str,
    session_id: str = "manual",
    tool: str = "cli",
    source: str = "heuristic",
    pinned: bool = False,
    meta: Optional[Dict[str, Any]] = None,
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Admit a claim. Returns {"action", "id", "text"} where action is one of
    new | seen | duplicate | subsumed.

    Dedup cascade:
      1. exact fingerprint on the canonical token set -- free, and doubles as
         the primary key
      2. (SimHash banding: deferred; plain scanning is fine below ~2k facts)
      3. token-set Jaccard >= 0.82, or containment >= 0.90 with a length ratio
         <= 0.60, in which case the longer wording wins and the shorter is
         archived with its evidence carried across

    difflib.SequenceMatcher is deliberately *not* the primary comparator: it is
    order- and character-sensitive, so two orderings of the same decision score
    poorly on ratio() and 1.0 on token-set Jaccard.
    """
    kind = kind if kind in KINDS else "entity"
    text = clean_text(text)
    # The two-token floor rejects prose fragments, but an `entity` fact is often
    # a bare verified path -- and "omnimem/core/facts.py" tokenises to a single
    # token, so applying the floor there silently dropped the whole Map section.
    min_tokens = 1 if kind in ("entity", "stack") else 2
    if not text or len(token_set(text)) < min_tokens:
        return {"action": "rejected", "id": None, "text": text}

    ts = ts if ts is not None else time.time()
    authority = AUTHORITY.get(source, 1.0)
    if pinned:
        authority = AUTHORITY["human"]
    fid = fingerprint(kind, text)
    meta_json = json.dumps(meta or {}, sort_keys=True)

    with db.write_txn(conn):
        row = conn.execute("SELECT id, status FROM facts WHERE id = ?", (fid,)).fetchone()
        if row is not None:
            if row["status"] != "active":
                conn.execute(
                    "UPDATE facts SET status='active', superseded_by=NULL WHERE id = ?",
                    (fid,),
                )
            if pinned:
                conn.execute("UPDATE facts SET pinned=1 WHERE id = ?", (fid,))
            _record_sighting(conn, fid, session_id, tool, ts, authority)
            return {"action": "seen", "id": fid, "text": text}

        new_tokens = token_set(text)
        subsumed: List[str] = []
        for cand in _candidates(conn, project, kind):
            cand_tokens = set(cand["norm"].split())
            if jaccard(new_tokens, cand_tokens) >= 0.82:
                _record_sighting(conn, cand["id"], session_id, tool, ts, authority)
                return {"action": "duplicate", "id": cand["id"], "text": cand["text"]}
            smaller, larger = (
                (new_tokens, cand_tokens)
                if len(new_tokens) <= len(cand_tokens)
                else (cand_tokens, new_tokens)
            )
            ratio = len(smaller) / max(1, len(larger))
            if containment(smaller, larger) >= 0.90 and ratio <= 0.60:
                if len(new_tokens) <= len(cand_tokens):
                    # The stored fact already says everything this one does.
                    _record_sighting(conn, cand["id"], session_id, tool, ts, authority)
                    return {"action": "duplicate", "id": cand["id"], "text": cand["text"]}
                subsumed.append(cand["id"])

        conn.execute(
            "INSERT INTO facts(id, project, kind, text, norm, status, pinned, first_seen,"
            " in_brief, meta) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, 0, ?)",
            (fid, project, kind, text, " ".join(sorted(new_tokens)),
             1 if pinned else 0, ts, meta_json),
        )
        _record_sighting(conn, fid, session_id, tool, ts, authority)

        for old_id in subsumed:
            # Carry the evidence over, or archiving would silently discard the
            # corroboration that made the shorter wording credible.
            conn.execute(
                "INSERT OR IGNORE INTO fact_sightings(fact_id, session_id, tool, ts, authority)"
                " SELECT ?, session_id, tool, ts, authority FROM fact_sightings"
                " WHERE fact_id = ?",
                (fid, old_id),
            )
            conn.execute(
                "UPDATE facts SET status='archived', superseded_by=? WHERE id = ?",
                (fid, old_id),
            )

    return {"action": "subsumed" if subsumed else "new", "id": fid, "text": text}


def set_pinned(conn: sqlite3.Connection, fact_id: str, pinned: bool = True) -> None:
    with db.write_txn(conn):
        conn.execute("UPDATE facts SET pinned=? WHERE id=?", (1 if pinned else 0, fact_id))


# --------------------------------------------------------------------------- reads

def load_facts(
    conn: sqlite3.Connection,
    project: str,
    kinds: Optional[Sequence[str]] = None,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """All active facts for a project, scored and sorted. One query, no N+1."""
    sql = [
        "SELECT f.id, f.kind, f.text, f.pinned, f.first_seen, f.meta,",
        "       COUNT(s.fact_id)              AS sightings,",
        "       COUNT(DISTINCT s.tool)        AS tools,",
        "       COALESCE(MAX(s.ts), f.first_seen)   AS last_seen,",
        "       COALESCE(MAX(s.authority), 1.0)     AS authority",
        "FROM facts f LEFT JOIN fact_sightings s ON s.fact_id = f.id",
        "WHERE f.project = ? AND f.status = 'active'",
    ]
    params: List[Any] = [project]
    if kinds:
        sql.append("AND f.kind IN (%s)" % ",".join("?" * len(kinds)))
        params.extend(kinds)
    sql.append("GROUP BY f.id")

    out: List[Dict[str, Any]] = []
    for r in conn.execute(" ".join(sql), params):
        item = {
            "id": r["id"],
            "kind": r["kind"],
            "text": r["text"],
            "pinned": bool(r["pinned"]),
            "sightings": int(r["sightings"]),
            "tools": int(r["tools"]),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "authority": r["authority"],
        }
        item["score"] = score_fact(
            kind=item["kind"],
            sightings=item["sightings"],
            tools=item["tools"],
            last_seen=item["last_seen"],
            authority=item["authority"],
            text=item["text"],
            pinned=item["pinned"],
            now=now,
        )
        out.append(item)

    # Total order: score, then id. Without the id tiebreak, two facts with equal
    # quantised scores could swap places between renders and churn the file.
    out.sort(key=lambda f: (-f["score"], f["id"]))
    return out


def group_by_kind(items: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["kind"], []).append(item)
    return grouped


def fact_counts(conn: sqlite3.Connection, project: str) -> Tuple[int, int]:
    row = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(status='active'), 0) AS active"
        " FROM facts WHERE project = ?",
        (project,),
    ).fetchone()
    return (int(row["n"]), int(row["active"])) if row else (0, 0)
