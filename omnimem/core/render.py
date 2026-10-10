"""
Rendering: turn facts into the markdown files.

Render is a *total function of database state*. That single property is what
buys idempotency, safe concurrent writers, and a stable file on disk:

* same state in -> byte-identical bytes out, so there is no diff/merge logic
  anywhere in this codebase;
* two processes racing to render produce the same bytes, so a lost write is
  harmless and the next digest regenerates it;
* the hash gate can therefore skip the write entirely, leaving mtime alone.

Section order is by injection value, because truncation eats from the bottom.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omnimem.core import facts as facts_mod

# (heading, kinds, char budget). Budgets sum to ~2140, leaving the rest of the
# 2400 for "Now", headings and the pointer footer.
SECTIONS: Sequence[Tuple[str, Tuple[str, ...], int]] = (
    ("Stack", ("stack",), 220),
    ("Map", ("entity",), 300),
    ("Decisions in force", ("decision",), 520),
    ("Constraints", ("constraint", "convention"), 320),
    ("Gotchas", ("gotcha",), 380),
    ("Open questions", ("open_question",), 400),
)

BRIEF_HEADER = "<!-- OmniMem brief — generated, do not edit. Edit PINNED.md instead. -->"


class Rendered:
    """
    Body and trailer are separate so the trailer can carry a timestamp without
    breaking the hash gate: only `body` is hashed, so a brief whose content is
    unchanged is not rewritten just because the clock moved.
    """

    def __init__(self, body: str, trailer: str = "", fact_ids: Optional[List[str]] = None):
        self.body = body
        self.trailer = trailer
        self.fact_ids = fact_ids or []

    @property
    def text(self) -> str:
        return self.body + ("\n" + self.trailer + "\n" if self.trailer else "")


def _bullet(text: str) -> str:
    return f"- {text}\n"


def _fill(
    items: List[Dict[str, Any]],
    budget: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Greedy fill in score order. Returns (kept, skipped, chars_used)."""
    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    used = 0
    for item in items:
        cost = len(_bullet(item["text"]))
        if used + cost <= budget:
            kept.append(item)
            used += cost
        else:
            skipped.append(item)
    return kept, skipped, used


def read_pinned(project_dir: str) -> str:
    """
    Human-authored block, passed through verbatim and *off-budget*.

    Off-budget on purpose: if you took the trouble to write it by hand, no
    scoring heuristic gets to evict it.
    """
    path = os.path.join(project_dir, "PINNED.md")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read().strip()
    except OSError:
        return ""
    return body[:1200]


def render_project_md(
    slug: str,
    title: str,
    all_facts: List[Dict[str, Any]],
    next_step: Optional[str] = None,
    pinned_block: str = "",
    budget: int = 2400,
    session_count: int = 0,
    last_tool: Optional[str] = None,
    now: Optional[float] = None,
) -> Rendered:
    now = now if now is not None else time.time()
    grouped = facts_mod.group_by_kind(all_facts)
    chosen: List[str] = []

    lines: List[str] = [BRIEF_HEADER, f"# {title}", ""]

    # ---- Now: the single most useful line in the file.
    goals = grouped.get("goal", [])
    now_lines: List[str] = []
    if next_step:
        now_lines.append(_bullet(f"**Next:** {next_step}").rstrip("\n"))
    for goal in goals[:2]:
        now_lines.append(_bullet(goal["text"]).rstrip("\n"))
        chosen.append(goal["id"])
    if now_lines:
        lines.append("## Now")
        lines.extend(now_lines)
        lines.append("")

    if pinned_block:
        lines.append("## Pinned (human)")
        lines.append(pinned_block)
        lines.append("")

    # ---- Budgeted sections, then an auction for the leftover.
    section_kept: Dict[str, List[Dict[str, Any]]] = {}
    leftovers: List[Dict[str, Any]] = []
    spent = 0
    allocated = 0
    for heading, kinds, section_budget in SECTIONS:
        pool: List[Dict[str, Any]] = []
        for kind in kinds:
            pool.extend(grouped.get(kind, []))
        pool.sort(key=lambda f: (-f["score"], f["id"]))
        kept, skipped, used = _fill(pool, section_budget)
        section_kept[heading] = kept
        leftovers.extend(skipped)
        spent += used
        allocated += section_budget

    # Unspent section budget goes to the best remaining facts regardless of kind.
    surplus = max(0, allocated - spent)
    if surplus and leftovers:
        leftovers.sort(key=lambda f: (-f["score"], f["id"]))
        by_kind_heading = {
            kind: heading for heading, kinds, _ in SECTIONS for kind in kinds
        }
        for item in leftovers:
            heading = by_kind_heading.get(item["kind"])
            if not heading:
                continue
            cost = len(_bullet(item["text"]))
            if cost <= surplus:
                section_kept[heading].append(item)
                surplus -= cost

    for heading, _kinds, _b in SECTIONS:
        kept = section_kept.get(heading) or []
        if not kept:
            continue
        kept.sort(key=lambda f: (-f["score"], f["id"]))
        lines.append(f"## {heading}")
        for item in kept:
            lines.append(_bullet(item["text"]).rstrip("\n"))
            chosen.append(item["id"])
        lines.append("")

    lines.append("---")
    lines.append(
        "Full log: `HANDOFF.md` (last session) · `DECISIONS.md` · `TASKS.md` · `sessions/`. "
        "Call `omnimem_search` for anything not here."
    )

    body = "\n".join(lines).rstrip() + "\n"
    if len(body) > budget * 2:
        body = body[: budget * 2].rstrip() + "\n"

    stamp = time.strftime("%Y-%m-%d", time.localtime(now))
    trailer = (
        f"<!-- omnimem: {len(all_facts)} facts · {len(chosen)} in brief · "
        f"{session_count} sessions · last tool {last_tool or 'n/a'} · {stamp} -->"
    )
    return Rendered(body, trailer, chosen)


def render_handoff_md(
    slug: str,
    title: str,
    last_session: Optional[Dict[str, Any]],
    next_step: Optional[str],
    recent_facts: List[Dict[str, Any]],
    open_tasks: List[str],
    now: Optional[float] = None,
) -> Rendered:
    now = now if now is not None else time.time()
    lines = [
        "<!-- OmniMem handoff — generated. State as of the last recorded session. -->",
        f"# Handoff — {title}",
        "",
    ]
    if last_session:
        when = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(last_session.get("started") or now)
        )
        lines.append(f"**Last session:** {last_session.get('tool', 'unknown')} · {when}")
        if last_session.get("summary"):
            lines.append("")
            lines.append(last_session["summary"].strip())
        lines.append("")
    if next_step:
        lines.append(f"**Pick up here:** {next_step}")
        lines.append("")
    if open_tasks:
        lines.append("## Open tasks")
        for task in open_tasks[:12]:
            lines.append(f"- [ ] {task}")
        lines.append("")
    if recent_facts:
        lines.append("## Most recent claims")
        for item in recent_facts[:10]:
            lines.append(f"- ({item['kind']}) {item['text']}")
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n"
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    return Rendered(body, f"<!-- omnimem handoff rendered {stamp} -->")


_TASK_LINE = re.compile(r"^\s*[-*]\s*\[( |x|X)\]\s*(.+?)\s*$")


def parse_tasks_md(path: str) -> Dict[str, bool]:
    """
    Read checkbox state back out of TASKS.md.

    TASKS.md is the one file in the store that is bidirectional -- people tick
    boxes in their editor, and that edit has to win. Keys are canonical token
    forms so reflowed or lightly reworded text still matches.
    """
    states: Dict[str, bool] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                match = _TASK_LINE.match(line)
                if match:
                    key = facts_mod.canonical(match.group(2))
                    if key:
                        states[key] = match.group(1).lower() == "x"
    except OSError:
        pass
    return states


def render_tasks_md(
    title: str,
    open_tasks: List[Dict[str, Any]],
    done_tasks: List[Dict[str, Any]],
) -> Rendered:
    lines = [
        "<!-- OmniMem tasks — tick a box here and OmniMem picks it up. -->",
        f"# Tasks — {title}",
        "",
    ]
    lines.append("## Open")
    if open_tasks:
        for task in open_tasks:
            lines.append(f"- [ ] {task['text']}")
    else:
        lines.append("_none_")
    lines.append("")
    if done_tasks:
        lines.append("## Done")
        for task in done_tasks[:25]:
            lines.append(f"- [x] {task['text']}")
        lines.append("")
    return Rendered("\n".join(lines).rstrip() + "\n")


def render_meta_json(
    slug: str,
    title: str,
    root: Optional[str],
    counts: Dict[str, int],
    session_count: int,
    last_tool: Optional[str],
    next_step: Optional[str],
    updated: float,
) -> Rendered:
    payload = {
        "slug": slug,
        "title": title,
        "root": root,
        "facts": counts,
        "sessions": session_count,
        "last_tool": last_tool,
        "next_step": next_step,
        "updated": round(updated, 3),
        "schema": 1,
    }
    return Rendered(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def decision_block(text: str, tool: str, ts: float, session_id: str) -> str:
    """One immutable block appended to DECISIONS.md."""
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    return (
        f"\n## {when} · {tool}\n"
        f"{text.strip()}\n"
        f"<!-- session: {session_id} -->\n"
    )
