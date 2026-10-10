"""
Facts store and render stability.

Two properties carry the whole architecture:

  * a fact's fingerprint is a stable function of its *meaning* -- not of the
    process that computed it, and not of word order; and
  * the markdown files are a total function of DB state, so rendering twice
    produces identical bytes and the hash gate holds.

If either breaks, re-ingesting a session stops being a no-op, every brief starts
churning on every write, and the memory files rot -- which is the failure mode
this design exists to avoid.
"""

import difflib
import json
import os
import subprocess
import sys
import time

import omnimem
from omnimem.core import extract, facts, ingest, locks, render, store

PROJECT = "factstest"


# --------------------------------------------------------------- fingerprints

def test_fingerprint_is_stable_across_processes():
    """
    blake2b, never the builtin hash().

    PYTHONHASHSEED is randomised per process, so a builtin-hash fingerprint
    computed in a hook subprocess would not match the one the server computes --
    silently duplicating every fact instead of recognising it. This runs a real
    subprocess under a pinned, different seed to prove the id does not move.
    """
    text = "We chose SQLite over Postgres for the prototype"
    here = facts.fingerprint("decision", text)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(omnimem.__file__)))
    code = (
        "from omnimem.core import facts;"
        f"print(facts.fingerprint('decision', {text!r}))"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONPATH=repo)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=repo, check=True,
    )
    assert out.stdout.strip() == here


def test_synonyms_fold_before_the_fingerprint():
    """Abbreviations and tense are noise; the claim underneath is the same one."""
    assert facts.fingerprint("decision", "we picked SQLite over Postgres") == \
        facts.fingerprint("decision", "we chose SQLite over Postgres")
    assert facts.fingerprint("convention", "use the db config") == \
        facts.fingerprint("convention", "use the database configuration")


def test_kind_is_part_of_the_key():
    """The same sentence as a goal and as a constraint are different facts."""
    assert facts.fingerprint("goal", "ship the browser extension") != \
        facts.fingerprint("task", "ship the browser extension")


def test_token_set_beats_sequence_matching_on_reordering():
    """
    Why difflib is not the primary comparator.

    SequenceMatcher is order- and character-sensitive, so a pure reordering of
    the same claim scores 0.56 -- well under any usable threshold -- while the
    token set recognises it as identical.
    """
    a = "use flock rather than lockf for the project lock"
    b = "for the project lock use flock rather than lockf"

    assert difflib.SequenceMatcher(None, a, b).ratio() < 0.7
    assert facts.jaccard(facts.token_set(a), facts.token_set(b)) == 1.0
    assert facts.fingerprint("convention", a) == facts.fingerprint("convention", b)


# ---------------------------------------------------------------------- dedup

def test_reingesting_the_same_fact_is_a_noop(conn):
    text = "The facts store is the source of truth and markdown is a render"
    first = facts.upsert_fact(conn, PROJECT, "decision", text)
    second = facts.upsert_fact(conn, PROJECT, "decision", text)

    assert first["action"] == "new"
    assert second["action"] == "seen"
    assert first["id"] == second["id"]
    assert conn.execute(
        "SELECT COUNT(*) FROM facts WHERE project = ?", (PROJECT,)
    ).fetchone()[0] == 1


def test_sightings_are_appended_not_incremented(conn):
    """
    Counters are derived from an append-only table.

    `UPDATE ... SET sightings = sightings + 1` would inflate a fact's score every
    time a session file was replayed. Counting rows with a composite primary key
    cannot: one row per (fact, session, tool), so a replay is a no-op and three
    genuinely different tools still count as three.
    """
    text = "Ingest must be idempotent so replaying a session is safe"
    for i in range(3):
        facts.upsert_fact(conn, PROJECT, "constraint", text, tool="tool-%d" % i)
    # Replay: same session, same tool, already recorded.
    facts.upsert_fact(conn, PROJECT, "constraint", text, tool="tool-0")

    fact_id = facts.fingerprint("constraint", text)
    rows, tools = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT tool) FROM fact_sightings WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    assert (rows, tools) == (3, 3)

    loaded = [f for f in facts.load_facts(conn, PROJECT) if f["id"] == fact_id][0]
    assert loaded["sightings"] == 3
    assert loaded["tools"] == 3


def test_longer_wording_subsumes_the_shorter_and_inherits_its_evidence(conn):
    short = "Use flock for the project lock"
    long = "Use flock for the project lock because lockf is per-process and unreliable"

    facts.upsert_fact(conn, PROJECT, "convention", short, session_id="s1", tool="cc")
    short_id = facts.fingerprint("convention", short)
    result = facts.upsert_fact(conn, PROJECT, "convention", long, session_id="s2", tool="cx")

    assert result["action"] == "subsumed"

    active = {f["text"] for f in facts.load_facts(conn, PROJECT)}
    assert active == {long}

    archived = conn.execute(
        "SELECT status, superseded_by FROM facts WHERE id = ?", (short_id,)
    ).fetchone()
    assert archived["status"] == "archived"
    assert archived["superseded_by"] == result["id"]

    # The corroboration that made the shorter wording credible is carried across,
    # not discarded -- otherwise subsumption would quietly destroy evidence.
    tools = {
        r["tool"] for r in conn.execute(
            "SELECT tool FROM fact_sightings WHERE fact_id = ?", (result["id"],)
        )
    }
    assert tools == {"cc", "cx"}


def test_a_reworded_duplicate_does_not_create_a_second_fact(conn):
    facts.upsert_fact(conn, PROJECT, "decision", "we picked SQLite over Postgres")
    again = facts.upsert_fact(conn, PROJECT, "decision", "we chose SQLite over Postgres")
    assert again["action"] == "seen"
    assert len(facts.load_facts(conn, PROJECT)) == 1


def test_fragments_are_rejected_but_a_bare_path_is_not(conn):
    """
    The two-token floor rejects prose fragments. `entity` is exempt: a verified
    path like `omnimem/core/facts.py` tokenises to one token, and applying the
    floor there silently emptied the whole Map section of the brief.
    """
    assert facts.upsert_fact(conn, PROJECT, "decision", "ok")["action"] == "rejected"
    assert facts.upsert_fact(conn, PROJECT, "entity", "omnimem/core/facts.py")["action"] == "new"


# -------------------------------------------------------------------- scoring

def test_decay_is_evaluated_at_read_time(conn):
    """
    The original defect: the decayed weight was computed at *write* time and
    frozen into a column, so nothing ever actually decayed. Scoring the same
    unchanged row at two different `now` values must give two different scores.
    """
    facts.upsert_fact(conn, PROJECT, "task", "Ship the extension before the landing page")

    now = time.time()
    fresh = facts.load_facts(conn, PROJECT, now=now)[0]["score"]
    later = facts.load_facts(conn, PROJECT, now=now + 60 * 86400)[0]["score"]
    assert later < fresh


def test_half_lives_are_per_kind():
    """A single global lambda is wrong: a task is stale in a week, a goal is not."""
    now = time.time()
    week = now - 7 * 86400
    assert facts.recency("task", week, now) < 0.25
    assert facts.recency("goal", week, now) > 0.95


def test_recency_never_falls_below_the_floor():
    """An old but corroborated constraint must not be evicted by today's chatter."""
    now = time.time()
    ancient = now - 3650 * 86400
    for kind in facts.KINDS:
        assert facts.recency(kind, ancient, now) == facts.RECENCY_FLOOR


def test_a_human_pinned_fact_outranks_a_heuristic_guess(conn):
    facts.upsert_fact(conn, PROJECT, "constraint", "Something a regex tier guessed at")
    human = facts.upsert_fact(
        conn, PROJECT, "constraint", "The store must never leave this machine",
        source="human",
    )
    facts.set_pinned(conn, human["id"], True)

    assert facts.load_facts(conn, PROJECT)[0]["id"] == human["id"]


def test_scores_are_quantised_and_ties_break_on_id(conn):
    """
    Float drift must not be able to reorder the brief between two renders, so
    scores are rounded and the sort carries a total-order tiebreak.
    """
    for i in range(6):
        facts.upsert_fact(conn, PROJECT, "gotcha", "Gotcha number %d in the writer path" % i)

    order = [f["id"] for f in facts.load_facts(conn, PROJECT)]
    assert order == [f["id"] for f in facts.load_facts(conn, PROJECT)]
    for item in facts.load_facts(conn, PROJECT):
        assert item["score"] == round(item["score"], 3)


# -------------------------------------------------------------- render output

def _seed(conn, project=PROJECT):
    for kind, text in [
        ("goal", "Build a cross-tool memory layer so no tool has to be re-briefed"),
        ("decision", "SQLite with FTS5 rather than a vector database"),
        ("constraint", "Zero ongoing cost and no GPU on the default path"),
        ("gotcha", "INSERT OR REPLACE skips the FTS delete trigger unless recursive_triggers is on"),
        ("open_question", "Should the extension ship its own project picker"),
    ]:
        facts.upsert_fact(conn, project, kind, text)


def test_render_is_byte_stable(conn):
    _seed(conn)
    loaded = facts.load_facts(conn, PROJECT)
    first = render.render_project_md(PROJECT, "Factstest", loaded)
    second = render.render_project_md(PROJECT, "Factstest", loaded)
    assert first.body == second.body
    assert first.fact_ids == second.fact_ids


def test_the_dated_trailer_is_outside_the_hashed_body(conn):
    """
    Why `Rendered` splits body from trailer: the brief carries a date stamp, and
    if that stamp were inside the hashed region the file would be rewritten every
    day whether or not anything had actually been learned.
    """
    _seed(conn)
    loaded = facts.load_facts(conn, PROJECT)
    now = time.time()
    today = render.render_project_md(PROJECT, "Factstest", loaded, now=now)
    later = render.render_project_md(PROJECT, "Factstest", loaded, now=now + 5 * 86400)

    assert today.body == later.body
    assert locks.body_hash(today.body) == locks.body_hash(later.body)
    assert today.trailer != later.trailer


def test_brief_stays_inside_the_hook_injection_cap(conn):
    """Claude Code truncates hook stdout at 10,000 chars; the brief targets 2400."""
    for i in range(200):
        facts.upsert_fact(
            conn, PROJECT, "decision",
            "Decision number %d about a subsystem with a reasonably long body" % i,
        )
    brief = render.render_project_md(PROJECT, "Factstest", facts.load_facts(conn, PROJECT))
    assert brief.body.startswith(render.BRIEF_HEADER)
    assert len(brief.text) < 10000
    # Every fact cannot fit, and the ones that did are recorded so the digest can
    # set in_brief without re-deriving the selection.
    assert 0 < len(brief.fact_ids) < 200


def test_the_pinned_human_block_is_verbatim_and_off_budget(conn):
    """If you took the trouble to write it by hand, no heuristic gets to evict it."""
    store.ensure_project(PROJECT)
    pinned = "Never touch the billing module without asking.\nRun `make check` first."
    with open(os.path.join(store.project_dir(PROJECT), "PINNED.md"), "w") as fh:
        fh.write(pinned)

    for i in range(120):
        facts.upsert_fact(conn, PROJECT, "decision", "Filler decision %d in the writer path" % i)

    brief = render.render_project_md(
        PROJECT, "Factstest", facts.load_facts(conn, PROJECT),
        pinned_block=render.read_pinned(store.project_dir(PROJECT)),
    )
    assert "Never touch the billing module without asking." in brief.body
    assert "Run `make check` first." in brief.body


# ------------------------------------------------------------------- idempotency

def test_digest_writes_once_then_leaves_mtime_alone(conn, home):
    """
    The hash gate must skip the write entirely, not rewrite identical bytes.

    Touching mtime on every digest would make every editor and file watcher think
    the brief changed -- exactly the churn this design exists to remove. `force`
    deliberately bypasses the dirty flag but *not* the hash gate.
    """
    root = os.path.join(home, "repo")
    os.makedirs(root)
    ingest.ingest_turn(
        conn, PROJECT,
        "I want to build a local memory layer for my coding tools",
        role="user", tool="pytest", root=root, digest_now=False,
    )
    first = ingest.digest(conn, PROJECT, root=root, force=True)
    assert "PROJECT.md" in first["written"]

    path = os.path.join(store.project_dir(PROJECT), "PROJECT.md")
    before = os.stat(path).st_mtime_ns
    body_before = open(path, encoding="utf-8").read()

    time.sleep(0.02)
    again = ingest.digest(conn, PROJECT, root=root, force=True)

    assert again["written"] == []
    assert os.stat(path).st_mtime_ns == before, "hash gate did not hold"
    assert open(path, encoding="utf-8").read() == body_before


def test_a_clean_project_short_circuits(conn, home):
    root = os.path.join(home, "repo")
    os.makedirs(root)
    ingest.ingest_turn(
        conn, PROJECT, "The goal is a shared brief every tool can read",
        role="user", tool="pytest", root=root,
    )
    ingest.digest(conn, PROJECT, root=root)
    assert ingest.digest(conn, PROJECT, root=root)["status"] == "clean"


def test_meta_json_is_valid_and_stable(conn):
    _seed(conn)
    counts = {"goal": 1, "decision": 1}
    args = dict(
        slug=PROJECT, title="Factstest", root="/tmp/repo", counts=counts,
        session_count=2, last_tool="claude-code", next_step="Write the tests",
        updated=1_700_000_000.0,
    )
    first = render.render_meta_json(**args)
    parsed = json.loads(first.text)
    assert parsed["slug"] == PROJECT
    assert parsed["facts"] == counts
    assert render.render_meta_json(**args).text == first.text


# --------------------------------------------------------------------- tasks

def test_a_ticked_checkbox_on_disk_wins(conn, home):
    """
    TASKS.md is the one bidirectional file in the store. Someone ticking a box in
    their editor is a human statement of fact, so the next digest has to read it
    back rather than overwrite it.
    """
    root = os.path.join(home, "repo")
    os.makedirs(root)
    ingest.ensure_state(conn, PROJECT, root=root)
    ingest.set_task(conn, PROJECT, "Wire up the landing page")
    ingest.digest(conn, PROJECT, root=root, force=True)

    path = os.path.join(store.project_dir(PROJECT), "TASKS.md")
    body = open(path, encoding="utf-8").read()
    assert "- [ ] Wire up the landing page" in body

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body.replace("- [ ]", "- [x]"))
    assert any(render.parse_tasks_md(path).values())

    ingest.digest(conn, PROJECT, root=root, force=True)
    state = conn.execute(
        "SELECT state FROM tasks WHERE project = ?", (PROJECT,)
    ).fetchone()["state"]
    assert state == "done"


# --------------------------------------------------------------- ground truth

def test_hallucinated_paths_are_filtered_by_one_syscall(conn, home):
    """
    An unverified path is worse than no path: it sends the next session hunting
    for a file that never existed. os.path.exists() is the entire filter.
    """
    root = os.path.join(home, "repo")
    os.makedirs(root)
    open(os.path.join(root, "real_module.py"), "w").write("# real\n")

    claims = extract.extract_entities(
        "We put the retry logic in real_module.py and the parser in imaginary_module.py",
        root,
    )
    assert [c[1] for c in claims] == ["real_module.py"]

    ingest.ingest_turn(
        conn, PROJECT,
        "We put the retry logic in real_module.py and the parser in imaginary_module.py",
        role="assistant", tool="pytest", root=root,
    )
    stored = " ".join(f["text"] for f in facts.load_facts(conn, PROJECT))
    assert "real_module.py" in stored
    assert "imaginary_module.py" not in stored


def test_stack_facts_come_from_the_manifests(conn, home):
    """Ground truth that needs no NLP: read the file the package manager reads."""
    root = os.path.join(home, "repo")
    os.makedirs(root)
    with open(os.path.join(root, "package.json"), "w") as fh:
        json.dump({"dependencies": {"react": "^18.0.0", "vite": "^5.0.0"}}, fh)

    stack = extract.scan_stack(root)
    assert stack and all(kind == "stack" for kind, _ in stack)
    assert any("react" in text for _, text in stack)
