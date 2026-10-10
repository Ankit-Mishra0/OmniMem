"""
Concurrency and durability.

Three IDEs, a browser extension and a handful of fire-and-exit hook subprocesses
all write this store, so none of the properties here are theoretical. The tests
use real processes rather than threads because the failure modes are per-process:
`BEGIN IMMEDIATE`'s snapshot upgrade, flock ownership, and the inbox's
`O_CREAT|O_EXCL` handoff all behave differently inside one interpreter.
"""

import multiprocessing as mp
import os
import sqlite3
import time

import pytest

from omnimem.core import db, facts, ingest, locks, store

PROJECT = "conctest"
WORKERS = 6
PER_WORKER = 8


# ------------------------------------------------------------------ workers
#
# Module level and argument-driven: the default start method on macOS is
# `spawn`, so a closure or a fixture-bound function could not be pickled, and
# the child re-imports this module from scratch. OMNIMEM_HOME is inherited
# through os.environ, which is what points the child at the test's store.

def _write_distinct(home: str, worker: int) -> int:
    os.environ["OMNIMEM_HOME"] = home
    conn = db.init_schema()
    written = 0
    for i in range(PER_WORKER):
        # The unique part is one fused token (`w3i5`), not two separate numbers.
        # Two loose numbers permute: "Worker 0 ... question 1" and "Worker 1 ...
        # question 0" have the same token set, so dedup correctly collapses them
        # and the workers would appear to lose writes they never made.
        result = facts.upsert_fact(
            conn, PROJECT, "decision",
            "Decision w%di%d pins the writer path in place" % (worker, i),
            session_id="s%d" % worker, tool="tool%d" % worker,
        )
        if result["action"] in ("new", "subsumed"):
            written += 1
    db.close_all()
    return written


def _write_identical(home: str, worker: int) -> str:
    """Every worker races to admit the *same* claim."""
    os.environ["OMNIMEM_HOME"] = home
    conn = db.init_schema()
    result = facts.upsert_fact(
        conn, PROJECT, "constraint",
        "The store must stay on this machine and cost nothing to run",
        session_id="shared", tool="shared-tool",
    )
    db.close_all()
    return result["action"]


def _hold_project_lock(home: str, seconds: float, ready, done) -> None:
    os.environ["OMNIMEM_HOME"] = home
    store.ensure_project(PROJECT)
    with locks.file_lock(store.lock_path(PROJECT), timeout=5.0):
        ready.set()
        done.wait(seconds)


# ------------------------------------------------------------ multi-process

def test_concurrent_processes_all_land_their_writes(home):
    """
    Every write path here is read-then-write: dedup SELECTs candidates, then
    INSERTs. Under a *deferred* transaction SQLite takes a read snapshot on the
    first SELECT and fails the upgrade with SQLITE_BUSY_SNAPSHOT -- returned
    immediately, without invoking the busy handler, so `busy_timeout` does not
    help and a blind retry would re-run the write against a stale read.
    `BEGIN IMMEDIATE` is what makes the retry safe.
    """
    db.init_schema()
    db.close_all()

    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        counts = pool.starmap(_write_distinct, [(home, w) for w in range(WORKERS)])

    assert counts == [PER_WORKER] * WORKERS, "a worker lost writes to contention"

    conn = db.init_schema()
    rows = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE project = ? AND status = 'active'", (PROJECT,)
    ).fetchone()[0]
    assert rows == WORKERS * PER_WORKER


def test_the_same_claim_from_every_process_is_stored_once(home):
    """Content addressing plus a composite sighting key, not a lock, is what
    makes this safe -- so the racing writers do not even need to coordinate."""
    db.init_schema()
    db.close_all()

    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        actions = pool.starmap(_write_identical, [(home, w) for w in range(WORKERS)])

    assert actions.count("new") == 1
    assert set(actions) <= {"new", "seen", "duplicate"}

    conn = db.init_schema()
    assert conn.execute(
        "SELECT COUNT(*) FROM facts WHERE project = ?", (PROJECT,)
    ).fetchone()[0] == 1
    # One (fact, session, tool) triple, no matter how many processes raced.
    assert conn.execute("SELECT COUNT(*) FROM fact_sightings").fetchone()[0] == 1


def test_write_txn_is_reentrant(conn):
    """
    ingest calls into facts.upsert_fact, which opens its own write_txn. A nested
    BEGIN IMMEDIATE would raise "cannot start a transaction within a
    transaction", so the inner call has to no-op.
    """
    with db.write_txn(conn):
        facts.upsert_fact(conn, PROJECT, "goal", "Nested transactions must not explode")
        with db.write_txn(conn):
            facts.upsert_fact(conn, PROJECT, "task", "And the inner one must no-op")

    assert len(facts.load_facts(conn, PROJECT)) == 2


def test_a_failed_write_txn_rolls_back(conn):
    class Boom(RuntimeError):
        pass

    facts.upsert_fact(conn, PROJECT, "goal", "This one survives the rollback")
    with pytest.raises(Boom):
        with db.write_txn(conn):
            conn.execute(
                "INSERT INTO facts(id, project, kind, text, norm, first_seen)"
                " VALUES ('doomed', ?, 'goal', 'rolled back', 'rolled back', 0)",
                (PROJECT,),
            )
            raise Boom()

    assert conn.execute("SELECT COUNT(*) FROM facts WHERE id='doomed'").fetchone()[0] == 0
    # And the connection is usable afterwards -- a leaked open transaction would
    # wedge every later write with "cannot start a transaction within a
    # transaction".
    facts.upsert_fact(conn, PROJECT, "goal", "Writes still work after a rollback")
    assert len(facts.load_facts(conn, PROJECT)) == 2


# -------------------------------------------------------------------- inbox

def test_hooks_enqueue_one_file_per_event(home):
    """
    One file per event, not a shared append log: O_APPEND guarantees
    single-write atomicity only up to PIPE_BUF (512-4096 bytes), and a captured
    assistant turn is comfortably larger, so a shared log *will* interleave.
    """
    paths = [
        ingest.enqueue({
            "kind": "turn", "project": PROJECT, "role": "user", "tool": "claude-code",
            "text": "Event %d: we need the retry to use jittered backoff" % i,
        })
        for i in range(5)
    ]
    assert len(set(paths)) == 5
    assert all(os.path.dirname(p) == store.inbox_dir() for p in paths)
    assert len([n for n in os.listdir(store.inbox_dir()) if n.endswith(".json")]) == 5

    # Written 0600: an inbox event is a verbatim slice of someone's conversation.
    assert all(os.stat(p).st_mode & 0o777 == 0o600 for p in paths)


def test_draining_twice_changes_nothing(conn):
    for i in range(4):
        ingest.enqueue({
            "kind": "turn", "project": PROJECT, "role": "user", "tool": "claude-code",
            "session_id": "hook-session",
            "text": "We must keep the digest idempotent across replays, item %d" % i,
        })

    first = ingest.drain_inbox(conn)
    assert first == {"processed": 4, "failed": 0}
    after_first = {f["id"]: f["sightings"] for f in facts.load_facts(conn, PROJECT)}
    assert after_first

    second = ingest.drain_inbox(conn)
    assert second == {"processed": 0, "failed": 0}
    assert {f["id"]: f["sightings"] for f in facts.load_facts(conn, PROJECT)} == after_first


def test_replaying_an_identical_event_does_not_inflate_evidence(conn):
    event = {
        "kind": "turn", "project": PROJECT, "role": "user", "tool": "claude-code",
        "session_id": "replayed", "ts": 1_700_000_000.0,
        "text": "The brief must never exceed the ten thousand character hook cap",
    }
    ingest.enqueue(dict(event))
    ingest.drain_inbox(conn)
    baseline = {f["id"]: f["sightings"] for f in facts.load_facts(conn, PROJECT)}

    ingest.enqueue(dict(event))
    ingest.drain_inbox(conn)
    assert {f["id"]: f["sightings"] for f in facts.load_facts(conn, PROJECT)} == baseline


def test_a_corrupt_event_is_quarantined_not_fatal(conn):
    good = ingest.enqueue({
        "kind": "turn", "project": PROJECT, "role": "user", "tool": "codex",
        "text": "A good event alongside a broken one must still be processed",
    })
    bad = os.path.join(store.inbox_dir(), "9999999999.000000-deadbeef.json")
    with open(bad, "w") as fh:
        fh.write("{not json")

    stats = ingest.drain_inbox(conn)
    assert stats["processed"] == 1 and stats["failed"] == 1
    assert not os.path.exists(good), "processed events are removed"
    assert os.path.exists(bad + ".bad"), "unparseable events are kept for inspection"
    assert facts.load_facts(conn, PROJECT)


def test_drain_yields_to_whoever_holds_the_lock(conn, home):
    """
    Two tools starting at once both call resume(), which drains. The loser must
    return immediately rather than block an editor's session start -- the events
    are still on disk, so the winner picks them up.
    """
    ingest.enqueue({
        "kind": "turn", "project": PROJECT, "role": "user", "tool": "cursor",
        "text": "A queued event that the blocked drainer must leave alone",
    })
    lock = os.path.join(store.inbox_dir(), ".drain.lock")

    with locks.file_lock(lock, timeout=5.0):
        started = time.monotonic()
        stats = ingest.drain_inbox(conn)
        elapsed = time.monotonic() - started

    assert stats == {"processed": 0, "failed": 0}
    assert elapsed < 1.0, "a contended drain must not stall the caller"
    assert [n for n in os.listdir(store.inbox_dir()) if n.endswith(".json")]

    # Lock released: the event is still there to be picked up.
    assert ingest.drain_inbox(conn)["processed"] == 1


# --------------------------------------------------------------------- locks

def test_flock_excludes_another_process(home):
    """
    flock, not lockf. POSIX record locks are owned by the process and dropped by
    *any* close() of *any* descriptor on the file, which breaks as soon as two
    threads in uvicorn's threadpool touch one project.
    """
    ctx = mp.get_context("spawn")
    ready, done = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_project_lock, args=(home, 5.0, ready, done))
    holder.start()
    try:
        assert ready.wait(10), "child never acquired the lock"
        with pytest.raises(locks.LockTimeout):
            with locks.file_lock(store.lock_path(PROJECT), timeout=0.2):
                pass
    finally:
        done.set()
        holder.join(10)

    # Released on the child's exit, so the next writer is not stuck behind a
    # dead process.
    with locks.file_lock(store.lock_path(PROJECT), timeout=2.0):
        pass


def test_atomic_write_leaves_no_temp_files(home):
    target = os.path.join(home, "projects", PROJECT, "PROJECT.md")
    locks.atomic_write(target, "first version\n")
    locks.atomic_write(target, "second version\n")

    assert open(target).read() == "second version\n"
    leftovers = [n for n in os.listdir(os.path.dirname(target)) if n.endswith(".tmp")]
    assert leftovers == []


def test_write_if_changed_reports_a_noop(home):
    target = os.path.join(home, "projects", PROJECT, "meta.json")
    written = locks.write_if_changed(target, '{"a": 1}\n')
    assert written is not None
    mtime = os.stat(target).st_mtime_ns

    time.sleep(0.02)
    assert locks.write_if_changed(target, '{"a": 1}\n') is None
    assert os.stat(target).st_mtime_ns == mtime
    assert locks.write_if_changed(target, '{"a": 2}\n') is not None


# ------------------------------------------------------------------ pragmas

def test_every_connection_gets_wal_and_a_busy_timeout(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
    # Per-connection and not persisted in the file, so it has to be re-asserted
    # on every connect -- easy to lose in a refactor, hence the assertion.
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.isolation_level is None


def test_connections_are_cached_per_path(home):
    first = db.connect()
    second = db.connect()
    assert first is second

    other = db.connect(os.path.join(home, "other.db"))
    assert other is not first

    db.close_all()
    assert db.connect() is not first


def test_maintenance_checkpoints_and_optimises(conn):
    facts.upsert_fact(conn, PROJECT, "goal", "Give the WAL something to checkpoint")
    ingest.ingest_turn(
        conn, PROJECT, "The maintenance pass compacts the WAL and the FTS index",
        role="user", tool="pytest", digest_now=False,
    )
    result = db.maintenance()
    assert "checkpoint_error" not in result
    assert result.get("fts_optimize") is True


def test_the_store_refuses_to_live_in_a_sync_folder():
    """
    Syncing a live WAL corrupts SQLite -- the client copies -wal and -shm out
    from under the writer -- and flock is unreliable on network filesystems.
    Refusing is the only safe answer.
    """
    for bad in (
        "/Users/someone/Dropbox/omnimem",
        "/Users/someone/Library/Mobile Documents/com~apple~CloudDocs/omnimem",
        "/Users/someone/OneDrive/omnimem",
        "/Users/someone/Google Drive/omnimem",
    ):
        with pytest.raises(store.StoreError):
            store.check_not_synced(bad)

    store.check_not_synced("/Users/someone/.omnimem")


def test_schema_init_is_safe_to_run_twice(home):
    first = db.init_schema()
    tables = {
        r["name"] for r in first.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"facts", "fact_sightings", "project_state", "sessions", "tasks", "notes"} <= tables

    db.close_all()
    second = db.init_schema()
    assert second.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    with pytest.raises(sqlite3.Error):
        # Sanity check that the assertion above would actually fail on a missing
        # table, rather than passing because the query is wrong.
        second.execute("SELECT 1 FROM definitely_not_a_table")
