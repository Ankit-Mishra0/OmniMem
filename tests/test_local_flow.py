import os
import time

import pytest
from fastapi.testclient import TestClient

from omnimem.core import store
from omnimem.database import (
    get_connected_notes,
    get_db_connection,
    get_notes_by_project,
    init_db,
    list_projects,
    save_note,
    search_notes,
)
from omnimem.memory_engine import (
    build_active_context,
    calculate_decayed_weight,
    extract_structured_summary,
    is_important,
    process_and_store_memory,
)
from omnimem.server import (
    app,
    get_active_project_state,
    query_project_memory,
    save_project_memory,
)

TEST_DB_PATH = os.path.join(os.environ["OMNIMEM_HOME"], "test_omnimem.db")


@pytest.fixture(autouse=True)
def setup_test_db():
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB_PATH + suffix
        if os.path.exists(path):
            os.remove(path)
    init_db(TEST_DB_PATH)
    yield
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB_PATH + suffix
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def client():
    """
    Authenticated REST client.

    Every /api/* route now requires the bearer token from ~/.omnimem/token.
    CORS alone would not be enough: a cross-origin POST with a simple content
    type is still *delivered* even though the browser hides the response, so a
    foreign page could write to the store without the token check.
    """
    token = store.ensure_token()
    with TestClient(app) as c:
        c.headers.update({"X-OmniMem-Token": token})
        yield c


def test_database_wal_and_fts5():
    conn = get_db_connection(TEST_DB_PATH)
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"

    note = save_note(
        project_id="OmniMemTest",
        content_summary="Refactored database layer to use SQLite WAL mode and FTS5 full-text indexing.",
        tags=["sqlite", "wal", "fts5"],
        platform="unit_test",
        db_path=TEST_DB_PATH,
    )
    assert note["id"] is not None

    results = search_notes("FTS5 indexing", project_id="OmniMemTest", db_path=TEST_DB_PATH)
    assert len(results) >= 1
    assert "FTS5" in results[0]["content_summary"]
    assert results[0]["project_id"] == "OmniMemTest"


def test_resave_does_not_create_fts_ghost_rows():
    """
    Regression: the old INSERT OR REPLACE left a ghost row in notes_fts on every
    re-save, because REPLACE's implicit delete does not fire an AFTER DELETE
    trigger unless recursive_triggers is on -- and it is off by default. Since
    search joins FTS -> notes, one note came back as duplicate hits forever and
    skewed BM25's document statistics.
    """
    conn = get_db_connection(TEST_DB_PATH)
    note = save_note(
        project_id="Ghosts",
        content_summary="First version of the deduplication cascade.",
        tags=["dedup"],
        platform="unit_test",
        db_path=TEST_DB_PATH,
    )
    for i in range(3):
        save_note(
            project_id="Ghosts",
            content_summary=f"Revision {i} of the deduplication cascade.",
            tags=["dedup"],
            platform="unit_test",
            note_id=note["id"],
            db_path=TEST_DB_PATH,
        )

    notes_rows = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    fts_rows = conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
    assert notes_rows == fts_rows == 1

    hits = search_notes("deduplication", project_id="Ghosts", db_path=TEST_DB_PATH)
    assert len({h["id"] for h in hits}) == len(hits)


def test_importance_gating():
    assert not is_important("hello")
    assert not is_important("thanks!")
    assert not is_important("ok got it")
    assert not is_important("good morning")

    assert is_important("Configured FastAPI REST bridge on port 8000 for cross-LLM prompt injection.")
    assert is_important("Fixed bug in SQLite schema migration script for links table.")


def test_structured_extraction_and_zettelkasten():
    content = "Implemented FastMCP stdio server and importance gating mechanism in memory_engine.py."
    summary = extract_structured_summary(content)
    assert summary["topic"] is not None
    assert len(summary["tags"]) > 0

    note1 = process_and_store_memory("ProjectAlpha", content, platform="test", db_path=TEST_DB_PATH)
    note2 = process_and_store_memory(
        "ProjectAlpha",
        "Added FastAPI REST bridge endpoint /api/memory/save in server.py",
        platform="test",
        db_path=TEST_DB_PATH,
    )
    assert note1 is not None and note2 is not None

    decayed = calculate_decayed_weight(initial_score=1.0, elapsed_seconds=3600, decay_lambda=0.0001)
    assert 0.0 < decayed < 1.0


def test_links_require_real_tag_overlap():
    """
    Links now mean tag overlap and nothing else.

    The old code applied a 0.3 floor to notes "in the same project" -- over a
    list already filtered by project, so the condition was always true, the
    floor always cleared the 0.2 gate, and every note linked to the 20 most
    recent. That is ~10k edges at 500 notes, all encoding only what the
    project_id column already says.
    """
    shared = process_and_store_memory(
        "LinkTest",
        "Set up the SQLite FTS5 index for note search.",
        tags=["sqlite", "fts5", "search"],
        platform="test",
        db_path=TEST_DB_PATH,
    )
    related = process_and_store_memory(
        "LinkTest",
        "Tuned the SQLite FTS5 ranking function.",
        tags=["sqlite", "fts5", "ranking"],
        platform="test",
        db_path=TEST_DB_PATH,
    )
    unrelated = process_and_store_memory(
        "LinkTest",
        "Chose an indigo and violet gradient for the popup theme.",
        tags=["css", "theme", "popup"],
        platform="test",
        db_path=TEST_DB_PATH,
    )

    linked_ids = {n["id"] for n in get_connected_notes(shared["id"], db_path=TEST_DB_PATH)}
    assert related["id"] in linked_ids
    assert unrelated["id"] not in linked_ids


def test_link_weight_decays_at_read_time():
    """
    Decay used to be computed at write time and frozen into links.weight, so the
    graph never actually aged. Weights are stored undecayed now and decayed when
    read.
    """
    from omnimem.core import notes as core_notes

    conn = get_db_connection(TEST_DB_PATH)
    a = save_note("DecayTest", "Note A about caching.", ["cache"], db_path=TEST_DB_PATH)
    b = save_note("DecayTest", "Note B about caching.", ["cache"], db_path=TEST_DB_PATH)

    now = time.time()
    core_notes.upsert_link(conn, a["id"], b["id"], "tag_overlap", 1.0, updated_at=now)

    fresh = core_notes.get_connected_notes(conn, a["id"], now=now)[0]["link_weight"]
    aged = core_notes.get_connected_notes(
        conn, a["id"], now=now + core_notes.LINK_HALF_LIFE_DAYS * 86400
    )[0]["link_weight"]

    assert fresh == pytest.approx(1.0, abs=1e-3)
    assert aged == pytest.approx(0.5, abs=1e-3)


def test_fastapi_rest_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    save_res = client.post("/api/memory/save", json={
        "project_id": "RESTTest",
        "content": "Implemented debounced DOM mutation observer in extension content.js to avoid browser lag.",
        "platform": "chatgpt",
    })
    assert save_res.status_code == 200
    assert save_res.json()["status"] == "saved"

    fluff_res = client.post("/api/memory/save", json={
        "project_id": "RESTTest", "content": "thanks!", "platform": "chatgpt",
    })
    assert fluff_res.status_code == 200
    assert fluff_res.json()["status"] == "skipped"

    query_res = client.get("/api/memory/query?query=observer&project_id=RESTTest")
    assert query_res.status_code == 200
    assert len(query_res.json()["results"]) >= 1

    state_res = client.get("/api/memory/active_state?project_id=RESTTest")
    assert state_res.status_code == 200
    assert "OmniMem Synced Context" in state_res.json()["context"]

    proj_res = client.get("/api/projects")
    assert proj_res.status_code == 200
    assert "RESTTest" in proj_res.json()["projects"]


def test_rest_requires_token():
    """
    Defect 4: the server used allow_origins=["*"] with allow_credentials=True and
    no auth, so any page you visited could read the whole memory store off
    127.0.0.1:8000.
    """
    with TestClient(app) as anon:
        assert anon.get("/api/projects").status_code == 401
        assert anon.get("/api/projects", headers={"X-OmniMem-Token": "wrong"}).status_code == 401
        # Health stays open so the extension can detect the server cheaply.
        assert anon.get("/health").status_code == 200


def test_handshake_is_extension_only():
    with TestClient(app) as anon:
        ok = anon.get("/api/handshake", headers={"Origin": "chrome-extension://abcdefghijk"})
        assert ok.status_code == 200 and ok.json()["token"]
        blocked = anon.get("/api/handshake", headers={"Origin": "https://evil.example"})
        assert blocked.status_code == 403


def test_agentic_endpoints(client):
    cap = client.post("/api/capture", json={
        "project": "AgenticTest",
        "tool": "chatgpt",
        "role": "user",
        "text": "I want to build a cross-tool memory layer. It must not cost anything to run.",
    })
    assert cap.status_code == 200 and cap.json()["status"] == "saved"

    sess = client.post("/api/session", json={
        "project": "AgenticTest",
        "tool": "chatgpt",
        "facts": [{"kind": "decision", "text": "Use SQLite with FTS5 rather than a vector database."}],
        "next_step": "Wire up the browser extension.",
    })
    assert sess.status_code == 200

    brief = client.get("/api/resume", params={"project": "AgenticTest"}).json()["brief"]
    assert "cross-tool memory layer" in brief
    assert "Wire up the browser extension." in brief
    assert "SQLite" in brief

    found = client.get("/api/search", params={"query": "vector database", "project": "AgenticTest"})
    assert found.status_code == 200
    assert found.json()["facts"]


def test_mcp_tool_functions():
    res_save = save_project_memory(
        project_id="MCPTest",
        content="Testing FastMCP tool integration over stdio protocol.",
        platform="mcp",
    )
    assert res_save["status"] == "saved"

    res_query = query_project_memory(query="FastMCP stdio", project_id="MCPTest")
    assert len(res_query) >= 1

    res_state = get_active_project_state(project_id="MCPTest")
    assert "OmniMem Synced Context" in res_state
