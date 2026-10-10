"""
The HTTP contract the browser extension and the landing page depend on.

These are not "does FastAPI work" tests. Every assertion here corresponds to a
line in `extension/background.js` or `extension/popup.js` that would break
silently -- the extension swallows transport errors on purpose, so a renamed
field or a shadowed route would show up as capture quietly doing nothing rather
than as an exception anyone sees.

Route *order* is load-bearing too: Starlette matches in registration order, so
the StaticFiles mount at "/" must stay registered after every /api route.
"""

import os
import re

import pytest
from fastapi.testclient import TestClient

from omnimem import server
from omnimem.core import db, redact, store


@pytest.fixture
def client(home, monkeypatch):
    """
    Authenticated client against an isolated store.

    `server._ready` is process-global -- schema init is deliberately deferred out
    of import time so that importing the package never writes to disk -- so a
    test that relocates OMNIMEM_HOME has to clear it, or the app keeps using the
    schema it initialised for the previous home.
    """
    monkeypatch.setattr(server, "_ready", False)
    db.init_schema()
    token = store.ensure_token()
    with TestClient(server.app) as c:
        c.headers.update({"X-OmniMem-Token": token})
        yield c


@pytest.fixture
def anon(home, monkeypatch):
    monkeypatch.setattr(server, "_ready", False)
    db.init_schema()
    store.ensure_token()
    with TestClient(server.app) as c:
        yield c


# ------------------------------------------------------------------- health

def test_health_is_open_and_carries_the_version(anon):
    """The extension pings this to decide whether the engine is up, before it has
    a token; the landing page uses the same response to light up its chip."""
    res = anon.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["version"] == server.VERSION
    assert body["mcp"] is True


# ---------------------------------------------------------------------- auth

def test_every_api_route_requires_the_token(anon):
    for method, path in [
        ("get", "/api/config"),
        ("get", "/api/resume"),
        ("get", "/api/overview"),
        ("get", "/api/projects"),
        ("get", "/api/search?query=x"),
        ("post", "/api/capture"),
        ("post", "/api/session"),
        ("post", "/api/config"),
    ]:
        res = getattr(anon, method)(path, **({"json": {}} if method == "post" else {}))
        assert res.status_code == 401, f"{method.upper()} {path} was not gated"


def test_a_wrong_token_is_rejected_including_a_valid_prefix(anon):
    """
    CORS alone is not enough: a cross-origin POST with a simple content type is
    still *delivered*, the browser only hides the response. So a foreign page
    could write to the store without this check.
    """
    real = store.read_token()
    assert real
    for bad in ("", "wrong", real[:-1], real + "x", real.upper() if real.islower() else real.lower()):
        res = anon.get("/api/overview", headers={"X-OmniMem-Token": bad})
        assert res.status_code == 401


def test_bearer_header_is_accepted_too(anon):
    token = store.read_token()
    res = anon.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200


def test_handshake_is_extension_only(anon):
    """
    The one unauthenticated write-adjacent route. It holds against web pages
    because browsers set Origin themselves and page JavaScript cannot forge it;
    it does not hold against other local processes, which is the accepted trade.
    """
    ok = anon.get("/api/handshake", headers={"Origin": "chrome-extension://abcdefghijk"})
    assert ok.status_code == 200
    assert ok.json()["token"] == store.read_token()

    for origin in ("https://chatgpt.com", "https://evil.example", "http://127.0.0.1:8000"):
        assert anon.get("/api/handshake", headers={"Origin": origin}).status_code == 403

    # No Origin at all is a native client (the CLI), not a page.
    assert anon.get("/api/handshake").status_code == 200


# ---------------------------------------------------- the extension's calls

def test_capture_accepts_the_exact_body_the_worker_sends(client):
    """Mirror of the `capture` branch in extension/background.js, nulls included."""
    res = client.post("/api/capture", json={
        "project": None,
        "tool": "chatgpt",
        "role": "user",
        "text": "I am building a local memory layer that must cost nothing to run.",
        "session_id": None,
        "url": None,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "saved"
    assert body["project"]           # inferred from cwd, never null


def test_capture_rejects_a_role_the_adapter_could_not_produce(client):
    res = client.post("/api/capture", json={
        "project": "shapes", "tool": "chatgpt", "role": "system", "text": "x" * 40,
    })
    assert res.status_code == 422


def test_fluff_is_skipped_not_stored(client):
    res = client.post("/api/capture", json={
        "project": "shapes", "tool": "chatgpt", "role": "user", "text": "thanks!",
    })
    assert res.status_code == 200
    assert res.json()["status"] == "skipped"


def test_session_accepts_the_exact_body_the_worker_sends(client):
    res = client.post("/api/session", json={
        "project": "shapes",
        "tool": "chatgpt",
        "facts": [{"kind": "decision", "text": "Render markdown from SQLite, never the reverse."}],
        "summary": None,
        "next_step": "Write the concurrency tests.",
        "session_id": None,
    })
    assert res.status_code == 200
    assert res.json()["status"] == "saved"


def test_resume_returns_project_and_brief(client):
    """background.js destructures exactly these two keys."""
    client.post("/api/session", json={
        "project": "shapes", "tool": "chatgpt",
        "facts": [{"kind": "goal", "text": "Carry the project brief between every tool."}],
        "next_step": "Load the extension unpacked.",
    })
    body = client.get("/api/resume", params={"project": "shapes"}).json()
    assert set(body) == {"project", "brief"}
    assert body["project"] == "shapes"
    assert "Carry the project brief between every tool." in body["brief"]
    assert "Load the extension unpacked." in body["brief"]


def test_overview_rows_carry_the_key_the_popup_reads(client):
    """popup.js sets `opt.value = p.project` for the datalist."""
    client.post("/api/session", json={
        "project": "shapes", "tool": "chatgpt",
        "facts": [{"kind": "goal", "text": "Populate the popup project picker."}],
    })
    rows = client.get("/api/overview").json()["projects"]
    assert rows
    assert "project" in rows[0]
    assert {"title", "facts", "sessions", "last_tool"} <= set(rows[0])


def test_config_round_trips_the_popup_toggles(client):
    before = client.get("/api/config").json()
    assert set(before) == {"capture", "redact", "paused_sites"}

    client.post("/api/config", json={
        "capture": False, "redact": True, "paused_sites": ["chatgpt.com"],
    })
    after = client.get("/api/config").json()
    assert after == {"capture": False, "redact": True, "paused_sites": ["chatgpt.com"]}

    # And it is on disk, so `omnimem doctor` and the CLI see the same state.
    assert store.load_config()["paused_sites"] == ["chatgpt.com"]


def test_an_unknown_config_key_is_ignored(client):
    client.post("/api/config", json={"capture": True, "port": 9999, "evil": "x"})
    assert "evil" not in store.load_config()


def test_search_resolves_the_project_slug(client):
    """
    Facts are keyed by slug, so passing an unslugified name straight through
    would silently match nothing stored under the slugified form.
    """
    client.post("/api/session", json={
        "project": "My Project", "tool": "chatgpt",
        "facts": [{"kind": "decision", "text": "Use SQLite with FTS5 rather than a vector database."}],
    })
    res = client.get("/api/search", params={"query": "vector database", "project": "My Project"})
    assert res.status_code == 200
    assert res.json()["facts"]


def test_search_limit_is_bounded(client):
    assert client.get("/api/search", params={"query": "x", "limit": 0}).status_code == 422
    assert client.get("/api/search", params={"query": "x", "limit": 500}).status_code == 422


# ---------------------------------------------------------------- redaction

def test_secrets_are_scrubbed_before_they_reach_the_store(client):
    """
    Redaction happens on the way *in*. Once a secret is in index.db it is also in
    a rendered brief that gets injected into three other tools.
    """
    client.post("/api/capture", json={
        "project": "secrets", "tool": "chatgpt", "role": "user",
        "text": (
            "The deploy failed. My key is sk-abcdefghijklmnopqrstuvwxyz012345 and "
            "you can reach me at dev@example.com to debug the auth token."
        ),
    })
    brief = client.get("/api/resume", params={"project": "secrets"}).json()["brief"]
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in brief
    assert "dev@example.com" not in brief

    found = client.get("/api/search", params={"query": "deploy failed", "project": "secrets"})
    dumped = found.text
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in dumped
    assert "dev@example.com" not in dumped


@pytest.mark.parametrize("secret", [
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "sk-ant-abcdefghijklmnopqrstuvwxyz0123",
    "xoxb-123456789012-abcdefghijkl",
    # A Google key is AIza + exactly 35 characters. Length is the whole signal
    # here -- there is no separator and no checksum -- so a fixture of the wrong
    # length tests nothing.
    "AIzaSyD-1234567890abcdefghijklmnopqrstu",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
    "someone@example.org",
])
def test_the_shapes_that_actually_show_up_in_chat(secret):
    scrubbed = redact.redact(f"here it is: {secret} -- please help")
    assert secret not in scrubbed
    assert "[redacted" in scrubbed
    assert redact.has_secret(secret)


def test_redaction_covers_assignments_and_url_credentials():
    assert "hunter2hunter2" not in redact.redact("password = hunter2hunter2")
    assert "s3cr3tvalue" not in redact.redact('api_key: "s3cr3tvalue"')
    assert "p4ssw0rd" not in redact.redact("postgres://admin:p4ssw0rd@db.internal:5432/app")
    assert redact.redact("the database is slow today") == "the database is slow today"


# ------------------------------------------------------------ route order

def test_the_landing_mount_does_not_shadow_the_api(anon):
    """
    The regression this guards: a StaticFiles mount at "/" registered before the
    API turns every /api route into a 404 from StaticFiles. Registration order is
    the only thing keeping these apart, and nothing else would fail loudly.
    """
    assert anon.get("/").status_code == 200
    assert "OmniMem" in anon.get("/").text
    # 401, not 404 -- the route still exists and is still gated.
    assert anon.get("/api/projects").status_code == 401
    assert anon.get("/health").status_code == 200


def test_docs_serves_the_landing_page_and_bare_docs_redirects(anon):
    """
    popup.js opens `http://127.0.0.1:<port>/docs`. A Starlette Mount at "/docs"
    matches only "/docs/...", so bare "/docs" needs the explicit redirect or the
    extension's Setup guide button lands on a 404.
    """
    redirected = anon.get("/docs", follow_redirects=False)
    assert redirected.status_code == 308
    assert redirected.headers["location"] == "/docs/"

    assert anon.get("/docs", follow_redirects=True).status_code == 200
    assert anon.get("/docs/styles.css").status_code == 200


def test_swagger_moved_off_docs(anon):
    assert anon.get("/api-docs").status_code == 200
    assert anon.get("/openapi.json").status_code == 200


def test_the_landing_page_makes_no_third_party_requests():
    """
    The page claims nothing leaves your machine. It should not need a CDN to say
    so -- and this is cheaper to assert than to remember.

    Checked against the *references* rather than the source text: an earlier
    version of this test grepped for the substring "analytics" and tripped over
    the comment promising there is no analytics.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(here, "landing", "index.html"), encoding="utf-8").read()
    css = open(os.path.join(here, "landing", "styles.css"), encoding="utf-8").read()

    markup = re.sub(r"<!--.*?-->", "", html, flags=re.S)

    for attr, value in re.findall(r"\b(src|href|action|poster)\s*=\s*[\"']([^\"']*)", markup):
        assert not re.match(r"(?i)(https?:)?//", value) or value.startswith(
            ("http://127.0.0.1", "http://localhost")
        ), f"{attr}={value!r} leaves the machine"

    # Webfonts and background images are references too, just spelled differently.
    for source in (markup, css):
        assert "@import" not in source
        assert not re.search(r"url\(\s*[\"']?(https?:)?//", source)

    # And no third-party script tags, inline or otherwise.
    assert "googletagmanager" not in markup
    assert not re.search(r"(?i)<script[^>]*\bsrc\s*=", markup)


# ------------------------------------------------------------------- CORS

def test_cors_is_restricted_to_known_chat_origins_and_extensions(anon):
    allowed = anon.options("/api/capture", headers={
        "Origin": "https://chatgpt.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-omnimem-token",
    })
    assert allowed.headers.get("access-control-allow-origin") == "https://chatgpt.com"
    # Never credentials: nothing may ride on ambient cookies.
    assert "access-control-allow-credentials" not in allowed.headers

    blocked = anon.options("/api/capture", headers={
        "Origin": "https://chatgpt.com.evil.example",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in blocked.headers
