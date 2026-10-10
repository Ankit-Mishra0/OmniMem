"""
OmniMem server: the MCP surface and the local REST bridge for the extension.

Both are served from one uvicorn process -- the FastMCP streamable-HTTP app is
mounted under /mcp on the FastAPI app, so the browser extension and MCP clients
share a port. `--mcp` still runs plain stdio for editors that launch the server
themselves.

The server is an *accelerator, not a dependency*. Every write path here also
exists as a direct SQLite write in `omnimem.core`, so a dead server degrades the
extension quietly instead of surfacing tool errors to a model -- a model that
sees `omnimem_save_session` fail once will stop calling it for the rest of the
session, which is the worst possible failure mode for a memory tool.

Security posture (this replaces `allow_origins=["*"]` + `allow_credentials=True`,
which let any page you visited read your entire memory store off localhost:8000):

* bound to 127.0.0.1;
* CORS restricted to the specific chat origins and extension schemes;
* `allow_credentials=False`, so nothing rides on ambient cookies;
* every /api/* route requires the bearer token from ~/.omnimem/token.

The one deliberate exception is /api/handshake, which hands the token to the
extension so install stays zero-friction. It is gated on the `Origin` header
being an extension scheme or absent. That check holds against web pages because
browsers set `Origin` themselves and page JavaScript cannot forge it; it does
not hold against other locally-installed extensions or native processes, which
is an accepted trade for a personal-machine prototype.
"""

import argparse
import hmac
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from omnimem.core import db, ingest, notes, store
from omnimem.core import facts as facts_mod

import omnimem

# Legacy note-level API, kept working for the REST endpoints the extension and
# the original tests already use.
from omnimem.database import list_projects, search_notes
from omnimem.memory_engine import build_active_context, is_important, process_and_store_memory

try:
    from fastmcp import FastMCP
    mcp: Optional[Any] = FastMCP("OmniMem")
except ImportError:  # pragma: no cover - fastmcp is a declared dependency
    mcp = None

VERSION = omnimem.__version__

# Origins allowed to talk to the bridge. Chat hosts are listed explicitly rather
# than by regex so a lookalike domain cannot slip in.
ALLOWED_ORIGINS = [
    "https://chatgpt.com",
    "https://chat.openai.com",
    "https://claude.ai",
    "https://gemini.google.com",
    "https://aistudio.google.com",
    "https://www.perplexity.ai",
    "https://perplexity.ai",
    "https://chat.deepseek.com",
    "https://grok.com",
    "https://x.com",
    "https://copilot.microsoft.com",
    "https://chat.mistral.ai",
]
EXTENSION_ORIGIN_RE = r"^(chrome-extension|moz-extension|safari-web-extension)://[a-z0-9\-]+$"

_ready = False
_ready_lock = threading.Lock()


def ensure_ready() -> None:
    """
    Idempotent schema init.

    Called per request rather than at import time: importing a module should
    never create directories on disk, and it made the package unimportable in
    any sandbox that could not write to $HOME.
    """
    global _ready
    if _ready:
        return
    with _ready_lock:
        if _ready:
            return
        store.ensure_home()
        db.init_schema()
        _ready = True


def _conn():
    ensure_ready()
    return db.connect()


def _resolve(project: Optional[str], cwd: Optional[str] = None) -> str:
    return store.resolve_project(project, cwd or os.getcwd())


# --------------------------------------------------------------------------- MCP

if mcp is not None:

    @mcp.tool()
    def omnimem_resume(project: Optional[str] = None) -> str:
        """
        Load the project brief. CALL THIS FIRST in any new session.

        Returns everything already known about the project -- goals, decisions in
        force, constraints, gotchas, open questions and where the last session
        stopped -- so the user does not have to re-explain the project. If
        `project` is omitted, it is inferred from the working directory (git
        remote name, then folder name).
        """
        conn = _conn()
        slug = _resolve(project)
        return ingest.resume(conn, slug, root=store.project_root(os.getcwd()))

    @mcp.tool()
    def omnimem_save_session(
        tool: str = "mcp",
        project: Optional[str] = None,
        facts: Optional[List[Dict[str, str]]] = None,
        summary: Optional[str] = None,
        next_step: Optional[str] = None,
        transcript: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record what this session established, so the next tool can continue.

        `facts` is the high-value argument: a list of {"kind", "text"} where kind
        is one of goal, decision, constraint, convention, stack, entity, gotcha,
        task, open_question. Keep each text under 180 characters and atomic --
        one claim per entry. `next_step` is what a fresh session should do first.
        """
        conn = _conn()
        slug = _resolve(project)
        return ingest.ingest_session(
            conn, slug, tool=tool, claims=facts, summary=summary,
            next_step=next_step, transcript=transcript, session_id=session_id,
            root=store.project_root(os.getcwd()),
        )

    @mcp.tool()
    def omnimem_search(
        query: str, project: Optional[str] = None, limit: int = 8
    ) -> Dict[str, Any]:
        """Search remembered facts and past conversation notes for this project."""
        conn = _conn()
        slug = _resolve(project) if project is not None else None
        return ingest.search(conn, query, project=slug, limit=limit)

    @mcp.tool()
    def omnimem_add_decision(
        text: str,
        project: Optional[str] = None,
        rationale: Optional[str] = None,
        tool: str = "mcp",
    ) -> Dict[str, Any]:
        """
        Record an architectural decision. Use this the moment one is made.

        Decisions are appended to DECISIONS.md immutably and surface in every
        future brief, which is what stops the next session re-litigating them.
        """
        conn = _conn()
        slug = _resolve(project)
        return ingest.add_decision(conn, slug, text, tool=tool, rationale=rationale)

    @mcp.tool()
    def omnimem_set_task(
        text: str, done: bool = False, project: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add an open task, or mark an existing one done (`done=true`)."""
        conn = _conn()
        slug = _resolve(project)
        return ingest.set_task(conn, slug, text, state="done" if done else "open")

    @mcp.tool()
    def omnimem_handoff(project: Optional[str] = None) -> str:
        """State as of the last recorded session: what happened and where to pick up."""
        conn = _conn()
        slug = _resolve(project)
        ingest.digest(conn, slug, root=store.project_root(os.getcwd()))
        path = os.path.join(store.project_dir(slug), "HANDOFF.md")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return f"No handoff recorded yet for '{slug}'."

    @mcp.tool()
    def omnimem_list_projects() -> List[Dict[str, Any]]:
        """Every project OmniMem is tracking, with fact counts and last-used tool."""
        return ingest.overview(_conn())

    @mcp.prompt(name="resume")
    def resume_prompt(project: str = "") -> str:
        """Pick up where the last session left off (any tool)."""
        conn = _conn()
        slug = _resolve(project or None)
        brief = ingest.resume(conn, slug, root=store.project_root(os.getcwd()))
        return (
            "Here is the OmniMem brief for this project. Read it, then continue "
            "the work from where it stopped. Do not ask me to re-explain "
            "anything that is already in the brief.\n\n" + brief
        )

    @mcp.resource("omnimem://projects")
    def projects_resource() -> str:
        import json
        return json.dumps(ingest.overview(_conn()), indent=2)

    # ---- Legacy tool names, kept so existing configs and tests keep working.

    @mcp.tool()
    def save_project_memory(
        project_id: str,
        content: str,
        platform: str = "mcp",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Deprecated: prefer omnimem_save_session. Saves one raw note."""
        ensure_ready()
        note = process_and_store_memory(
            project_id=project_id, content=content, platform=platform, tags=tags
        )
        if note:
            return {"status": "saved", "note": note}
        return {"status": "skipped", "reason": "Content filtered by importance gating"}

    @mcp.tool()
    def query_project_memory(
        query: str, project_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Deprecated: prefer omnimem_search. FTS5/BM25 over raw notes."""
        ensure_ready()
        return search_notes(query=query, project_id=project_id, limit=limit)

    @mcp.tool()
    def get_active_project_state(project_id: str, limit: int = 5) -> str:
        """Deprecated: prefer omnimem_resume. Formatted context block."""
        ensure_ready()
        return build_active_context(project_id=project_id, limit=limit)

else:  # pragma: no cover
    def save_project_memory(*a, **k):  # type: ignore[misc]
        raise RuntimeError("fastmcp is not installed")

    def query_project_memory(*a, **k):  # type: ignore[misc]
        raise RuntimeError("fastmcp is not installed")

    def get_active_project_state(*a, **k):  # type: ignore[misc]
        raise RuntimeError("fastmcp is not installed")


# --------------------------------------------------------------------------- auth

def require_token(
    authorization: Optional[str] = Header(default=None),
    x_omnimem_token: Optional[str] = Header(default=None),
) -> None:
    """
    Bearer check on every /api/* route.

    CORS alone is not enough: a cross-origin POST with a simple content type is
    *sent* before the browser hides the response, so a foreign page could still
    write to the store. compare_digest keeps the check constant-time.
    """
    ensure_ready()
    expected = store.read_token()
    if not expected:
        raise HTTPException(status_code=503, detail="No token provisioned; run `omnimem serve` once.")
    supplied = x_omnimem_token or ""
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing OmniMem token")


# --------------------------------------------------------------------------- REST

class SaveMemoryRequest(BaseModel):
    project_id: str
    content: str
    platform: str = "web_ui"
    tags: Optional[List[str]] = None


class CaptureRequest(BaseModel):
    project: Optional[str] = None
    tool: str = "browser"
    role: str = Field(default="user", pattern="^(user|assistant)$")
    text: str
    session_id: Optional[str] = None
    url: Optional[str] = None


class SessionRequest(BaseModel):
    project: Optional[str] = None
    tool: str = "browser"
    facts: Optional[List[Dict[str, str]]] = None
    summary: Optional[str] = None
    next_step: Optional[str] = None
    session_id: Optional[str] = None


def _build_app() -> FastAPI:
    mcp_app = mcp.http_app(
        path="/",
        allowed_origins=ALLOWED_ORIGINS,
    ) if mcp is not None else None

    application = FastAPI(
        title="OmniMem Local Engine",
        description="Local-first cross-tool project memory for LLMs.",
        version=VERSION,
        # Swagger moves off /docs to make room for the landing page, which is
        # what the extension's "Setup guide" link and a curious user both expect
        # to find there. The API reference is for whoever goes looking for it.
        docs_url="/api-docs",
        redoc_url=None,
        # FastMCP's session manager lives in its own lifespan; chaining it here
        # is what makes the mounted app usable rather than 500ing on connect.
        lifespan=mcp_app.lifespan if mcp_app is not None else None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=EXTENSION_ORIGIN_RE,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-OmniMem-Token"],
    )

    if mcp_app is not None:
        application.mount("/mcp", mcp_app)

    return application


app = _build_app()


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "ok", "service": "omnimem", "version": VERSION,
            "mcp": mcp is not None}


@app.get("/api/handshake")
def api_handshake(request: Request) -> Dict[str, str]:
    """Hand the bearer token to a browser extension. See the module docstring."""
    ensure_ready()
    origin = request.headers.get("origin") or ""
    if origin and not origin.split("://")[0] in (
        "chrome-extension", "moz-extension", "safari-web-extension"
    ):
        raise HTTPException(status_code=403, detail="Handshake is extension-only")
    return {"token": store.ensure_token(), "version": VERSION}


@app.get("/api/config", dependencies=[Depends(require_token)])
def api_config() -> Dict[str, Any]:
    cfg = store.load_config()
    return {"capture": cfg.get("capture", True), "redact": cfg.get("redact", True),
            "paused_sites": cfg.get("paused_sites", [])}


@app.post("/api/config", dependencies=[Depends(require_token)])
def api_set_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    cfg = store.load_config()
    for key in ("capture", "redact", "paused_sites"):
        if key in patch:
            cfg[key] = patch[key]
    store.save_config(cfg)
    return {"status": "ok", "config": cfg}


@app.get("/api/resume", dependencies=[Depends(require_token)])
def api_resume(project: Optional[str] = None) -> Dict[str, Any]:
    conn = _conn()
    slug = _resolve(project)
    return {"project": slug, "brief": ingest.resume(conn, slug, with_footer=False)}


@app.post("/api/capture", dependencies=[Depends(require_token)])
def api_capture(req: CaptureRequest) -> Dict[str, Any]:
    conn = _conn()
    slug = _resolve(req.project)
    return ingest.ingest_turn(
        conn, slug, text=req.text, role=req.role, tool=req.tool,
        session_id=req.session_id or "browser", digest_now=True,
    )


@app.post("/api/session", dependencies=[Depends(require_token)])
def api_session(req: SessionRequest) -> Dict[str, Any]:
    conn = _conn()
    slug = _resolve(req.project)
    return ingest.ingest_session(
        conn, slug, tool=req.tool, claims=req.facts, summary=req.summary,
        next_step=req.next_step, session_id=req.session_id,
    )


@app.get("/api/search", dependencies=[Depends(require_token)])
def api_search(
    query: str,
    project: Optional[str] = None,
    limit: int = Query(default=8, ge=1, le=50),
) -> Dict[str, Any]:
    # Resolve, don't pass through: facts are keyed by slug, so an unslugified
    # "MyProject" would silently match nothing stored under "myproject".
    slug = _resolve(project) if project is not None else None
    return ingest.search(_conn(), query, project=slug, limit=limit)


@app.get("/api/overview", dependencies=[Depends(require_token)])
def api_overview() -> Dict[str, Any]:
    return {"projects": ingest.overview(_conn())}


# ---- Legacy note-level endpoints. Still used by the original extension build.

@app.post("/api/memory/save", dependencies=[Depends(require_token)])
def api_save_memory(req: SaveMemoryRequest) -> Dict[str, Any]:
    ensure_ready()
    note = process_and_store_memory(
        project_id=req.project_id, content=req.content,
        platform=req.platform, tags=req.tags,
    )
    if note:
        return {"status": "saved", "note": note}
    return {"status": "skipped", "reason": "Filtered by importance gating"}


@app.get("/api/memory/query", dependencies=[Depends(require_token)])
def api_query_memory(
    query: str,
    project_id: Optional[str] = None,
    limit: int = Query(default=5, ge=1, le=50),
) -> Dict[str, Any]:
    ensure_ready()
    results = search_notes(query=query, project_id=project_id, limit=limit)
    return {"query": query, "project_id": project_id, "results": results}


@app.get("/api/memory/active_state", dependencies=[Depends(require_token)])
def api_active_state(
    project_id: str = Query(..., description="Project workspace identifier"),
    limit: int = Query(default=5, ge=1, le=20),
) -> Dict[str, Any]:
    ensure_ready()
    return {"project_id": project_id, "context": build_active_context(project_id, limit)}


@app.get("/api/projects", dependencies=[Depends(require_token)])
def api_list_projects() -> Dict[str, Any]:
    ensure_ready()
    return {"projects": list_projects()}


# ------------------------------------------------------------------ landing page

# Mounted last, on purpose.
#
# Starlette matches routes in registration order, so a mount at "/" registered
# here loses to every explicit /api/... route and to the /mcp mount above it.
# Registering it earlier would shadow the whole API with a 404 from StaticFiles.
#
# It is served from the engine rather than only opened from disk so that the
# setup instructions a user reads always match the version they have installed,
# and so the page's same-origin /health check can light up the "engine running"
# chip without asking the browser to make a cross-origin request to localhost.
_LANDING = Path(__file__).resolve().parent.parent / "landing"

if _LANDING.is_dir():
    # A Mount at "/docs" only matches "/docs/..." -- bare "/docs" falls through to
    # a 404, and bare "/docs" is exactly what the extension's Setup guide link and
    # anyone typing the URL by hand will use. Redirect rather than serving
    # index.html here, so relative asset paths resolve against /docs/ either way.
    @app.get("/docs", include_in_schema=False)
    def docs_redirect() -> RedirectResponse:
        return RedirectResponse("/docs/", status_code=308)

    _pages = StaticFiles(directory=str(_LANDING), html=True)
    app.mount("/docs", _pages, name="docs")
    app.mount("/", _pages, name="landing")


# --------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OmniMem context engine")
    parser.add_argument("--mcp", action="store_true", help="run as a stdio MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    ensure_ready()
    if args.mcp:
        if mcp is None:
            print("fastmcp is not installed", file=sys.stderr)
            return 1
        # stdout is the protocol channel here -- never print to it.
        mcp.run()
        return 0

    import uvicorn
    cfg = store.load_config()
    port = args.port or int(cfg.get("port", 8000))
    print(f"OmniMem on http://{args.host}:{port}  (MCP at /mcp/)", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
