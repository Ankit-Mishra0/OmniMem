"""
OmniMem command line: serve, install, hook, digest, resume.

Nothing heavy is imported at module scope. This file is the entry point for
Claude Code hooks, which run on *every prompt submission* -- importing FastAPI,
pydantic and fastmcp there would add 300 ms+ of latency to every keystroke-to-
response cycle. `omnimem.core` is stdlib-only and imports in ~20 ms, and the
server is imported lazily inside `cmd_serve`.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

from omnimem.core import db, ingest, store

HOOK_EVENTS = ("session-start", "user-prompt", "stop", "session-end")

# Kept in sync with the SessionStart injection cap in Claude Code.
MAX_INJECT_CHARS = 10000

BEGIN_MARK = "<!-- omnimem:begin -->"
END_MARK = "<!-- omnimem:end -->"

RULES_TEXT = """\
## OmniMem — cross-tool project memory

OmniMem is connected as an MCP server. It remembers this project across tools
(Claude Code, Codex, Cursor, Antigravity, ChatGPT, Gemini), so the user should
never have to re-explain the project.

1. **At the start of every session, call `omnimem_resume` before doing anything
   else.** It returns the project brief: goals, decisions already made,
   constraints, known gotchas, and where the last session stopped. Read it and
   continue from there. Do not ask the user to repeat anything it already says.
2. **When a decision is made, call `omnimem_add_decision` immediately.** That is
   what stops the next session re-litigating it.
3. **Before the session ends, or after any substantial piece of work, call
   `omnimem_save_session`** with `facts` (a list of `{kind, text}`, one atomic
   claim each, under 180 characters) and `next_step` (what a fresh session should
   do first). Valid kinds: goal, decision, constraint, convention, stack, entity,
   gotcha, task, open_question.
4. Use `omnimem_search` instead of asking the user about earlier work.

Recording a gotcha ("the delete trigger does not fire under REPLACE") is the
single highest-value thing you can save: it costs a dozen tokens and saves the
next session an hour.
"""


# --------------------------------------------------------------------------- util

def _out(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")


def _repo_root() -> str:
    """Directory containing the `omnimem` package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _launcher() -> Tuple[str, List[str], Dict[str, str]]:
    """
    (command, args, env) that starts the stdio MCP server.

    The interpreter is spelled out absolutely and PYTHONPATH is set explicitly,
    because OmniMem is run from a checkout rather than installed on PATH -- and
    MCP hosts spawn servers with a minimal environment where `python` may not
    resolve to the virtualenv at all.

    `OMNIMEM_HOME` is propagated for the same reason: a GUI-launched host (Claude
    Desktop, Antigravity, Cursor) inherits the *login* environment, not the shell
    the install ran in. Without it a relocated store would be silently ignored
    and the tool would read an empty default one. Only written when it is
    actually set, so the common case leaves no redundant entry behind.
    """
    env = {"PYTHONPATH": _repo_root()}
    custom = os.environ.get("OMNIMEM_HOME")
    if custom:
        env["OMNIMEM_HOME"] = os.path.abspath(os.path.expanduser(custom))
    return sys.executable, ["-m", "omnimem.cli", "mcp"], env


def _hook_command(event: str) -> str:
    cmd, _args, env = _launcher()
    prefix = " ".join(f"{k}={_shq(v)}" for k, v in env.items())
    return f"{prefix} {_shq(cmd)} -m omnimem.cli hook {event}"


def _shq(value: str) -> str:
    return value if re.fullmatch(r"[\w@%+=:,./-]+", value) else "'" + value.replace("'", "'\\''") + "'"


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read().strip()
    except OSError:
        return {}
    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError:
        # Never clobber a file we cannot parse -- back it up and start clean.
        os.replace(path, path + ".omnimem-backup")
        _err(f"  ! {path} was not valid JSON; moved to {path}.omnimem-backup")
        return {}


def _write_json(path: str, data: Dict[str, Any]) -> None:
    from omnimem.core import locks
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Hash-gated: Cursor and VS Code watch their MCP config and restart every
    # server in it when the file changes, so rewriting identical bytes on a
    # re-install would needlessly bounce the user's other MCP servers.
    locks.write_if_changed(path, json.dumps(data, indent=2) + "\n")


def _splice(body: str, block: str) -> str:
    """
    Insert or replace a marker-delimited block.

    Marker-delimited rather than append-only so re-running `install` is
    idempotent instead of stacking duplicate copies of the rules.

    The tail is re-normalised rather than kept verbatim: `payload` already ends
    in a newline, so splicing the old tail back on unchanged added one more blank
    line to the file on every single install.
    """
    payload = f"{BEGIN_MARK}\n{block.rstrip()}\n{END_MARK}\n"
    if BEGIN_MARK in body and END_MARK in body:
        head = body.split(BEGIN_MARK, 1)[0]
        tail = body.split(END_MARK, 1)[1].lstrip("\n")
        return head + payload + (("\n" + tail) if tail.strip() else "")
    sep = "" if not body.strip() else ("\n" if body.endswith("\n") else "\n\n")
    return body + sep + payload


def _write_rules(path: str) -> None:
    from omnimem.core import locks
    try:
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
    except OSError:
        body = ""
    # Cursor only applies an .mdc rule automatically when its front matter says
    # so; without alwaysApply the file is inert and the agent never resumes.
    # The front matter is prepended *outside* the markers -- inside, the next
    # `install` would replace the marker block and silently delete it.
    if path.endswith(".mdc") and not body.strip():
        body = ("---\ndescription: Cross-tool project memory via OmniMem\n"
                "alwaysApply: true\n---\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    locks.write_if_changed(path, _splice(body, RULES_TEXT))


# --------------------------------------------------------------------------- tools

def _mcp_json_entry() -> Dict[str, Any]:
    cmd, args, env = _launcher()
    return {"command": cmd, "args": args, "env": env}


HOME = os.path.expanduser("~")


def _tool_targets() -> Dict[str, Dict[str, Any]]:
    """
    Where each host keeps its MCP config, and which key it uses.

    The key differences are not cosmetic and are the usual reason a server
    silently fails to appear: VS Code uses `servers`, Zed uses `context_servers`,
    Codex uses TOML tables, and Continue uses a YAML *list*. Antigravity and
    Gemini CLI share ~/.gemini/settings.json.
    """
    return {
        "claude-code": {
            "label": "Claude Code",
            "config": os.path.join(HOME, ".claude.json"),
            "key": "mcpServers",
            "hooks": os.path.join(HOME, ".claude", "settings.json"),
            "push": "hooks (fully automatic)",
        },
        "antigravity": {
            "label": "Antigravity",
            "config": os.path.join(HOME, ".gemini", "settings.json"),
            "key": "mcpServers",
            # Antigravity ignores AGENTS.md; its always-on global rules file is
            # ~/.gemini/GEMINI.md (or .agents/rules/*.md set to "Always On").
            "rules": os.path.join(HOME, ".gemini", "GEMINI.md"),
            "push": "always-on rules file",
        },
        "gemini-cli": {
            "label": "Gemini CLI",
            "config": os.path.join(HOME, ".gemini", "settings.json"),
            "key": "mcpServers",
            "rules": os.path.join(HOME, ".gemini", "GEMINI.md"),
            "push": "always-on rules file",
        },
        "codex": {
            "label": "Codex CLI",
            "config": os.path.join(HOME, ".codex", "config.toml"),
            "format": "toml",
            "rules": os.path.join(HOME, ".codex", "AGENTS.md"),
            "push": "AGENTS.md",
        },
        "claude-desktop": {
            "label": "Claude Desktop",
            "config": os.path.join(
                HOME, "Library", "Application Support", "Claude",
                "claude_desktop_config.json",
            ),
            "key": "mcpServers",
            "push": "manual / slash command (no rules file)",
        },
        "cursor": {
            "label": "Cursor",
            "config": os.path.join(HOME, ".cursor", "mcp.json"),
            "key": "mcpServers",
            "rules": os.path.join(HOME, ".cursor", "rules", "omnimem.mdc"),
            "push": "always-apply rule",
        },
        "windsurf": {
            "label": "Windsurf",
            "config": os.path.join(HOME, ".codeium", "windsurf", "mcp_config.json"),
            "key": "mcpServers",
            "rules": os.path.join(HOME, ".codeium", "windsurf", "memories",
                                  "omnimem.md"),
            "push": "rules file",
        },
        "vscode": {
            "label": "VS Code (Copilot)",
            "config": os.path.join(os.getcwd(), ".vscode", "mcp.json"),
            "key": "servers",
            "push": "manual",
        },
        "zed": {
            "label": "Zed",
            "config": os.path.join(HOME, ".config", "zed", "settings.json"),
            "key": "context_servers",
            "push": "manual",
        },
    }


def _install_json(path: str, key: str) -> None:
    data = _read_json(path)
    section = data.get(key)
    if not isinstance(section, dict):
        section = {}
    section["omnimem"] = _mcp_json_entry()
    data[key] = section
    _write_json(path, data)


_TOML_BLOCK_RE = re.compile(
    r"(?ms)^\[mcp_servers\.omnimem\].*?(?=^\[(?!mcp_servers\.omnimem)|\Z)"
)


def _install_toml(path: str) -> None:
    """
    Splice a `[mcp_servers.omnimem]` table into config.toml.

    Text splicing rather than parse-and-rewrite: stdlib `tomllib` is read-only,
    and a hand-rolled writer would reformat and strip the comments out of a file
    the user maintains by hand.
    """
    from omnimem.core import locks
    cmd, args, env = _launcher()
    block = (
        "[mcp_servers.omnimem]\n"
        f'command = {json.dumps(cmd)}\n'
        f"args = [{', '.join(json.dumps(a) for a in args)}]\n"
        "startup_timeout_sec = 30\n"
        "\n[mcp_servers.omnimem.env]\n"
        + "".join(f"{k} = {json.dumps(v)}\n" for k, v in env.items())
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
    except OSError:
        body = ""
    if _TOML_BLOCK_RE.search(body):
        body = _TOML_BLOCK_RE.sub(block + "\n", body, count=1)
    else:
        body = (body.rstrip() + "\n\n" if body.strip() else "") + block
    # Exactly one trailing newline. The substitution above deliberately leaves a
    # blank line so a following table is never glued to ours, but at the end of
    # the file that same blank line would accumulate on every install.
    body = body.rstrip("\n") + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    locks.write_if_changed(path, body)


def _install_hooks(path: str) -> None:
    """
    Claude Code hooks: the only true *push* capture path.

    * SessionStart prints the brief; plain stdout from this event is added to the
      context Claude can see, so no tool call and no model cooperation is needed.
      Matched on startup|resume|clear so a compaction does not re-inject it.
    * UserPromptSubmit receives the `prompt` field -- this is what captures what
      the user is actually trying to achieve.
    * Stop receives `last_assistant_message`. The transcript file lags the live
      conversation, so the field is used and transcript_path is never parsed.

    Every capture hook exits 0 unconditionally: `exit 2` on Stop *prevents Claude
    from stopping*, so a memory bug must never be able to wedge a coding session.
    """
    data = _read_json(path)
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}

    spec = {
        "SessionStart": ("startup|resume|clear", "session-start", 15),
        "UserPromptSubmit": (None, "user-prompt", 30),
        "Stop": (None, "stop", 15),
        "SessionEnd": (None, "session-end", 5),
    }
    for event, (matcher, sub, timeout) in spec.items():
        entry: Dict[str, Any] = {
            "hooks": [{"type": "command", "command": _hook_command(sub),
                       "timeout": timeout}]
        }
        if matcher:
            entry["matcher"] = matcher
        existing = [
            e for e in (hooks.get(event) or [])
            if "omnimem.cli hook" not in json.dumps(e)
        ]
        hooks[event] = existing + [entry]

    data["hooks"] = hooks
    _write_json(path, data)


def cmd_install(args: argparse.Namespace) -> int:
    store.ensure_home()
    store.ensure_token()
    db.init_schema()

    targets = _tool_targets()
    names: List[str]
    if args.all:
        names = [n for n, t in targets.items() if os.path.exists(
            os.path.dirname(t["config"]))]
        if not names:
            _err("No supported tools detected.")
            return 1
    else:
        names = args.tools or []
        unknown = [n for n in names if n not in targets]
        if unknown:
            _err(f"Unknown tool(s): {', '.join(unknown)}")
            _err("Known: " + ", ".join(sorted(targets)))
            return 2

    for name in names:
        target = targets[name]
        path = target["config"]
        try:
            if target.get("format") == "toml":
                _install_toml(path)
            else:
                _install_json(path, target["key"])
            _out(f"✓ {target['label']}: MCP server registered in {path}")

            if name == "claude-code":
                _install_hooks(target["hooks"])
                _out(f"✓ {target['label']}: capture hooks added to {target['hooks']}")
            if target.get("rules"):
                _write_rules(target["rules"])
                _out(f"✓ {target['label']}: rules written to {target['rules']}")
        except OSError as exc:
            _err(f"✗ {target['label']}: {exc}")

    _out("")
    _out("Restart the tool(s) so they pick up the new configuration.")
    return 0


def cmd_list_tools(_args: argparse.Namespace) -> int:
    targets = _tool_targets()
    width = max(len(t["label"]) for t in targets.values())
    for name, target in sorted(targets.items()):
        present = os.path.exists(os.path.dirname(target["config"]))
        mark = "detected" if present else "not found"
        _out(f"  {target['label']:<{width}}  {name:<15} {mark:<10} capture: {target['push']}")
    return 0


# --------------------------------------------------------------------------- hooks

def _hook_payload() -> Dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    try:
        return json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}


def _hook_project(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    cwd = payload.get("cwd") or os.getcwd()
    return store.resolve_project(None, cwd), store.project_root(cwd)


def _capture(payload: Dict[str, Any], role: str, text: str) -> None:
    """
    Write the turn straight to SQLite, falling back to the inbox.

    Direct is preferred because an inbox event is not reflected in the brief
    until something drains it, which in a hooks-only setup means the *next*
    session -- a day of latency on the facts that matter most. The write is one
    BEGIN IMMEDIATE transaction, so contention is bounded by busy_timeout; if it
    still fails, the event is queued rather than lost.
    """
    if not text or not text.strip():
        return
    project, root = _hook_project(payload)
    session_id = str(payload.get("session_id") or "hook")
    event = {
        "kind": "turn", "project": project, "root": root, "role": role,
        "text": text, "tool": "claude-code", "session_id": session_id,
    }
    try:
        conn = db.init_schema()
        ingest.ingest_turn(
            conn, project, text=text, role=role, tool="claude-code",
            session_id=session_id, root=root, digest_now=False,
        )
    except Exception:
        try:
            ingest.enqueue(event)
        except Exception:
            pass


def cmd_hook(args: argparse.Namespace) -> int:
    """
    All hook subcommands exit 0 no matter what.

    On `Stop`, exit code 2 tells Claude Code *not* to stop -- so a crash in
    OmniMem would trap the user in a loop. Swallowing errors here is the correct
    trade: a memory tool may lose a fact, but it must never wedge a session.
    """
    event = args.event
    payload = _hook_payload()
    try:
        store.ensure_home()

        if event == "session-start":
            source = payload.get("source") or "startup"
            if source not in ("startup", "resume", "clear"):
                return 0
            project, root = _hook_project(payload)
            conn = db.init_schema()
            try:
                ingest.drain_inbox(conn)
            except Exception:
                pass
            brief = ingest.resume(conn, project, root=root)
            # stdout on SessionStart is added to the context Claude can see, so
            # printing the brief *is* the injection. Nothing else is required.
            _out(brief[:MAX_INJECT_CHARS])

        elif event == "user-prompt":
            _capture(payload, "user", str(payload.get("prompt") or ""))
            # Print nothing: this fires on every prompt, and anything written
            # here is injected into the context of every single turn.

        elif event == "stop":
            _capture(payload, "assistant", str(payload.get("last_assistant_message") or ""))

        elif event == "session-end":
            # SessionEnd hooks share a ~1.5 s budget, so this only flips the
            # dirty flag. The render happens at the next session start.
            project, _root = _hook_project(payload)
            conn = db.init_schema()
            ingest.mark_dirty(conn, project)
    except Exception:
        pass
    finally:
        try:
            db.close_all()
        except Exception:
            pass
    return 0


# --------------------------------------------------------------------------- misc

def cmd_serve(args: argparse.Namespace) -> int:
    store.ensure_home()
    store.ensure_token()
    from omnimem import server
    return server.main(
        ["--host", args.host] + (["--port", str(args.port)] if args.port else [])
    )


def cmd_mcp(_args: argparse.Namespace) -> int:
    store.ensure_home()
    from omnimem import server
    return server.main(["--mcp"])


def cmd_digest(args: argparse.Namespace) -> int:
    conn = db.init_schema()
    ingest.drain_inbox(conn)
    if args.project or not args.all:
        project = store.resolve_project(args.project, os.getcwd())
        result = ingest.digest(conn, project, root=store.project_root(os.getcwd()),
                              force=args.force)
        _out(json.dumps(result, indent=2))
    else:
        _out(json.dumps(ingest.digest_all(conn, force=args.force), indent=2))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    conn = db.init_schema()
    ingest.drain_inbox(conn)
    project = store.resolve_project(args.project, os.getcwd())
    _out(ingest.resume(conn, project, root=store.project_root(os.getcwd()),
                       with_footer=False))
    return 0


def cmd_projects(_args: argparse.Namespace) -> int:
    conn = db.init_schema()
    rows = ingest.overview(conn)
    if not rows:
        _out("No projects tracked yet.")
        return 0
    width = max(len(r["project"]) for r in rows)
    for row in rows:
        _out(f"  {row['project']:<{width}}  {row['facts']:>4} facts  "
             f"{row['sessions']:>3} sessions  last: {row['last_tool'] or '-'}")
    return 0


def cmd_token(_args: argparse.Namespace) -> int:
    _out(store.ensure_token())
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    problems = 0
    _out(f"home           {store.home()}")
    try:
        store.check_not_synced()
        _out("sync check     ok (not inside a cloud-synced folder)")
    except store.StoreError as exc:
        _out(f"sync check     FAIL — {exc}")
        problems += 1

    # Doctor is the command people run *because* something is broken, so an
    # unwritable or corrupt store has to be reported as a finding rather than
    # raised as a traceback -- and reported before the tool-wiring checks, which
    # are still useful information when the store itself is the problem.
    try:
        store.ensure_home()
        conn = db.init_schema()
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        _out(f"database       {store.db_path()}  (schema {ver}, journal {mode})")
        if mode.lower() != "wal":
            problems += 1
    except (OSError, sqlite3.Error) as exc:
        _out(f"database       FAIL — {store.db_path()}: {exc}")
        problems += 1

    pending = len([f for f in os.listdir(store.inbox_dir())
                   if f.endswith(".json")]) if os.path.isdir(store.inbox_dir()) else 0
    _out(f"inbox          {pending} pending event(s)")
    _out(f"token          {'present' if store.read_token() else 'MISSING'}")

    for name, target in sorted(_tool_targets().items()):
        path = target["config"]
        if not os.path.exists(path):
            continue
        if target.get("format") == "toml":
            try:
                with open(path, encoding="utf-8") as fh:
                    wired = "[mcp_servers.omnimem]" in fh.read()
            except OSError:
                wired = False
        else:
            wired = "omnimem" in (_read_json(path).get(target["key"]) or {})
        _out(f"{name:<15}{'connected' if wired else 'present, not connected'}")

    _out("")
    _out("ok" if not problems else f"{problems} problem(s) found")
    return 1 if problems else 0


def cmd_maintain(_args: argparse.Namespace) -> int:
    _out(json.dumps(db.maintenance(), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnimem",
        description="Local-first cross-tool project memory for LLMs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the local bridge (REST + MCP over HTTP)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="run as a stdio MCP server (used by editors)")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("install", help="wire OmniMem into a tool's config")
    p.add_argument("tools", nargs="*", help="tool names; see `omnimem tools`")
    p.add_argument("--all", action="store_true", help="every detected tool")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("tools", help="list supported tools and detection status")
    p.set_defaults(func=cmd_list_tools)

    p = sub.add_parser("hook", help="internal: Claude Code hook entry point")
    p.add_argument("event", choices=HOOK_EVENTS)
    p.set_defaults(func=cmd_hook)

    p = sub.add_parser("digest", help="re-render a project's markdown files")
    p.add_argument("project", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("resume", help="print a project's brief")
    p.add_argument("project", nargs="?")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("projects", help="list tracked projects")
    p.set_defaults(func=cmd_projects)

    p = sub.add_parser("token", help="print the local API token")
    p.set_defaults(func=cmd_token)

    p = sub.add_parser("doctor", help="check the install for problems")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("maintain", help="checkpoint the WAL and optimise the FTS index")
    p.set_defaults(func=cmd_maintain)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except store.StoreError as exc:
        _err(f"omnimem: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
