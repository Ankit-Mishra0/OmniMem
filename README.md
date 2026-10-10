# OmniMem

**Local-first cross-tool project memory for LLM coding agents.**

You explain your project to Claude Code. Then you switch to Codex and explain it
again. Then you open ChatGPT to ask one question and explain it a third time.

OmniMem watches what you're working on, keeps one folder per project with a
living summary, and hands that summary to whichever tool you open next. The new
tool starts with *"continue from where I left off"* instead of a wall of context.

Everything runs on your machine. No account, no API key, no cloud, no GPU, no
ongoing cost.

```
~/.omnimem/projects/payments-api/
├── PROJECT.md      the ≤2400-char brief that gets injected into every tool
├── DECISIONS.md    append-only log of what was decided and why
├── TASKS.md        checkboxes you can tick by hand
├── HANDOFF.md      "state as of the last session"
└── sessions/       one file per session, with its extracted facts
```

---

## Status: working prototype

Daily-usable, not shippable to strangers. 116 tests pass. The parts that are
deliberately left for later are listed under [Not built yet](#not-built-yet).

---

## Install

Python 3.10+ and a checkout. No packaging step yet.

```bash
git clone <your-fork> OmniMem && cd OmniMem
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Wire it into everything it can detect on this machine:

```bash
PYTHONPATH=. .venv/bin/python -m omnimem.cli install --all
```

Then check what it found:

```bash
PYTHONPATH=. .venv/bin/python -m omnimem.cli doctor
```

Restart your editor. That's the whole setup — re-running `install` is a no-op, so
it's safe to run again any time.

---

## Supported tools

Capture is layered by how much a host will let you push into it. MCP tools alone
aren't enough: they're **pull**-based, so the model has to *choose* to call them.

| Tool | How it captures | Automatic? |
| --- | --- | --- |
| **Claude Code** | Lifecycle hooks — the brief is injected at session start, your prompts and the replies are captured as they happen | **Fully.** No tool call, no cooperation from the model |
| **Antigravity** | Always-on rules file (`~/.gemini/GEMINI.md`) + MCP | In practice, yes |
| **Gemini CLI** | Same rules file + MCP | In practice, yes |
| **Codex CLI** | `~/.codex/AGENTS.md` + MCP over TOML config | In practice, yes |
| **Cursor** | Always-apply `.mdc` rule + MCP | In practice, yes |
| **Windsurf** | Rules file + MCP | In practice, yes |
| **Claude Desktop** | MCP — call `omnimem_resume`, or use the `/resume` prompt | On demand |
| **VS Code (Copilot)** | MCP (`.vscode/mcp.json`) | On demand |
| **Zed** | MCP (`context_servers`) | On demand |
| **ChatGPT · Claude.ai · Gemini · Perplexity · DeepSeek · Grok · Copilot · Mistral · Poe** | Browser extension — scrapes the conversation, injects the brief into the prompt box | **Fully** |

`omnimem tools` prints this list with live detection status.

### Per-tool config, if you'd rather do it by hand

<details>
<summary>Claude Code, Claude Desktop, Cursor, Windsurf, Antigravity, Gemini CLI</summary>

`mcpServers` in `~/.claude.json`, `~/.cursor/mcp.json`,
`~/.codeium/windsurf/mcp_config.json`, `~/.gemini/settings.json`, or
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "omnimem": {
      "command": "/absolute/path/to/OmniMem/.venv/bin/python",
      "args": ["-m", "omnimem.cli", "mcp"],
      "env": { "PYTHONPATH": "/absolute/path/to/OmniMem" }
    }
  }
}
```
</details>

<details>
<summary>VS Code and Zed use a different key</summary>

This is the usual reason a server "doesn't show up": VS Code reads `servers`
(`.vscode/mcp.json`), Zed reads `context_servers`
(`~/.config/zed/settings.json`). Same object, different key.
</details>

<details>
<summary>Codex CLI is TOML</summary>

`~/.codex/config.toml`:

```toml
[mcp_servers.omnimem]
command = "/absolute/path/to/OmniMem/.venv/bin/python"
args = ["-m", "omnimem.cli", "mcp"]
startup_timeout_sec = 30

[mcp_servers.omnimem.env]
PYTHONPATH = "/absolute/path/to/OmniMem"
```
</details>

### Browser extension

```bash
PYTHONPATH=. .venv/bin/python -m omnimem.cli serve   # leave running
```

Then `chrome://extensions` → **Developer mode** → **Load unpacked** →
pick `extension/`. Works in Chrome, Edge, Brave, Arc, and any Chromium browser.

The extension needs the bridge running; the editors don't. Capture is on by
default, with a global pause and a per-site toggle in the popup.

---

## Commands

| Command | What it does |
| --- | --- |
| `omnimem serve` | Local bridge: REST for the extension + MCP over HTTP, on 127.0.0.1 |
| `omnimem mcp` | stdio MCP server — this is what editors spawn |
| `omnimem install <tool>` / `--all` | Merge config into a host. Idempotent |
| `omnimem tools` | Supported tools and whether each was detected |
| `omnimem resume [project]` | Print the brief |
| `omnimem digest [project]` | Re-render the markdown from the database |
| `omnimem projects` | Tracked projects |
| `omnimem doctor` | Diagnose a broken install |
| `omnimem maintain` | Checkpoint the WAL, optimise the FTS index |
| `omnimem token` | The local API token |

Prefix with `PYTHONPATH=. .venv/bin/python -m omnimem.cli` until it's packaged.

## MCP tools agents can call

`omnimem_resume` · `omnimem_save_session` · `omnimem_search` ·
`omnimem_add_decision` · `omnimem_set_task` · `omnimem_handoff` ·
`omnimem_list_projects`, plus a `resume` prompt that shows up as
`/mcp__omnimem__resume`.

`omnimem_resume` returns the brief *and* asks the agent to call
`omnimem_save_session` with structured facts when it's done. That's how the
summaries get to be LLM-quality at zero API cost: the agent in the loop is
already an LLM, so it writes them.

---

## How it works

**The database is the source of truth; the markdown files are renders of it.**
`PROJECT.md` is a deterministic function of what's in SQLite, so rendering twice
produces byte-identical output and re-ingesting a session is a true no-op. That
one inversion is what stops the memory files from rotting.

A **fact** is one atomic claim under 180 characters, typed
(`goal`, `decision`, `constraint`, `convention`, `stack`, `entity`, `gotcha`,
`task`, `open_question`), and keyed by a content hash of its normalised token
set — so the same claim reworded is recognised as the same claim. Evidence is
append-only: counters are derived by counting rows, never incremented, which is
why replaying a session can't inflate a score.

Facts are ranked at **read** time with per-kind half-lives (a task is stale in
three days, a goal in eight months) and an authority multiplier — a fact you
pinned by hand outranks a regex's guess by 4×. The top ~2400 characters become
the brief, ordered so that the most useful sections survive truncation.

Two sources need no NLP and are simply true: **Stack** comes from reading
`package.json` / `requirements.txt` / `pyproject.toml` / `go.mod` / `Dockerfile`,
and file paths are filtered through `os.path.exists()` — one syscall kills every
hallucinated path.

### Footprint

Regex and SQLite on the default path. No model weights, no vector database, no
GPU. ~60–90 MB RSS for the bridge, ~0% CPU idle, FTS query under 5 ms, render
10–20 ms at 2000 facts. The Claude Code hook path is ~5 ms because it writes one
file into an inbox and exits — `omnimem/core/` is stdlib-only, enforced by a
test, so the hook never pays to import FastAPI.

### Privacy

Nothing leaves your machine. The bridge binds 127.0.0.1 only, requires a token
from `~/.omnimem/token` on every API route, and restricts CORS to the specific
chat origins. Emails, API keys, tokens, JWTs and URL credentials are scrubbed on
the way *in* — before a secret can reach the database and get injected into three
other tools.

---

## Development

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

`tests/test_concurrency.py` spawns real processes, because the failure modes it
covers (`BEGIN IMMEDIATE` snapshot upgrades, `flock` ownership, the inbox's
`O_CREAT|O_EXCL` handoff) all behave differently inside one interpreter.
`tests/test_core_purity.py` is what keeps the hook path fast.

`OMNIMEM_HOME` relocates the whole store — set it to a scratch directory to
experiment without touching your real memory.

## Not built yet

Designed, deliberately deferred:

1. **SimHash banding** for dedup — plain Jaccard is fine below ~2000 facts per project.
2. **Hysteresis** at the brief's cut line, so a fact near the boundary can't flip in and out.
3. **Supersession rules** — until they land, two contradictory decisions can both
   appear in the brief. The `status` / `superseded_by` columns already exist, so
   adding the rules needs no migration.
4. **Archival** of old sessions.

Also missing: `pipx` packaging, and a Firefox port.

## Licence

MIT — see [LICENSE](LICENSE).
