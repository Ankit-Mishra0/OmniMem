"""
The installer's contract with files the user also edits by hand.

Every target here is a config file with a life of its own: `~/.claude.json`
holds the user's other MCP servers, `~/.codex/config.toml` holds hand-written
comments, `~/.gemini/GEMINI.md` holds their own instructions. So the two things
that matter are that installing twice changes nothing the second time, and that
nothing already in the file is lost.

The per-host key differences (`servers` vs `mcpServers` vs `context_servers` vs a
TOML table) are asserted too, because getting one wrong fails *silently* -- the
host reads the file, finds no server under the name it looks for, and says
nothing.
"""

import argparse
import json
import os

import pytest

from omnimem import cli


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """
    Relocate every install target.

    `HOME` is a module constant read by `_tool_targets()` at call time, so
    patching the attribute is enough -- and `cwd` has to move too, because the
    VS Code target is project-relative.
    """
    monkeypatch.setattr(cli, "HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return str(tmp_path)


def _install(*tools):
    return cli.cmd_install(argparse.Namespace(tools=list(tools), all=False))


def _snapshot(root):
    """Path -> (bytes, mtime_ns) for everything under root."""
    out = {}
    for base, _dirs, names in os.walk(root):
        for name in names:
            path = os.path.join(base, name)
            with open(path, "rb") as fh:
                out[os.path.relpath(path, root)] = (fh.read(), os.stat(path).st_mtime_ns)
    return out


# ------------------------------------------------------------------ idempotency

@pytest.mark.parametrize("tool", [
    "claude-code", "antigravity", "gemini-cli", "codex", "claude-desktop",
    "cursor", "windsurf", "vscode", "zed",
])
def test_installing_twice_is_a_no_op(tool, fake_home, capsys):
    """
    Run it again and nothing moves -- not the bytes, not the mtimes.

    The failure this guards is duplication rather than corruption: an
    append-based rules writer stacks a second copy of the instructions, and a
    list-based hook writer adds a second identical hook that then runs twice on
    every prompt.

    Three runs rather than two on purpose. Both real bugs this caught -- a blank
    line accreting in the marker splice and in the TOML substitution -- were
    off-by-one-newline differences between the *first* write and every subsequent
    one, so a two-run test would have flagged them while a fix that only
    stabilised from run 2 onward would still have passed.
    """
    assert _install(tool) == 0
    first = _snapshot(fake_home)
    assert first, "install wrote nothing"

    for run in (2, 3):
        assert _install(tool) == 0
        again = _snapshot(fake_home)

        assert set(again) == set(first), f"install #{run} created or removed files"
        for path, (body, mtime) in first.items():
            assert again[path][0] == body, f"{path} changed on install #{run}"
            assert again[path][1] == mtime, f"{path} was rewritten identically on #{run}"
    capsys.readouterr()


def test_a_third_install_still_leaves_one_hook_per_event(fake_home, capsys):
    """
    Hooks live in a JSON *list*, which is the shape that silently accumulates.
    Two SessionStart entries means the brief is injected twice, doubling the
    context cost of every session start.
    """
    for _ in range(3):
        _install("claude-code")
    capsys.readouterr()

    data = json.load(open(os.path.join(fake_home, ".claude", "settings.json")))
    for event in ("SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"):
        ours = [
            e for e in data["hooks"][event]
            if "omnimem.cli hook" in json.dumps(e)
        ]
        assert len(ours) == 1, f"{event} accumulated {len(ours)} OmniMem hooks"


def test_the_rules_block_is_replaced_not_appended(fake_home, capsys):
    path = os.path.join(fake_home, ".gemini", "GEMINI.md")
    for _ in range(3):
        _install("gemini-cli")
    capsys.readouterr()

    body = open(path, encoding="utf-8").read()
    assert body.count(cli.BEGIN_MARK) == 1
    assert body.count(cli.END_MARK) == 1
    assert body.count("omnimem_resume") == cli.RULES_TEXT.count("omnimem_resume")


# --------------------------------------------------------- preserving user data

def test_other_mcp_servers_survive(fake_home, capsys):
    path = os.path.join(fake_home, ".claude.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({
            "mcpServers": {"playwright": {"command": "npx", "args": ["-y", "@pw/mcp"]}},
            "theme": "dark",
        }, fh)

    _install("claude-code")
    capsys.readouterr()

    data = json.load(open(path))
    assert data["mcpServers"]["playwright"]["command"] == "npx"
    assert data["theme"] == "dark", "an unrelated top-level key was dropped"
    assert "omnimem" in data["mcpServers"]


def test_a_users_own_rules_are_kept_above_the_block(fake_home, capsys):
    path = os.path.join(fake_home, ".gemini", "GEMINI.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# My rules\n\nAlways write tests first.\n")

    _install("gemini-cli")
    _install("gemini-cli")
    capsys.readouterr()

    body = open(path, encoding="utf-8").read()
    assert body.startswith("# My rules")
    assert body.count("Always write tests first.") == 1
    assert cli.BEGIN_MARK in body


def test_toml_comments_and_other_tables_survive(fake_home, capsys):
    """
    Why the TOML path splices text instead of parsing: stdlib `tomllib` is
    read-only, and a hand-rolled writer would reformat the whole file and drop
    the comments the user put there.
    """
    path = os.path.join(fake_home, ".codex", "config.toml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "# my codex config\n"
            'model = "gpt-5"\n\n'
            "[mcp_servers.other]\n"
            'command = "other-server"\n'
        )

    _install("codex")
    _install("codex")
    capsys.readouterr()

    body = open(path, encoding="utf-8").read()
    assert "# my codex config" in body
    assert '[mcp_servers.other]' in body
    assert body.count("[mcp_servers.omnimem]") == 1
    assert "startup_timeout_sec = 30" in body


def test_unparseable_json_is_backed_up_not_clobbered(fake_home, capsys):
    path = os.path.join(fake_home, ".cursor", "mcp.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("{ this is not json")

    _install("cursor")
    capsys.readouterr()

    assert json.load(open(path))["mcpServers"]["omnimem"]
    assert open(path + ".omnimem-backup").read() == "{ this is not json"


# ------------------------------------------------------------- per-host details

def test_each_host_gets_the_key_it_actually_reads(fake_home, capsys):
    """
    The single most common reason an MCP server "doesn't show up". VS Code reads
    `servers`, Zed reads `context_servers`, everyone else reads `mcpServers`.
    """
    for tool, key in [("claude-code", "mcpServers"), ("vscode", "servers"),
                      ("zed", "context_servers")]:
        _install(tool)
        path = cli._tool_targets()[tool]["config"]
        data = json.load(open(path))
        assert "omnimem" in data[key], f"{tool} config is missing {key}.omnimem"
    capsys.readouterr()


def test_the_cursor_rule_keeps_its_front_matter(fake_home, capsys):
    """
    Cursor only applies an .mdc rule automatically when the front matter says
    `alwaysApply: true`; without it the file is inert and the agent never
    resumes. The front matter sits *outside* the markers, so the second install
    -- which replaces the marker block -- must not take it with it.
    """
    _install("cursor")
    _install("cursor")
    capsys.readouterr()

    body = open(os.path.join(fake_home, ".cursor", "rules", "omnimem.mdc"),
               encoding="utf-8").read()
    assert body.startswith("---\n")
    assert "alwaysApply: true" in body
    assert body.count("alwaysApply") == 1
    # Front matter first, rules after: Cursor reads the block at the top of the
    # file and stops.
    assert body.index("---\n") < body.index(cli.BEGIN_MARK)


def test_the_launcher_survives_a_relocated_store(fake_home, capsys):
    """The registered command has to carry OMNIMEM_HOME explicitly: a GUI-launched
    host does not inherit the shell environment the install ran in."""
    _install("claude-code")
    capsys.readouterr()

    entry = json.load(open(os.path.join(fake_home, ".claude.json")))["mcpServers"]["omnimem"]
    assert os.path.isabs(entry["command"])
    assert "OMNIMEM_HOME" in entry["env"]
    assert os.path.isabs(entry["env"]["OMNIMEM_HOME"])


def test_an_unknown_tool_is_an_error_not_a_silent_skip(fake_home, capsys):
    assert cli.cmd_install(argparse.Namespace(tools=["emacs"], all=False)) == 2
    assert _snapshot(fake_home) == {}, "a failed install still wrote files"
    assert "Unknown tool" in capsys.readouterr().err


def test_install_all_only_touches_hosts_that_exist(fake_home, capsys):
    """`--all` is detection-driven: creating config for a tool that is not
    installed leaves confusing dead files in the user's home directory."""
    os.makedirs(os.path.join(fake_home, ".codex"))

    assert cli.cmd_install(argparse.Namespace(tools=[], all=True)) == 0
    capsys.readouterr()

    assert os.path.exists(os.path.join(fake_home, ".codex", "config.toml"))
    assert not os.path.exists(os.path.join(fake_home, ".cursor"))
    assert not os.path.exists(os.path.join(fake_home, ".config", "zed"))


def test_list_tools_reports_every_target(fake_home, capsys):
    """The landing page's supported-tools grid and this list must not drift."""
    assert cli.cmd_list_tools(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    for target in cli._tool_targets().values():
        assert target["label"] in out
        assert target["push"] in out
