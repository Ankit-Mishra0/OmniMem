"""
Import purity of the hook path.

Claude Code runs `python -m omnimem.cli hook user-prompt` on *every prompt
submission*, synchronously, before the model sees the turn. That process pays
the full cost of whatever the import graph drags in: FastAPI + pydantic +
fastmcp is 300 ms+, `omnimem.core` alone is a few tens of milliseconds. The
difference is felt as editor latency on every keystroke-to-response cycle, and
it is invisible in every other test -- nothing fails, it just gets slower.

So the constraint is asserted directly: `omnimem/core/` imports nothing outside
the standard library, `omnimem.cli` imports the server lazily, and importing any
of it touches no disk.
"""

import ast
import os
import subprocess
import sys

import pytest

import omnimem

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO, "omnimem", "core")

CORE_MODULES = sorted(
    n[:-3] for n in os.listdir(CORE) if n.endswith(".py") and n != "__init__.py"
)

# The graph the hook must not touch. `starlette` and `uvicorn` are listed
# separately from `fastapi` because a stray `starlette.responses` import would
# pull the same weight without naming FastAPI.
HEAVY = ("fastapi", "starlette", "uvicorn", "pydantic", "fastmcp", "mcp", "httpx", "click")


def _toplevel_imports(path):
    """Every module name imported by a file, flattened to its top-level package."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, necessarily inside the package
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_the_core_modules_are_the_ones_we_think_they_are():
    """A new file in core/ must be caught by the purity check below, not skipped
    silently because this test hard-coded a list that went stale."""
    assert set(CORE_MODULES) == {
        "db", "extract", "facts", "ingest", "locks", "notes", "redact", "render", "store",
    }


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_imports_nothing_outside_the_standard_library(module):
    """
    Checked statically as well as dynamically: a lazy import inside a function
    body would pass the sys.modules check below (because nothing calls it during
    the test) while still costing 300 ms the first time a hook hits that branch.
    """
    for name in _toplevel_imports(os.path.join(CORE, module + ".py")):
        assert name in sys.stdlib_module_names or name == "omnimem", (
            f"omnimem/core/{module}.py imports {name!r}, which is not stdlib"
        )


def test_importing_core_does_not_drag_in_the_server():
    """
    The dynamic half. A fresh interpreter, because this one has already imported
    FastAPI via the HTTP contract tests -- checking sys.modules in-process would
    pass no matter what core did.
    """
    code = (
        "import sys;"
        "import omnimem.core.ingest, omnimem.core.render, omnimem.core.facts;"
        "print(','.join(sorted(m for m in sys.modules if m in %r)))" % (HEAVY,)
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO,
        env=dict(os.environ, PYTHONPATH=REPO), check=True,
    )
    assert out.stdout.strip() == "", f"core pulled in {out.stdout.strip()}"


def test_the_cli_hook_entry_point_stays_light():
    """
    `omnimem.cli` is what the hook actually executes, so its module scope is on
    the latency path too -- the server has to be imported inside `cmd_serve`, not
    at the top of the file.
    """
    for name in _toplevel_imports(os.path.join(REPO, "omnimem", "cli.py")):
        assert name in sys.stdlib_module_names or name == "omnimem", (
            f"omnimem/cli.py imports {name!r} at module scope"
        )

    code = (
        "import sys, omnimem.cli;"
        "print(','.join(sorted(m for m in sys.modules if m in %r)))" % (HEAVY,)
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO,
        env=dict(os.environ, PYTHONPATH=REPO), check=True,
    )
    assert out.stdout.strip() == "", f"the CLI pulled in {out.stdout.strip()}"


def test_importing_omnimem_writes_nothing_to_disk():
    """
    Schema init is deferred out of import time on purpose. If importing the
    package created ~/.omnimem, then `pytest --collect-only`, a stray `python -c
    "import omnimem"`, and any tool that merely *inspects* the module would each
    materialise a store -- and a test that forgot to relocate OMNIMEM_HOME would
    quietly write to the developer's real one.
    """
    code = (
        "import os, omnimem.cli, omnimem.core.ingest;"
        "print(os.path.exists(os.environ['OMNIMEM_HOME']))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO,
        env=dict(os.environ, PYTHONPATH=REPO,
                 OMNIMEM_HOME=os.path.join(REPO, ".pytest-should-not-exist")),
        check=True,
    )
    assert out.stdout.strip() == "False"
    assert not os.path.exists(os.path.join(REPO, ".pytest-should-not-exist"))


def test_the_version_is_declared_in_one_place():
    """Two version constants drift, and the one the extension reads off /health is
    the one users quote in bug reports."""
    from omnimem import server

    assert server.VERSION == omnimem.__version__
