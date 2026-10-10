"""
OmniMem core — the stdlib-only memory kernel.

Nothing in this package may import a third-party module. Hooks and CLI
subcommands import from here on every editor event, so import cost is a
latency budget: stdlib-only keeps cold start around 40ms, whereas pulling in
FastAPI/pydantic/fastmcp costs 300ms+ per keystroke-triggered hook.

`tests/test_core_purity.py` enforces this.
"""

__all__ = ["store", "locks", "db", "facts", "extract", "render", "ingest", "redact"]
