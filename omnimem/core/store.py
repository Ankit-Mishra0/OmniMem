"""
Filesystem layout, configuration, and project-slug resolution.

    ~/.omnimem/
      config.json
      token                     mode 0600, bearer for mutating REST endpoints
      index.db                  facts + sightings + notes/FTS5
      inbox/<uuid>.json         hook drop-box
      projects/<slug>/...       rendered brief + append-only logs
"""

import json
import os
import re
import secrets
import stat
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8000,
    "capture": True,          # master switch for browser auto-capture
    "redact": True,           # scrub secrets before anything is stored
    "brief_chars": 2400,      # PROJECT.md budget (~600 tokens)
    "paused_sites": [],       # per-site opt-out, e.g. ["gemini"]
    "ollama": False,          # opt-in local summariser, off by default
}

# Live SQLite in a synced folder is a corruption generator: the sync client
# copies -wal and -shm out from under the writer, and flock is unreliable on
# network filesystems. Refuse rather than silently lose data.
_SYNC_MARKERS = (
    "dropbox",
    "google drive",
    "googledrive",
    "onedrive",
    "library/mobile documents",  # iCloud Drive
    "icloud",
    "pcloud",
    "sync.com",
    "mega",
)


class StoreError(RuntimeError):
    pass


# --------------------------------------------------------------------------- paths

def home() -> str:
    """Root of the store. Override with OMNIMEM_HOME (used by every test)."""
    override = os.environ.get("OMNIMEM_HOME")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), ".omnimem")


def db_path() -> str:
    return os.environ.get("OMNIMEM_DB_PATH") or os.path.join(home(), "index.db")


def config_path() -> str:
    return os.path.join(home(), "config.json")


def token_path() -> str:
    return os.path.join(home(), "token")


def inbox_dir() -> str:
    return os.path.join(home(), "inbox")


def projects_dir() -> str:
    return os.path.join(home(), "projects")


def project_dir(slug: str) -> str:
    return os.path.join(projects_dir(), slug)


def sessions_dir(slug: str) -> str:
    return os.path.join(project_dir(slug), "sessions")


def lock_path(slug: str) -> str:
    return os.path.join(project_dir(slug), ".lock")


# --------------------------------------------------------------------------- guards

def check_not_synced(path: Optional[str] = None) -> None:
    """Raise if the store lives inside a file-sync folder."""
    target = (path or home()).lower()
    for marker in _SYNC_MARKERS:
        if marker in target:
            raise StoreError(
                "OmniMem store resolves inside a file-sync folder (matched "
                f"{marker!r}: {path or home()}).\nSQLite WAL corrupts when synced. "
                "Move it, or set OMNIMEM_HOME to a local path."
            )


# --------------------------------------------------------------------------- init

def ensure_home() -> str:
    """Create the store if missing and return its root. Idempotent."""
    root = home()
    check_not_synced(root)
    os.makedirs(root, exist_ok=True)
    os.makedirs(inbox_dir(), exist_ok=True)
    os.makedirs(projects_dir(), exist_ok=True)
    if not os.path.exists(config_path()):
        save_config(DEFAULT_CONFIG)
    ensure_token()
    return root


def ensure_project(slug: str) -> str:
    d = project_dir(slug)
    os.makedirs(sessions_dir(slug), exist_ok=True)
    return d


def ensure_token() -> str:
    """Create the bearer token once, 0600. Returns it."""
    path = token_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read().strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    return token


def read_token() -> Optional[str]:
    try:
        with open(token_path(), "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


# --------------------------------------------------------------------------- config

def load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(config_path(), "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    from omnimem.core import locks

    os.makedirs(home(), exist_ok=True)
    locks.atomic_write(config_path(), json.dumps(cfg, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- slugs

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(name: str, fallback: str = "default") -> str:
    slug = _SLUG_STRIP.sub("-", (name or "").strip().lower()).strip("-")
    return slug[:48].strip("-") or fallback


def _git_repo_name(start: str) -> Optional[str]:
    """
    Read the origin URL straight out of .git/config, walking upward.

    Parsed rather than shelled out to: `git remote get-url` costs ~25ms of
    process spawn, and this runs on the UserPromptSubmit hook path.
    """
    cur = os.path.abspath(start)
    while True:
        cfg = os.path.join(cur, ".git", "config")
        if os.path.isfile(cfg):
            try:
                with open(cfg, "r", encoding="utf-8", errors="replace") as fh:
                    in_origin = False
                    for line in fh:
                        line = line.strip()
                        if line.startswith("["):
                            in_origin = line.replace(" ", "").lower() == '[remote"origin"]'
                        elif in_origin and line.lower().startswith("url"):
                            url = line.split("=", 1)[-1].strip()
                            base = url.rstrip("/").rsplit("/", 1)[-1]
                            if base.endswith(".git"):
                                base = base[:-4]
                            return base or None
            except OSError:
                return None
            # .git found but no origin — fall back to the working-tree name
            return os.path.basename(cur) or None
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _git_root(start: str) -> Optional[str]:
    cur = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def resolve_project(explicit: Optional[str] = None, cwd: Optional[str] = None) -> str:
    """
    Slug resolution, most-authoritative first:
      1. explicit argument (an agent naming the project, or --project)
      2. OMNIMEM_PROJECT environment variable
      3. git origin repo name  -- stable across clones and worktrees
      4. containing folder name
      5. "default"
    """
    if explicit and explicit.strip():
        return slugify(explicit)
    env = os.environ.get("OMNIMEM_PROJECT")
    if env and env.strip():
        return slugify(env)
    base = cwd or os.getcwd()
    repo = _git_repo_name(base)
    if repo:
        return slugify(repo)
    return slugify(os.path.basename(os.path.abspath(base)))


def project_root(cwd: Optional[str] = None) -> str:
    """Best guess at the code root for a project, used for manifest scanning."""
    base = os.path.abspath(cwd or os.getcwd())
    return _git_root(base) or base
