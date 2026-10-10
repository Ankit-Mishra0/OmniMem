"""
Advisory file locking and atomic writes for the rendered markdown.

Two choices worth stating, because both were bugs waiting to happen:

* `fcntl.flock`, not `lockf`/`fcntl.lock`. POSIX record locks are owned by the
  *process* and are dropped by *any* close() of *any* descriptor on the file,
  which breaks the moment two threads in uvicorn's threadpool touch the same
  project. flock is per-descriptor and is released by the kernel on process
  death including SIGKILL -- exactly what fire-and-exit hooks need.

* The temp file for the atomic swap is created in the *destination directory*.
  A temp file in $TMPDIR may be on a different volume, which silently degrades
  os.replace() into copy+unlink and loses atomicity.
"""

import errno
import hashlib
import os
import random
import time
from contextlib import contextmanager
from typing import Iterator, Optional


class LockTimeout(RuntimeError):
    pass


@contextmanager
def file_lock(path: str, timeout: float = 10.0) -> Iterator[int]:
    """Exclusive advisory lock on `path`, polled so we can bound the wait."""
    import fcntl

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    delay = 0.005
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise LockTimeout(f"could not lock {path} within {timeout}s")
                time.sleep(delay + random.random() * delay)
                delay = min(delay * 2, 0.25)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def atomic_write(path: str, data: str, mode: int = 0o644) -> None:
    """Write `data` to `path` atomically. Never leaves a partial file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def body_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def write_if_changed(path: str, text: str, known_hash: Optional[str] = None) -> Optional[str]:
    """
    The hash gate. Returns the new hash if written, None if the content was
    already identical.

    Not touching mtime on a no-op matters more than it looks: editors, file
    watchers and `git status` all key off mtime, so a digest that rewrote
    byte-identical content every session would light up as constant churn.
    """
    new_hash = body_hash(text)
    if known_hash == new_hash and os.path.exists(path):
        return None
    if known_hash is None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if body_hash(fh.read()) == new_hash:
                    return None
        except OSError:
            pass
    atomic_write(path, text)
    return new_hash


def append_line(path: str, text: str) -> None:
    """Append to a log file, creating parents. Used for the session log."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
