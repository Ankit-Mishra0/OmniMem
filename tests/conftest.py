"""
Test isolation.

`OMNIMEM_HOME` is set here, before anything imports `omnimem`, because
`omnimem.core.store` reads it at call time and several modules resolve paths
during import. Without this the suite would read and write the developer's real
`~/.omnimem` store -- which is both destructive and makes tests order-dependent
on whatever happens to be in it.
"""

import os
import tempfile

_TMP_HOME = os.path.join(tempfile.gettempdir(), "omnimem-tests")
os.environ["OMNIMEM_HOME"] = _TMP_HOME
os.makedirs(_TMP_HOME, exist_ok=True)

import pytest  # noqa: E402

from omnimem.core import db as _db  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_connections():
    """
    Drop cached connections between tests.

    Connections are cached per thread and keyed by path; a test that deletes its
    database file would otherwise leave the next test holding a handle to an
    unlinked inode.
    """
    yield
    _db.close_all()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """
    A private store per test: own index.db, own projects/, own config.

    The suite-wide `_TMP_HOME` above is shared, so tests that assert on rendered
    file *contents* or on fact counts would otherwise see each other's writes.
    `store.home()` reads the environment on every call, so overriding the
    variable is enough -- but the cached connection is keyed by the old path, so
    it has to be dropped on the way in as well as out.
    """
    target = tmp_path / "store"
    target.mkdir()
    _db.close_all()
    monkeypatch.setenv("OMNIMEM_HOME", str(target))
    yield str(target)
    _db.close_all()


@pytest.fixture
def conn(home):
    """Schema-initialised connection against the private store."""
    return _db.init_schema()
