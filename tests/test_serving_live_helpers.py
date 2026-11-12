"""Helpers of the opt-in live tests that must work without RabbitMQ/GPU (run in the CPU suite)."""

import hashlib
import importlib.util
import os
import sqlite3
import stat
from pathlib import Path

import pytest

_LIVE = Path(__file__).resolve().parent / "live" / "test_integration_live.py"


@pytest.fixture(scope="module")
def live_module():
    spec = importlib.util.spec_from_file_location("cengbot_live_helpers", _LIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines tests/fixtures only; nothing runs
    return module


def _make_wal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE raw_data (id INTEGER PRIMARY KEY, question TEXT)")
    conn.execute("INSERT INTO raw_data (question) VALUES ('Staj zorunlu mu?')")
    conn.commit()
    conn.close()  # the last connection checkpoints and removes -wal/-shm


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _side_files(path: Path):
    return [p.name for p in path.parent.iterdir() if p.name.startswith(path.name + "-")]


def test_backup_of_a_wal_database_leaves_no_side_files(live_module, tmp_path):
    """A mode=ro connection to a WAL database left university_bot.db-wal/-shm behind."""
    db = tmp_path / "real.db"
    _make_wal_db(db)
    before = _digest(db)
    assert _side_files(db) == []
    source = live_module.open_db_read_only(db)
    dest = sqlite3.connect(tmp_path / "copy.db")
    try:
        source.backup(dest)
        with pytest.raises(sqlite3.OperationalError):
            source.execute("DELETE FROM raw_data")  # query_only
    finally:
        dest.close()
        source.close()
    assert _side_files(db) == [] and _digest(db) == before
    copy = sqlite3.connect(tmp_path / "copy.db")
    assert copy.execute("SELECT question FROM raw_data").fetchall() == [("Staj zorunlu mu?",)]
    copy.close()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file permissions")
def test_unwritable_database_is_opened_immutable(live_module, tmp_path):
    folder = tmp_path / "ro dir with space"  # the URI path is quoted
    folder.mkdir()
    db = folder / "real.db"
    _make_wal_db(db)
    db.chmod(stat.S_IRUSR)
    try:
        conn = live_module.open_db_read_only(db)
        try:
            assert conn.execute("SELECT count(*) FROM raw_data").fetchone() == (1,)
        finally:
            conn.close()
        assert _side_files(db) == []
    finally:
        db.chmod(stat.S_IRUSR | stat.S_IWUSR)
