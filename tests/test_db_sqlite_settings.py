"""SQLite connection settings of database_models (WAL, synchronous, busy timeout).

Every test uses its own temporary database file and engine (never the real
university_bot.db, never the conftest engine).
"""

import logging
import sqlite3

import pytest
from sqlalchemy import text

import database_models as dm

SETTING_VARS = ("SQLITE_JOURNAL_MODE", "SQLITE_SYNCHRONOUS", "SQLITE_BUSY_TIMEOUT_MS")


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch, tmp_path):
    """No SQLITE_* variables and no .env fallback (tests set what they need)."""
    for name in SETTING_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dm, "BASE_DIR", str(tmp_path / "no_project"))


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def _make(name="settings.db"):
        eng = dm._create_engine_for(f"sqlite:///{tmp_path / name}")
        engines.append(eng)
        return eng

    yield _make
    for eng in engines:
        eng.dispose()


def pragmas(engine):
    with engine.connect() as conn:
        return {
            "journal_mode": conn.execute(text("PRAGMA journal_mode")).scalar(),
            "synchronous": conn.execute(text("PRAGMA synchronous")).scalar(),
            "busy_timeout": conn.execute(text("PRAGMA busy_timeout")).scalar(),
        }


def test_defaults_are_wal_normal_5s(make_engine):
    assert dm.sqlite_connection_settings() == {
        "journal_mode": "wal", "synchronous": "normal", "busy_timeout_ms": 5000,
    }
    assert pragmas(make_engine()) == {"journal_mode": "wal", "synchronous": 1, "busy_timeout": 5000}


def test_wal_is_persistent_for_other_connections(make_engine, tmp_path):
    make_engine().connect().close()
    plain = sqlite3.connect(tmp_path / "settings.db")  # e.g. a script using sqlite3 directly
    try:
        assert plain.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        plain.close()


def test_env_overrides(monkeypatch, make_engine):
    monkeypatch.setenv("SQLITE_JOURNAL_MODE", "DELETE")
    monkeypatch.setenv("SQLITE_SYNCHRONOUS", "full")
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "1234")
    assert pragmas(make_engine()) == {"journal_mode": "delete", "synchronous": 2, "busy_timeout": 1234}


def test_dotenv_fallback(monkeypatch, tmp_path, make_engine):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("SQLITE_JOURNAL_MODE=truncate\nSQLITE_BUSY_TIMEOUT_MS=250\n")
    monkeypatch.setattr(dm, "BASE_DIR", str(project))
    settings = dm.sqlite_connection_settings()
    assert settings["journal_mode"] == "truncate" and settings["busy_timeout_ms"] == 250
    assert settings["synchronous"] is None  # SQLite's default outside WAL


def test_keep_leaves_the_journal_mode_alone(monkeypatch, tmp_path, make_engine):
    plain = sqlite3.connect(tmp_path / "settings.db")
    plain.execute("CREATE TABLE t (x)")  # an existing rollback-journal database
    plain.close()
    monkeypatch.setenv("SQLITE_JOURNAL_MODE", "keep")
    result = pragmas(make_engine())
    assert result["journal_mode"] == "delete"  # not switched to WAL
    assert result["synchronous"] == 2  # untouched (SQLite default FULL)


@pytest.mark.parametrize("name,value", [
    ("SQLITE_JOURNAL_MODE", "turbo"),
    ("SQLITE_JOURNAL_MODE", "wal; DROP TABLE raw_data"),
    ("SQLITE_SYNCHRONOUS", "sometimes"),
    ("SQLITE_BUSY_TIMEOUT_MS", "-5"),
    ("SQLITE_BUSY_TIMEOUT_MS", "soon"),
])
def test_invalid_values_fall_back_to_defaults(monkeypatch, caplog, name, value):
    monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING, logger="database_models"):
        settings = dm.sqlite_connection_settings()
    assert settings == {"journal_mode": "wal", "synchronous": "normal", "busy_timeout_ms": 5000}
    assert name in caplog.text


def test_memory_and_non_sqlite_urls_get_no_pragmas():
    for url in ("sqlite://", "sqlite:///:memory:"):
        eng = dm._create_engine_for(url)
        try:
            with eng.connect() as conn:
                assert conn.execute(text("PRAGMA journal_mode")).scalar() == "memory"
        finally:
            eng.dispose()
    assert not dm._is_file_sqlite(dm.make_url("sqlite:///file:x?mode=memory&uri=true"))
    assert dm._is_file_sqlite(dm.make_url("sqlite:////tmp/x.db"))


def test_wal_readers_do_not_block_the_writer(make_engine, tmp_path):
    """The point of WAL: an open read transaction (export, statistics) does not
    make the bot's commit fail with "database is locked" (busy timeout 0)."""
    eng = make_engine()
    dm.Base.metadata.create_all(bind=eng)
    path = tmp_path / "settings.db"
    reader = sqlite3.connect(path, isolation_level=None)
    writer = sqlite3.connect(path, timeout=0)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM raw_data").fetchone()[0] == 0
        writer.execute("INSERT INTO raw_data (telegram_id, question) VALUES (1, 'q')")
        writer.commit()  # rollback-journal mode: sqlite3.OperationalError: database is locked
        # the reader keeps its consistent snapshot until it ends its transaction
        assert reader.execute("SELECT count(*) FROM raw_data").fetchone()[0] == 0
        reader.execute("COMMIT")
        assert reader.execute("SELECT count(*) FROM raw_data").fetchone()[0] == 1
    finally:
        reader.close()
        writer.close()


def test_rollback_journal_mode_would_block(monkeypatch, make_engine, tmp_path):
    """Control for the test above: the same sequence fails without WAL."""
    monkeypatch.setenv("SQLITE_JOURNAL_MODE", "delete")
    eng = make_engine()
    dm.Base.metadata.create_all(bind=eng)
    path = tmp_path / "settings.db"
    reader = sqlite3.connect(path, isolation_level=None)
    writer = sqlite3.connect(path, timeout=0)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM raw_data").fetchone()
        writer.execute("INSERT INTO raw_data (telegram_id, question) VALUES (1, 'q')")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            writer.commit()
    finally:
        reader.close()
        writer.close()


def test_locked_database_does_not_break_connect(monkeypatch, tmp_path, caplog):
    """Switching to WAL needs a moment of exclusive access; if another process
    holds a lock the connection still opens (and switches next time)."""
    path = tmp_path / "locked.db"
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("CREATE TABLE t (x)")
    blocker.execute("BEGIN EXCLUSIVE")
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "50")
    eng = dm._create_engine_for(f"sqlite:///{path}")
    try:
        with caplog.at_level(logging.WARNING, logger="database_models"):
            conn = eng.connect()  # must not raise
        conn.close()
        assert "journal_mode" in caplog.text
        blocker.execute("COMMIT")
        eng.dispose()
        with eng.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    finally:
        blocker.close()
        eng.dispose()


def test_session_round_trip_on_wal_engine(make_engine):
    eng = make_engine()
    dm.Base.metadata.create_all(bind=eng)
    Session = dm.sessionmaker(bind=eng)
    with Session() as db:
        row = dm.process_new_question(db, {"telegram_id": 1, "question": "Staj ne zaman başlıyor?"})
        assert db.get(dm.RawData, row.id).question == "Staj ne zaman başlıyor?"
