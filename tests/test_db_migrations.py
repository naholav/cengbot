"""Tests for src/db_migrations.py (index migration of existing databases).

Legacy databases are built in tmp_path with the exact index set of the
production database before the migration; the real university_bot.db is only
ever copied (read-only).
"""

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

import database_models as dm
import db_migrations as mig

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "university_bot.db"


def copy_real_db(target: Path) -> None:
    """Copy the real DB read-only; in WAL mode recent commits may still be in
    the -wal file, so copy that too (SQLite replays it when the copy is opened)."""
    shutil.copyfile(REAL_DB, target)
    wal = REAL_DB.with_name(REAL_DB.name + "-wal")
    if wal.exists():
        shutil.copyfile(wal, target.with_name(target.name + "-wal"))

# The indexes of the production database before this migration (sqlite_master).
LEGACY_INDEXES = (
    "CREATE INDEX idx_raw_data_admin_approved ON raw_data (admin_approved)",
    "CREATE INDEX idx_raw_data_composite ON raw_data (telegram_id, created_at)",
    "CREATE INDEX idx_raw_data_created_at ON raw_data (created_at)",
    "CREATE INDEX idx_raw_data_is_duplicate ON raw_data (is_duplicate)",
    "CREATE INDEX idx_raw_data_language ON raw_data (language)",
    'CREATE INDEX idx_raw_data_like ON raw_data ("like")',
    "CREATE INDEX idx_raw_data_telegram_id ON raw_data (telegram_id)",
    "CREATE INDEX ix_raw_data_id ON raw_data (id)",
    "CREATE INDEX ix_raw_data_telegram_id ON raw_data (telegram_id)",
    "CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name)",
    "CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp)",
    "CREATE INDEX ix_system_metrics_id ON system_metrics (id)",
    "CREATE INDEX ix_system_metrics_metric_name ON system_metrics (metric_name)",
    "CREATE INDEX idx_training_data_created_at ON training_data (created_at)",
    "CREATE INDEX idx_training_data_is_active ON training_data (is_active)",
    "CREATE INDEX idx_training_data_is_answer_duplicate ON training_data(is_answer_duplicate)",
    "CREATE INDEX idx_training_data_language ON training_data (language)",
    "CREATE INDEX idx_training_data_review_status ON training_data (review_status)",
    "CREATE INDEX idx_training_data_source_id ON training_data (source_id)",
    "CREATE INDEX ix_training_data_id ON training_data (id)",
    "CREATE INDEX idx_user_analytics_telegram_id ON user_analytics(telegram_id)",
    "CREATE INDEX ix_user_analytics_id ON user_analytics (id)",
    "CREATE UNIQUE INDEX ix_user_analytics_telegram_id ON user_analytics (telegram_id)",
    "CREATE INDEX idx_user_votes_composite ON user_votes (raw_data_id, telegram_user_id)",
    "CREATE INDEX idx_user_votes_raw_data_id ON user_votes(raw_data_id)",
    "CREATE INDEX idx_user_votes_telegram_user_id ON user_votes(telegram_user_id)",
    "CREATE UNIQUE INDEX idx_user_votes_unique ON user_votes(raw_data_id, telegram_user_id)",
    "CREATE INDEX ix_user_votes_id ON user_votes (id)",
    "CREATE INDEX ix_user_votes_telegram_user_id ON user_votes (telegram_user_id)",
)
NEW_INDEXES = {"idx_raw_data_duplicate_of_id", "idx_training_data_duplicate_answer_of_id"}
REDUNDANT = {name for name, _table, _cols in mig.REDUNDANT_INDEXES}


def _create_tables(path: Path) -> None:
    eng = create_engine(f"sqlite:///{path}")
    dm.Base.metadata.create_all(bind=eng)
    eng.dispose()


def _drop_all_indexes(conn: sqlite3.Connection) -> None:
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' "
                                "AND name NOT LIKE 'sqlite_autoindex_%'").fetchall():
        conn.execute(f'DROP INDEX "{name}"')


def build_legacy_db(path: Path, indexes=LEGACY_INDEXES, seed=True) -> Path:
    _create_tables(path)
    conn = sqlite3.connect(path)
    _drop_all_indexes(conn)
    for statement in indexes:
        conn.execute(statement)
    if seed:
        for i in range(1, 31):
            conn.execute("INSERT INTO raw_data (id, telegram_id, question, answer, language, admin_approved, "
                         "is_duplicate, duplicate_of_id, created_at) VALUES (?, ?, ?, ?, 'TR', ?, ?, ?, "
                         "datetime('2025-01-01', ?))",
                         (i, 100 + i % 4, f"Soru {i}", f"Cevap {i}", i % 2, i > 20, 1 if i > 20 else None,
                          f"+{i} minutes"))
            conn.execute("INSERT INTO user_votes (raw_data_id, telegram_user_id, current_vote) VALUES (?, ?, ?)",
                         (i, 7, 1 if i % 3 else -1))
        conn.execute("INSERT INTO training_data (source_id, question, answer) VALUES (1, 'Soru 1', 'Cevap 1')")
        conn.execute("INSERT INTO training_data (source_id, question, answer, duplicate_answer_of_id) "
                     "VALUES (2, 'Soru 2', 'Cevap 1', 1)")
    conn.commit()
    conn.close()
    return path


def index_signature(path: Path):
    """{index name: (table, unique, columns)} incl. constraint autoindexes."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = {}
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            for _seq, name, unique, _origin, _partial in conn.execute(f'PRAGMA index_list("{table}")'):
                columns = tuple(r[2] for r in conn.execute(f'PRAGMA index_info("{name}")'))
                result[name] = (table, bool(unique), columns)
        return result
    finally:
        conn.close()


def table_dump(path: Path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f'SELECT * FROM "{t}" ORDER BY rowid').fetchall()
                for t in ("raw_data", "training_data", "user_votes", "user_analytics", "system_metrics")}
    finally:
        conn.close()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fresh_signature(tmp_path):
    path = tmp_path / "fresh.db"
    _create_tables(path)
    return index_signature(path)


@pytest.fixture
def legacy_db(tmp_path):
    return build_legacy_db(tmp_path / "legacy.db")


def test_fresh_schema_has_no_redundant_index(fresh_signature):
    """Guard for the models: no index duplicates the rowid or is a prefix of another one."""
    by_table = {}
    for name, (table, unique, columns) in fresh_signature.items():
        by_table.setdefault(table, []).append((name, unique, columns))
    for table, indexes in by_table.items():
        for name, unique, columns in indexes:
            assert columns != ("id",), f"{name} duplicates the INTEGER PRIMARY KEY of {table}"
            if unique:
                continue
            for other, _u, other_columns in indexes:
                assert other == name or other_columns[:len(columns)] != columns, \
                    f"{name} {columns} is redundant with {other} {other_columns}"


def test_legacy_db_migrates_to_the_fresh_schema(legacy_db, fresh_signature):
    before_rows = table_dump(legacy_db)
    result = mig.migrate(str(legacy_db))

    assert set(result.created) == NEW_INDEXES
    assert {name for name, _ in result.dropped} == REDUNDANT
    assert result.kept == [] and result.warnings == []
    assert index_signature(legacy_db) == fresh_signature
    assert table_dump(legacy_db) == before_rows  # no row touched
    conn = sqlite3.connect(legacy_db)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    # every drop names something that still exists (or the rowid)
    remaining = set(index_signature(legacy_db))
    for _name, covered_by in result.dropped:
        assert covered_by in remaining or covered_by.startswith("INTEGER PRIMARY KEY")


def test_migration_is_idempotent(legacy_db, fresh_signature):
    mig.migrate(str(legacy_db))
    digest = sha256(legacy_db)
    again = mig.migrate(str(legacy_db))
    assert not again.changed and again.created == [] and again.dropped == []
    assert index_signature(legacy_db) == fresh_signature
    assert sha256(legacy_db) == digest


def test_fresh_database_needs_nothing(tmp_path):
    path = tmp_path / "fresh.db"
    _create_tables(path)
    assert not mig.migrate(str(path)).changed


def test_dry_run_changes_nothing_and_reports_the_plan(legacy_db, tmp_path):
    digest = sha256(legacy_db)
    planned = mig.migrate(str(legacy_db), dry_run=True)
    assert sha256(legacy_db) == digest
    assert not Path(str(legacy_db) + "-journal").exists()
    applied = mig.migrate(str(build_legacy_db(tmp_path / "other.db")))
    assert planned.created == applied.created and planned.dropped == applied.dropped
    assert planned.indexes_after == applied.indexes_after


def test_index_with_unexpected_definition_is_kept(tmp_path):
    odd = tuple(s for s in LEGACY_INDEXES if "idx_raw_data_telegram_id" not in s) + (
        "CREATE INDEX idx_raw_data_telegram_id ON raw_data (language, telegram_id)",)
    path = build_legacy_db(tmp_path / "odd.db", indexes=odd)
    result = mig.migrate(str(path))
    assert [name for name, _ in result.kept] == ["idx_raw_data_telegram_id"]
    assert "idx_raw_data_telegram_id" in index_signature(path)


def test_unique_index_over_duplicate_rows_is_skipped_and_coverage_kept(tmp_path):
    """Legacy double votes: the UNIQUE vote index cannot be created, so the
    composite index that serves the raw_data_id lookups must stay."""
    no_unique = tuple(s for s in LEGACY_INDEXES if "idx_user_votes_unique" not in s)
    path = build_legacy_db(tmp_path / "dupvotes.db", indexes=no_unique)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO user_votes (raw_data_id, telegram_user_id, current_vote) VALUES (1, 7, -1)")
    conn.commit()
    conn.close()

    result = mig.migrate(str(path))
    assert any("idx_user_votes_unique" in w for w in result.warnings)
    assert ("idx_user_votes_composite", "no other index covers it") in result.kept
    signature = index_signature(path)
    assert "idx_user_votes_unique" not in signature
    assert any(t == "user_votes" and cols[:1] == ("raw_data_id",) for t, _u, cols in signature.values())


def test_failure_rolls_everything_back(legacy_db, monkeypatch):
    before = index_signature(legacy_db)
    real_create = mig._create_statement
    calls = []

    def flaky(index):
        calls.append(index.name)
        if len(calls) == 2:
            return "CREATE INDEX broken ON no_such_table (x)"
        return real_create(index)

    monkeypatch.setattr(mig, "_create_statement", flaky)
    with pytest.raises(sqlite3.OperationalError):
        mig.migrate(str(legacy_db))
    assert index_signature(legacy_db) == before


def test_missing_file_is_not_created(tmp_path):
    path = tmp_path / "nope.db"
    with pytest.raises(mig.MigrationError):
        mig.migrate(str(path))
    assert not path.exists()
    assert mig.main(["--db", str(path)]) == 2
    assert not path.exists()


def test_not_a_cengbot_database(tmp_path, capsys):
    path = tmp_path / "other.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE something (x)")
    conn.close()
    assert mig.main(["--db", str(path)]) == 2
    assert "not a CengBot database" in capsys.readouterr().err


def test_cli_dry_run_and_apply(legacy_db, capsys):
    digest = sha256(legacy_db)
    assert mig.main(["--db", str(legacy_db), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "+ create index idx_raw_data_duplicate_of_id" in out
    assert "- drop index ix_raw_data_id" in out
    assert sha256(legacy_db) == digest

    assert mig.main(["--db", str(legacy_db)]) == 0
    assert "migration applied" in capsys.readouterr().out
    assert mig.main(["--db", str(legacy_db)]) == 0
    assert "Nothing to do" in capsys.readouterr().out


def test_cli_defaults_to_database_url(legacy_db, monkeypatch, capsys):
    monkeypatch.setattr(dm, "DATABASE_URL", f"sqlite:///{legacy_db}")
    assert mig.main(["--dry-run"]) == 0
    assert str(legacy_db) in capsys.readouterr().out
    monkeypatch.setattr(dm, "DATABASE_URL", "postgresql://u:secret@db/cengbot")
    assert mig.main(["--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "use --db" in err and "secret" not in err


def test_wal_database_migrates(legacy_db, fresh_signature):
    conn = sqlite3.connect(legacy_db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.close()
    result = mig.migrate(str(legacy_db))
    assert result.journal_mode == "wal" and result.changed
    assert index_signature(legacy_db) == fresh_signature


def test_migrated_indexes_are_used_by_the_queries(legacy_db):
    mig.migrate(str(legacy_db))
    conn = sqlite3.connect(legacy_db)
    try:
        def plan(sql, params=()):
            return " | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params))

        # delete of an original question / of a training answer clears the links
        assert "idx_raw_data_duplicate_of_id" in plan(
            "UPDATE raw_data SET duplicate_of_id = NULL WHERE duplicate_of_id = ?", (1,))
        assert "idx_training_data_duplicate_answer_of_id" in plan(
            "UPDATE training_data SET duplicate_answer_of_id = NULL WHERE duplicate_answer_of_id = ?", (1,))
        # vote lookups still use an index after idx_user_votes_raw_data_id is gone
        assert "idx_user_votes_unique" in plan(
            "SELECT sum(current_vote) FROM user_votes WHERE raw_data_id = ?", (1,))
        assert "SCAN raw_data" not in plan("SELECT * FROM raw_data WHERE telegram_id = ?", (101,))
    finally:
        conn.close()


@pytest.mark.skipif(not REAL_DB.exists(), reason="real university_bot.db not present")
def test_copy_of_the_real_database_migrates_to_the_fresh_schema(tmp_path, fresh_signature):
    copy = tmp_path / "real_copy.db"
    copy_real_db(copy)  # read-only access to the real file
    before_rows = table_dump(copy)
    mig.migrate(str(copy))
    assert index_signature(copy) == fresh_signature
    assert table_dump(copy) == before_rows
    assert not mig.migrate(str(copy)).changed
