#!/usr/bin/env python3
"""
Index migration for the CengBot SQLite database
===============================================

Brings an existing database to the index set declared in
``src/database_models.py`` (the set a freshly created database gets):

1. creates the model indexes the database lacks (``CREATE INDEX IF NOT EXISTS``),
   e.g. ``raw_data(duplicate_of_id)`` and ``training_data(duplicate_answer_of_id)``;
2. drops the redundant indexes older versions created: indexes on the
   ``INTEGER PRIMARY KEY`` (the rowid needs none), exact duplicates of a
   UNIQUE constraint / another index, and single-column indexes that are the
   leading column of a composite index. An index is only dropped when it
   still has exactly the expected definition and another index (or the
   primary key) that stays covers it.

Everything runs in one transaction (all or nothing); no table or row is
modified. Safe to run again: a migrated database needs no further action.
Stop the services and take a backup first (``scripts/backup_data.sh``).

Usage::

    python src/db_migrations.py --dry-run          # show what would change
    python src/db_migrations.py                     # migrate the DATABASE_URL database
    python src/db_migrations.py --db /path/university_bot.db [--dry-run]

Exit codes: 0 success (or nothing to do), 1 migration failed, 2 bad input.
"""

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logger = logging.getLogger(__name__)

# Indexes created by older versions that the current models no longer declare.
# (name, table, columns). Each is dropped only if it is still redundant.
REDUNDANT_INDEXES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    # Column(..., primary_key=True, index=True): duplicates the rowid
    ("ix_raw_data_id", "raw_data", ("id",)),
    ("ix_training_data_id", "training_data", ("id",)),
    ("ix_user_votes_id", "user_votes", ("id",)),
    ("ix_user_analytics_id", "user_analytics", ("id",)),
    ("ix_system_metrics_id", "system_metrics", ("id",)),
    # two identical single-column indexes, both a prefix of idx_raw_data_composite
    ("idx_raw_data_telegram_id", "raw_data", ("telegram_id",)),
    ("ix_raw_data_telegram_id", "raw_data", ("telegram_id",)),
    # duplicates of UNIQUE indexes
    ("idx_training_data_source_id", "training_data", ("source_id",)),
    ("idx_user_analytics_telegram_id", "user_analytics", ("telegram_id",)),
    ("idx_user_votes_composite", "user_votes", ("raw_data_id", "telegram_user_id")),
    # prefix of idx_user_votes_unique (raw_data_id, telegram_user_id)
    ("idx_user_votes_raw_data_id", "user_votes", ("raw_data_id",)),
    # duplicates of the model's ix_* indexes
    ("idx_user_votes_telegram_user_id", "user_votes", ("telegram_user_id",)),
    ("idx_system_metrics_name", "system_metrics", ("metric_name",)),
)


@dataclass
class IndexInfo:
    name: str
    table: str
    columns: Tuple[Optional[str], ...]
    unique: bool
    origin: str  # 'c' = CREATE INDEX, 'u' = UNIQUE constraint, 'pk' = PRIMARY KEY
    partial: bool


@dataclass
class MigrationResult:
    database: str
    dry_run: bool
    journal_mode: str = ""
    created: List[str] = field(default_factory=list)
    dropped: List[Tuple[str, str]] = field(default_factory=list)  # (index, covered by)
    kept: List[Tuple[str, str]] = field(default_factory=list)  # (index, reason)
    warnings: List[str] = field(default_factory=list)
    indexes_before: Dict[str, List[str]] = field(default_factory=dict)
    indexes_after: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.created or self.dropped)


class MigrationError(Exception):
    """The database cannot be migrated (nothing was changed)."""


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]


def _indexes(conn: sqlite3.Connection, tables: Sequence[str]) -> Dict[str, IndexInfo]:
    result: Dict[str, IndexInfo] = {}
    for table in tables:
        for _seq, name, unique, origin, partial in conn.execute(f"PRAGMA index_list({_quote(table)})"):
            columns = tuple(row[2] for row in conn.execute(f"PRAGMA index_info({_quote(name)})"))
            result[name] = IndexInfo(name, table, columns, bool(unique), origin, bool(partial))
    return result


def _rowid_alias(conn: sqlite3.Connection, table: str) -> Optional[str]:
    """Name of the INTEGER PRIMARY KEY column (an alias of the rowid), if any."""
    pk_columns = [(row[1], (row[2] or "").upper()) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
                  if row[5]]
    if len(pk_columns) == 1 and pk_columns[0][1] == "INTEGER":
        return pk_columns[0][0]
    return None


def _model_indexes():
    """The indexes declared on the models (one source of truth for fresh and old DBs)."""
    import database_models  # imported lazily: heavy (sklearn) and reads DATABASE_URL

    for table in database_models.Base.metadata.sorted_tables:
        for index in sorted(table.indexes, key=lambda i: i.name):
            yield table.name, index


def _create_statement(index) -> str:
    from sqlalchemy.dialects import sqlite as sqlite_dialect
    from sqlalchemy.schema import CreateIndex

    return str(CreateIndex(index, if_not_exists=True).compile(dialect=sqlite_dialect.dialect()))


def _has_duplicates(conn: sqlite3.Connection, table: str, columns: Sequence[str]) -> bool:
    cols = ", ".join(_quote(c) for c in columns)
    not_null = " AND ".join(f"{_quote(c)} IS NOT NULL" for c in columns)
    row = conn.execute(f"SELECT 1 FROM {_quote(table)} WHERE {not_null} GROUP BY {cols} "
                       f"HAVING count(*) > 1 LIMIT 1").fetchone()
    return row is not None


def _covering_index(conn, info: IndexInfo, indexes: Dict[str, IndexInfo], dropping: set) -> Optional[str]:
    """Name of what makes ``info`` redundant (another kept index or the rowid), else None."""
    if len(info.columns) == 1 and info.columns[0] == _rowid_alias(conn, info.table):
        return "INTEGER PRIMARY KEY (rowid)"
    candidates = []
    for other in indexes.values():
        if other.name == info.name or other.name in dropping or other.table != info.table or other.partial:
            continue
        if other.columns[:len(info.columns)] == info.columns:
            candidates.append(other)
    if not candidates:
        return None
    # Prefer an index that is not itself a drop candidate, then one with exactly
    # the same columns, then UNIQUE ones, then by name.
    redundant = {name for name, _table, _columns in REDUNDANT_INDEXES}
    candidates.sort(key=lambda o: (o.name in redundant, len(o.columns) != len(info.columns),
                                   not o.unique, o.name))
    return candidates[0].name


def _group(indexes: Dict[str, IndexInfo]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for info in indexes.values():
        grouped.setdefault(info.table, []).append(info.name)
    return {table: sorted(names) for table, names in sorted(grouped.items())}


def migrate(db_path: str, dry_run: bool = False, busy_timeout: float = 30.0) -> MigrationResult:
    """Migrate the SQLite file ``db_path`` (see the module docstring)."""
    db_path = os.path.abspath(db_path)
    if not os.path.isfile(db_path):
        raise MigrationError(f"database file not found: {db_path}")
    result = MigrationResult(database=db_path, dry_run=dry_run)

    if dry_run:  # read-only: guaranteed to change nothing
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=busy_timeout,
                               isolation_level=None)
    else:
        conn = sqlite3.connect(db_path, timeout=busy_timeout, isolation_level=None)
    try:
        tables = _tables(conn)
        if "raw_data" not in tables:
            raise MigrationError(f"{db_path} is not a CengBot database (no raw_data table)")
        check = conn.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise MigrationError(f"PRAGMA quick_check failed: {check} - restore a backup first")
        result.journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        if not dry_run:
            # Write lock for the whole plan + DDL: nobody changes the schema meanwhile.
            conn.execute("BEGIN IMMEDIATE")
        try:
            before = _indexes(conn, tables)
            result.indexes_before = _group(before)
            after = dict(before)

            # 1) model indexes the database lacks
            for table, index in _model_indexes():
                if table not in tables:
                    result.warnings.append(f"table {table} is missing (run init_db); skipped {index.name}")
                    continue
                columns = tuple(c.name for c in index.columns)
                existing = after.get(index.name)
                if existing is not None:
                    if existing.columns != columns or existing.unique != bool(index.unique):
                        result.warnings.append(
                            f"index {index.name} exists with a different definition "
                            f"{existing.columns} (expected {columns}); left unchanged")
                    continue
                if index.unique and _has_duplicates(conn, table, columns):
                    result.warnings.append(
                        f"cannot create UNIQUE index {index.name}: {table} has duplicate "
                        f"({', '.join(columns)}) rows; remove them and run again")
                    continue
                if not dry_run:
                    conn.execute(_create_statement(index))
                result.created.append(index.name)
                after[index.name] = IndexInfo(index.name, table, columns, bool(index.unique), "c", False)

            # 2) redundant legacy indexes (checked against the post-create state)
            model_names = {index.name for _table, index in _model_indexes()}
            dropping: set = set()
            for name, table, columns in REDUNDANT_INDEXES:
                info = after.get(name)
                if info is None:
                    continue
                if name in model_names:  # never drop what the models declare
                    result.kept.append((name, "declared by the models"))
                    continue
                if (info.table != table or info.columns != columns or info.unique
                        or info.origin != "c" or info.partial):
                    result.kept.append((name, f"unexpected definition {info.table}{info.columns}"))
                    continue
                covered_by = _covering_index(conn, info, after, dropping)
                if covered_by is None:
                    result.kept.append((name, "no other index covers it"))
                    continue
                dropping.add(name)
                if not dry_run:
                    conn.execute(f"DROP INDEX {_quote(name)}")
                result.dropped.append((name, covered_by))
            for name in dropping:
                after.pop(name, None)

            if not dry_run:
                conn.execute("COMMIT")
                after = _indexes(conn, tables)
        except BaseException:
            if not dry_run and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        result.indexes_after = _group(after)
        return result
    finally:
        conn.close()


def _default_db_path() -> str:
    from sqlalchemy.engine import make_url

    import database_models

    url = make_url(database_models.DATABASE_URL)
    if url.get_backend_name() != "sqlite" or not url.database or ":memory:" in url.database:
        raise MigrationError(f"DATABASE_URL is not an SQLite file: "
                             f"{url.render_as_string(hide_password=True)}; use --db")
    return url.database


def _print_report(result: MigrationResult) -> None:
    mode = "DRY RUN - nothing was changed" if result.dry_run else "migration applied"
    print(f"Database: {result.database} (journal_mode={result.journal_mode})")
    print(f"Mode:     {mode}")
    for name in result.created:
        print(f"  + create index {name}")
    for name, covered_by in result.dropped:
        print(f"  - drop index {name}  (redundant: covered by {covered_by})")
    for name, reason in result.kept:
        print(f"  = keep index {name}  ({reason})")
    for warning in result.warnings:
        print(f"  ! {warning}")
    if not result.changed:
        print("Nothing to do: the indexes already match the models.")
    print("Indexes after:" if not result.dry_run else "Indexes after (planned):")
    for table, names in result.indexes_after.items():
        print(f"  {table}: {', '.join(names) if names else '-'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create missing / drop redundant indexes of the CengBot SQLite database "
                    "(back it up first; stop the services).")
    parser.add_argument("--db", help="SQLite database file (default: the DATABASE_URL database)")
    parser.add_argument("--dry-run", action="store_true", help="only show what would change")
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
    try:
        db_path = args.db or _default_db_path()
        result = migrate(db_path, dry_run=args.dry_run)
    except MigrationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except sqlite3.Error as e:
        print(f"ERROR: migration failed, nothing was changed: {e}", file=sys.stderr)
        return 1
    _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
