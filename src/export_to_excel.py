#!/usr/bin/env python3
"""
Raw Database to Excel Export Script for CengBot
===============================================

Exports the raw_data table (default) -- or any of the other tables -- to an
Excel workbook with an ``export_summary`` sheet. Exports raw user
interactions data for analysis and backup purposes.

Usage::

    python src/export_to_excel.py                       # raw_data -> excel/cengbot_database_export_<ts>.xlsx
    python src/export_to_excel.py --tables raw_data,user_votes
    python src/export_to_excel.py --all-tables
    python src/export_to_excel.py --output /path/file.xlsx [--overwrite]

Details:
- uses the application's database (``database_models.engine`` / ``SessionLocal``,
  i.e. ``DATABASE_URL``), never a separate hard-coded database path;
- rows are streamed in primary-key chunks into a write-only workbook, so
  memory stays flat and SQLite writers are never blocked for long;
- text is made Excel-safe: XML-illegal control characters are removed, cells
  longer than Excel's 32,767-character limit are truncated with a visible
  marker, and text starting with ``=`` is stored as text, not as a formula;
- timezone-aware datetimes are converted to naive UTC (Excel has no timezones;
  the application stores UTC);
- an existing file is never overwritten silently: the default timestamped name
  gets a ``_1``, ``_2``... suffix, and an explicit ``--output`` that exists is
  an error unless ``--overwrite`` is given. The workbook is written to a hidden
  temporary file first and moved into place only when complete.

Author: naholav
"""

import argparse
import logging
import math
import os
import re
import sys
import uuid
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from sqlalchemy import select

# Import sibling modules from src/ (one database_models instance for everyone)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import database_models  # noqa: E402
from database_models import (  # noqa: E402
    RawData,
    SystemMetrics,
    TrainingData,
    UserAnalytics,
    UserVotes,
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(SRC_DIR)
EXCEL_DIR = os.path.join(BASE_DIR, 'excel')
EXPORT_PREFIX = "cengbot_database_export"

# Excel limits
EXCEL_MAX_CELL_CHARS = 32767          # characters (UTF-16 code units) per cell
EXCEL_MAX_ROWS = 1048576              # rows per sheet, header included
EXCEL_MAX_SHEET_NAME = 31
DEFAULT_CHUNK_SIZE = 1000
TRUNCATION_MARKER = " ... [truncated by export, original length {length} characters]"

# Characters XML 1.0 (and therefore xlsx) cannot store. openpyxl raises
# IllegalCharacterError for the C0 controls, which aborts the whole export.
_ILLEGAL_XML_CHARS_RE = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]')

# table name -> (model, description); order = sheet order for --all-tables
TABLES: Dict[str, Tuple[type, str]] = {
    'raw_data': (RawData, 'User interactions and questions'),
    'training_data': (TrainingData, 'Curated question/answer pairs for training'),
    'user_votes': (UserVotes, 'Per-user like/dislike votes'),
    'user_analytics': (UserAnalytics, 'Per-user usage analytics'),
    'system_metrics': (SystemMetrics, 'Stored system metrics'),
}
DEFAULT_TABLES = ('raw_data',)


def get_database_connection():
    """Return the application's engine and a new session (``DATABASE_URL``)."""
    try:
        return database_models.engine, database_models.SessionLocal()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def table_columns(model) -> List[str]:
    """Column names of a model's table, in schema order."""
    return [column.name for column in model.__table__.columns]


def iter_table_rows(db_session, model, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[tuple]:
    """Yield all rows of ``model``'s table as tuples (``table_columns`` order).

    Uses keyset pagination on the primary key: each chunk is a short, separate
    query, so memory is bounded and no long-lived read cursor blocks writers.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    table = model.__table__
    pk = list(table.primary_key.columns)[0]
    pk_index = list(table.columns).index(pk)
    last_key = None
    while True:
        stmt = select(*table.columns).order_by(pk).limit(chunk_size)
        if last_key is not None:
            stmt = stmt.where(pk > last_key)
        rows = db_session.execute(stmt).all()
        for row in rows:
            yield tuple(row)
        if len(rows) < chunk_size:
            return
        last_key = rows[-1][pk_index]


def _table_dataframe(db_session, table_name: str) -> pd.DataFrame:
    model, _ = TABLES[table_name]
    try:
        return pd.DataFrame(list(iter_table_rows(db_session, model)),
                            columns=table_columns(model))
    except Exception as e:
        logger.error(f"Error exporting {table_name}: {e}")
        raise


def export_raw_data(db_session):
    """Export raw_data table to DataFrame."""
    return _table_dataframe(db_session, 'raw_data')


def export_training_data(db_session):
    """Export training_data table to DataFrame."""
    return _table_dataframe(db_session, 'training_data')


def export_user_votes(db_session):
    """Export user_votes table to DataFrame."""
    return _table_dataframe(db_session, 'user_votes')


def export_user_analytics(db_session):
    """Export user_analytics table to DataFrame."""
    return _table_dataframe(db_session, 'user_analytics')


def export_system_metrics(db_session):
    """Export system_metrics table to DataFrame."""
    return _table_dataframe(db_session, 'system_metrics')


# ---------------------------------------------------------------------------
# Excel-safe values
# ---------------------------------------------------------------------------

def _utf16_len(text: str) -> int:
    return len(text.encode('utf-16-le', 'surrogatepass')) // 2


def sanitize_text(text: str, max_chars: Optional[int] = None) -> str:
    """Make ``text`` storable in one Excel cell.

    Removes XML-illegal characters and truncates (with a visible marker) to
    the cell limit, counted in UTF-16 code units like Excel does. Turkish and
    other non-ASCII characters are kept as they are.
    """
    limit = EXCEL_MAX_CELL_CHARS if max_chars is None else max_chars
    text = _ILLEGAL_XML_CHARS_RE.sub('', text)
    if 2 * len(text) <= limit or _utf16_len(text) <= limit:
        return text
    marker = TRUNCATION_MARKER.format(length=len(text))
    budget = max(limit - _utf16_len(marker), 0)
    cut = budget
    while cut > 0:
        excess = _utf16_len(text[:cut]) - budget
        if excess <= 0:
            break
        cut -= excess
    return text[:max(cut, 0)] + marker


def excel_value(value, worksheet=None):
    """Convert one database value into something openpyxl can always write."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = sanitize_text(value)
        if text.startswith('=') and worksheet is not None:
            # openpyxl would store this as a formula (formula injection from
            # user-supplied text); force a plain string cell instead.
            cell = WriteOnlyCell(worksheet, value=text)
            cell.data_type = 's'
            return cell
        return text
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            if value.utcoffset() is not None:
                value = value.astimezone(timezone.utc)
            value = value.replace(tzinfo=None)
        return value
    if isinstance(value, time):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (int, date, Decimal)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return sanitize_text(bytes(value).decode('utf-8', errors='replace'))
    return sanitize_text(str(value))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _write_table(workbook, sheet_name: str, columns: Sequence[str], rows) -> Tuple[int, List[str]]:
    """Stream ``rows`` into one or more sheets; returns (row count, sheet names)."""
    max_data_rows = EXCEL_MAX_ROWS - 1  # one row per sheet is the header
    sheets = [sheet_name[:EXCEL_MAX_SHEET_NAME]]
    worksheet = workbook.create_sheet(sheets[0])
    worksheet.append(list(columns))
    in_sheet = total = 0
    for row in rows:
        if in_sheet >= max_data_rows:
            suffix = f"_{len(sheets) + 1}"
            name = sheet_name[:EXCEL_MAX_SHEET_NAME - len(suffix)] + suffix
            sheets.append(name)
            worksheet = workbook.create_sheet(name)
            worksheet.append(list(columns))
            in_sheet = 0
        worksheet.append([excel_value(v, worksheet) for v in row])
        in_sheet += 1
        total += 1
    return total, sheets


def _database_label(db_session) -> str:
    try:
        url = db_session.get_bind().url
    except Exception:
        return 'unknown'
    if url.get_backend_name() == 'sqlite':
        return os.path.basename(url.database or '') or ':memory:'
    return url.render_as_string(hide_password=True)


def _write_summary(workbook, exported_at: datetime, database: str,
                   counts: Dict[str, int], sheets: Dict[str, List[str]]):
    worksheet = workbook.create_sheet('export_summary')
    worksheet.append(['Export Information', 'Values'])
    worksheet.append(['Export Date', exported_at.strftime("%Y-%m-%d")])
    worksheet.append(['Export Time', exported_at.strftime("%H:%M:%S")])
    worksheet.append(['Database File', sanitize_text(database)])
    worksheet.append(['Tables Exported', ', '.join(counts)])
    worksheet.append(['Timestamps', 'UTC (as stored in the database)'])
    worksheet.append([])
    worksheet.append(['Table Name', 'Record Count', 'Description', 'Sheets'])
    for name, count in counts.items():
        worksheet.append([name, count, TABLES[name][1], ', '.join(sheets[name])])


def _discard_workbook(workbook):
    """Close a half-written write-only workbook and delete its temp files."""
    for worksheet in getattr(workbook, 'worksheets', []):
        try:
            if not worksheet.closed:
                worksheet.close()
            writer = getattr(worksheet, '_writer', None)
            if writer is not None:
                writer.cleanup()
        except Exception:
            pass


def _reserve_output_path(output_dir: str, stamp: str) -> str:
    """Create (exclusively) and return a new, unused export file name."""
    os.makedirs(output_dir, exist_ok=True)
    for n in range(1000):
        suffix = f"_{n}" if n else ""
        path = os.path.join(output_dir, f"{EXPORT_PREFIX}_{stamp}{suffix}.xlsx")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return path
    raise FileExistsError(f"No free export file name for {stamp} in {output_dir}")


def _parse_tables(tables) -> List[str]:
    if isinstance(tables, str):
        tables = [t for t in (part.strip() for part in tables.split(',')) if t]
    tables = list(dict.fromkeys(tables))  # de-duplicate, keep order
    unknown = [t for t in tables if t not in TABLES]
    if unknown or not tables:
        raise ValueError(f"Unknown or no table(s) {unknown}; choose from {', '.join(TABLES)}")
    return tables


def export_database(output_path: Optional[str] = None, output_dir: Optional[str] = None,
                    tables=DEFAULT_TABLES, db_session=None,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    overwrite: bool = False) -> Tuple[str, Dict[str, int]]:
    """Export ``tables`` to an .xlsx file; returns (file path, row counts).

    ``output_path`` -- exact target file (must not exist unless ``overwrite``);
    otherwise a timestamped file is created in ``output_dir`` (default ``excel/``).
    """
    tables = _parse_tables(tables)
    exported_at = datetime.now()

    reserved = False
    if output_path:
        final_path = os.path.abspath(output_path)
        if os.path.exists(final_path) and not overwrite:
            raise FileExistsError(f"{final_path} already exists (use --overwrite to replace it)")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
    else:
        final_path = _reserve_output_path(os.path.abspath(output_dir or EXCEL_DIR),
                                          exported_at.strftime("%Y%m%d_%H%M%S"))
        reserved = True

    # Hidden temp file next to the target (same filesystem -> atomic rename);
    # it does not match the cengbot_database_export_*.xlsx glob used by scripts.
    tmp_path = os.path.join(os.path.dirname(final_path),
                            f".{os.path.basename(final_path)}.{uuid.uuid4().hex[:8]}.partial")

    own_session = db_session is None
    session = database_models.SessionLocal() if own_session else db_session
    workbook = None
    try:
        workbook = Workbook(write_only=True)
        counts: Dict[str, int] = {}
        sheets: Dict[str, List[str]] = {}
        for name in tables:
            model, _ = TABLES[name]
            logger.info(f"Exporting {name} table...")
            counts[name], sheets[name] = _write_table(
                workbook, name, table_columns(model),
                iter_table_rows(session, model, chunk_size))
            if counts[name]:
                logger.info(f"Exported {counts[name]} {name} records")
            else:
                logger.warning(f"No {name} records found to export")
        _write_summary(workbook, exported_at, _database_label(session), counts, sheets)

        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        with os.fdopen(fd, 'wb') as f:
            workbook.save(f)
        os.replace(tmp_path, final_path)
        return final_path, counts
    except BaseException:
        if workbook is not None:
            _discard_workbook(workbook)
        for path in [tmp_path] + ([final_path] if reserved else []):
            try:
                os.remove(path)
            except OSError:
                pass
        raise
    finally:
        if own_session:
            session.close()


def main(argv: Optional[List[str]] = None):
    """Main export function. Returns True on success."""
    parser = argparse.ArgumentParser(description="Export the CengBot database to Excel")
    parser.add_argument('--output-dir', default=EXCEL_DIR,
                        help=f"directory for the timestamped export file (default {EXCEL_DIR})")
    parser.add_argument('--output', default=None,
                        help="exact output file (refuses to overwrite unless --overwrite)")
    parser.add_argument('--overwrite', action='store_true',
                        help="allow --output to replace an existing file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tables', default=','.join(DEFAULT_TABLES),
                       help=f"comma-separated tables (default raw_data; available: {', '.join(TABLES)})")
    group.add_argument('--all-tables', action='store_true', help="export every table")
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    tables = list(TABLES) if args.all_tables else args.tables
    try:
        logger.info("Starting database export")
        filepath, counts = export_database(
            output_path=args.output, output_dir=args.output_dir, tables=tables,
            chunk_size=args.chunk_size, overwrite=args.overwrite)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        print(f"❌ Export failed: {e}")
        return False

    total = sum(counts.values())
    details = ', '.join(f"{n}={c}" for n, c in counts.items())
    logger.info(f"Database export completed: {filepath} ({details})")
    print("✅ Database export completed successfully!")
    print(f"📁 File: {os.path.basename(filepath)}")
    print(f"📍 Location: {filepath}")
    print(f"📊 Records: {total} ({details})")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
