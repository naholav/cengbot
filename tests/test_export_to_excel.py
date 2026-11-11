"""Tests for src/export_to_excel.py (temp DB from conftest, output in tmp_path)."""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook
from sqlalchemy import event

import database_models
import export_to_excel as ex
from database_models import RawData, TrainingData, UserVotes

TURKISH_Q = "Çukurova Üniversitesi'nde ders kaydı ne zaman? ğüşıöç İĞÜŞÖÇ"
TURKISH_A = "Ders kayıtları Eylül'ün ilk haftasında başlar. Şimdi öğrenci işlerine başvurun."
ORIGINAL_RAW_COLUMNS = [
    'id', 'telegram_id', 'telegram_message_id', 'username', 'question', 'answer',
    'language', 'like', 'admin_approved', 'is_duplicate', 'duplicate_of_id',
    'similarity_score', 'created_at', 'answered_at', 'message_thread_id',
    'processing_time', 'model_version', 'context_length', 'response_length',
    'quality_score', 'sentiment_score', 'complexity_score', 'topic_category',
    'keywords', 'admin_notes', 'last_updated',
]
CREATED = datetime(2025, 10, 1, 14, 52, 3, 123000)


@pytest.fixture
def seeded(db_session):
    rows = [
        RawData(telegram_id=111, username="öğrenci_1", question=TURKISH_Q, answer=TURKISH_A,
                language="TR", like=1, admin_approved=1, created_at=CREATED,
                answered_at=CREATED + timedelta(seconds=5), similarity_score=0.5),
        RawData(telegram_id=222, question="bell\x07 null\x00 esc\x1b tab\tnew\nline\r end",
                answer="x" * 40000, language="EN", admin_approved=0, is_duplicate=True),
        RawData(telegram_id=333, question='=HYPERLINK("http://evil.example","click")',
                answer="=1+1", language="EN"),
        RawData(telegram_id=444, question="emoji 😀 test", answer="😀" * 20000),
    ]
    db_session.add_all(rows)
    db_session.flush()
    db_session.add(TrainingData(source_id=rows[0].id, question=TURKISH_Q, answer=TURKISH_A,
                                language="TR"))
    db_session.add(UserVotes(raw_data_id=rows[0].id, telegram_user_id=999, current_vote=1))
    db_session.commit()
    return rows


def _rows_by_id(ws):
    values = list(ws.values)
    header = list(values[0])
    return header, {row[0]: dict(zip(header, row)) for row in values[1:]}


def _utf16(text):
    return len(text.encode("utf-16-le")) // 2


# --------------------------------------------------------------------------
# configuration / imports (decisions d, e)
# --------------------------------------------------------------------------

def test_uses_application_engine_and_single_module():
    assert ex.database_models is sys.modules["database_models"]
    engine, session = ex.get_database_connection()
    try:
        assert engine is database_models.engine
        assert session.get_bind() is database_models.engine
    finally:
        session.close()
    source = Path(ex.__file__).read_text(encoding="utf-8")
    assert "src.database_models" not in source
    assert "create_engine" not in source


def test_export_without_session_uses_session_local(seeded, tmp_path):
    path, counts = ex.export_database(output_dir=tmp_path)
    assert counts == {"raw_data": 4}
    assert os.path.dirname(path) == str(tmp_path)


def test_default_output_dir_is_project_excel_dir():
    root = Path(__file__).resolve().parent.parent
    assert Path(ex.EXCEL_DIR) == root / "excel"


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------

def test_export_round_trip(seeded, db_session, tmp_path):
    path, counts = ex.export_database(output_dir=tmp_path, db_session=db_session)

    name = os.path.basename(path)
    assert name.startswith("cengbot_database_export_") and name.endswith(".xlsx")
    assert counts == {"raw_data": 4}

    wb = load_workbook(path)
    assert wb.sheetnames == ["raw_data", "export_summary"]
    header, rows = _rows_by_id(wb["raw_data"])
    assert header == ORIGINAL_RAW_COLUMNS
    assert len(rows) == 4

    tr = rows[seeded[0].id]
    assert tr["question"] == TURKISH_Q            # Turkish characters intact
    assert tr["answer"] == TURKISH_A
    assert tr["username"] == "öğrenci_1"
    assert tr["created_at"] == CREATED            # real datetime cells
    assert tr["answered_at"] == CREATED + timedelta(seconds=5)
    assert tr["is_duplicate"] is False

    ctrl = rows[seeded[1].id]
    # control characters removed; tab and newlines kept (XML turns a bare
    # carriage return into a newline when the file is read back)
    assert ctrl["question"] in ("bell null esc tab\tnew\nline\r end",
                                "bell null esc tab\tnew\nline\n end")
    assert _utf16(ctrl["answer"]) <= ex.EXCEL_MAX_CELL_CHARS
    assert ctrl["answer"].startswith("x" * 1000)
    assert ctrl["answer"].endswith("original length 40000 characters]")
    assert ctrl["is_duplicate"] is True

    emoji = rows[seeded[3].id]
    assert _utf16(emoji["answer"]) <= ex.EXCEL_MAX_CELL_CHARS
    assert "truncated" in emoji["answer"]


def test_formula_like_text_is_stored_as_text(seeded, db_session, tmp_path):
    path, _ = ex.export_database(output_dir=tmp_path, db_session=db_session)
    ws = load_workbook(path)["raw_data"]
    header = [c.value for c in ws[1]]
    q_col, a_col = header.index("question") + 1, header.index("answer") + 1
    for row in ws.iter_rows(min_row=2):
        if row[0].value == seeded[2].id:
            assert row[q_col - 1].data_type == "s"
            assert row[q_col - 1].value == '=HYPERLINK("http://evil.example","click")'
            assert row[a_col - 1].data_type == "s"
            assert row[a_col - 1].value == "=1+1"
            break
    else:
        pytest.fail("row not exported")


def test_pandas_can_read_export(seeded, db_session, tmp_path):
    path, _ = ex.export_database(output_dir=tmp_path, db_session=db_session)
    df = pd.read_excel(path, sheet_name="raw_data")
    assert len(df) == 4
    assert TURKISH_Q in df["question"].tolist()


def test_summary_sheet(seeded, db_session, tmp_path):
    path, _ = ex.export_database(output_dir=tmp_path, db_session=db_session,
                                 tables=["raw_data", "user_votes"])
    values = list(load_workbook(path)["export_summary"].values)
    info = {row[0]: row[1] for row in values[:6]}
    assert info["Database File"] == "test_university_bot.db"
    assert info["Tables Exported"] == "raw_data, user_votes"
    table_rows = {row[0]: row for row in values if row[0] in ex.TABLES}
    assert table_rows["raw_data"][1] == 4
    assert table_rows["user_votes"][1] == 1


def test_all_tables_and_empty_table_has_header(seeded, db_session, tmp_path):
    path, counts = ex.export_database(output_dir=tmp_path, db_session=db_session,
                                      tables=list(ex.TABLES))
    assert counts == {"raw_data": 4, "training_data": 1, "user_votes": 1,
                      "user_analytics": 0, "system_metrics": 0}
    wb = load_workbook(path)
    assert wb.sheetnames == list(ex.TABLES) + ["export_summary"]
    header = [c.value for c in next(wb["user_analytics"].iter_rows(max_row=1))]
    assert header == ex.table_columns(database_models.UserAnalytics)
    _, training = _rows_by_id(wb["training_data"])
    assert list(training.values())[0]["question"] == TURKISH_Q


# --------------------------------------------------------------------------
# large tables
# --------------------------------------------------------------------------

def test_rows_are_read_in_bounded_chunks(db_session, test_engine):
    db_session.add_all([RawData(telegram_id=i, question=f"q{i}") for i in range(25)])
    db_session.commit()
    selects = []

    def count(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    event.listen(test_engine, "before_cursor_execute", count)
    try:
        rows = list(ex.iter_table_rows(db_session, RawData, chunk_size=7))
    finally:
        event.remove(test_engine, "before_cursor_execute", count)

    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    assert len(rows) == 25
    assert len(selects) == 4  # 7 + 7 + 7 + 4
    assert all("LIMIT" in s.upper() for s in selects)


def test_iter_table_rows_rejects_bad_chunk_size(db_session):
    with pytest.raises(ValueError):
        list(ex.iter_table_rows(db_session, RawData, chunk_size=0))


def test_sheets_are_split_at_excel_row_limit(db_session, tmp_path, monkeypatch):
    db_session.add_all([RawData(telegram_id=i, question=f"q{i}") for i in range(10)])
    db_session.commit()
    monkeypatch.setattr(ex, "EXCEL_MAX_ROWS", 4)  # 3 data rows per sheet

    path, counts = ex.export_database(output_dir=tmp_path, db_session=db_session,
                                      chunk_size=3)
    assert counts == {"raw_data": 10}
    wb = load_workbook(path)
    assert wb.sheetnames == ["raw_data", "raw_data_2", "raw_data_3", "raw_data_4",
                             "export_summary"]
    ids = []
    for name in wb.sheetnames[:-1]:
        values = list(wb[name].values)
        assert values[0][0] == "id"
        ids += [row[0] for row in values[1:]]
    assert len(ids) == 10 and len(set(ids)) == 10


# --------------------------------------------------------------------------
# value sanitising
# --------------------------------------------------------------------------

def test_sanitize_text():
    assert ex.sanitize_text(TURKISH_Q) == TURKISH_Q
    assert ex.sanitize_text("a\x00b\x0bc\x1fd￾\tf\n") == "abcd\tf\n"
    out = ex.sanitize_text("é" * 500, max_chars=100)
    assert len(out) <= 100
    assert out.startswith("é") and out.endswith("original length 500 characters]")
    long = ex.sanitize_text("ğ" * 40000)
    assert _utf16(long) <= ex.EXCEL_MAX_CELL_CHARS
    assert long.startswith("ğ" * 30000)
    astral = ex.sanitize_text("😀" * 17000)  # 34000 UTF-16 units
    assert _utf16(astral) <= ex.EXCEL_MAX_CELL_CHARS
    assert "\ud83d" not in astral  # no split surrogate pair
    exact = "a" * ex.EXCEL_MAX_CELL_CHARS
    assert ex.sanitize_text(exact) == exact


def test_excel_value_conversions():
    aware = datetime(2025, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=3)))
    assert ex.excel_value(aware) == datetime(2025, 1, 1, 9, 0)
    assert ex.excel_value(aware).tzinfo is None
    naive = datetime(2025, 1, 1, 12, 0)
    assert ex.excel_value(naive) == naive
    assert ex.excel_value(float("inf")) == "inf"
    assert ex.excel_value(float("nan")) == "nan"
    assert ex.excel_value(1.5) == 1.5
    assert ex.excel_value(True) is True
    assert ex.excel_value(None) is None
    assert ex.excel_value(12345678901) == 12345678901
    assert ex.excel_value(b"caf\xc3\xa9\x01") == "café"
    assert ex.excel_value({"k": "ş"}) == "{'k': 'ş'}"


def test_tz_aware_datetimes_can_be_written(tmp_path):
    from openpyxl import Workbook
    wb = Workbook(write_only=True)
    rows = [(1, datetime(2025, 5, 5, 10, 0, tzinfo=timezone.utc), "ç\x01")]
    count, sheets = ex._write_table(wb, "t", ["id", "ts", "text"], rows)
    out = tmp_path / "t.xlsx"
    wb.save(out)
    values = list(load_workbook(out)["t"].values)
    assert values[1] == (1, datetime(2025, 5, 5, 10, 0), "ç")
    assert (count, sheets) == (1, ["t"])


# --------------------------------------------------------------------------
# never overwrite silently / failures leave no junk
# --------------------------------------------------------------------------

def test_reserved_names_never_collide(tmp_path):
    first = ex._reserve_output_path(str(tmp_path), "20250101_000000")
    Path(first).write_bytes(b"previous export")
    second = ex._reserve_output_path(str(tmp_path), "20250101_000000")
    third = ex._reserve_output_path(str(tmp_path), "20250101_000000")
    assert os.path.basename(first) == "cengbot_database_export_20250101_000000.xlsx"
    assert os.path.basename(second) == "cengbot_database_export_20250101_000000_1.xlsx"
    assert os.path.basename(third) == "cengbot_database_export_20250101_000000_2.xlsx"
    assert Path(first).read_bytes() == b"previous export"


def test_two_exports_in_the_same_second_keep_both(seeded, db_session, tmp_path):
    p1, _ = ex.export_database(output_dir=tmp_path, db_session=db_session)
    p2, _ = ex.export_database(output_dir=tmp_path, db_session=db_session)
    assert p1 != p2
    assert load_workbook(p1).sheetnames == load_workbook(p2).sheetnames


def test_explicit_output_is_not_overwritten(seeded, db_session, tmp_path):
    target = tmp_path / "mine.xlsx"
    target.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        ex.export_database(output_path=str(target), db_session=db_session)
    assert target.read_bytes() == b"keep me"

    path, _ = ex.export_database(output_path=str(target), db_session=db_session,
                                 overwrite=True)
    assert path == str(target)
    assert load_workbook(target).sheetnames == ["raw_data", "export_summary"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mine.xlsx"]


def test_failed_export_leaves_no_files(seeded, db_session, tmp_path, monkeypatch):
    def broken_rows(session, model, chunk_size=1000):
        yield (1,) + (None,) * 25
        raise RuntimeError("database disappeared")

    monkeypatch.setattr(ex, "iter_table_rows", broken_rows)
    with pytest.raises(RuntimeError):
        ex.export_database(output_dir=tmp_path, db_session=db_session)
    assert list(tmp_path.iterdir()) == []

    assert ex.main(["--output-dir", str(tmp_path)]) is False
    assert list(tmp_path.iterdir()) == []


def test_dataframe_helpers(seeded, db_session):
    df = ex.export_raw_data(db_session)
    assert list(df.columns) == ORIGINAL_RAW_COLUMNS
    assert len(df) == 4
    # these used to fail with NameError (models not imported) and silently
    # return an empty frame
    assert len(ex.export_training_data(db_session)) == 1
    assert len(ex.export_user_votes(db_session)) == 1
    assert ex.export_user_analytics(db_session).empty
    assert ex.export_system_metrics(db_session).empty


def test_dataframe_helpers_propagate_errors(db_session, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no such table")

    monkeypatch.setattr(db_session, "execute", broken)
    with pytest.raises(RuntimeError):
        ex.export_raw_data(db_session)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_main_success(seeded, tmp_path, capsys):
    assert ex.main(["--output-dir", str(tmp_path)]) is True
    files = list(tmp_path.glob("cengbot_database_export_*.xlsx"))
    assert len(files) == 1
    out = capsys.readouterr().out
    assert str(files[0]) in out and "raw_data=4" in out
    assert not list(tmp_path.glob(".*"))  # no leftover temp file


def test_main_all_tables_and_explicit_output(seeded, tmp_path):
    target = tmp_path / "sub" / "all.xlsx"
    assert ex.main(["--all-tables", "--output", str(target)]) is True
    assert load_workbook(target).sheetnames == list(ex.TABLES) + ["export_summary"]
    assert ex.main(["--all-tables", "--output", str(target)]) is False  # exists
    assert ex.main(["--all-tables", "--output", str(target), "--overwrite"]) is True


def test_main_rejects_unknown_table(seeded, tmp_path):
    assert ex.main(["--tables", "raw_data,nope", "--output-dir", str(tmp_path)]) is False
    assert list(tmp_path.iterdir()) == []
