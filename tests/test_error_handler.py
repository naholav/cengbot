"""Tests for src/error_handler.py: rotation, dedup of repeated errors, handlers."""

import io
import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

import error_handler as eh
from error_handler import ErrorHandler, ErrorLevel

ROOT = Path(__file__).resolve().parent.parent


def _raise(exc):
    try:
        raise exc
    except Exception as caught:  # noqa: BLE001
        return caught


@pytest.fixture
def make_handler(tmp_path):
    created = []

    def factory(**kwargs):
        name = f"CengBotTest.{uuid.uuid4().hex}"
        kwargs.setdefault("log_file", str(tmp_path / "error.log"))
        handler = ErrorHandler(name, **kwargs)
        _capture_console(handler)  # keep test output quiet
        created.append(handler)
        return handler

    yield factory
    for handler in created:
        for h in list(handler.logger.handlers):
            h.close()
            handler.logger.removeHandler(h)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _flush(handler):
    for h in handler.logger.handlers:
        h.flush()


def _console(handler):
    (console,) = [h for h in handler.logger.handlers if isinstance(h, eh._StdoutHandler)]
    return console


def _capture_console(handler):
    buf = io.StringIO()
    _console(handler).setStream(buf)
    return buf


def test_uses_size_rotating_file_handler(make_handler, tmp_path):
    handler = make_handler()
    file_handlers = [h for h in handler.logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    fh = file_handlers[0]
    assert fh.maxBytes == 10 * 1024 * 1024
    assert fh.backupCount == 5
    assert Path(fh.baseFilename) == tmp_path / "error.log"
    assert fh.level == logging.WARNING


def test_default_rotation_is_10mb_x5_on_logs_error_log(monkeypatch, tmp_path):
    for var in ("ERROR_LOG_FILE", "LOG_DIR", "ERROR_LOG_MAX_BYTES", "ERROR_LOG_BACKUP_COUNT"):
        monkeypatch.delenv(var, raising=False)
    # default target is <root>/logs/error.log (checked without opening it) ...
    assert Path(eh._default_log_file()) == ROOT / "logs" / "error.log"
    # ... with 10 MB x 5 rotation (handler built on a tmp file, never the real log)
    handler = ErrorHandler(f"CengBotTest.{uuid.uuid4().hex}", log_file=str(tmp_path / "e.log"))
    try:
        (fh,) = [h for h in handler.logger.handlers if isinstance(h, RotatingFileHandler)]
        assert (fh.maxBytes, fh.backupCount) == (10 * 1024 * 1024, 5)
        assert fh.stream is None  # delay=True: not opened until the first record
        assert eh.error_handler.max_bytes == 10 * 1024 * 1024
        assert eh.error_handler.backup_count == 5
    finally:
        for h in list(handler.logger.handlers):
            h.close()
            handler.logger.removeHandler(h)


def test_log_file_created_lazily(make_handler, tmp_path):
    make_handler()
    assert not (tmp_path / "error.log").exists()


def test_rotation_bounds_disk_usage(make_handler, tmp_path):
    handler = make_handler(max_bytes=2000, backup_count=2, dedup_window=0)
    for i in range(200):
        handler.handle_error(_raise(RuntimeError(f"boom {i}")), "rotation", ErrorLevel.HIGH)
    _flush(handler)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["error.log", "error.log.1", "error.log.2"]
    for p in tmp_path.iterdir():
        assert p.stat().st_size <= 2000 + 4000  # one record may overshoot the limit


def test_no_duplicate_handlers_when_set_up_repeatedly(make_handler, tmp_path):
    handler = make_handler()
    before = list(handler.logger.handlers)
    handler.setup_logging()
    handler.setup_logging()
    ErrorHandler(handler.logger.name, log_file=str(tmp_path / "other.log"))
    assert handler.logger.handlers == before
    assert len(before) == 2  # one rotating file handler + one console handler


def test_setup_ignores_foreign_handlers(make_handler):
    """A handler added by someone else (e.g. pytest) must not suppress ours."""
    name = f"CengBotTest.{uuid.uuid4().hex}"
    logger = logging.getLogger(name)
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    try:
        handler = ErrorHandler(name, log_file=str(Path(make_handler().log_file).with_name("f.log")))
        assert any(isinstance(h, RotatingFileHandler) for h in handler.logger.handlers)
    finally:
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)


def test_traceback_only_in_file_console_gets_one_line(make_handler, tmp_path):
    """stdout is redirected into logs/<service>.log by the start scripts, so the
    traceback must not be printed there (it is already in error.log)."""
    handler = make_handler()
    console = _capture_console(handler)
    handler.handle_error(_raise(ValueError("bad thing\nsecond line")), "ctx", ErrorLevel.HIGH)
    _flush(handler)
    out = console.getvalue()
    assert out.count("\n") == 1
    assert "HIGH ERROR in ctx: bad thing second line" in out
    assert "Traceback" not in out
    assert "(traceback in error log)" in out
    text = _read(tmp_path / "error.log")
    assert text.count("Traceback (most recent call last)") == 1


def test_no_propagation_to_root_basicconfig_handlers(make_handler):
    """ai_model_worker / llama_model_handler call logging.basicConfig(); a root
    handler must not receive (and re-print) CengBot records."""
    handler = make_handler()
    _capture_console(handler)
    root_buf = io.StringIO()
    root_handler = logging.StreamHandler(root_buf)
    root = logging.getLogger()
    root.addHandler(root_handler)
    try:
        assert handler.logger.propagate is False
        handler.handle_error(_raise(RuntimeError("only once")), "ctx", ErrorLevel.CRITICAL)
    finally:
        root.removeHandler(root_handler)
    assert root_buf.getvalue() == ""


def test_console_level_and_summary_for_repeats(make_handler):
    handler = make_handler()
    console = _capture_console(handler)
    handler.handle_error(_raise(ValueError("low")), "ctx", ErrorLevel.LOW)
    for _ in range(3):
        handler.handle_error(_raise(ValueError("medium")), "ctx", ErrorLevel.MEDIUM)
    lines = console.getvalue().splitlines()
    assert len(lines) == 3  # LOW (INFO) is not printed
    assert all("Traceback" not in l for l in lines)
    assert lines[-1].endswith("[repeated x2 within 300s; traceback suppressed]")


def test_console_follows_current_stdout(make_handler, monkeypatch):
    handler = make_handler()
    _console(handler).setStream(eh.sys.stdout)  # un-pin: follow sys.stdout again
    buf = io.StringIO()
    monkeypatch.setattr(eh.sys, "stdout", buf)
    handler.handle_error(_raise(ValueError("to current stdout")), "ctx", ErrorLevel.HIGH)
    assert "to current stdout" in buf.getvalue()


def test_first_error_full_traceback_then_one_line_summary(make_handler, tmp_path):
    handler = make_handler(dedup_window=300)
    log = tmp_path / "error.log"
    for _ in range(5):
        handler.handle_error(_raise(OSError("DNS lookup failed for huggingface.co")),
                             "model_load_model", ErrorLevel.HIGH)
    _flush(handler)
    text = _read(log)
    assert text.count("Traceback (most recent call last)") == 1
    assert "_raise" in text  # traceback of the actual exception
    summaries = [l for l in text.splitlines() if "repeated x" in l]
    assert len(summaries) == 4
    assert summaries[-1].endswith("[repeated x4 within 300s; traceback suppressed]")
    assert "OSError: DNS lookup failed" in summaries[0]


def test_distinct_errors_are_not_deduplicated(make_handler, tmp_path):
    handler = make_handler()
    handler.handle_error(_raise(ValueError("a")), "ctx", ErrorLevel.HIGH)
    handler.handle_error(_raise(ValueError("b")), "ctx", ErrorLevel.HIGH)
    handler.handle_error(_raise(KeyError("a")), "ctx", ErrorLevel.HIGH)
    handler.handle_error(_raise(ValueError("a")), "other_ctx", ErrorLevel.HIGH)
    _flush(handler)
    assert _read(tmp_path / "error.log").count("Traceback (most recent call last)") == 4


def test_full_traceback_again_after_window(make_handler, tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(eh.time, "monotonic", lambda: clock[0])
    handler = make_handler(dedup_window=60)
    err = _raise(RuntimeError("same"))
    handler.handle_error(err, "ctx", ErrorLevel.HIGH)
    clock[0] += 30
    handler.handle_error(err, "ctx", ErrorLevel.HIGH)
    clock[0] += 61
    handler.handle_error(err, "ctx", ErrorLevel.HIGH)
    _flush(handler)
    text = _read(tmp_path / "error.log")
    assert text.count("Traceback (most recent call last)") == 2
    assert text.count("repeated x1") == 1


def test_summary_is_single_line_and_truncated(make_handler, tmp_path):
    handler = make_handler()
    message = "line1\nline2\n" + "x" * 2000
    for _ in range(2):
        handler.handle_error(_raise(RuntimeError(message)), "ctx", ErrorLevel.HIGH)
    _flush(handler)
    summary = [l for l in _read(tmp_path / "error.log").splitlines() if "repeated x1" in l]
    assert len(summary) == 1
    assert "line1 line2" in summary[0]
    assert len(summary[0]) < 600


def test_dedup_memory_is_bounded(make_handler):
    handler = make_handler()
    for i in range(eh.MAX_TRACKED_ERRORS + 50):
        handler.handle_error(ValueError(f"e{i}"), "ctx", ErrorLevel.LOW)
    assert len(handler._recent) == eh.MAX_TRACKED_ERRORS


def test_error_outside_except_block_has_its_own_traceback(make_handler, tmp_path):
    handler = make_handler()
    err = _raise(ZeroDivisionError("division"))  # handled later, outside except
    handler.handle_error(err, "ctx", ErrorLevel.CRITICAL)
    _flush(handler)
    text = _read(tmp_path / "error.log")
    assert "ZeroDivisionError: division" in text
    assert "NoneType: None" not in text


def test_medium_traceback_in_file_low_not(make_handler, tmp_path):
    handler = make_handler()
    _capture_console(handler)
    handler.handle_error(_raise(ValueError("medium")), "ctx", ErrorLevel.MEDIUM)
    handler.handle_error(_raise(ValueError("low")), "ctx", ErrorLevel.LOW)
    _flush(handler)
    text = _read(tmp_path / "error.log")
    assert "MEDIUM ERROR in ctx: medium" in text
    assert "Traceback (most recent call last)" in text
    assert "low" not in text


def test_response_shape_unchanged(make_handler):
    handler = make_handler()
    resp = handler.handle_error(_raise(ValueError("bad")), "ctx", ErrorLevel.HIGH,
                                user_message="nice message")
    assert resp["success"] is False
    assert resp["error"]["type"] == "ValueError"
    assert resp["error"]["message"] == "nice message"
    assert resp["error"]["context"] == "ctx"
    assert resp["error"]["level"] == "high"
    assert "timestamp" in resp["error"]
    # repeated call returns the same shape (only logging is deduplicated)
    resp2 = handler.handle_error(_raise(ValueError("bad")), "ctx", ErrorLevel.HIGH)
    assert resp2["error"]["message"] == "bad"


def test_specialised_helpers(make_handler):
    handler = make_handler()
    assert handler.handle_database_error(ValueError("x"), "select", "raw_data")["error"]["context"] == "database_select_raw_data"
    assert handler.handle_api_error(ValueError("x"), "/raw-data", "POST")["error"]["context"] == "api_post__raw-data"
    assert handler.handle_model_error(ValueError("x"))["error"]["level"] == "high"
    assert handler.handle_training_error(ValueError("x"), "train")["error"]["level"] == "critical"
    assert handler.handle_telegram_error(ValueError("x"), 1, 2)["error"]["context"] == "telegram_bot"
    v = handler.handle_validation_error("age", "abc", "int")
    assert v["error"]["type"] == "ValidationError" and v["error"]["field"] == "age"


def test_env_overrides_for_log_file_and_rotation(monkeypatch, tmp_path):
    monkeypatch.setenv("ERROR_LOG_FILE", str(tmp_path / "custom" / "err.log"))
    monkeypatch.setenv("ERROR_LOG_MAX_BYTES", "1234")
    monkeypatch.setenv("ERROR_LOG_BACKUP_COUNT", "2")
    monkeypatch.setenv("ERROR_DEDUP_WINDOW", "7")
    name = f"CengBotTest.{uuid.uuid4().hex}"
    handler = ErrorHandler(name)
    try:
        fh = handler.logger.handlers[0]
        assert Path(fh.baseFilename) == tmp_path / "custom" / "err.log"
        assert (fh.maxBytes, fh.backupCount) == (1234, 2)
        assert handler.dedup_window == 7
    finally:
        for h in list(handler.logger.handlers):
            h.close()
            handler.logger.removeHandler(h)


def test_default_log_file_is_project_logs_dir(monkeypatch):
    monkeypatch.delenv("ERROR_LOG_FILE", raising=False)
    monkeypatch.delenv("LOG_DIR", raising=False)
    assert Path(eh._default_log_file()) == ROOT / "logs" / "error.log"
    monkeypatch.setenv("LOG_DIR", "var/log")
    assert Path(eh._default_log_file()) == ROOT / "var" / "log" / "error.log"


def test_module_convenience_functions_exist():
    for name in ("handle_error", "handle_validation_error", "handle_database_error",
                 "handle_api_error", "handle_model_error", "handle_telegram_error",
                 "handle_training_error"):
        assert callable(getattr(eh, name))
    assert isinstance(eh.error_handler, ErrorHandler)
