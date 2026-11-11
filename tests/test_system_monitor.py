"""Tests for src/system_monitor.py (psutil, nvidia-smi and the DB are mocked
or pointed at temp locations; big log files are sparse, never written)."""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

import system_monitor as sm
from database_models import RawData, TrainingData

MB = 1024 ** 2
GB = 1024 ** 3

DiskUsage = namedtuple("DiskUsage", "total used free percent")
VirtualMemory = namedtuple("VirtualMemory", "total available percent used free")


@pytest.fixture(autouse=True)
def _reset_monitor_logger():
    yield
    logger = logging.getLogger("SystemMonitor")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def logs_dir(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def plenty_of_disk(monkeypatch):
    calls = []

    def fake_disk_usage(path):
        calls.append(path)
        return DiskUsage(total=400 * GB, used=100 * GB, free=300 * GB, percent=25.0)

    monkeypatch.setattr(sm.psutil, "disk_usage", fake_disk_usage)
    return calls


@pytest.fixture
def monitor(logs_dir):
    return sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=False,
                            expected_services=[])


def sparse_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)
    return path


def make_system_metrics(**overrides):
    values = dict(timestamp=datetime.now(), cpu_percent=10.0, memory_percent=20.0,
                  memory_available_gb=30.0, disk_usage_percent=40.0, disk_free_gb=100.0,
                  gpu_available=False, disk_path="/data")
    values.update(overrides)
    return sm.SystemMetrics(**values)


# --------------------------------------------------------------------------
# import / construction
# --------------------------------------------------------------------------

def test_paths_are_relative_to_project_not_hardcoded():
    root = Path(__file__).resolve().parent.parent
    assert sm.PROJECT_ROOT == root
    assert sm.DEFAULT_LOGS_DIR == root / "logs"
    assert "/home/ceng/cu_ceng_bot/" not in Path(sm.__file__).read_text(encoding="utf-8")


def test_constructor_creates_nothing_until_first_write(logs_dir):
    mon = sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=False)
    assert not (logs_dir / "monitoring").exists()
    mon.log_alert({"timestamp": datetime.now().isoformat(), "type": "x"})
    assert (logs_dir / "monitoring" / "alerts.jsonl").exists()


def test_global_monitor_is_lazy(monkeypatch):
    created = []

    class FakeMonitor:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(sm, "_monitor", None)
    monkeypatch.setattr(sm, "SystemMonitor", FakeMonitor)
    assert created == []
    first = sm.monitor
    assert sm.monitor is first
    assert len(created) == 1
    with pytest.raises(AttributeError):
        sm.does_not_exist  # noqa: B018


def test_file_logging_rotates_and_does_not_propagate(logs_dir):
    mon = sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=True)
    handlers = [h for h in mon.logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.handlers.RotatingFileHandler)
    assert handlers[0].maxBytes > 0
    assert mon.logger.propagate is False
    # a second instance for the same file must not add a duplicate handler
    sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=True)
    assert len([h for h in mon.logger.handlers if isinstance(h, logging.FileHandler)]) == 1


# --------------------------------------------------------------------------
# log size check (decision h)
# --------------------------------------------------------------------------

def test_scan_log_files_uses_stat_only(logs_dir, monkeypatch):
    sparse_file(logs_dir / "worker.log", 900 * MB)
    sparse_file(logs_dir / "train" / "nested.log", 600 * MB)
    sparse_file(logs_dir / "bot.log", 10 * MB)
    (logs_dir / "link.log").symlink_to(logs_dir / "worker.log")

    def no_open(*args, **kwargs):
        raise AssertionError("log files must never be opened")

    monkeypatch.setattr("builtins.open", no_open)
    monkeypatch.setattr(os, "open", no_open)
    report = sm.scan_log_files(logs_dir, 500 * MB)

    assert report["exists"] is True
    assert report["file_count"] == 3  # symlink ignored
    assert report["total_bytes"] == (900 + 600 + 10) * MB
    assert [Path(i["path"]).name for i in report["oversized"]] == ["worker.log", "nested.log"]
    assert report["oversized"][0]["size_bytes"] == 900 * MB


def test_scan_log_files_missing_dir(tmp_path):
    report = sm.scan_log_files(tmp_path / "nope", 1)
    assert report["exists"] is False
    assert report["oversized"] == [] and report["total_bytes"] == 0


def test_log_size_threshold_from_env(monkeypatch, logs_dir):
    monkeypatch.setenv("MONITOR_LOG_SIZE_THRESHOLD_MB", "100")
    monkeypatch.setenv("MONITOR_MIN_FREE_DISK_GB", "7.5")
    mon = sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=False)
    assert mon.alert_thresholds["log_file_size_mb"] == 100.0
    assert mon.alert_thresholds["disk_free_gb"] == 7.5
    # explicit argument wins over env
    mon = sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=False,
                           log_size_threshold_mb=50)
    assert mon.alert_thresholds["log_file_size_mb"] == 50.0


@pytest.mark.parametrize("raw", ["abc", "-5", "0", "nan"])
def test_invalid_env_threshold_falls_back_to_default(monkeypatch, logs_dir, raw):
    monkeypatch.setenv("MONITOR_LOG_SIZE_THRESHOLD_MB", raw)
    mon = sm.SystemMonitor(logs_dir=logs_dir, enable_file_logging=False)
    assert mon.alert_thresholds["log_file_size_mb"] == sm.DEFAULT_LOG_SIZE_THRESHOLD_MB


def test_check_logs_and_disk_alerts(logs_dir, monitor, monkeypatch):
    big = sparse_file(logs_dir / "worker.log", 2 * GB)
    sparse_file(logs_dir / "error.log", 100 * MB)
    monitor.alert_thresholds["logs_total_gb"] = 1.0
    seen_paths = []

    def low_disk(path):
        seen_paths.append(path)
        return DiskUsage(total=100 * GB, used=99 * GB, free=int(0.5 * GB), percent=99.0)

    monkeypatch.setattr(sm.psutil, "disk_usage", low_disk)
    result = monitor.check_logs_and_disk()

    types_ = sorted(a["type"] for a in result["alerts"])
    assert types_ == ["disk_high", "disk_low", "log_file_large", "logs_total_high"]
    large = [a for a in result["alerts"] if a["type"] == "log_file_large"]
    assert large[0]["path"] == str(big)
    # disk figures refer to the partition holding the logs, not "/"
    assert seen_paths == [str(logs_dir)]
    assert result["disk"]["free_gb"] == pytest.approx(0.5)


def test_check_logs_and_disk_ok(logs_dir, monitor, plenty_of_disk):
    sparse_file(logs_dir / "bot.log", 1 * MB)
    result = monitor.check_logs_and_disk()
    assert result["alerts"] == []


def test_cli_check_logs_exit_codes_and_read_only(logs_dir, plenty_of_disk, capsys):
    sparse_file(logs_dir / "bot.log", 1 * MB)
    assert sm.main(["--check-logs", "--logs-dir", str(logs_dir)]) == 0
    assert "OK" in capsys.readouterr().out

    big = sparse_file(logs_dir / "worker.log", 600 * MB)
    assert sm.main(["--check-logs", "--logs-dir", str(logs_dir)]) == 1
    out = capsys.readouterr().out
    assert str(big) in out and "ALERT" in out

    assert sm.main(["--check-logs", "--logs-dir", str(logs_dir),
                    "--log-size-threshold-mb", "1000", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["alerts"] == []

    # one-shot mode must not create files in the log directory
    assert sorted(p.name for p in logs_dir.iterdir()) == ["bot.log", "worker.log"]


def test_cli_once_prints_json_status(logs_dir, plenty_of_disk, db_session, monkeypatch, capsys):
    monkeypatch.setattr(sm.psutil, "cpu_percent", lambda interval=None: 5.0)
    monkeypatch.setattr(sm.psutil, "virtual_memory",
                        lambda: VirtualMemory(64 * GB, 60 * GB, 6.0, 4 * GB, 60 * GB))
    monkeypatch.setattr(sm, "query_gpu_stats", lambda: None)
    monkeypatch.setattr(sm, "find_service_processes",
                        lambda: {"ai_worker": [1], "telegram_bot": [2], "admin_api": [3]})

    assert sm.main(["--once", "--logs-dir", str(logs_dir)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["system"]["cpu_percent"] == 5.0
    assert status["database"]["raw_data_count"] == 0
    assert status["alerts"] == []
    assert list(logs_dir.iterdir()) == []

    sparse_file(logs_dir / "worker.log", 600 * MB)
    assert sm.main(["--once", "--logs-dir", str(logs_dir)]) == 1
    status = json.loads(capsys.readouterr().out)
    assert [a["type"] for a in status["alerts"]] == ["log_file_large"]


def test_cli_rejects_bad_interval():
    with pytest.raises(SystemExit):
        sm.main(["--interval", "0"])


# --------------------------------------------------------------------------
# system metrics / GPU
# --------------------------------------------------------------------------

def test_collect_system_metrics_non_blocking_cpu(monitor, monkeypatch, plenty_of_disk, logs_dir):
    cpu_calls = []

    def fake_cpu(interval=None):
        cpu_calls.append(interval)
        return 42.0

    monkeypatch.setattr(sm.psutil, "cpu_percent", fake_cpu)
    monkeypatch.setattr(sm.psutil, "virtual_memory",
                        lambda: VirtualMemory(64 * GB, 16 * GB, 75.0, 48 * GB, 16 * GB))
    monkeypatch.setattr(sm, "query_gpu_stats", lambda: {
        "count": 1, "memory_used_gb": 12.0, "memory_total_gb": 24.0, "utilization_percent": 50.0})
    monitor._cpu_sampled_at = time.monotonic() - 60

    metrics = monitor.collect_system_metrics()

    assert cpu_calls == [None]  # no 1-second blocking sample
    assert metrics.cpu_percent == 42.0
    assert metrics.memory_percent == 75.0
    assert metrics.memory_available_gb == pytest.approx(16.0)
    assert metrics.disk_free_gb == pytest.approx(300.0)
    assert metrics.disk_path == str(logs_dir)
    assert metrics.gpu_available is True
    assert metrics.gpu_memory_percent == pytest.approx(50.0)
    assert json.loads(json.dumps(metrics.to_dict()))["gpu_memory_total"] == 24.0


def test_collect_system_metrics_error_returns_none(monitor, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("psutil broke")

    monkeypatch.setattr(sm.psutil, "virtual_memory", boom)
    assert monitor.collect_system_metrics() is None


def _completed(stdout, returncode=0):
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=returncode,
                                       stdout=stdout, stderr="")


def test_query_gpu_stats_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(sm.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    runs = []

    def fake_run(cmd, **kwargs):
        runs.append((cmd, kwargs))
        return _completed("15, 24576, 0\n1009, 24576, [N/A]\n")

    monkeypatch.setattr(sm.subprocess, "run", fake_run)
    stats = sm.query_gpu_stats()
    assert stats["count"] == 2
    assert stats["memory_used_gb"] == pytest.approx(1024 / 1024)
    assert stats["memory_total_gb"] == pytest.approx(48.0)
    assert stats["utilization_percent"] == 0.0
    assert runs[0][1]["timeout"] > 0


def test_query_gpu_stats_never_initialises_cuda(monkeypatch):
    """No nvidia-smi + torch imported but CUDA not initialised -> None, and
    no CUDA call is made from the monitor process."""
    monkeypatch.setattr(sm.shutil, "which", lambda name: None)
    cuda = mock.Mock()
    cuda.is_initialized.return_value = False
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert sm.query_gpu_stats() is None
    cuda.mem_get_info.assert_not_called()
    cuda.is_available.assert_not_called()
    cuda.synchronize.assert_not_called()


def test_query_gpu_stats_uses_existing_cuda_context(monkeypatch):
    monkeypatch.setattr(sm.shutil, "which", lambda name: None)
    cuda = mock.Mock()
    cuda.is_initialized.return_value = True
    cuda.mem_get_info.return_value = (20 * GB, 24 * GB)
    cuda.device_count.return_value = 1
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    stats = sm.query_gpu_stats()
    assert stats["memory_used_gb"] == pytest.approx(4.0)
    assert stats["memory_total_gb"] == pytest.approx(24.0)


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
    FileNotFoundError("gone"),
])
def test_query_gpu_stats_handles_nvidia_smi_failures(monkeypatch, failure):
    monkeypatch.setattr(sm.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def fake_run(*a, **k):
        raise failure

    monkeypatch.setattr(sm.subprocess, "run", fake_run)
    assert sm.query_gpu_stats() is None


def test_query_gpu_stats_nonzero_exit(monkeypatch):
    monkeypatch.setattr(sm.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sm.subprocess, "run", lambda *a, **k: _completed("", returncode=9))
    assert sm.query_gpu_stats() is None


# --------------------------------------------------------------------------
# database metrics
# --------------------------------------------------------------------------

def test_collect_database_metrics_counts(db_session, monitor):
    rows = [
        RawData(telegram_id=1, question="Ders kaydı ne zaman?", answer="a", admin_approved=1),
        RawData(telegram_id=2, question="q2", admin_approved=0),
        RawData(telegram_id=3, question="q3", admin_approved=0, is_duplicate=True),
        RawData(telegram_id=4, question="q4", admin_approved=1, is_duplicate=True),
    ]
    db_session.add_all(rows)
    db_session.flush()
    db_session.add(TrainingData(source_id=rows[0].id, question="q", answer="a"))
    db_session.commit()

    metrics = monitor.collect_database_metrics()
    assert metrics.raw_data_count == 4
    assert metrics.approved_count == 2
    assert metrics.pending_count == 2
    assert metrics.duplicate_count == 2
    assert metrics.training_data_count == 1


def test_collect_database_metrics_empty_db(db_session, monitor):
    metrics = monitor.collect_database_metrics()
    assert (metrics.raw_data_count, metrics.approved_count, metrics.pending_count,
            metrics.duplicate_count, metrics.training_data_count) == (0, 0, 0, 0, 0)


def test_collect_database_metrics_closes_session_on_error(monitor, monkeypatch):
    session = mock.Mock()
    session.query.side_effect = RuntimeError("db is locked")
    monkeypatch.setattr(sm, "SessionLocal", lambda: session)
    assert monitor.collect_database_metrics() is None
    session.close.assert_called_once()


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_repeated_errors_are_rate_limited(monitor, monkeypatch):
    session = mock.Mock()
    session.query.side_effect = RuntimeError("db is locked")
    monkeypatch.setattr(sm, "SessionLocal", lambda: session)
    handler = ListHandler()
    monitor.logger.addHandler(handler)
    for _ in range(5):
        assert monitor.collect_database_metrics() is None
    errors = [r for r in handler.records if "database metrics" in r.getMessage()]
    assert len(errors) == 1
    assert session.close.call_count == 5


# --------------------------------------------------------------------------
# process discovery
# --------------------------------------------------------------------------

class FakeProc:
    def __init__(self, pid, cmdline, error=None):
        self._info = {"pid": pid, "cmdline": cmdline}
        self._error = error

    @property
    def info(self):
        if self._error is not None:
            raise self._error
        return self._info


def test_find_service_processes(monkeypatch):
    procs = [
        FakeProc(10, ["python3", "src/ai_model_worker.py"]),
        FakeProc(11, ["/x/.venv/bin/python", "-u", "/x/src/telegram_bot_rabbitmq.py"]),
        FakeProc(12, ["vim", "src/admin_rest_api.py"]),             # editor, not a service
        FakeProc(13, ["tail", "-f", "logs/ai_model_worker.py"]),     # not python
        FakeProc(14, None),                                          # AccessDenied -> None
        FakeProc(15, ["python3", "x.py"], error=sm.psutil.NoSuchProcess(15)),
        FakeProc(16, ["python3", "y.py"], error=sm.psutil.AccessDenied(16)),
        FakeProc(17, ["python3.11", "-m", "admin_rest_api"]),
        FakeProc(99, ["python3", "src/ai_model_worker.py"]),         # ourselves
    ]
    monkeypatch.setattr(sm.psutil, "process_iter", lambda attrs=None: iter(procs))
    found = sm.find_service_processes(exclude_pid=99)
    assert found == {"ai_worker": [10], "telegram_bot": [11], "admin_api": [17]}


def test_service_down_alert(monitor):
    monitor.expected_services = ["ai_worker", "admin_api"]
    alerts = monitor.evaluate_alerts(None, None, None,
                                     {"ai_worker": [], "admin_api": [5], "telegram_bot": []})
    assert [(a["type"], a["service"]) for a in alerts] == [("service_down", "ai_worker")]
    # unknown status (scan failed) -> no alert
    assert monitor.evaluate_alerts(None, None, None, None) == []


# --------------------------------------------------------------------------
# alerts / JSONL files
# --------------------------------------------------------------------------

def test_alert_thresholds(monitor):
    metrics = make_system_metrics(cpu_percent=95.0, memory_percent=90.0,
                                  disk_usage_percent=95.0, disk_free_gb=1.0,
                                  gpu_available=True, gpu_memory_used=23.0,
                                  gpu_memory_total=24.0)
    db = sm.DatabaseMetrics(timestamp=datetime.now(), raw_data_count=500,
                            training_data_count=0, approved_count=0, pending_count=101,
                            duplicate_count=0)
    types_ = {a["type"] for a in monitor.evaluate_alerts(metrics, db)}
    assert types_ == {"cpu_high", "memory_high", "disk_high", "disk_low",
                      "gpu_memory_high", "pending_high"}

    calm = make_system_metrics()
    assert monitor.evaluate_alerts(calm, None) == []


def test_check_alerts_writes_valid_jsonl_with_cooldown(monitor):
    hot = make_system_metrics(cpu_percent=99.0)
    for _ in range(3):
        alerts = monitor.check_alerts(hot, None)
        assert [a["type"] for a in alerts] == ["cpu_high"]

    lines = monitor.alerts_log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # re-logged only after the cooldown
    assert json.loads(lines[0])["type"] == "cpu_high"

    monitor.alert_cooldown_seconds = 0
    monitor.check_alerts(hot, None)
    lines = monitor.alerts_log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(line) for line in lines)


def test_log_file_alerts_are_per_path(monitor):
    report = {"logs_dir": "/l", "total_bytes": 0, "oversized": [
        {"path": "/l/a.log", "size_bytes": 2 * GB, "size_mb": 2048.0},
        {"path": "/l/b.log", "size_bytes": 1 * GB, "size_mb": 1024.0},
    ]}
    monitor.check_alerts(None, None, report)
    lines = monitor.alerts_log_file.read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["path"] for line in lines) == ["/l/a.log", "/l/b.log"]


def test_log_metrics_writes_one_json_object_per_line(monitor):
    db = sm.DatabaseMetrics(datetime.now(), 1, 2, 3, 4, 5)
    monitor.log_metrics(make_system_metrics(), db)
    monitor.log_metrics(make_system_metrics(cpu_percent=11.0), None)
    sys_lines = monitor.system_log_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["cpu_percent"] for line in sys_lines] == [10.0, 11.0]
    db_lines = monitor.database_log_file.read_text(encoding="utf-8").splitlines()
    assert json.loads(db_lines[0])["duplicate_count"] == 5


def test_cleanup_old_logs(monitor):
    monitor.monitoring_dir.mkdir(parents=True)
    old = (datetime.now() - timedelta(days=monitor.max_log_age_days + 1)).isoformat()
    new = datetime.now().isoformat()
    monitor.system_log_file.write_text(
        json.dumps({"timestamp": old, "v": 1}) + "\n"
        + "not json\n"
        + json.dumps({"timestamp": new, "v": 2}) + "\n"
        + json.dumps({"no_timestamp": True}) + "\n"
        + json.dumps({"timestamp": new, "v": 3}),  # last line without newline
        encoding="utf-8")

    monitor.cleanup_old_logs()

    lines = monitor.system_log_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["v"] for line in lines] == [2, 3]
    assert not list(monitor.monitoring_dir.glob("*.tmp"))


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------

def test_run_checks_persists_and_cleans_up_once(monitor, monkeypatch, plenty_of_disk):
    monkeypatch.setattr(monitor, "collect_system_metrics", lambda: make_system_metrics(cpu_percent=99.0))
    monkeypatch.setattr(monitor, "collect_database_metrics", lambda: None)
    monkeypatch.setattr(monitor, "collect_service_status", lambda: {})
    cleanups = []
    monkeypatch.setattr(monitor, "cleanup_old_logs", lambda: cleanups.append(1))

    result = monitor.run_checks()
    monitor.run_checks()

    assert [a["type"] for a in result["alerts"]] == ["cpu_high"]
    assert len(monitor.system_log_file.read_text(encoding="utf-8").splitlines()) == 2
    assert cleanups == [1]  # once per day, not on every iteration


def test_stop_interrupts_the_wait(monitor, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(monitor, "run_checks", lambda: ran.set())
    monitor.log_interval = 3600
    monitor.start()
    assert ran.wait(5)
    started = time.monotonic()
    monitor.stop()
    assert time.monotonic() - started < 3
    assert not monitor.monitor_thread.is_alive()
    assert monitor.running is False


def test_loop_survives_errors(monitor, monkeypatch):
    calls = []

    def flaky():
        calls.append(1)
        raise RuntimeError("transient")

    monkeypatch.setattr(monitor, "run_checks", flaky)
    monitor.log_interval = 3600
    monitor.start()
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    monitor.stop()
    assert calls and not monitor.monitor_thread.is_alive()
