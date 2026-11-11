#!/usr/bin/env python3
"""
System monitoring and logging for CengBot
=========================================

Collects host metrics (CPU, memory, disk, GPU), database statistics, the
state of the CengBot service processes and the size of the files in
``logs/``; writes metrics/alerts as JSON lines to ``logs/monitoring/`` and
raises (rate-limited) alerts when thresholds are exceeded.

Usage::

    python src/system_monitor.py                 # run the monitoring loop (service mode)
    python src/system_monitor.py --once          # one JSON status report, then exit
    python src/system_monitor.py --check-logs    # only log sizes + free disk, then exit

``--once`` and ``--check-logs`` exit with status 1 when an alert is active,
so they can be used from cron or ``scripts/health_check.sh``.

Configuration (environment, all optional):

    MONITOR_LOG_SIZE_THRESHOLD_MB    alert when one file in logs/ is bigger (default 500)
    MONITOR_LOGS_TOTAL_THRESHOLD_GB  alert when logs/ as a whole is bigger (default 5)
    MONITOR_MIN_FREE_DISK_GB         alert when free disk space is lower (default 2)

Log files are only ``stat``-ed, never opened, so multi-GB logs are cheap to check.
GPU statistics come from ``nvidia-smi``; this process never initialises CUDA.
"""

import argparse
import json
import logging
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import psutil

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sqlalchemy import case, func  # noqa: E402

# Database imports
from database_models import SessionLocal, RawData, TrainingData  # noqa: E402

PROJECT_ROOT = SRC_DIR.parent
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_LOG_SIZE_THRESHOLD_MB = 500.0
DEFAULT_LOGS_TOTAL_THRESHOLD_GB = 5.0
DEFAULT_MIN_FREE_DISK_GB = 2.0
DEFAULT_ALERT_COOLDOWN_SECONDS = 30 * 60
CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60

MONITOR_LOG_MAX_BYTES = 10 * 1024 * 1024
MONITOR_LOG_BACKUP_COUNT = 3

GB = 1024 ** 3
MB = 1024 ** 2

# Service name -> script file names that identify its process.
SERVICE_SCRIPTS: Dict[str, tuple] = {
    "ai_worker": ("ai_model_worker.py",),
    "telegram_bot": ("telegram_bot_rabbitmq.py", "telegram_bot.py"),
    "admin_api": ("admin_rest_api.py",),
}


def _env_float(name: str, default: float) -> float:
    """Positive float from the environment, or ``default`` if unset/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.getLogger("SystemMonitor").warning(
            "Ignoring invalid %s=%r (using %s)", name, raw, default
        )
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def _format_bytes(size: float) -> str:
    for unit, factor in (("GB", GB), ("MB", MB), ("KB", 1024)):
        if size >= factor:
            return f"{size / factor:.1f} {unit}"
    return f"{int(size)} B"


@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_available: bool
    gpu_memory_used: Optional[float] = None   # GB, whole device (all processes)
    gpu_memory_total: Optional[float] = None  # GB
    gpu_utilization: Optional[float] = None   # percent
    disk_path: Optional[str] = None           # mount point / path the disk figures refer to

    @property
    def gpu_memory_percent(self) -> Optional[float]:
        if self.gpu_memory_used is None or not self.gpu_memory_total:
            return None
        return self.gpu_memory_used / self.gpu_memory_total * 100.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_gb": self.memory_available_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "gpu_available": self.gpu_available,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
            "gpu_utilization": self.gpu_utilization,
            "disk_path": self.disk_path,
        }


@dataclass
class DatabaseMetrics:
    """Database metrics data structure"""
    timestamp: datetime
    raw_data_count: int
    training_data_count: int
    approved_count: int
    pending_count: int
    duplicate_count: int

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "raw_data_count": self.raw_data_count,
            "training_data_count": self.training_data_count,
            "approved_count": self.approved_count,
            "pending_count": self.pending_count,
            "duplicate_count": self.duplicate_count
        }


def scan_log_files(logs_dir, threshold_bytes: float) -> Dict:
    """Size report for every regular file below ``logs_dir``.

    Uses ``os.lstat`` only -- files are never opened or read, so this is
    cheap even for logs that are tens of gigabytes. Symlinks are ignored.
    """
    logs_dir = Path(logs_dir)
    report = {
        "logs_dir": str(logs_dir),
        "exists": logs_dir.is_dir(),
        "threshold_bytes": int(threshold_bytes),
        "total_bytes": 0,
        "file_count": 0,
        "oversized": [],
    }
    if not report["exists"]:
        return report

    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                st = os.lstat(path)
            except OSError:
                continue  # vanished (rotated) or unreadable directory entry
            if not stat.S_ISREG(st.st_mode):
                continue
            report["total_bytes"] += st.st_size
            report["file_count"] += 1
            if st.st_size > threshold_bytes:
                report["oversized"].append({
                    "path": path,
                    "size_bytes": st.st_size,
                    "size_mb": round(st.st_size / MB, 1),
                })

    report["oversized"].sort(key=lambda item: item["size_bytes"], reverse=True)
    return report


def _script_names_in_cmdline(cmdline: List[str]) -> set:
    """Script basenames a Python interpreter command line is running."""
    if not cmdline or not os.path.basename(str(cmdline[0])).lower().startswith("python"):
        return set()
    names = set()
    args = [str(a) for a in cmdline[1:]]
    for i, arg in enumerate(args):
        if arg.endswith(".py"):
            names.add(os.path.basename(arg))
        elif arg == "-m" and i + 1 < len(args):
            names.add(args[i + 1].rsplit(".", 1)[-1] + ".py")
    return names


def find_service_processes(services: Optional[Dict[str, Iterable[str]]] = None,
                           exclude_pid: Optional[int] = None) -> Dict[str, List[int]]:
    """PIDs of the CengBot service processes, by service name.

    A process matches when its command line is a Python interpreter running
    one of the service's scripts (``python src/ai_model_worker.py``), so
    ``tail -f logs/ai_model_worker.log`` or an editor do not count.
    Processes that vanish or are not accessible while scanning are skipped.
    """
    services = SERVICE_SCRIPTS if services is None else services
    exclude_pid = os.getpid() if exclude_pid is None else exclude_pid
    found: Dict[str, List[int]] = {name: [] for name in services}

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            info = proc.info
            pid = info.get("pid")
            cmdline = info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if pid == exclude_pid:
            continue
        scripts = _script_names_in_cmdline(cmdline)
        if not scripts:
            continue
        for name, script_names in services.items():
            if scripts.intersection(script_names):
                found[name].append(pid)
    return found


def query_gpu_stats(timeout: float = 5.0) -> Optional[Dict]:
    """Device-wide GPU memory/utilisation, or None when no GPU is visible.

    Uses ``nvidia-smi`` so the monitor never creates a CUDA context (which
    would cost ~300 MB of GPU memory and could race with the AI worker).
    Falls back to torch only if torch is already imported *and* CUDA is
    already initialised in this process.
    """
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run(
                [smi, "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0:
            used_mb = total_mb = 0.0
            utilisation = []
            count = 0
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    used, total = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                used_mb += used
                total_mb += total
                count += 1
                if len(parts) > 2:
                    try:
                        utilisation.append(float(parts[2]))
                    except ValueError:  # "[N/A]" on some boards
                        pass
            if count:
                return {
                    "count": count,
                    "memory_used_gb": used_mb / 1024.0,
                    "memory_total_gb": total_mb / 1024.0,
                    "utilization_percent": max(utilisation) if utilisation else None,
                }

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info()
                return {
                    "count": torch.cuda.device_count(),
                    "memory_used_gb": (total - free) / GB,
                    "memory_total_gb": total / GB,
                    "utilization_percent": None,
                }
        except Exception:
            pass
    return None


class SystemMonitor:
    """System monitoring and logging class"""

    def __init__(self, log_interval: int = 60, max_log_age_days: int = 30,
                 logs_dir=None, log_size_threshold_mb: Optional[float] = None,
                 alert_thresholds: Optional[Dict] = None,
                 alert_cooldown_seconds: float = DEFAULT_ALERT_COOLDOWN_SECONDS,
                 expected_services: Optional[Iterable[str]] = None,
                 enable_file_logging: bool = True):
        self.log_interval = log_interval
        self.max_log_age_days = max_log_age_days
        self.running = False
        self.monitor_thread = None
        self._stop_event = threading.Event()

        self.logs_dir = Path(logs_dir) if logs_dir else DEFAULT_LOGS_DIR
        self.enable_file_logging = enable_file_logging

        # Setup logging
        self.setup_logging()

        # Monitoring directory (created on first write, so read-only uses such
        # as --once / --check-logs never create files)
        self.monitoring_dir = self.logs_dir / "monitoring"

        # Log files
        self.system_log_file = self.monitoring_dir / "system_metrics.jsonl"
        self.database_log_file = self.monitoring_dir / "database_metrics.jsonl"
        self.alerts_log_file = self.monitoring_dir / "alerts.jsonl"

        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "disk_free_gb": _env_float("MONITOR_MIN_FREE_DISK_GB", DEFAULT_MIN_FREE_DISK_GB),
            "gpu_memory_percent": 90.0,
            "pending_count": 100,
            "log_file_size_mb": (
                float(log_size_threshold_mb) if log_size_threshold_mb
                else _env_float("MONITOR_LOG_SIZE_THRESHOLD_MB", DEFAULT_LOG_SIZE_THRESHOLD_MB)
            ),
            "logs_total_gb": _env_float("MONITOR_LOGS_TOTAL_THRESHOLD_GB",
                                        DEFAULT_LOGS_TOTAL_THRESHOLD_GB),
        }
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)

        # The same alert/error is logged at most once per cooldown period so a
        # persistent condition cannot flood the logs (cf. the 124 GB of logs).
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self._last_logged: Dict[str, float] = {}
        self.expected_services = (
            list(SERVICE_SCRIPTS) if expected_services is None else list(expected_services)
        )
        self._last_cleanup: Optional[float] = None

        # Prime psutil so later cpu_percent(interval=None) calls are non-blocking
        # and measure the whole interval since the previous sample.
        psutil.cpu_percent(interval=None)
        self._cpu_sampled_at = time.monotonic()

        self.logger.info("System monitor initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("SystemMonitor")
        self.logger.setLevel(logging.INFO)
        # Other modules (e.g. the services' basicConfig) may configure the root logger; don't log everything twice.
        self.logger.propagate = False

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if not any(type(h) is logging.StreamHandler for h in self.logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.enable_file_logging:
            log_file = os.path.abspath(str(self.logs_dir / "system_monitor.log"))
            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file
                       for h in self.logger.handlers):
                try:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    file_handler = RotatingFileHandler(
                        log_file, maxBytes=MONITOR_LOG_MAX_BYTES,
                        backupCount=MONITOR_LOG_BACKUP_COUNT, encoding="utf-8", delay=True,
                    )
                except OSError as e:
                    self.logger.warning(f"File logging disabled ({log_file}): {e}")
                else:
                    file_handler.setLevel(logging.INFO)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)

    def _should_log(self, key: str) -> bool:
        """True if ``key`` was not logged within the cooldown period."""
        now = time.monotonic()
        last = self._last_logged.get(key)
        if last is not None and now - last < self.alert_cooldown_seconds:
            return False
        self._last_logged[key] = now
        return True

    def _log_error(self, key: str, message: str):
        if self._should_log(f"error:{key}"):
            self.logger.error(message)

    def _disk_path(self) -> Path:
        """The partition that holds the logs and the database (not '/')."""
        return self.logs_dir if self.logs_dir.exists() else PROJECT_ROOT

    def _cpu_percent(self) -> float:
        elapsed = time.monotonic() - self._cpu_sampled_at
        # Non-blocking unless the previous sample is too recent to be meaningful.
        value = psutil.cpu_percent(interval=None if elapsed >= 0.5 else 0.5)
        self._cpu_sampled_at = time.monotonic()
        return value

    def collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system metrics"""
        try:
            # CPU and Memory
            cpu_percent = self._cpu_percent()
            memory = psutil.virtual_memory()
            disk_path = self._disk_path()
            disk = psutil.disk_usage(str(disk_path))

            # GPU metrics (device-wide, via nvidia-smi; no CUDA init here)
            gpu = query_gpu_stats()

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / GB,
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / GB,
                gpu_available=gpu is not None,
                gpu_memory_used=gpu["memory_used_gb"] if gpu else None,
                gpu_memory_total=gpu["memory_total_gb"] if gpu else None,
                gpu_utilization=gpu["utilization_percent"] if gpu else None,
                disk_path=str(disk_path),
            )
        except Exception as e:
            self._log_error("system_metrics", f"Error collecting system metrics: {e}")
            return None

    def collect_database_metrics(self) -> Optional[DatabaseMetrics]:
        """Collect database metrics"""
        db = None
        try:
            db = SessionLocal()

            # One aggregate pass over raw_data instead of four COUNT queries.
            raw_count, approved, pending, duplicates = db.query(
                func.count(RawData.id),
                func.coalesce(func.sum(case((RawData.admin_approved == 1, 1), else_=0)), 0),
                func.coalesce(func.sum(case((RawData.admin_approved == 0, 1), else_=0)), 0),
                func.coalesce(func.sum(case((RawData.is_duplicate == True, 1), else_=0)), 0),  # noqa: E712
            ).one()
            training_data_count = db.query(func.count(TrainingData.id)).scalar() or 0

            return DatabaseMetrics(
                timestamp=datetime.now(),
                raw_data_count=int(raw_count or 0),
                training_data_count=int(training_data_count),
                approved_count=int(approved or 0),
                pending_count=int(pending or 0),
                duplicate_count=int(duplicates or 0)
            )
        except Exception as e:
            self._log_error("database_metrics", f"Error collecting database metrics: {e}")
            return None
        finally:
            if db is not None:
                db.close()

    def collect_log_report(self) -> Optional[Dict]:
        """Sizes of the files in the logs directory (stat only)."""
        threshold_bytes = self.alert_thresholds["log_file_size_mb"] * MB
        try:
            return scan_log_files(self.logs_dir, threshold_bytes)
        except Exception as e:
            self._log_error("log_report", f"Error scanning log directory {self.logs_dir}: {e}")
            return None

    def collect_service_status(self) -> Optional[Dict[str, List[int]]]:
        """PIDs of the CengBot service processes."""
        try:
            return find_service_processes()
        except Exception as e:
            self._log_error("service_status", f"Error scanning processes: {e}")
            return None

    def evaluate_alerts(self, system_metrics: Optional[SystemMetrics],
                        database_metrics: Optional[DatabaseMetrics],
                        log_report: Optional[Dict] = None,
                        service_status: Optional[Dict[str, List[int]]] = None) -> List[Dict]:
        """Alert conditions for the given metrics (pure; nothing is logged)."""
        t = self.alert_thresholds
        alerts = []

        # System alerts
        if system_metrics:
            if system_metrics.cpu_percent > t["cpu_percent"]:
                alerts.append({
                    "type": "cpu_high",
                    "message": f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                    "value": system_metrics.cpu_percent,
                    "threshold": t["cpu_percent"]
                })

            if system_metrics.memory_percent > t["memory_percent"]:
                alerts.append({
                    "type": "memory_high",
                    "message": f"High memory usage: {system_metrics.memory_percent:.1f}%",
                    "value": system_metrics.memory_percent,
                    "threshold": t["memory_percent"]
                })

            if system_metrics.disk_usage_percent > t["disk_usage_percent"]:
                alerts.append({
                    "type": "disk_high",
                    "message": f"High disk usage: {system_metrics.disk_usage_percent:.1f}%"
                               f" on {system_metrics.disk_path}",
                    "value": system_metrics.disk_usage_percent,
                    "threshold": t["disk_usage_percent"]
                })

            if system_metrics.disk_free_gb < t["disk_free_gb"]:
                alerts.append({
                    "type": "disk_low",
                    "message": f"Low disk space: {system_metrics.disk_free_gb:.1f}GB free"
                               f" on {system_metrics.disk_path}",
                    "value": system_metrics.disk_free_gb,
                    "threshold": t["disk_free_gb"]
                })

            gpu_percent = system_metrics.gpu_memory_percent
            if gpu_percent is not None and gpu_percent > t["gpu_memory_percent"]:
                alerts.append({
                    "type": "gpu_memory_high",
                    "message": f"High GPU memory usage: {gpu_percent:.1f}%",
                    "value": gpu_percent,
                    "threshold": t["gpu_memory_percent"]
                })

        # Database alerts
        if database_metrics:
            if database_metrics.pending_count > t["pending_count"]:
                alerts.append({
                    "type": "pending_high",
                    "message": f"High pending questions: {database_metrics.pending_count}",
                    "value": database_metrics.pending_count,
                    "threshold": t["pending_count"]
                })

        # Log size alerts
        if log_report:
            for item in log_report.get("oversized", []):
                alerts.append({
                    "type": "log_file_large",
                    "message": f"Log file too large: {item['path']} is "
                               f"{_format_bytes(item['size_bytes'])} "
                               f"(threshold {t['log_file_size_mb']:.0f} MB)",
                    "value": item["size_mb"],
                    "threshold": t["log_file_size_mb"],
                    "path": item["path"],
                })
            total_gb = log_report.get("total_bytes", 0) / GB
            if total_gb > t["logs_total_gb"]:
                alerts.append({
                    "type": "logs_total_high",
                    "message": f"Log directory {log_report['logs_dir']} holds "
                               f"{total_gb:.1f}GB (threshold {t['logs_total_gb']:.1f}GB)",
                    "value": round(total_gb, 2),
                    "threshold": t["logs_total_gb"],
                })

        # Service alerts
        if service_status is not None:
            for name in self.expected_services:
                if name in service_status and not service_status[name]:
                    alerts.append({
                        "type": "service_down",
                        "message": f"Service not running: {name}",
                        "value": 0,
                        "threshold": 1,
                        "service": name,
                    })

        return alerts

    def check_alerts(self, system_metrics: Optional[SystemMetrics],
                     database_metrics: Optional[DatabaseMetrics],
                     log_report: Optional[Dict] = None,
                     service_status: Optional[Dict[str, List[int]]] = None) -> List[Dict]:
        """Check for alert conditions, log new ones and return all active ones.

        An alert that is still active is re-logged only after
        ``alert_cooldown_seconds``.
        """
        alerts = self.evaluate_alerts(system_metrics, database_metrics,
                                      log_report, service_status)

        # Log alerts
        for alert in alerts:
            key = ":".join(str(alert.get(k, "")) for k in ("type", "path", "service"))
            if not self._should_log(f"alert:{key}"):
                continue
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "warning",
                **alert
            }

            self.log_alert(alert_data)
            self.logger.warning(f"ALERT: {alert['message']}")
        return alerts

    def _ensure_monitoring_dir(self):
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, system_metrics: SystemMetrics, database_metrics: DatabaseMetrics):
        """Log metrics to files (one JSON object per line)"""
        try:
            self._ensure_monitoring_dir()
            # Log system metrics
            if system_metrics:
                with open(self.system_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(system_metrics.to_dict()) + '\n')

            # Log database metrics
            if database_metrics:
                with open(self.database_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(database_metrics.to_dict()) + '\n')

        except Exception as e:
            self._log_error("log_metrics", f"Error logging metrics: {e}")

    def log_alert(self, alert_data: Dict):
        """Log alert to file"""
        try:
            self._ensure_monitoring_dir()
            with open(self.alerts_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            self._log_error("log_alert", f"Error logging alert: {e}")

    def cleanup_old_logs(self):
        """Drop entries older than ``max_log_age_days`` from the JSONL files.

        Streams each file into a temporary file and atomically replaces it,
        so memory use is constant and a crash cannot truncate the file.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_log_age_days)

            # Clean up monitoring logs
            for log_file in [self.system_log_file, self.database_log_file, self.alerts_log_file]:
                if not log_file.exists():
                    continue
                tmp_file = log_file.with_name(log_file.name + ".tmp")
                kept = dropped = 0
                with open(log_file, 'r', encoding='utf-8', errors='replace') as src, \
                        open(tmp_file, 'w', encoding='utf-8') as dst:
                    for line in src:
                        try:
                            data = json.loads(line)
                            timestamp = datetime.fromisoformat(data['timestamp'])
                        except (ValueError, KeyError, TypeError):
                            dropped += 1  # malformed line
                            continue
                        if timestamp > cutoff_date:
                            dst.write(line if line.endswith('\n') else line + '\n')
                            kept += 1
                        else:
                            dropped += 1
                os.replace(tmp_file, log_file)
                if dropped:
                    self.logger.info(f"{log_file.name}: kept {kept}, removed {dropped} entries")

            self.logger.info(f"Cleaned up logs older than {self.max_log_age_days} days")

        except Exception as e:
            self._log_error("cleanup", f"Error cleaning up logs: {e}")

    def check_logs_and_disk(self) -> Dict:
        """Log directory sizes and free disk space only (fast, no DB/GPU)."""
        log_report = self.collect_log_report()
        disk = None
        system_metrics = None
        try:
            disk_path = self._disk_path()
            usage = psutil.disk_usage(str(disk_path))
            disk = {
                "path": str(disk_path),
                "usage_percent": usage.percent,
                "free_gb": usage.free / GB,
                "total_gb": usage.total / GB,
            }
            # Only the disk figures are meaningful here; CPU/memory are zero so
            # they can never trigger an alert.
            system_metrics = SystemMetrics(
                timestamp=datetime.now(), cpu_percent=0.0, memory_percent=0.0,
                memory_available_gb=0.0, disk_usage_percent=usage.percent,
                disk_free_gb=usage.free / GB, gpu_available=False, disk_path=str(disk_path),
            )
        except Exception as e:
            self._log_error("disk_usage", f"Error reading disk usage: {e}")
        alerts = self.evaluate_alerts(system_metrics, None, log_report)
        return {"logs": log_report, "disk": disk, "alerts": alerts}

    def run_checks(self) -> Dict:
        """One monitoring iteration: collect, persist, alert."""
        system_metrics = self.collect_system_metrics()
        database_metrics = self.collect_database_metrics()
        log_report = self.collect_log_report()
        service_status = self.collect_service_status()

        self.log_metrics(system_metrics, database_metrics)
        alerts = self.check_alerts(system_metrics, database_metrics, log_report, service_status)

        if self._last_cleanup is None or \
                time.monotonic() - self._last_cleanup >= CLEANUP_INTERVAL_SECONDS:
            self.cleanup_old_logs()
            self._last_cleanup = time.monotonic()

        return {
            "system": system_metrics,
            "database": database_metrics,
            "logs": log_report,
            "services": service_status,
            "alerts": alerts,
        }

    def get_system_status(self) -> Dict:
        """Get current system status (nothing is written to disk)."""
        system_metrics = self.collect_system_metrics()
        database_metrics = self.collect_database_metrics()
        log_report = self.collect_log_report()
        service_status = self.collect_service_status()
        alerts = self.evaluate_alerts(system_metrics, database_metrics, log_report, service_status)

        return {
            "system": system_metrics.to_dict() if system_metrics else None,
            "database": database_metrics.to_dict() if database_metrics else None,
            "logs": log_report,
            "services": service_status,
            "alerts": alerts,
            "monitoring": {
                "running": self.running,
                "log_interval": self.log_interval,
                "max_log_age_days": self.max_log_age_days,
                "alert_thresholds": dict(self.alert_thresholds),
            }
        }

    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting monitoring loop")

        while self.running:
            try:
                self.run_checks()
                wait = self.log_interval
            except Exception as e:
                self._log_error("monitor_loop", f"Error in monitoring loop: {e}")
                wait = max(self.log_interval, 60)  # back off before retrying
            # Event.wait instead of sleep so stop() takes effect immediately.
            if self._stop_event.wait(wait):
                break

    def start(self):
        """Start monitoring"""
        if self.running:
            self.logger.warning("Monitor is already running")
            return

        self.running = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("System monitor started")

    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()
        if self.monitor_thread and self.monitor_thread is not threading.current_thread():
            self.monitor_thread.join(timeout=10)

        self.logger.info("System monitor stopped")


# Global monitor instance, created on first use (importing this module must not
# create directories or log handlers).
_monitor: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor


def __getattr__(name):
    # Backward compatibility for ``from system_monitor import monitor``.
    if name == "monitor":
        return get_monitor()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutdown signal received, stopping monitor...")
    if _monitor is not None:
        _monitor.stop()
    sys.exit(0)


def _print_log_report(result: Dict):
    logs = result.get("logs") or {}
    disk = result.get("disk")
    if logs.get("exists"):
        print(f"Logs directory: {logs['logs_dir']} "
              f"({_format_bytes(logs['total_bytes'])} in {logs['file_count']} files)")
    else:
        print(f"Logs directory: {logs.get('logs_dir')} (missing)")
    if disk:
        print(f"Disk: {disk['free_gb']:.1f} GB free of {disk['total_gb']:.1f} GB "
              f"({disk['usage_percent']:.1f}% used) on {disk['path']}")
    oversized = logs.get("oversized") or []
    if oversized:
        print(f"Files over {_format_bytes(logs['threshold_bytes'])}:")
        for item in oversized:
            print(f"  {_format_bytes(item['size_bytes']):>10}  {item['path']}")
    for alert in result.get("alerts", []):
        print(f"ALERT: {alert['message']}")
    if not result.get("alerts"):
        print("OK: no log size or disk space alerts")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CengBot system monitor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                      help="collect all metrics once, print a JSON report and exit "
                           "(exit status 1 if any alert is active)")
    mode.add_argument("--check-logs", action="store_true",
                      help="only check log file sizes and free disk space, then exit "
                           "(exit status 1 if a threshold is exceeded)")
    parser.add_argument("--json", action="store_true", help="JSON output for --check-logs")
    parser.add_argument("--interval", type=int, default=60,
                        help="seconds between samples in service mode (default 60)")
    parser.add_argument("--logs-dir", default=None,
                        help=f"log directory to monitor (default {DEFAULT_LOGS_DIR})")
    parser.add_argument("--log-size-threshold-mb", type=float, default=None,
                        help="per-file log size alert threshold in MB "
                             f"(default env MONITOR_LOG_SIZE_THRESHOLD_MB or "
                             f"{DEFAULT_LOG_SIZE_THRESHOLD_MB:.0f})")
    parser.add_argument("--max-log-age-days", type=int, default=30)
    args = parser.parse_args(argv)

    if args.interval <= 0:
        parser.error("--interval must be positive")

    one_shot = args.once or args.check_logs
    global _monitor
    monitor = SystemMonitor(
        log_interval=args.interval,
        max_log_age_days=args.max_log_age_days,
        logs_dir=args.logs_dir,
        log_size_threshold_mb=args.log_size_threshold_mb,
        # one-shot reports are read-only: no log file, no monitoring dir
        enable_file_logging=not one_shot,
    )
    _monitor = monitor

    if args.check_logs:
        result = monitor.check_logs_and_disk()
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _print_log_report(result)
        return 1 if result["alerts"] else 0

    if args.once:
        status = monitor.get_system_status()
        print(json.dumps(status, indent=2, default=str))
        return 1 if status["alerts"] else 0

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting CengBot System Monitor...")
    print("Press Ctrl+C to stop")

    # Start monitoring
    monitor.start()

    # Keep main thread alive
    try:
        while monitor.running and monitor.monitor_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
