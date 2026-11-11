#!/usr/bin/env python3
"""
Centralized error handling for CengBot system

Logging policy (see decision (h) in the review notes - unbounded error logs
once filled the disk):
- ``logs/error.log`` is a size-rotated file (10 MB x 5 backups by default;
  override with ``ERROR_LOG_FILE``, ``ERROR_LOG_MAX_BYTES``,
  ``ERROR_LOG_BACKUP_COUNT``).
- The first occurrence of an error (same context + type + message) is logged
  with its full traceback; identical errors within ``ERROR_DEDUP_WINDOW``
  seconds (default 300) are logged as a single line with a repeat counter.
- Full tracebacks go ONLY to the rotating file (WARNING and above). The
  console handler (stdout, which the start scripts redirect into
  logs/<service>.log) gets a one-line summary per record. The logger does not
  propagate to the root logger: the services' ``logging.basicConfig()``
  handlers would otherwise print the traceback a second time.
- Handlers are attached only once per logger, however often the module is
  imported (``error_handler`` vs ``src.error_handler``) or set up.
"""

import logging
import sys
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from enum import Enum

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_DEDUP_WINDOW = 300.0  # seconds
MAX_TRACKED_ERRORS = 256
_SUMMARY_MAX_CHARS = 300
_CONSOLE_MAX_CHARS = 500
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Marker attribute set on handlers installed by this module.
_HANDLER_MARKER = "_cengbot_error_handler"
_setup_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
        return value if value >= 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
        return value if value >= 0 else default
    except ValueError:
        return default


def _default_log_file() -> str:
    explicit = os.environ.get("ERROR_LOG_FILE", "").strip()
    if explicit:
        path = explicit
    else:
        log_dir = os.environ.get("LOG_DIR", "").strip() or "logs"
        path = os.path.join(log_dir, "error.log")
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path


class OneLineFormatter(logging.Formatter):
    """Formatter for console output: no traceback, newlines collapsed."""

    def format(self, record: logging.LogRecord) -> str:
        saved = (record.exc_info, record.exc_text, record.stack_info)
        record.exc_info = record.exc_text = record.stack_info = None
        try:
            text = super().format(record)
        finally:
            record.exc_info, record.exc_text, record.stack_info = saved
        text = " ".join(text.split())
        if len(text) > _CONSOLE_MAX_CHARS:
            text = text[:_CONSOLE_MAX_CHARS] + "..."
        if saved[0]:
            text += " (traceback in error log)"
        return text


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler writing to the *current* ``sys.stdout`` (resolved at emit
    time, so a replaced/closed stdout - e.g. under pytest - does not break it).
    ``setStream()`` can still pin an explicit stream."""

    def __init__(self):
        self._pinned = None
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return self._pinned if self._pinned is not None else sys.stdout

    @stream.setter
    def stream(self, value):
        self._pinned = None if value is sys.stdout else value


class ErrorLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_LEVEL_TO_LOGGING = {
    ErrorLevel.CRITICAL: logging.CRITICAL,
    ErrorLevel.HIGH: logging.ERROR,
    ErrorLevel.MEDIUM: logging.WARNING,
    ErrorLevel.LOW: logging.INFO,
}


class ErrorHandler:
    """Centralized error handling and logging"""

    def __init__(
        self,
        logger_name: str = "CengBot",
        log_file: Optional[str] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
        dedup_window: Optional[float] = None,
    ):
        self.logger = logging.getLogger(logger_name)
        self.log_file = log_file or _default_log_file()
        self.max_bytes = max_bytes if max_bytes is not None else _env_int(
            "ERROR_LOG_MAX_BYTES", DEFAULT_MAX_BYTES)
        self.backup_count = backup_count if backup_count is not None else _env_int(
            "ERROR_LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT)
        self.dedup_window = dedup_window if dedup_window is not None else _env_float(
            "ERROR_DEDUP_WINDOW", DEFAULT_DEDUP_WINDOW)
        # key -> [last_full_log_time, repeats_since_full_log]
        self._recent: "OrderedDict[tuple, list]" = OrderedDict()
        self._recent_lock = threading.Lock()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration (idempotent: handlers are added once)."""
        with _setup_lock:
            if any(getattr(h, _HANDLER_MARKER, False) for h in self.logger.handlers):
                return

            formatter = logging.Formatter(_LOG_FORMAT)

            # Size-rotated error file with full tracebacks; delay=True -> the
            # file is only opened by the first record (import creates nothing).
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                    encoding="utf-8",
                    delay=True,
                )
                file_handler.setLevel(logging.WARNING)
                file_handler.setFormatter(formatter)
                setattr(file_handler, _HANDLER_MARKER, True)
                self.logger.addHandler(file_handler)
            except OSError as exc:  # read-only FS, permissions, ...
                sys.stderr.write(f"error_handler: cannot open {self.log_file}: {exc}\n")

            # Console: one line per record, never the traceback.
            console_handler = _StdoutHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(OneLineFormatter(_LOG_FORMAT))
            setattr(console_handler, _HANDLER_MARKER, True)
            self.logger.addHandler(console_handler)

            self.logger.setLevel(logging.DEBUG)
            # Root handlers (basicConfig in the worker/bot/API) would print the
            # full traceback again into the same redirected stdout log.
            self.logger.propagate = False

    def _register_occurrence(self, key: tuple) -> Optional[int]:
        """
        Record an occurrence of ``key``.

        Returns None if the error should be logged in full (first time, or
        the dedup window expired), else the number of repeats so far.
        """
        now = time.monotonic()
        with self._recent_lock:
            entry = self._recent.get(key)
            if entry is None or self.dedup_window <= 0 or now - entry[0] >= self.dedup_window:
                self._recent[key] = [now, 0]
                self._recent.move_to_end(key)
                while len(self._recent) > MAX_TRACKED_ERRORS:
                    self._recent.popitem(last=False)
                return None
            entry[1] += 1
            self._recent.move_to_end(key)
            return entry[1]

    def handle_error(
        self,
        error: Exception,
        context: str,
        level: ErrorLevel = ErrorLevel.MEDIUM,
        user_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle errors with appropriate logging and response generation

        Args:
            error: The exception that occurred
            context: Context where error occurred (e.g., "telegram_bot", "model_loading")
            level: Error severity level
            user_message: Optional user message
            additional_data: Optional additional context data

        Returns:
            Dict containing error response information
        """
        if not isinstance(level, ErrorLevel):
            try:
                level = ErrorLevel(str(level).lower())
            except ValueError:
                level = ErrorLevel.MEDIUM

        timestamp = datetime.now().isoformat()
        error_type = type(error).__name__
        error_message = str(error)
        log_level = _LEVEL_TO_LOGGING[level]
        prefix = f"{level.name} ERROR in {context}: "

        if self.logger.isEnabledFor(log_level):
            key = (context, level.value, error_type, error_message[:500])
            repeats = self._register_occurrence(key)
            if repeats is None:
                # Full report: traceback of *this* exception (not whatever
                # sys.exc_info() happens to hold), in the same log record.
                exc_info = None
                if level != ErrorLevel.LOW and isinstance(error, BaseException):
                    exc_info = (type(error), error, error.__traceback__)
                self.logger.log(log_level, f"{prefix}{error_message}", exc_info=exc_info)
            else:
                summary = " ".join(error_message.split())
                if len(summary) > _SUMMARY_MAX_CHARS:
                    summary = summary[:_SUMMARY_MAX_CHARS] + "..."
                self.logger.log(
                    log_level,
                    f"{prefix}{error_type}: {summary} "
                    f"[repeated x{repeats} within {int(self.dedup_window)}s; traceback suppressed]",
                )

        # Return response
        return {
            "success": False,
            "error": {
                "type": error_type,
                "message": user_message or error_message,
                "context": context,
                "level": level.value,
                "timestamp": timestamp
            }
        }

    def handle_validation_error(
        self,
        field: str,
        value: Any,
        expected: str,
        context: str = "validation"
    ) -> Dict[str, Any]:
        """Handle validation errors"""
        error_msg = f"Validation failed for field '{field}': expected {expected}, got {type(value).__name__}"

        return {
            "success": False,
            "error": {
                "type": "ValidationError",
                "message": error_msg,
                "field": field,
                "value": str(value),
                "expected": expected,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        }

    def handle_database_error(
        self,
        error: Exception,
        operation: str,
        table: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle database-specific errors"""
        context = f"database_{operation}"
        if table:
            context += f"_{table}"

        return self.handle_error(
            error,
            context,
            level=ErrorLevel.HIGH,
            user_message="Database operation failed. Please try again.",
            additional_data={"operation": operation, "table": table}
        )

    def handle_api_error(
        self,
        error: Exception,
        endpoint: str,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """Handle API-specific errors"""
        context = f"api_{method.lower()}_{endpoint.replace('/', '_')}"

        return self.handle_error(
            error,
            context,
            level=ErrorLevel.MEDIUM,
            user_message="API request failed. Please try again.",
            additional_data={"endpoint": endpoint, "method": method}
        )

    def handle_model_error(
        self,
        error: Exception,
        operation: str = "inference"
    ) -> Dict[str, Any]:
        """Handle model-specific errors"""
        context = f"model_{operation}"

        return self.handle_error(
            error,
            context,
            level=ErrorLevel.HIGH,
            user_message="Model operation failed. Please try again.",
            additional_data={"operation": operation}
        )

    def handle_telegram_error(
        self,
        error: Exception,
        user_id: Optional[int] = None,
        message_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Handle Telegram bot errors"""
        context = "telegram_bot"

        return self.handle_error(
            error,
            context,
            level=ErrorLevel.MEDIUM,
            user_message="Bot operation failed. Please try again.",
            additional_data={"user_id": user_id, "message_id": message_id}
        )

    def handle_training_error(
        self,
        error: Exception,
        step: str,
        epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """Handle training-specific errors"""
        context = f"training_{step}"

        return self.handle_error(
            error,
            context,
            level=ErrorLevel.CRITICAL,
            user_message="Training operation failed.",
            additional_data={"step": step, "epoch": epoch}
        )

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions
def handle_error(error: Exception, context: str, level: ErrorLevel = ErrorLevel.MEDIUM, **kwargs):
    """Convenience function for error handling"""
    return error_handler.handle_error(error, context, level, **kwargs)

def handle_validation_error(field: str, value: Any, expected: str, context: str = "validation"):
    """Convenience function for validation errors"""
    return error_handler.handle_validation_error(field, value, expected, context)

def handle_database_error(error: Exception, operation: str, table: Optional[str] = None):
    """Convenience function for database errors"""
    return error_handler.handle_database_error(error, operation, table)

def handle_api_error(error: Exception, endpoint: str, method: str = "GET"):
    """Convenience function for API errors"""
    return error_handler.handle_api_error(error, endpoint, method)

def handle_model_error(error: Exception, operation: str = "inference"):
    """Convenience function for model errors"""
    return error_handler.handle_model_error(error, operation)

def handle_telegram_error(error: Exception, user_id: Optional[int] = None, message_id: Optional[int] = None):
    """Convenience function for Telegram errors"""
    return error_handler.handle_telegram_error(error, user_id, message_id)

def handle_training_error(error: Exception, step: str, epoch: Optional[int] = None):
    """Convenience function for training errors"""
    return error_handler.handle_training_error(error, step, epoch)
