"""
Environment Configuration Loader for CengBot
============================================

This module handles loading environment variables from .env files
and provides a centralized configuration management system.

Notes:
- Variables that are already present in ``os.environ`` always win over the
  values in the .env file (python-dotenv ``override=False``).
- An empty (or whitespace-only) value is treated as "not set", so the
  documented default applies.
- Relative paths are resolved against the project root, never against the
  current working directory.
- ``CENGBOT_ENV_FILE`` may point to an alternative .env file (useful for
  systemd units and tests); the default is ``<project root>/.env``.

Author: naholav
"""

import os
import re
import logging
import threading
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TRUE_VALUES = {"true", "1", "yes", "on", "enabled", "y"}
_FALSE_VALUES = {"false", "0", "no", "off", "disabled", "n"}
_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
_MODEL_PRECISIONS = {"float16", "float32", "bfloat16"}
_SHA256_HEX_RE = re.compile(r"[0-9a-fA-F]{64}")
# Values copied verbatim from .env.example (e.g. ``your_admin_password_here``)
_PLACEHOLDER_RE = re.compile(
    r"(your[_\-].*|.*[_\-]here|change[_\-]?me|placeholder|<.*>)", re.IGNORECASE
)

DEFAULT_MODEL_PATH = "models/active-model"

# .env files that were already loaded in this process (avoid re-parsing and
# re-logging every time a module calls load_config()).
_loaded_env_files = set()
_load_lock = threading.Lock()
# Keys we already warned about (warn once per process, not on every access).
_warned = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)


def is_placeholder(value: Optional[str]) -> bool:
    """Return True if ``value`` looks like an unedited .env.example placeholder."""
    if value is None:
        return False
    return bool(_PLACEHOLDER_RE.fullmatch(value.strip()))


class EnvironmentConfig:
    """
    Environment configuration manager for CengBot.

    This class loads configuration from environment variables and .env files,
    providing default values and type conversion for different settings.
    """

    def __init__(self, env_file: Optional[Union[str, os.PathLike]] = None):
        """
        Initialize environment configuration.

        Args:
            env_file: Optional path to .env file (str or Path). If None, uses
                ``$CENGBOT_ENV_FILE`` or ``.env`` in the project root.
                Relative paths are tried relative to the current directory
                first and then relative to the project root.
        """
        self.project_root = PROJECT_ROOT
        if env_file is None:
            env_file = os.environ.get("CENGBOT_ENV_FILE") or (self.project_root / ".env")
        self.env_file = self._resolve_env_file(env_file)
        self._load_env_file()

    def _resolve_env_file(self, env_file: Union[str, os.PathLike]) -> Path:
        path = Path(env_file).expanduser()
        if not path.is_absolute() and not path.exists():
            candidate = self.project_root / path
            if candidate.exists():
                return candidate
        return path

    def _load_env_file(self) -> None:
        key = str(self.env_file.resolve()) if self.env_file.exists() else str(self.env_file)
        with _load_lock:
            first_time = key not in _loaded_env_files
            _loaded_env_files.add(key)
        if self.env_file.is_file():
            # override=False: real environment variables always win.
            load_dotenv(self.env_file, override=False)
            if first_time:
                logger.info(f"Loaded environment from {self.env_file}")
        elif first_time:
            logger.warning(f"Environment file not found: {self.env_file}")

    # -------------------------------------------------------------------------
    # Typed getters
    # -------------------------------------------------------------------------

    @staticmethod
    def _raw(key: str) -> Optional[str]:
        """Raw value, or None if unset/empty/whitespace-only."""
        value = os.environ.get(key)
        if value is None or not value.strip():
            return None
        return value.strip()

    def get_str(self, key: str, default: str = "") -> str:
        """Get string value from environment (empty value -> default)."""
        value = self._raw(key)
        return default if value is None else value

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value from environment."""
        value = self._raw(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            _warn_once(key, f"Invalid integer value for {key}, using default: {default}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value from environment."""
        value = self._raw(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            _warn_once(key, f"Invalid float value for {key}, using default: {default}")
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment (unknown value -> default)."""
        value = self._raw(key)
        if value is None:
            return default
        lowered = value.lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        _warn_once(key, f"Invalid boolean value for {key}, using default: {default}")
        return default

    def get_list(self, key: str, default: list = None, separator: str = ",") -> list:
        """Get list value from environment (comma-separated by default)."""
        if default is None:
            default = []
        value = self._raw(key)
        if value is None:
            return list(default)
        items = [item.strip() for item in value.split(separator) if item.strip()]
        return items if items else list(default)

    def get_path(self, key: str, default: str = "") -> Path:
        """Get Path object from environment (relative -> project root)."""
        path = Path(self.get_str(key, default)).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path  # Relative to project root

    # =============================================================================
    # TELEGRAM BOT CONFIGURATION
    # =============================================================================

    @property
    def telegram_bot_token(self) -> str:
        """Telegram bot token from environment."""
        token = self.get_str("TELEGRAM_BOT_TOKEN")
        if not token or is_placeholder(token):
            raise ValueError("TELEGRAM_BOT_TOKEN is required but not set")
        return token

    @property
    def telegram_topic_id(self) -> Optional[int]:
        """Optional Telegram topic ID for group chats."""
        topic_id = self.get_str("TELEGRAM_TOPIC_ID")
        if not topic_id:
            return None
        try:
            return int(topic_id)
        except ValueError:
            _warn_once(
                "TELEGRAM_TOPIC_ID",
                "Invalid TELEGRAM_TOPIC_ID (not an integer); topic filtering disabled",
            )
            return None

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================

    @property
    def database_url(self) -> str:
        """Database connection URL (relative sqlite paths -> project root)."""
        url = self.get_str("DATABASE_URL", "sqlite:///university_bot.db")
        prefix = "sqlite:///"
        if url.startswith(prefix):
            rest = url[len(prefix):]
            path_part, sep, query = rest.partition("?")
            if path_part and path_part != ":memory:" and not path_part.startswith("/"):
                path_part = str(self.project_root / path_part)
                url = f"{prefix}{path_part}{sep}{query}"
        return url

    @property
    def db_pool_size(self) -> int:
        """Database connection pool size."""
        return self.get_int("DB_POOL_SIZE", 10)

    @property
    def db_max_overflow(self) -> int:
        """Database connection pool max overflow."""
        return self.get_int("DB_MAX_OVERFLOW", 20)

    # =============================================================================
    # RABBITMQ CONFIGURATION
    # =============================================================================

    @property
    def rabbitmq_url(self) -> str:
        """RabbitMQ connection URL."""
        return self.get_str("RABBITMQ_URL", "amqp://localhost:5672")

    @property
    def rabbitmq_host(self) -> str:
        """RabbitMQ host."""
        return self.get_str("RABBITMQ_HOST", "localhost")

    @property
    def rabbitmq_port(self) -> int:
        """RabbitMQ port."""
        return self.get_int("RABBITMQ_PORT", 5672)

    @property
    def rabbitmq_user(self) -> str:
        """RabbitMQ user (default ``guest``, accepted by RabbitMQ on localhost only)."""
        return self.get_str("RABBITMQ_USER", "guest")

    @property
    def rabbitmq_password(self) -> str:
        """RabbitMQ password (default ``guest``; not stripped)."""
        value = os.environ.get("RABBITMQ_PASSWORD", "")
        return value if value.strip() else "guest"
    
    @property
    def questions_queue(self) -> str:
        """RabbitMQ questions queue name."""
        return self.get_str("QUESTIONS_QUEUE", "questions")

    @property
    def answers_queue(self) -> str:
        """RabbitMQ answers queue name."""
        return self.get_str("ANSWERS_QUEUE", "answers")

    # =============================================================================
    # AI MODEL CONFIGURATION
    # =============================================================================

    @property
    def base_model_name(self) -> str:
        """Base model name from Hugging Face."""
        return self.get_str("BASE_MODEL_NAME", "meta-llama/Llama-3.2-3B")

    @property
    def model_path(self) -> Path:
        """
        Active model directory (PEFT adapter or full model).

        Defaults to the ``models/active-model`` symlink managed by
        ``scripts/switch_model.sh``; ``MODEL_PATH`` may override it.
        The symlink is intentionally NOT resolved here.
        """
        return self.get_path("MODEL_PATH", DEFAULT_MODEL_PATH)

    @property
    def lora_model_path(self) -> Path:
        """
        DEPRECATED - use :attr:`model_path`. Kept for backward compatibility
        and now returns exactly ``model_path`` (it used to default to the
        non-existent ``models/active-model/method1``).

        ``LORA_MODEL_PATH`` is ignored (a stale value would silently downgrade
        the production model); a one-time warning is logged if it is set to a
        different location.
        """
        path = self.model_path
        legacy = self._raw("LORA_MODEL_PATH")
        if legacy is not None:
            legacy_path = Path(legacy).expanduser()
            if not legacy_path.is_absolute():
                legacy_path = self.project_root / legacy_path
            if legacy_path != path:
                _warn_once(
                    "LORA_MODEL_PATH",
                    f"LORA_MODEL_PATH ({legacy}) is legacy and ignored; using "
                    f"{path} (set MODEL_PATH to override, or use scripts/switch_model.sh)",
                )
        return path

    @property
    def model_cache_dir(self) -> Path:
        """Model cache directory."""
        return self.get_path("MODEL_CACHE_DIR", "model_cache")

    @property
    def model_temperature(self) -> float:
        """Model generation temperature."""
        return self.get_float("MODEL_TEMPERATURE", 0.7)

    @property
    def model_max_new_tokens(self) -> int:
        """Maximum new tokens to generate."""
        return self.get_int("MODEL_MAX_NEW_TOKENS", 200)

    @property
    def model_top_p(self) -> float:
        """Model top-p sampling parameter."""
        return self.get_float("MODEL_TOP_P", 0.95)

    @property
    def model_top_k(self) -> int:
        """Model top-k sampling parameter."""
        return self.get_int("MODEL_TOP_K", 50)

    @property
    def model_repetition_penalty(self) -> float:
        """Model repetition penalty."""
        return self.get_float("MODEL_REPETITION_PENALTY", 1.1)

    @property
    def prompt_guard_enabled(self) -> bool:
        """Enable the jailbreak / prompt-injection guard (src/prompt_guard.py)."""
        return self.get_bool("PROMPT_GUARD_ENABLED", True)

    # =============================================================================
    # API SERVER CONFIGURATION
    # =============================================================================

    @property
    def api_host(self) -> str:
        """API server host."""
        return self.get_str("API_HOST", "127.0.0.1")  # loopback by default; expose via a reverse proxy

    @property
    def api_port(self) -> int:
        """API server port."""
        return self.get_int("API_PORT", 8001)

    @property
    def api_reload(self) -> bool:
        """Enable API auto-reload for development."""
        return self.get_bool("API_RELOAD", False)

    @property
    def cors_origins(self) -> list:
        """CORS allowed origins."""
        return self.get_list("CORS_ORIGINS", ["http://localhost:3000"])

    # =============================================================================
    # ADMIN AUTHENTICATION
    # =============================================================================

    @property
    def admin_username(self) -> str:
        """Admin panel username (default ``admin``)."""
        return self.get_str("ADMIN_USERNAME", "admin")

    @property
    def admin_password(self) -> str:
        """
        Plain-text admin password, or "" if unset / placeholder.

        Not stripped: surrounding whitespace is part of the password.
        """
        value = os.environ.get("ADMIN_PASSWORD", "")
        if not value.strip() or is_placeholder(value):
            return ""
        return value

    @property
    def admin_password_hash(self) -> str:
        """
        SHA-256 hex digest of the admin password (lower-case), or "" if
        unset, a placeholder or malformed (anything but 64 hex characters).
        """
        value = self.get_str("ADMIN_PASSWORD_HASH")
        if not value or is_placeholder(value):
            return ""
        if not _SHA256_HEX_RE.fullmatch(value):
            _warn_once(
                "ADMIN_PASSWORD_HASH",
                "ADMIN_PASSWORD_HASH is not a 64-character hex SHA-256 digest; ignoring it",
            )
            return ""
        return value.lower()

    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================

    @property
    def log_level(self) -> str:
        """Logging level (always a valid ``logging`` level name)."""
        level = self.get_str("LOG_LEVEL", "INFO").upper()
        if level == "WARN":
            level = "WARNING"
        if level not in _LOG_LEVELS:
            _warn_once("LOG_LEVEL", f"Invalid LOG_LEVEL {level!r}, using INFO")
            return "INFO"
        return level

    @property
    def log_dir(self) -> Path:
        """Log directory path."""
        return self.get_path("LOG_DIR", "logs")

    @property
    def worker_log_file(self) -> Path:
        """Worker log file path."""
        return self.get_path("WORKER_LOG_FILE", "logs/worker.log")

    @property
    def bot_log_file(self) -> Path:
        """Bot log file path."""
        return self.get_path("BOT_LOG_FILE", "logs/bot.log")

    @property
    def api_log_file(self) -> Path:
        """API log file path."""
        return self.get_path("API_LOG_FILE", "logs/admin.log")

    # =============================================================================
    # PERFORMANCE CONFIGURATION
    # =============================================================================

    @property
    def use_cuda(self) -> bool:
        """Enable CUDA acceleration."""
        return self.get_bool("USE_CUDA", True)

    @property
    def model_precision(self) -> str:
        """Model precision: float16, float32, bfloat16."""
        precision = self.get_str("MODEL_PRECISION", "bfloat16").lower()
        if precision not in _MODEL_PRECISIONS:
            _warn_once("MODEL_PRECISION", f"Invalid MODEL_PRECISION {precision!r}, using bfloat16")
            return "bfloat16"
        return precision

    @property
    def use_4bit_quantization(self) -> bool:
        """Enable 4-bit quantization."""
        return self.get_bool("USE_4BIT_QUANTIZATION", False)

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent model inference requests."""
        return self.get_int("MAX_CONCURRENT_REQUESTS", 3)

    # =============================================================================
    # DEVELOPMENT CONFIGURATION
    # =============================================================================

    @property
    def debug(self) -> bool:
        """Enable debug mode."""
        return self.get_bool("DEBUG", False)

    @property
    def skip_model_loading(self) -> bool:
        """Skip model loading for testing."""
        return self.get_bool("SKIP_MODEL_LOADING", False)


# Global configuration instance (loads the default .env once at import time)
config = EnvironmentConfig()


def load_config(env_file: Optional[Union[str, os.PathLike]] = None) -> EnvironmentConfig:
    """
    Load configuration from environment file.

    Args:
        env_file: Optional path to .env file. When omitted, the global
            (already loaded) configuration instance is returned.

    Returns:
        EnvironmentConfig instance
    """
    if env_file is None:
        return config
    return EnvironmentConfig(env_file)


def get_config() -> EnvironmentConfig:
    """
    Get the global configuration instance.

    Returns:
        Global EnvironmentConfig instance
    """
    return config
