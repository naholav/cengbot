"""Tests for config/env_loader.py (offline, no real .env values are used)."""

import hashlib
import logging
import os
from pathlib import Path

import pytest

from config import env_loader
from config.env_loader import EnvironmentConfig, get_config, load_config


ROOT = Path(__file__).resolve().parent.parent

# Every variable a test in this file touches; cleared before each test so the
# values from the developer's real .env (loaded at import) never leak in.
_KEYS = [
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_TOPIC_ID", "DATABASE_URL", "QUESTIONS_QUEUE",
    "ANSWERS_QUEUE", "MODEL_PATH", "LORA_MODEL_PATH", "PROMPT_GUARD_ENABLED",
    "ADMIN_USERNAME", "ADMIN_PASSWORD", "ADMIN_PASSWORD_HASH", "LOG_LEVEL",
    "USE_CUDA", "MODEL_PRECISION", "MODEL_TEMPERATURE", "MODEL_MAX_NEW_TOKENS",
    "CORS_ORIGINS", "LOG_DIR", "CENGBOT_TEST_ONLY_VAR", "CENGBOT_ENV_FILE",
    "API_PORT",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    # load_dotenv() writes straight into os.environ (not via monkeypatch), so
    # snapshot and restore the whole environment around every test.
    saved = dict(os.environ)
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env_loader, "_warned", set())
    empty = tmp_path / "empty.env"
    empty.write_text("")
    yield EnvironmentConfig(empty)
    os.environ.clear()
    os.environ.update(saved)


# --- construction / .env loading -------------------------------------------

def test_env_file_accepts_str_and_path(clean_env, tmp_path, monkeypatch):
    env = tmp_path / "custom.env"
    env.write_text("CENGBOT_TEST_ONLY_VAR=from_file\n")
    cfg = EnvironmentConfig(str(env))  # used to crash: str has no .exists()
    assert isinstance(cfg.env_file, Path)
    assert cfg.get_str("CENGBOT_TEST_ONLY_VAR") == "from_file"
    monkeypatch.delenv("CENGBOT_TEST_ONLY_VAR")
    cfg = EnvironmentConfig(env)
    assert cfg.get_str("CENGBOT_TEST_ONLY_VAR") == "from_file"


def test_existing_environment_wins_over_env_file(clean_env, tmp_path, monkeypatch):
    env = tmp_path / "custom.env"
    env.write_text("CENGBOT_TEST_ONLY_VAR=from_file\n")
    monkeypatch.setenv("CENGBOT_TEST_ONLY_VAR", "from_env")
    cfg = EnvironmentConfig(env)
    assert cfg.get_str("CENGBOT_TEST_ONLY_VAR") == "from_env"


def test_crlf_env_file_values_have_no_carriage_return(clean_env, tmp_path):
    env = tmp_path / "crlf.env"
    env.write_bytes(b"CENGBOT_TEST_ONLY_VAR=value\r\nQUESTIONS_QUEUE=q1\r\n")
    cfg = EnvironmentConfig(env)
    assert cfg.get_str("CENGBOT_TEST_ONLY_VAR") == "value"
    assert cfg.questions_queue == "q1"


def test_missing_env_file_does_not_crash(clean_env, tmp_path, caplog):
    missing = tmp_path / "does-not-exist.env"
    with caplog.at_level(logging.WARNING, logger="config.env_loader"):
        cfg = EnvironmentConfig(missing)
    assert cfg.env_file == missing
    assert "not found" in caplog.text


def test_env_file_from_cengbot_env_file_variable(clean_env, tmp_path, monkeypatch):
    env = tmp_path / "alt.env"
    env.write_text("CENGBOT_TEST_ONLY_VAR=alt\n")
    monkeypatch.setenv("CENGBOT_ENV_FILE", str(env))
    cfg = EnvironmentConfig()
    assert cfg.env_file == env
    assert cfg.get_str("CENGBOT_TEST_ONLY_VAR") == "alt"


def test_project_root_is_repo_root(clean_env):
    assert clean_env.project_root == ROOT
    assert (clean_env.project_root / "config" / "env_loader.py").is_file()


def test_get_config_and_load_config_share_singleton():
    assert get_config() is env_loader.config
    assert load_config() is get_config()


def test_load_config_with_file_returns_new_instance(clean_env, tmp_path):
    env = tmp_path / "x.env"
    env.write_text("")
    cfg = load_config(env)
    assert cfg is not get_config()
    assert cfg.env_file == env


# --- typed getters -----------------------------------------------------------

def test_empty_values_fall_back_to_defaults(clean_env, monkeypatch):
    monkeypatch.setenv("QUESTIONS_QUEUE", "")
    monkeypatch.setenv("USE_CUDA", "")
    monkeypatch.setenv("API_PORT", "   ")
    assert clean_env.questions_queue == "questions"
    assert clean_env.use_cuda is True  # empty used to mean False
    assert clean_env.api_port == 8001


def test_int_and_float_conversion(clean_env, monkeypatch):
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", " 256 ")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.2")
    assert clean_env.model_max_new_tokens == 256
    assert clean_env.model_temperature == pytest.approx(0.2)
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "lots")
    monkeypatch.setenv("MODEL_TEMPERATURE", "hot")
    assert clean_env.model_max_new_tokens == 200
    assert clean_env.model_temperature == pytest.approx(0.7)


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("1", True), ("YES", True), ("on", True),
    ("false", False), ("0", False), ("No", False), ("off", False),
    ("garbage", True),  # unknown -> default (True for USE_CUDA)
])
def test_bool_conversion(clean_env, monkeypatch, raw, expected):
    monkeypatch.setenv("USE_CUDA", raw)
    assert clean_env.use_cuda is expected


def test_list_conversion(clean_env, monkeypatch):
    assert clean_env.cors_origins == ["http://localhost:3000"]
    monkeypatch.setenv("CORS_ORIGINS", "http://a, http://b ,,")
    assert clean_env.cors_origins == ["http://a", "http://b"]


def test_get_path_relative_and_absolute(clean_env, monkeypatch, tmp_path):
    assert clean_env.log_dir == ROOT / "logs"
    monkeypatch.setenv("LOG_DIR", "var/logs")
    assert clean_env.log_dir == ROOT / "var" / "logs"
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    assert clean_env.log_dir == tmp_path


def test_telegram_token_required_and_placeholder_rejected(clean_env, monkeypatch):
    with pytest.raises(ValueError):
        clean_env.telegram_bot_token
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here")
    with pytest.raises(ValueError):
        clean_env.telegram_bot_token
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    assert clean_env.telegram_bot_token == "123:abc"


def test_telegram_topic_id(clean_env, monkeypatch):
    assert clean_env.telegram_topic_id is None
    monkeypatch.setenv("TELEGRAM_TOPIC_ID", "42")
    assert clean_env.telegram_topic_id == 42
    monkeypatch.setenv("TELEGRAM_TOPIC_ID", "not-a-number")
    assert clean_env.telegram_topic_id is None  # used to raise ValueError


def test_log_level_is_always_valid(clean_env, monkeypatch):
    assert clean_env.log_level == "INFO"
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert clean_env.log_level == "DEBUG"
    monkeypatch.setenv("LOG_LEVEL", "warn")
    assert clean_env.log_level == "WARNING"
    monkeypatch.setenv("LOG_LEVEL", "verbose")
    assert clean_env.log_level == "INFO"
    assert hasattr(logging, clean_env.log_level)


def test_model_precision_validated(clean_env, monkeypatch):
    monkeypatch.setenv("MODEL_PRECISION", "FLOAT16")
    assert clean_env.model_precision == "float16"
    monkeypatch.setenv("MODEL_PRECISION", "int3")
    assert clean_env.model_precision == "bfloat16"


# --- database ------------------------------------------------------------------

def test_database_url_relative_sqlite_resolves_to_project_root(clean_env, monkeypatch):
    assert clean_env.database_url == f"sqlite:///{ROOT / 'university_bot.db'}"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///data/x.db?timeout=5")
    assert clean_env.database_url == f"sqlite:///{ROOT / 'data' / 'x.db'}?timeout=5"


def test_database_url_absolute_and_other_urls_unchanged(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/a.db")
    assert clean_env.database_url == f"sqlite:///{tmp_path}/a.db"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    assert clean_env.database_url == "sqlite:///:memory:"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
    assert clean_env.database_url == "postgresql://u@h/db"


# --- queues / model / guard (decisions c, f) -----------------------------------

def test_queue_defaults_and_override(clean_env, monkeypatch):
    assert (clean_env.questions_queue, clean_env.answers_queue) == ("questions", "answers")
    monkeypatch.setenv("QUESTIONS_QUEUE", "q_in")
    monkeypatch.setenv("ANSWERS_QUEUE", "q_out")
    assert (clean_env.questions_queue, clean_env.answers_queue) == ("q_in", "q_out")


def test_model_path_default_is_active_model_symlink(clean_env):
    assert clean_env.model_path == ROOT / "models" / "active-model"


def test_model_path_override(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", "models/final-best-model-v1.2")
    assert clean_env.model_path == ROOT / "models" / "final-best-model-v1.2"
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    assert clean_env.model_path == tmp_path


def test_stale_lora_model_path_is_ignored_with_one_warning(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("LORA_MODEL_PATH", "models/final-best-model-v1/method1")
    with caplog.at_level(logging.WARNING, logger="config.env_loader"):
        first = clean_env.lora_model_path
        second = clean_env.lora_model_path
    assert first == second == ROOT / "models" / "active-model"
    warnings = [r for r in caplog.records if "LORA_MODEL_PATH" in r.getMessage()]
    assert len(warnings) == 1


def test_lora_model_path_same_as_model_path_no_warning(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("LORA_MODEL_PATH", "models/active-model")
    with caplog.at_level(logging.WARNING, logger="config.env_loader"):
        assert clean_env.lora_model_path == clean_env.model_path
    assert "LORA_MODEL_PATH" not in caplog.text


@pytest.mark.parametrize("raw,expected", [(None, True), ("false", False), ("0", False), ("true", True)])
def test_prompt_guard_enabled(clean_env, monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("PROMPT_GUARD_ENABLED", raw)
    assert clean_env.prompt_guard_enabled is expected


# --- admin auth (decision b) -----------------------------------------------------

def test_admin_defaults_are_unset(clean_env):
    assert clean_env.admin_username == "admin"
    assert clean_env.admin_password == ""
    assert clean_env.admin_password_hash == ""


def test_admin_password_kept_verbatim(clean_env, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "root")
    monkeypatch.setenv("ADMIN_PASSWORD", " s3cret ")
    assert clean_env.admin_username == "root"
    assert clean_env.admin_password == " s3cret "


@pytest.mark.parametrize("placeholder", [
    "your_admin_password_here", "secure_password_here", "changeme", "   ",
])
def test_admin_password_placeholder_is_unset(clean_env, monkeypatch, placeholder):
    monkeypatch.setenv("ADMIN_PASSWORD", placeholder)
    assert clean_env.admin_password == ""


def test_admin_password_hash_valid_is_lowercased(clean_env, monkeypatch):
    digest = hashlib.sha256(b"pw").hexdigest()
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", digest.upper())
    assert clean_env.admin_password_hash == digest


@pytest.mark.parametrize("bad", [
    "your_admin_password_hash_here",  # placeholder
    "abc123",                          # too short
    "z" * 64,                          # not hex
    "$2b$12$" + "a" * 53,              # bcrypt is not supported
])
def test_admin_password_hash_malformed_is_unset(clean_env, monkeypatch, bad):
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", bad)
    assert clean_env.admin_password_hash == ""


# --- .env.example template ------------------------------------------------------

def _template_values(path):
    from dotenv import dotenv_values
    return dotenv_values(path)


@pytest.mark.parametrize("template", [".env.example", "config/env/.env.example"])
def test_env_example_template_is_safe(clean_env, template):
    path = ROOT / template
    if not path.exists():
        pytest.skip(f"{template} not present")
    values = _template_values(path)
    # no default/known admin password, nothing that silently pins an old model
    assert not values.get("ADMIN_PASSWORD")
    assert not values.get("ADMIN_PASSWORD_HASH")
    assert "LORA_MODEL_PATH" not in values
    assert not (values.get("ADMIN_PASSWORD") or "").strip() or "your_" in values.get("ADMIN_PASSWORD", "")
    assert values.get("QUESTIONS_QUEUE") == "questions"
    assert values.get("ANSWERS_QUEUE") == "answers"
    for key in ("MODEL_PATH", "PROMPT_GUARD_ENABLED", "ADMIN_USERNAME", "TELEGRAM_BOT_TOKEN"):
        assert key in values
    # loading the template as-is gives a config that fails closed
    saved = dict(os.environ)
    try:
        cfg = EnvironmentConfig(path)
        with pytest.raises(ValueError):
            cfg.telegram_bot_token  # placeholder is rejected
        assert cfg.admin_password == "" and cfg.admin_password_hash == ""
        assert cfg.model_path == ROOT / "models" / "active-model"
        assert cfg.prompt_guard_enabled is True
    finally:  # load_dotenv wrote into os.environ: restore it exactly
        os.environ.clear()
        os.environ.update(saved)


def test_root_and_config_env_examples_in_sync():
    root = ROOT / ".env.example"
    copy = ROOT / "config" / "env" / ".env.example"
    if not copy.exists():
        pytest.skip("config/env/.env.example not present (gitignored)")
    assert root.read_text(encoding="utf-8") in copy.read_text(encoding="utf-8")


def test_rabbitmq_credentials(clean_env, monkeypatch):
    monkeypatch.delenv("RABBITMQ_USER", raising=False)
    monkeypatch.delenv("RABBITMQ_PASSWORD", raising=False)
    assert (clean_env.rabbitmq_user, clean_env.rabbitmq_password) == ("guest", "guest")
    monkeypatch.setenv("RABBITMQ_USER", "cengbot")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "p w")
    assert (clean_env.rabbitmq_user, clean_env.rabbitmq_password) == ("cengbot", "p w")


def test_api_host_defaults_to_loopback(clean_env, monkeypatch):
    """The admin API must not listen on every interface unless explicitly configured."""
    monkeypatch.delenv("API_HOST", raising=False)
    assert clean_env.api_host == "127.0.0.1"
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    assert clean_env.api_host == "0.0.0.0"
