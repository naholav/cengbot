"""
Shared pytest configuration for CengBot.

- Puts the project root and ``src/`` on ``sys.path`` (modules import each other
  as ``from database_models import ...``).
- Sets safe dummy environment variables *before* any project module (and
  therefore python-dotenv) is imported, so the real ``.env`` secrets
  (Telegram token, Claude key, ...) are never used by tests. python-dotenv
  does not override variables that already exist in ``os.environ``.
- Redirects the database to a throw-away SQLite file and guarantees that the
  real ``university_bot.db`` is not modified by the test session.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_TMP_DIR = Path(tempfile.mkdtemp(prefix="cengbot_tests_"))
TEST_DB_PATH = _TMP_DIR / "test_university_bot.db"
TEST_DB_URL = f"sqlite:///{TEST_DB_PATH}"

_TEST_ENV = {
    "DATABASE_URL": TEST_DB_URL,
    "TELEGRAM_BOT_TOKEN": "123456789:TEST_TOKEN_NOT_REAL",
    "CLAUDE_API_KEY": "test-claude-key-not-real",
    "ANTHROPIC_API_KEY": "test-anthropic-key-not-real",
    "HUGGING_FACE_TOKEN": "",
    "HF_TOKEN": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "ADMIN_USERNAME": "admin",
    "ADMIN_PASSWORD": "test-admin-password",
    "ADMIN_PASSWORD_HASH": "",
    "ADMIN_API_KEY": "test-admin-api-key",
    "RABBITMQ_HOST": "localhost",
    # Never let tests write/rotate the real logs/ files.
    "LOG_DIR": str(_TMP_DIR / "logs"),
    "ERROR_LOG_FILE": str(_TMP_DIR / "logs" / "error.log"),
}
(_TMP_DIR / "logs").mkdir(parents=True, exist_ok=True)
for _k, _v in _TEST_ENV.items():
    os.environ[_k] = _v

REAL_DB_PATH = ROOT / "university_bot.db"


def _fingerprint(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_REAL_DB_FINGERPRINT = _fingerprint(REAL_DB_PATH)

# --- Point the ORM at the temporary database --------------------------------
from sqlalchemy import create_engine  # noqa: E402

import database_models  # noqa: E402

TEST_ENGINE = create_engine(
    TEST_DB_URL, echo=False, connect_args={"check_same_thread": False}
)
database_models.engine = TEST_ENGINE
database_models.SessionLocal.configure(bind=TEST_ENGINE)

# Some modules import ``src.database_models``; make sure that resolves to the
# very same module object (one Base / one engine / one SessionLocal).
import src  # noqa: E402  (namespace package)

sys.modules["src.database_models"] = database_models
src.database_models = database_models


def reset_schema():
    """Drop and recreate all tables on the temporary database."""
    # Pooled SQLite connections left over from earlier tests (e.g. TestClient
    # worker threads) can keep a stale view of the schema; start clean.
    TEST_ENGINE.dispose()
    database_models.Base.metadata.drop_all(bind=TEST_ENGINE)
    database_models.Base.metadata.create_all(bind=TEST_ENGINE)
    TEST_ENGINE.dispose()


@pytest.fixture
def test_engine():
    return TEST_ENGINE


@pytest.fixture
def db_session():
    """A session on a freshly created, empty schema (temp SQLite DB)."""
    reset_schema()
    session = database_models.SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def tmp_project_dir(tmp_path):
    """Scratch directory for tests that need to write files."""
    return tmp_path


def pytest_sessionfinish(session, exitstatus):
    after = _fingerprint(REAL_DB_PATH)
    if after != _REAL_DB_FINGERPRINT:
        sys.stderr.write(
            "\n\n!!! The real university_bot.db was MODIFIED during the test "
            "session. Tests must only use the temporary database. !!!\n\n"
        )
        session.exitstatus = 1
