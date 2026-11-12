"""
CengBot admin REST API (FastAPI).

Every endpoint that reads or modifies data requires HTTP Basic auth
(``verify_credentials``). Intentionally public endpoints: ``GET /`` (static
banner), ``GET /health`` (status only; system/GPU details need credentials)
and FastAPI's own ``/docs``, ``/redoc`` and ``/openapi.json`` (schema only).

Credentials come from the environment and are read at request time:

* ``ADMIN_USERNAME`` (default ``admin``)
* ``ADMIN_PASSWORD_HASH`` - sha256 hex digest of the password (preferred), or
* ``ADMIN_PASSWORD`` - the plain password (used when no valid hash is set).

There is no default password: if neither is configured every login is
rejected with 401 and an ERROR is logged at startup.

Failed logins are rate limited per client address (in memory, per process):
after ``ADMIN_LOGIN_MAX_FAILURES`` (default 10) wrong passwords within
``ADMIN_LOGIN_WINDOW_SECONDS`` (default 900) that address gets HTTP 429 with
``Retry-After`` for ``ADMIN_LOGIN_LOCKOUT_SECONDS`` (default 900), even with
the right password; ``ADMIN_LOGIN_MAX_FAILURES=0`` disables the limiter. The
client address is the ASGI peer address: behind a reverse proxy it is the
proxy's, unless uvicorn's proxy-header support resolves the real client
(``FORWARDED_ALLOW_IPS``, uvicorn's default trusts only 127.0.0.1/::1 - set it
to the proxy's address; ``X-Forwarded-For`` from anyone else is ignored). The
proxy must set ``X-Forwarded-For`` itself on every route it forwards.
Wrong passwords on ``/health`` are limited separately, for all clients
together, and only ever disable the admin view of ``/health``.
"""

import os
import sys

# Make ``from database_models import ...`` work no matter how this module is
# started (``python src/admin_rest_api.py``, ``uvicorn admin_rest_api:app`` from
# ``src/`` or ``from src.admin_rest_api import app`` from the project root).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
for _p in (SRC_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hashlib  # noqa: E402
import ipaddress  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict, deque  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path as FsPath, PurePosixPath  # noqa: E402
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar  # noqa: E402

import psutil  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query, Request, Response, status  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.security import HTTPBasic, HTTPBasicCredentials  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: E402
from sqlalchemy import case, func, or_, select, text, update  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import database_models  # noqa: E402
from database_models import (  # noqa: E402
    RawData,
    SessionLocal,
    TrainingData,
    UserVotes,
    mark_duplicate_answers,
    mark_duplicate_questions,
)
from config.env_loader import get_config  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")

# Load environment variables (.env is loaded by config.env_loader on import)
config = get_config()

# =============================================================================
# Authentication
# =============================================================================

security = HTTPBasic()
_optional_security = HTTPBasic(auto_error=False)

DEFAULT_ADMIN_USERNAME = "admin"
_SHA256_HEX_RE = re.compile(r"[0-9a-fA-F]{64}")
_PLACEHOLDER_RE = re.compile(
    r"^(your[_-].*|change[_-]?me|<.*>|placeholder|x{3,}|\*+)$", re.IGNORECASE
)
_EMPTY_PASSWORD_SHA256 = hashlib.sha256(b"").hexdigest()
_unconfigured_warned = False


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(value.strip()))


def get_admin_auth_config() -> Dict[str, Optional[str]]:
    """Read the admin credentials from the environment (no caching).

    Returns a dict with ``username``, ``mode`` (``"hash"``, ``"plain"`` or
    ``None`` when no usable password is configured) and ``secret``.
    """
    username = (os.getenv("ADMIN_USERNAME") or "").strip() or DEFAULT_ADMIN_USERNAME
    raw_hash = (os.getenv("ADMIN_PASSWORD_HASH") or "").strip()
    password = os.getenv("ADMIN_PASSWORD") or ""

    if (
        raw_hash
        and _SHA256_HEX_RE.fullmatch(raw_hash)
        and raw_hash.lower() != _EMPTY_PASSWORD_SHA256
    ):
        return {"username": username, "mode": "hash", "secret": raw_hash.lower()}
    if password.strip() and not _is_placeholder(password):
        return {"username": username, "mode": "plain", "secret": password}
    return {"username": username, "mode": None, "secret": None}


def log_auth_configuration() -> None:
    """Log (once, at startup) how admin authentication is configured."""
    cfg = get_admin_auth_config()
    raw_hash = (os.getenv("ADMIN_PASSWORD_HASH") or "").strip()
    if raw_hash and cfg["mode"] != "hash":
        logger.warning(
            "ADMIN_PASSWORD_HASH is set but is not a usable sha256 hex digest "
            "(64 hex characters, not the hash of an empty password); ignoring it."
        )
    if cfg["mode"] is None:
        logger.error(
            "Admin API authentication is NOT configured: set ADMIN_PASSWORD_HASH "
            "(sha256 hex digest) or ADMIN_PASSWORD in .env. Every login will be "
            "rejected with 401 until one of them is set."
        )
    else:
        if cfg["mode"] == "plain" and not cfg["secret"].isascii():
            logger.warning(
                "ADMIN_PASSWORD contains non-ASCII characters; HTTP Basic clients "
                "(browsers) cannot send it, so logins will fail."
            )
        logger.info(
            "Admin API authentication enabled for user '%s' (%s password).",
            cfg["username"],
            "hashed" if cfg["mode"] == "hash" else "plain-text",
        )


def credentials_are_valid(credentials: Optional[HTTPBasicCredentials]) -> bool:
    """Constant-time check of HTTP Basic credentials against the env config."""
    global _unconfigured_warned
    if credentials is None:
        return False
    cfg = get_admin_auth_config()
    if cfg["mode"] is None:
        if not _unconfigured_warned:
            _unconfigured_warned = True
            logger.error(
                "Admin login rejected: no ADMIN_PASSWORD_HASH / ADMIN_PASSWORD configured."
            )
        return False

    # Compare bytes: compare_digest() raises TypeError for non-ASCII str.
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), cfg["username"].encode("utf-8")
    )
    if cfg["mode"] == "hash":
        password_ok = secrets.compare_digest(
            hash_password(credentials.password).encode("ascii"),
            cfg["secret"].encode("ascii"),
        )
    else:
        password_ok = secrets.compare_digest(
            credentials.password.encode("utf-8"), cfg["secret"].encode("utf-8")
        )
    return username_ok and password_ok


# =============================================================================
# Failed-login rate limiting
# =============================================================================

DEFAULT_LOGIN_MAX_FAILURES = 10
DEFAULT_LOGIN_WINDOW_SECONDS = 900.0
DEFAULT_LOGIN_LOCKOUT_SECONDS = 900.0
LOGIN_LIMITER_MAX_CLIENTS = 10_000  # bounded memory: least recently seen clients are forgotten


class LoginRateLimiter:
    """Counts failed logins per client key; locks a key out after too many.

    ``max_failures`` failures within ``window_seconds`` lock the key for
    ``lockout_seconds``. A successful login clears only its own key. At most
    ``max_clients`` keys are tracked (least recently seen ones are dropped).
    Thread-safe (sync endpoints run in a thread pool).
    """

    def __init__(
        self,
        max_failures: int = DEFAULT_LOGIN_MAX_FAILURES,
        window_seconds: float = DEFAULT_LOGIN_WINDOW_SECONDS,
        lockout_seconds: float = DEFAULT_LOGIN_LOCKOUT_SECONDS,
        max_clients: int = LOGIN_LIMITER_MAX_CLIENTS,
        clock=time.monotonic,
    ):
        self.max_failures = max(0, int(max_failures))
        self.window_seconds = float(window_seconds)
        self.lockout_seconds = float(lockout_seconds)
        self.max_clients = max(1, int(max_clients))
        self._clock = clock
        self._lock = threading.Lock()
        # key -> [deque of failure times, locked_until]
        self._clients: "OrderedDict[str, list]" = OrderedDict()

    @classmethod
    def from_env(cls) -> "LoginRateLimiter":
        return cls(
            max_failures=_env_number("ADMIN_LOGIN_MAX_FAILURES", DEFAULT_LOGIN_MAX_FAILURES, int),
            window_seconds=_env_number("ADMIN_LOGIN_WINDOW_SECONDS", DEFAULT_LOGIN_WINDOW_SECONDS, float, 1.0),
            lockout_seconds=_env_number("ADMIN_LOGIN_LOCKOUT_SECONDS", DEFAULT_LOGIN_LOCKOUT_SECONDS, float, 1.0),
        )

    @property
    def enabled(self) -> bool:
        return self.max_failures > 0

    @property
    def tracked_clients(self) -> int:
        return len(self._clients)

    def _seconds_left(self, locked_until: float, now: float) -> int:
        return max(1, math.ceil(locked_until - now))

    def retry_after(self, key: str) -> int:
        """Seconds until ``key`` may try again (0 = not locked out)."""
        if not self.enabled:
            return 0
        now = self._clock()
        with self._lock:
            state = self._clients.get(key)
            if state is None or state[1] <= now:
                return 0
            return self._seconds_left(state[1], now)

    def record_failure(self, key: str) -> int:
        """Count a failed login; returns the lockout in seconds if ``key`` is now locked, else 0."""
        if not self.enabled:
            return 0
        now = self._clock()
        with self._lock:
            state = self._clients.get(key)
            if state is None:
                while len(self._clients) >= self.max_clients:
                    self._clients.popitem(last=False)
                state = [deque(maxlen=self.max_failures), 0.0]
                self._clients[key] = state
            else:
                self._clients.move_to_end(key)
            failures, locked_until = state
            if locked_until > now:
                return self._seconds_left(locked_until, now)
            while failures and failures[0] <= now - self.window_seconds:
                failures.popleft()
            failures.append(now)
            if len(failures) < self.max_failures:
                return 0
            failures.clear()
            state[1] = now + self.lockout_seconds
        logger.warning(
            "Admin login: %d failed attempts from %s within %ds; blocked for %ds.",
            self.max_failures, key, int(self.window_seconds), int(self.lockout_seconds),
        )
        return self._seconds_left(now + self.lockout_seconds, now)

    def record_success(self, key: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._clients.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._clients.clear()


def _env_number(name: str, default, cast, minimum=0):
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
        if not math.isfinite(value) or value < minimum:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Invalid %s=%r; using the default %s.", name, raw, default)
        return default


login_rate_limiter = LoginRateLimiter.from_env()
# Wrong passwords sent to /health are counted here, for all clients together
# (see _health_authenticated); they never lock anybody out of the API.
health_credentials_limiter = LoginRateLimiter.from_env()
HEALTH_LIMIT_KEY = "/health (all clients)"
INVALID_CLIENT_KEY = "invalid"


def client_key(request: Request) -> str:
    """Rate-limit key of the client: its IP address (IPv6: its /64 network).

    ``request.client`` is the TCP peer, or - only for peers listed in uvicorn's
    ``FORWARDED_ALLOW_IPS`` - the client named in ``X-Forwarded-For``. The
    header itself is never read here, so other clients cannot spoof it.
    Anything that is not an IP address (a forged header passed on by a
    misconfigured proxy) maps to one shared key, so arbitrary strings can
    neither create new keys nor grow the limiter's memory.
    """
    host = request.client.host if request.client else ""
    if not host:
        return "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return INVALID_CLIENT_KEY
    if address.version == 6:
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        # one host usually owns a whole /64: count it as one client
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def _too_many_attempts(retry_after: int) -> HTTPException:
    # No WWW-Authenticate header: browsers must not pop up a login dialog.
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
        headers={"Retry-After": str(retry_after)},
    )


def verify_credentials(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> str:
    """FastAPI dependency: 401 unless the HTTP Basic credentials are valid,
    429 while the client is locked out after too many failed attempts."""
    key = client_key(request)
    retry_after = login_rate_limiter.retry_after(key)
    if retry_after:
        raise _too_many_attempts(retry_after)
    if not credentials_are_valid(credentials):
        retry_after = login_rate_limiter.record_failure(key)
        if retry_after:
            raise _too_many_attempts(retry_after)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    login_rate_limiter.record_success(key)
    return credentials.username


async def optional_credentials(request: Request) -> Optional[HTTPBasicCredentials]:
    """Like ``HTTPBasic(auto_error=False)`` but never raises (malformed header -> None)."""
    try:
        return await _optional_security(request)
    except HTTPException:
        return None


# =============================================================================
# Database session dependency
# =============================================================================


def get_db():
    """Yield a session that is always rolled back on error and closed."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def handle_database_error(error: Exception, operation: str, table: Optional[str] = None):
    """Delegate to error_handler (imported lazily: no log-file side effects on import)."""
    from error_handler import handle_database_error as _handle

    return _handle(error, operation, table)


def _db_error(error: Exception, operation: str, table: str) -> HTTPException:
    """Log a database error and build a generic 500 (no internals leaked)."""
    message = "Database operation failed. Please try again."
    try:
        message = handle_database_error(error, operation, table)["error"]["message"]
    except Exception:  # pragma: no cover - logging must never break the API
        logger.exception("Database error during %s on %s", operation, table)
    return HTTPException(status_code=500, detail=message)


# =============================================================================
# App + CORS
# =============================================================================


def _cors_settings(origins: Iterable[str]) -> Dict[str, object]:
    """Normalise configured origins; never combine '*' with credentials."""
    cleaned = []
    for origin in origins or []:
        origin = (origin or "").strip().rstrip("/")
        if origin and origin not in cleaned:
            cleaned.append(origin)
    if not cleaned:
        cleaned = ["http://localhost:3000"]
    if "*" in cleaned:
        logger.warning(
            "CORS_ORIGINS contains '*': any origin is allowed, credentials are disabled."
        )
        return {"allow_origins": ["*"], "allow_credentials": False}
    return {"allow_origins": cleaned, "allow_credentials": True}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log_auth_configuration()
    yield


app = FastAPI(title="University Bot Admin API", lifespan=lifespan)

_cors = _cors_settings(config.cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors["allow_origins"],
    allow_credentials=_cors["allow_credentials"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    # readable by the panel: login lockout, /duplicates total, /docs/read info
    expose_headers=["Retry-After", "X-Total-Count", "X-Content-Truncated", "X-File-Size"],
)

# Every route registered on this router requires valid admin credentials.
protected = APIRouter(dependencies=[Depends(verify_credentials)])

# =============================================================================
# Pydantic models
# =============================================================================

MAX_PAGE = 1_000_000
MAX_ANSWER_LENGTH = 20_000
MAX_DB_ID = 2**63 - 1


def ItemId():
    """Fresh path-parameter spec for numeric primary keys (1 .. 2**63-1)."""
    return Path(..., ge=1, le=MAX_DB_ID)


class RawDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    telegram_id: int
    telegram_message_id: Optional[int] = None
    username: Optional[str] = None
    question: str
    answer: Optional[str] = None
    language: Optional[str] = None
    like: Optional[int] = None
    admin_approved: int = 0
    is_duplicate: bool = False
    duplicate_of_id: Optional[int] = None
    similarity_score: Optional[float] = None
    created_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    message_thread_id: Optional[int] = None
    likes: Optional[int] = 0
    dislikes: Optional[int] = 0
    total_votes: Optional[int] = 0
    vote_score: Optional[int] = 0


class TrainingDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_id: Optional[int] = None  # Can be NULL for standalone training data
    question: str
    answer: str
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    point: Optional[int] = None  # 1 liked / -1 disliked / None, from source raw_data
    is_answer_duplicate: Optional[bool] = None
    duplicate_answer_of_id: Optional[int] = None
    answer_similarity_score: Optional[float] = None


class UpdateAnswerRequest(BaseModel):
    answer: str = Field(..., max_length=MAX_ANSWER_LENGTH)

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must not be empty")
        return value


class ApproveResponse(BaseModel):
    success: bool
    message: str
    training_data_id: Optional[int] = None


class PaginatedResponse(BaseModel, Generic[T]):
    data: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def _paginate(total: int, page: int, page_size: int) -> Tuple[int, int]:
    """Return (total_pages, offset)."""
    total_pages = math.ceil(total / page_size) if total > 0 else 0
    return total_pages, (page - 1) * page_size


# =============================================================================
# Vote helpers (votes live in user_votes; raw_data.like is a legacy column)
# =============================================================================

_SQL_IN_CHUNK = 500


def _vote_stats_for(db: Session, raw_ids: Sequence[int]) -> Dict[int, Dict[str, int]]:
    """Batched equivalent of ``database_models.get_vote_statistics``."""
    stats: Dict[int, Dict[str, int]] = {}
    ids = list(dict.fromkeys(i for i in raw_ids if i is not None))
    for start in range(0, len(ids), _SQL_IN_CHUNK):
        chunk = ids[start:start + _SQL_IN_CHUNK]
        rows = (
            db.query(
                UserVotes.raw_data_id,
                func.sum(case((UserVotes.current_vote == 1, 1), else_=0)),
                func.sum(case((UserVotes.current_vote == -1, 1), else_=0)),
                func.count(UserVotes.id),
            )
            .filter(UserVotes.raw_data_id.in_(chunk))
            .group_by(UserVotes.raw_data_id)
            .all()
        )
        for raw_id, likes, dislikes, total in rows:
            likes, dislikes = int(likes or 0), int(dislikes or 0)
            stats[raw_id] = {
                "likes": likes,
                "dislikes": dislikes,
                "total_votes": int(total or 0),
                "score": likes - dislikes,
            }
    return stats


_NO_VOTES = {"likes": 0, "dislikes": 0, "total_votes": 0, "score": 0}


def _feedback_counts(db: Session) -> Tuple[int, int]:
    """(liked, disliked) question counts with the rule of ``_point``: the legacy
    ``raw_data.like`` value wins, rows without one count by their vote score.

    Only rows whose ``like`` is NULL need their votes summed (database_models
    keeps ``like`` in sync with the votes), so the vote aggregate is computed
    once and only for those rows - not over the whole user_votes table twice.
    """
    vote_sign = case((UserVotes.current_vote == 1, 1), (UserVotes.current_vote == -1, -1), else_=0)
    null_like_scores = (
        select(func.sum(vote_sign).label("score"))
        .select_from(RawData)
        .join(UserVotes, UserVotes.raw_data_id == RawData.id)
        .where(RawData.like.is_(None))
        .group_by(UserVotes.raw_data_id)
        .cte("null_like_scores")
    )

    def with_like(value: int):
        return select(func.count()).select_from(RawData).where(RawData.like == value).scalar_subquery()

    def with_score(condition):
        return select(func.count()).select_from(null_like_scores).where(condition).scalar_subquery()

    liked, disliked = db.execute(
        select(
            with_like(1) + with_score(null_like_scores.c.score > 0),
            with_like(-1) + with_score(null_like_scores.c.score < 0),
        )
    ).one()
    return int(liked or 0), int(disliked or 0)


def _point(legacy_like: Optional[int], vote_score: int) -> Optional[int]:
    """1 = liked, -1 = disliked, None = no feedback (legacy column wins)."""
    if legacy_like is not None:
        return legacy_like
    if vote_score > 0:
        return 1
    if vote_score < 0:
        return -1
    return None


def _raw_to_response(item: RawData, votes: Optional[Dict[str, int]] = None) -> RawDataResponse:
    votes = votes or _NO_VOTES
    return RawDataResponse(
        id=item.id,
        telegram_id=item.telegram_id,
        telegram_message_id=item.telegram_message_id,
        username=item.username,
        question=item.question,
        answer=item.answer,
        language=item.language,
        like=item.like,
        admin_approved=item.admin_approved or 0,
        is_duplicate=bool(item.is_duplicate),
        duplicate_of_id=item.duplicate_of_id,
        similarity_score=item.similarity_score,
        created_at=item.created_at,
        answered_at=item.answered_at,
        message_thread_id=item.message_thread_id,
        likes=votes["likes"],
        dislikes=votes["dislikes"],
        total_votes=votes["total_votes"],
        vote_score=votes["score"],
    )


# =============================================================================
# Duplicate detection
#
# The similarity measure, thresholds and matching rules are those of
# database_models (mark_duplicate_questions / mark_duplicate_answers), which
# the bot uses when a question arrives - so markings made by the bot and by
# the admin re-check (/detect-duplicates) mean the same thing. The re-check
# computes everything in memory and writes once (one transaction, no
# per-row query + commit).
# =============================================================================

QUESTION_DUPLICATE_THRESHOLD = 0.45  # as database_models.mark_duplicate_questions
ANSWER_DUPLICATE_THRESHOLD = 0.85  # as database_models.mark_duplicate_answers
QUESTION_WINDOW = 100  # as database_models.find_similar_questions(limit=100)
MIN_DUPLICATE_TEXT_LENGTH = 10  # only texts with len(strip()) > 10 are checked


def _meaningful(text_value: Optional[str]) -> bool:
    return bool(text_value) and len(text_value.strip()) > MIN_DUPLICATE_TEXT_LENGTH


def _similarity(text1: str, text2: str) -> float:
    return database_models.calculate_cosine_similarity(text1, text2)


def find_question_duplicates(
    rows: Sequence[Tuple[int, Optional[str]]],
) -> Dict[int, Tuple[int, float]]:
    """In-memory equivalent of clearing all markings and then running
    ``mark_duplicate_questions`` for every meaningful question in id order.

    Each question is compared with the ``QUESTION_WINDOW`` questions right
    before it; the oldest match is followed to its original (flat groups, no
    cycles). Returns ``{duplicate_id: (original_id, similarity)}``.
    """
    rows = sorted(rows, key=lambda r: r[0])
    result: Dict[int, Tuple[int, float]] = {}
    for idx, (qid, question) in enumerate(rows):
        if not _meaningful(question):
            continue
        matches = []
        for cid, candidate in rows[max(0, idx - QUESTION_WINDOW):idx]:
            if not candidate or not candidate.strip():
                continue
            score = _similarity(question, candidate)
            if score >= QUESTION_DUPLICATE_THRESHOLD:
                matches.append((cid, score))
        if not matches:
            continue
        match_id, score = min(matches)  # oldest matching question
        root, seen = match_id, {qid}
        while root in result and root not in seen:
            seen.add(root)
            root = result[root][0]
        result[qid] = (root, score)
    return result


def find_answer_duplicates(
    rows: Sequence[Tuple[int, Optional[str]]],
) -> Dict[int, Tuple[int, float]]:
    """In-memory equivalent of running ``mark_duplicate_answers`` for every
    meaningful answer in id order: each one points to the most similar older
    answer (ties -> oldest). Returns ``{duplicate_id: (original_id, similarity)}``.
    """
    rows = sorted(rows, key=lambda r: r[0])
    result: Dict[int, Tuple[int, float]] = {}
    for idx, (tid, answer) in enumerate(rows):
        if not _meaningful(answer):
            continue
        best: Optional[Tuple[float, int]] = None
        for cid, candidate in rows[:idx]:
            if not candidate or not candidate.strip():
                continue
            score = _similarity(answer, candidate)
            if score >= ANSWER_DUPLICATE_THRESHOLD and (best is None or score > best[0]):
                best = (score, cid)
        if best is not None:
            result[tid] = (best[1], best[0])
    return result


_CLEAR_RAW_DUPLICATE = {
    RawData.is_duplicate: False,
    RawData.duplicate_of_id: None,
    RawData.similarity_score: None,
}
_CLEAR_TRAINING_DUPLICATE = {
    TrainingData.is_answer_duplicate: False,
    TrainingData.duplicate_answer_of_id: None,
    TrainingData.answer_similarity_score: None,
}

# ai_model_worker tags rows it answered itself (apology after repeated failures,
# prompt_guard refusals) with "[auto] ..." in admin_notes.
AUTO_ANSWER_MARKER = "[auto]"
AUTO_ANSWER_EDITED_MARKER = "[auto-edited]"


def _has_auto_answer_marker(raw_data) -> bool:
    return AUTO_ANSWER_MARKER in (raw_data.admin_notes or "")


# =============================================================================
# Public endpoints
# =============================================================================


@app.get("/")
def root():
    return {"message": "University Bot Admin API"}


# Prime psutil so cpu_percent(interval=None) is meaningful and non-blocking.
psutil.cpu_percent(interval=None)

_GPU_CACHE_TTL = 30.0
_gpu_cache: Dict[str, object] = {"ts": 0.0, "value": None}


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def _gpu_info() -> Dict[str, object]:
    """GPU info via nvidia-smi (no torch import, no CUDA context in this process)."""
    now = time.monotonic()
    cached = _gpu_cache.get("value")
    if cached is not None and now - float(_gpu_cache["ts"]) < _GPU_CACHE_TTL:
        return dict(cached)

    info: Dict[str, object] = {"available": False}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [
                    smi,
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            devices = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                devices.append(
                    {
                        "index": int(parts[0]) if parts[0].isdigit() else len(devices),
                        "name": parts[1],
                        "memory_used_mb": _to_float(parts[2]),
                        "memory_total_mb": _to_float(parts[3]),
                        "utilization_percent": _to_float(parts[4]),
                    }
                )
            if devices:
                info = {
                    "available": True,
                    "device_count": len(devices),
                    "current_device": 0,
                    "device_name": devices[0]["name"],
                    "devices": devices,
                }
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.debug("nvidia-smi query failed: %s", exc)
    _gpu_cache.update(ts=now, value=dict(info))
    return info


def _model_path() -> str:
    model_path = os.getenv("MODEL_PATH") or "models/active-model"
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    return model_path


def _health_authenticated(credentials: Optional[HTTPBasicCredentials]) -> bool:
    """Credentials check for the admin view of /health.

    Without a limit /health would be an unlimited password oracle. Its wrong
    passwords are counted in one bucket for all clients - not per address,
    because a proxy route that forwards the client's own X-Forwarded-For would
    make the address attacker-chosen - and never in the API's per-address
    buckets, so forged /health failures cannot lock the admin out of the API.
    While the bucket is locked /health ignores credentials (public view only);
    the API and the public health status are unaffected.
    """
    if credentials is None:
        return False
    if health_credentials_limiter.retry_after(HEALTH_LIMIT_KEY):
        return False
    if credentials_are_valid(credentials):
        return True
    health_credentials_limiter.record_failure(HEALTH_LIMIT_KEY)
    return False


@app.get("/health")
def health_check(credentials: Optional[HTTPBasicCredentials] = Depends(optional_credentials)):
    """System health check.

    Public: status, timestamp, database and model availability (HTTP 503 when
    the database is unreachable). System and GPU metrics are only included for
    authenticated admins.
    """
    db_healthy = True
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Health check: database unreachable: %s", exc)
        db_healthy = False

    body: Dict[str, object] = {
        "status": "healthy" if db_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": {"healthy": db_healthy},
        "model": {"available": os.path.exists(_model_path())},
    }

    if _health_authenticated(credentials):
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(PROJECT_ROOT)
            body["system"] = {
                "cpu_usage_percent": psutil.cpu_percent(interval=None),
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
            }
        except Exception as exc:
            logger.warning("Health check: system metrics failed: %s", exc)
            body["system"] = {}
        body["gpu"] = _gpu_info()

    return JSONResponse(body, status_code=200 if db_healthy else 503)


@app.post("/auth/login")
def login(username: str = Depends(verify_credentials)):
    """Verify login credentials"""
    return {"success": True, "username": username}


# =============================================================================
# Protected endpoints (router-level auth)
# =============================================================================


@protected.get("/raw-data", response_model=PaginatedResponse[RawDataResponse])
def get_raw_data(
    page: int = Query(1, ge=1, le=MAX_PAGE),
    page_size: int = Query(20, ge=1, le=100),
    only_unapproved: bool = False,
    db: Session = Depends(get_db),
):
    """Get all raw data with page-based pagination (newest first)."""
    try:
        query = db.query(RawData)
        if only_unapproved:
            # == "admin_approved = 0 OR admin_approved IS NULL", written so that
            # SQLite walks idx_raw_data_created_at in ORDER BY order and stops
            # after one page, instead of collecting every pending row through
            # the (low-selectivity) admin_approved index and sorting them all.
            query = query.filter(func.coalesce(RawData.admin_approved, 0) == 0)

        total = query.with_entities(func.count(RawData.id)).scalar() or 0
        total_pages, skip = _paginate(total, page, page_size)

        items = (
            query.order_by(RawData.created_at.desc(), RawData.id.desc())
            .offset(skip)
            .limit(page_size)
            .all()
        )
        votes = _vote_stats_for(db, [item.id for item in items])
        data = [_raw_to_response(item, votes.get(item.id)) for item in items]

        return PaginatedResponse[RawDataResponse](
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise _db_error(e, "select", "raw_data")


@protected.put("/raw-data/{item_id}")
def update_answer(
    request: UpdateAnswerRequest,
    item_id: int = ItemId(),
    db: Session = Depends(get_db),
):
    """Update the answer of a raw data entry (and of its training copy, if any)."""
    try:
        raw_data = db.get(RawData, item_id)
        if not raw_data:
            raise HTTPException(status_code=404, detail="Item not found")

        now = datetime.utcnow()
        raw_data.answer = request.answer
        raw_data.answered_at = now
        raw_data.last_updated = now
        if _has_auto_answer_marker(raw_data):
            # The admin replaced the worker's automatic refusal/error answer,
            # so the row may now be approved for training.
            raw_data.admin_notes = raw_data.admin_notes.replace(
                AUTO_ANSWER_MARKER, AUTO_ANSWER_EDITED_MARKER
            )

        training = db.query(TrainingData).filter(TrainingData.source_id == item_id).first()
        training_updated = False
        if training is not None and training.answer != request.answer:
            # Keep the approved training copy in sync with the edited answer.
            training.answer = request.answer
            training.last_updated = now
            # Duplicate markings computed for the old answer are stale now.
            training.is_answer_duplicate = False
            training.duplicate_answer_of_id = None
            training.answer_similarity_score = None
            db.query(TrainingData).filter(
                TrainingData.duplicate_answer_of_id == training.id
            ).update(_CLEAR_TRAINING_DUPLICATE, synchronize_session=False)
            training_updated = True
        db.commit()

        if training_updated:
            training_id = training.id
            try:  # best effort, the edit itself is saved
                mark_duplicate_answers(db, training_id, request.answer)
            except Exception as exc:
                db.rollback()
                logger.warning("Answer duplicate check failed for training %s: %s", training_id, exc)

        return {
            "success": True,
            "message": "Answer updated successfully",
            "training_data_updated": training_updated,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise _db_error(e, "update", "raw_data")


@protected.post("/approve/{item_id}", response_model=ApproveResponse)
def approve_to_training(
    item_id: int = ItemId(),
    db: Session = Depends(get_db),
    username: str = Depends(verify_credentials),
):
    """Approve raw data: copy it to training data and check for duplicate answers."""
    try:
        raw_data = db.get(RawData, item_id)
        if not raw_data:
            raise HTTPException(status_code=404, detail="Item not found")

        if not raw_data.answer or not raw_data.answer.strip():
            raise HTTPException(status_code=400, detail="Cannot approve without answer")

        if _has_auto_answer_marker(raw_data):
            # Worker-generated apology / prompt-guard refusal: never train on it as-is.
            raise HTTPException(
                status_code=400,
                detail="This is an automatic error/refusal answer; edit the answer before approving",
            )

        existing = db.query(TrainingData.id).filter(TrainingData.source_id == item_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Already in training data")

        now = datetime.utcnow()
        training_data = TrainingData(
            source_id=raw_data.id,
            question=raw_data.question,
            answer=raw_data.answer,
            language=raw_data.language,
            source_type="original",
            review_status="approved",
            reviewer_id=username,
            review_date=now,
        )
        db.add(training_data)
        raw_data.admin_approved = 1
        raw_data.last_updated = now
        try:
            db.commit()
        except IntegrityError:
            # Concurrent approval of the same item (UNIQUE(source_id)).
            db.rollback()
            raise HTTPException(status_code=400, detail="Already in training data")
        db.refresh(training_data)
        training_id = training_data.id
        question, answer = raw_data.question, training_data.answer

        # Duplicate detection - best effort, the approval is already committed.
        # (database_models only links to *older* rows, so approving an
        # original can no longer create A<->B cycles.)
        try:
            mark_duplicate_questions(db, item_id, question)
            mark_duplicate_answers(db, training_id, answer)
        except Exception as exc:
            db.rollback()
            logger.warning("Duplicate check failed for approved item %s: %s", item_id, exc)

        return ApproveResponse(
            success=True,
            message="Successfully approved to training data with duplicate detection",
            training_data_id=training_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise _db_error(e, "insert", "training_data")


@protected.get("/stats")
def get_statistics(db: Session = Depends(get_db)):
    """Get basic statistics"""
    try:
        total_questions = db.query(func.count(RawData.id)).scalar() or 0
        approved_questions = (
            db.query(func.count(RawData.id)).filter(RawData.admin_approved == 1).scalar() or 0
        )

        liked_questions, disliked_questions = _feedback_counts(db)
        training_data_count = db.query(func.count(TrainingData.id)).scalar() or 0
        duplicate_count = (
            db.query(func.count(RawData.id)).filter(RawData.is_duplicate.is_(True)).scalar() or 0
        )
        unanswered_questions = (
            db.query(func.count(RawData.id))
            .filter(or_(RawData.answer.is_(None), func.trim(RawData.answer) == ""))
            .scalar()
            or 0
        )

        return {
            "total_questions": total_questions,
            "approved_questions": approved_questions,
            "liked_questions": liked_questions,
            "disliked_questions": disliked_questions,
            "training_data_count": training_data_count,
            "duplicate_count": duplicate_count,
            "unanswered_questions": unanswered_questions,
            "approval_rate": f"{(approved_questions / total_questions * 100):.1f}%"
            if total_questions > 0
            else "0%",
        }
    except Exception as e:
        raise _db_error(e, "select", "raw_data")


MAX_DUPLICATE_GROUPS = 1000


@protected.get("/duplicates")
def get_duplicate_groups(
    response: Response,
    limit: int = Query(500, ge=1, le=MAX_DUPLICATE_GROUPS),
    offset: int = Query(0, ge=0, le=MAX_DB_ID),
    db: Session = Depends(get_db),
):
    """Get duplicate question groups: [{original, duplicates: [...]}, ...]

    Groups are ordered by original id; ``limit``/``offset`` page through them
    and the ``X-Total-Count`` header holds the total number of groups.
    """
    try:
        group_ids = select(RawData.duplicate_of_id).where(RawData.duplicate_of_id.isnot(None))
        originals_query = db.query(RawData).filter(RawData.id.in_(group_ids))
        total = originals_query.with_entities(func.count(RawData.id)).scalar() or 0
        response.headers["X-Total-Count"] = str(total)
        originals = originals_query.order_by(RawData.id).offset(offset).limit(limit).all()
        if not originals:
            return []

        by_original: Dict[int, List[RawData]] = {}
        original_ids = [original.id for original in originals]
        for start in range(0, len(original_ids), _SQL_IN_CHUNK):
            chunk = original_ids[start:start + _SQL_IN_CHUNK]
            for dup in (
                db.query(RawData)
                .filter(RawData.duplicate_of_id.in_(chunk))
                .order_by(RawData.duplicate_of_id, RawData.id)
            ):
                by_original.setdefault(dup.duplicate_of_id, []).append(dup)
        votes = _vote_stats_for(
            db, original_ids + [d.id for dups in by_original.values() for d in dups]
        )

        return [
            {
                "original": _raw_to_response(original, votes.get(original.id)),
                "duplicates": [
                    _raw_to_response(d, votes.get(d.id)) for d in by_original.get(original.id, [])
                ],
            }
            for original in originals
        ]
    except Exception as e:
        raise _db_error(e, "select", "raw_data")


@protected.get("/training-data", response_model=PaginatedResponse[TrainingDataResponse])
def get_training_data(
    page: int = Query(1, ge=1, le=MAX_PAGE),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get all training data with page-based pagination (newest first)."""
    try:
        total = db.query(func.count(TrainingData.id)).scalar() or 0
        total_pages, skip = _paginate(total, page, page_size)

        # LEFT JOIN: source_id may be NULL (standalone training data)
        rows = (
            db.query(
                TrainingData.id,
                TrainingData.source_id,
                TrainingData.question,
                TrainingData.answer,
                TrainingData.language,
                TrainingData.created_at,
                TrainingData.is_answer_duplicate,
                TrainingData.duplicate_answer_of_id,
                TrainingData.answer_similarity_score,
                RawData.id.label("raw_id"),
                RawData.like.label("legacy_like"),
            )
            .outerjoin(RawData, TrainingData.source_id == RawData.id)
            .order_by(TrainingData.created_at.desc(), TrainingData.id.desc())
            .offset(skip)
            .limit(page_size)
            .all()
        )
        votes = _vote_stats_for(db, [r.raw_id for r in rows if r.raw_id is not None])

        training_data = [
            TrainingDataResponse(
                id=r.id,
                source_id=r.source_id,
                question=r.question,
                answer=r.answer,
                language=r.language,
                created_at=r.created_at,
                point=(
                    _point(r.legacy_like, votes.get(r.raw_id, _NO_VOTES)["score"])
                    if r.raw_id is not None
                    else None
                ),
                is_answer_duplicate=r.is_answer_duplicate,
                duplicate_answer_of_id=r.duplicate_answer_of_id,
                answer_similarity_score=r.answer_similarity_score,
            )
            for r in rows
        ]

        return PaginatedResponse[TrainingDataResponse](
            data=training_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    except Exception as e:
        raise _db_error(e, "select", "training_data")


@protected.delete("/raw-data/{item_id}")
def delete_raw_data(item_id: int = ItemId(), db: Session = Depends(get_db)):
    """Delete a raw data entry together with its training copy and votes."""
    try:
        raw_data = db.get(RawData, item_id)
        if not raw_data:
            raise HTTPException(status_code=404, detail="Raw data not found")

        training_ids = [
            tid for (tid,) in db.query(TrainingData.id).filter(TrainingData.source_id == item_id)
        ]
        if training_ids:
            db.query(TrainingData).filter(
                TrainingData.duplicate_answer_of_id.in_(training_ids)
            ).update(_CLEAR_TRAINING_DUPLICATE, synchronize_session=False)
            db.query(TrainingData).filter(TrainingData.id.in_(training_ids)).delete(
                synchronize_session=False
            )

        # No dangling references to the deleted row.
        db.query(RawData).filter(RawData.duplicate_of_id == item_id).update(
            _CLEAR_RAW_DUPLICATE, synchronize_session=False
        )
        db.query(UserVotes).filter(UserVotes.raw_data_id == item_id).delete(
            synchronize_session=False
        )
        db.delete(raw_data)
        db.commit()

        return {"success": True, "message": "Raw data deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise _db_error(e, "delete", "raw_data")


@protected.delete("/training-data/{item_id}")
def delete_training_data(item_id: int = ItemId(), db: Session = Depends(get_db)):
    """Remove from training data (the raw entry goes back to 'pending')."""
    try:
        training_data = db.get(TrainingData, item_id)
        if not training_data:
            raise HTTPException(status_code=404, detail="Training data not found")

        if training_data.source_id is not None:
            raw_data = db.get(RawData, training_data.source_id)
            if raw_data:
                raw_data.admin_approved = 0
                raw_data.last_updated = datetime.utcnow()

        db.query(TrainingData).filter(TrainingData.duplicate_answer_of_id == item_id).update(
            _CLEAR_TRAINING_DUPLICATE, synchronize_session=False
        )
        db.delete(training_data)
        db.commit()

        return {"success": True, "message": "Training data removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise _db_error(e, "delete", "training_data")


@protected.post("/detect-duplicates")
def detect_duplicates(db: Session = Depends(get_db)):
    """Recompute duplicate questions (raw data) and answers (training data).

    Everything is computed first and then written in one transaction, so a
    failure leaves the previous markings untouched.
    """
    try:
        raw_rows = db.query(RawData.id, RawData.question).order_by(RawData.id).all()
        logger.info("Starting duplicate detection for %d raw data entries", len(raw_rows))
        question_dups = find_question_duplicates(raw_rows)

        training_rows = db.query(TrainingData.id, TrainingData.answer).order_by(TrainingData.id).all()
        logger.info("Starting duplicate detection for %d training data entries", len(training_rows))
        answer_dups = find_answer_duplicates(training_rows)

        db.query(RawData).update(_CLEAR_RAW_DUPLICATE, synchronize_session=False)
        db.query(TrainingData).update(_CLEAR_TRAINING_DUPLICATE, synchronize_session=False)
        if question_dups:
            db.execute(
                update(RawData),
                [
                    {"id": dup_id, "is_duplicate": True, "duplicate_of_id": orig, "similarity_score": score}
                    for dup_id, (orig, score) in question_dups.items()
                ],
            )
        if answer_dups:
            db.execute(
                update(TrainingData),
                [
                    {
                        "id": dup_id,
                        "is_answer_duplicate": True,
                        "duplicate_answer_of_id": orig,
                        "answer_similarity_score": score,
                    }
                    for dup_id, (orig, score) in answer_dups.items()
                ],
            )
        db.commit()

        return {
            "success": True,
            "message": (
                f"Duplicate detection completed. Found {len(question_dups)} question duplicates "
                f"and {len(answer_dups)} answer duplicates"
            ),
            "question_duplicates": len(question_dups),
            "answer_duplicates": len(answer_dups),
        }
    except Exception as e:
        db.rollback()
        logger.error("Error in duplicate detection: %s", e)
        raise _db_error(e, "update", "raw_data")


# =============================================================================
# Documentation / log viewer (protected)
# =============================================================================

DOCS_MAX_BYTES = 1024 * 1024  # never load more than 1 MiB of a file (logs reach 80+ GB)
_DOC_EXCLUDED_DIRS = {
    "node_modules",
    "__pycache__",
    "model_cache",
    "build",
    "site-packages",
    "venv",
    "env",
}


def _project_root() -> FsPath:
    return FsPath(PROJECT_ROOT).resolve()


def classify_document(rel_path: PurePosixPath) -> Optional[str]:
    """Return 'documentation' / 'log' / 'script' for viewable files, else None.

    Only Markdown/README docs, ``logs/*.log`` and ``scripts/*.sh`` are
    viewable - never ``.env``, the database, model files, etc.
    """
    parts = rel_path.parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        return None
    if len(parts) == 2 and parts[0] == "logs" and rel_path.suffix == ".log":
        return "log"
    if len(parts) == 2 and parts[0] == "scripts" and rel_path.suffix == ".sh":
        return "script"
    dirs = parts[:-1]
    if any(d.startswith(".") or d in _DOC_EXCLUDED_DIRS for d in dirs):
        return None
    if rel_path.suffix.lower() == ".md" or rel_path.name == "README":
        return "documentation"
    return None


def _iter_documents(root: FsPath):
    """Yield (relative posix path, absolute path, type) for listable files."""
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = FsPath(dirpath).relative_to(root)
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in _DOC_EXCLUDED_DIRS
        )
        for name in sorted(filenames):
            rel = PurePosixPath(*rel_dir.parts, name)
            doc_type = classify_document(rel)
            if not doc_type:
                continue
            absolute = FsPath(dirpath) / name
            if absolute.is_symlink():
                try:
                    absolute.resolve().relative_to(root)
                except (ValueError, OSError):
                    continue  # symlink pointing outside the project
            yield rel, absolute, doc_type


@protected.get("/docs/list")
def list_documents():
    """List viewable documentation, log and script files."""
    root = _project_root()
    docs = []
    for rel, absolute, doc_type in _iter_documents(root):
        try:
            size = absolute.stat().st_size
        except OSError:
            continue  # broken symlink / race
        docs.append({"path": str(rel), "name": rel.name, "type": doc_type, "size": size})
    order = {"documentation": 0, "log": 1, "script": 2}
    return sorted(docs, key=lambda d: (order.get(d["type"], 9), d["path"]))


def _format_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num} B"  # pragma: no cover


@protected.get("/docs/read", response_class=PlainTextResponse)
def read_document(path: str = Query(..., min_length=1, max_length=1024)):
    """Read a documentation/log/script file (logs: last 1 MiB, others: first 1 MiB)."""
    if "\x00" in path or path.startswith(("/", "\\")) or ".." in PurePosixPath(path).parts:
        raise HTTPException(status_code=400, detail="Invalid path")

    root = _project_root()
    try:
        full_path = (root / path).resolve()
        rel = PurePosixPath(*full_path.relative_to(root).parts)
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")

    doc_type = classify_document(rel)
    if doc_type is None:
        raise HTTPException(status_code=403, detail="File type not viewable")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    try:
        size = full_path.stat().st_size
        truncated = size > DOCS_MAX_BYTES
        with open(full_path, "rb") as f:
            if truncated and doc_type == "log":
                f.seek(size - DOCS_MAX_BYTES)
                data = f.read(DOCS_MAX_BYTES)
                newline = data.find(b"\n")
                if 0 <= newline < len(data) - 1:
                    data = data[newline + 1:]  # drop the partial first line
            else:
                data = f.read(DOCS_MAX_BYTES)
        content = data.decode("utf-8", errors="replace")
    except OSError as e:
        logger.warning("Error reading document %s: %s", rel, e)
        raise HTTPException(status_code=500, detail="Error reading file")

    if truncated:
        note = (
            f"[... file is {_format_bytes(size)}; showing "
            f"{'last' if doc_type == 'log' else 'first'} {_format_bytes(len(data))} ...]"
        )
        content = f"{note}\n{content}" if doc_type == "log" else f"{content}\n{note}"

    return PlainTextResponse(
        content,
        headers={"X-Content-Truncated": "true" if truncated else "false", "X-File-Size": str(size)},
    )


app.include_router(protected)

if __name__ == "__main__":
    if config.api_reload:
        logger.warning("API_RELOAD is enabled - development mode only.")
        uvicorn.run(
            "admin_rest_api:app",
            host=config.api_host,
            port=config.api_port,
            reload=True,
            app_dir=SRC_DIR,
            reload_dirs=[SRC_DIR],
        )
    else:
        uvicorn.run(app, host=config.api_host, port=config.api_port)
