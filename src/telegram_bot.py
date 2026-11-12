"""
Telegram Bot for CengBot - AI Assistant (LEGACY entrypoint)
===========================================================

NOTE: the production bot is ``src/telegram_bot_rabbitmq.py`` (started by
``scripts/start_system.sh``). This module is an older, alternative RabbitMQ
entrypoint kept for reference / manual use. Do NOT run it at the same time as
the production bot: both would poll Telegram with the same token and both
would consume the same ``answers`` queue, stealing each other's answers.

This module implements a Telegram bot that handles user interactions,
processes messages, and communicates with the AI model through RabbitMQ.

Features:
- Telegram message handling
- Language detection (Turkish/English)
- Duplicate question detection
- User feedback system (like/dislike)
- RabbitMQ integration for AI processing
- SQLite database integration

Author: naholav
"""

import os
import sys

# Make ``python3 src/telegram_bot.py`` work from the project root: ``src/`` for
# sibling modules (``database_models``) and the project root for ``config``.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

import asyncio
import html
import json
import logging
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pika
from langdetect import DetectorFactory, LangDetectException, detect
from sqlalchemy import text
from sqlalchemy.orm import Session
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Import project modules
from database_models import RawData, SessionLocal, TrainingData, init_db
from config.env_loader import get_config

# Get configuration
config = get_config()

# langdetect is non-deterministic unless seeded.
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Seconds to wait for the AI worker's answer before telling the user to retry.
ANSWER_TIMEOUT_SECONDS = 30.0

# Telegram hard limit for a text message (counted in UTF-16 code units).
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
# How much of the user's question is echoed back in the reply.
MAX_QUESTION_ECHO_LENGTH = 500

# Rate limiting (same policy as the production bot).
RATE_LIMIT_MESSAGES = 5
RATE_LIMIT_WINDOW = 60.0

TECHNICAL_DIFFICULTY_MESSAGE = (
    "⚠️ Sorry, I'm experiencing technical difficulties. Please try again later "
    "or contact the department office: +90 322 338 60 10"
)
TIMEOUT_MESSAGE = (
    "⏰ Response timeout. Please try again later or contact the department "
    "office for assistance."
)
RATE_LIMIT_MESSAGE = (
    "⚠️ You're sending messages too quickly. Please wait a moment before "
    "sending another message."
)

_TURKISH_CHARS = set("çğışöüÇĞİŞÖÜ")
_CALLBACK_RE = re.compile(r"(like|dislike)_(\d{1,18})")


# =============================================================================
# Helpers
# =============================================================================

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text: Input text to analyze

    Returns:
        Language code ('TR' for Turkish, 'EN' for English)
    """
    if not text or not text.strip():
        return 'TR'
    if any(ch in _TURKISH_CHARS for ch in text):
        return 'TR'
    try:
        detected_lang = detect(text)
        return 'TR' if detected_lang == 'tr' else 'EN'
    except LangDetectException:
        logger.debug("Could not detect language for text: %r", text[:50])
        return 'TR'  # Default to Turkish for Cukurova University


def _utf16_len(value: str) -> int:
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in value)


def truncate_text(value: str, limit: int, suffix: str = "…") -> str:
    """Truncate ``value`` to at most ``limit`` UTF-16 code units."""
    if limit <= 0:
        return ""
    if _utf16_len(value) <= limit:
        return value
    budget = limit - _utf16_len(suffix)
    if budget <= 0:
        return suffix[:limit]
    out = []
    used = 0
    for ch in value:
        width = 2 if ord(ch) > 0xFFFF else 1
        if used + width > budget:
            break
        out.append(ch)
        used += width
    return "".join(out) + suffix


def _display_name(user) -> str:
    return (getattr(user, "username", None) or getattr(user, "first_name", None)
            or str(getattr(user, "id", "user")))


def parse_callback_data(data) -> Optional[Tuple[str, int]]:
    """Validate inline-button data of the form ``like_<id>`` / ``dislike_<id>``."""
    if not isinstance(data, str):
        return None
    match = _CALLBACK_RE.fullmatch(data)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _feedback_keyboard(raw_data_id: int, like: Optional[int] = None) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("👍 Helpful" + (" ✓" if like == 1 else ""),
                             callback_data=f"like_{raw_data_id}"),
        InlineKeyboardButton("👎 Not Helpful" + (" ✓" if like == -1 else ""),
                             callback_data=f"dislike_{raw_data_id}"),
    ]])


def _topic_id() -> Optional[int]:
    try:
        return config.telegram_topic_id
    except ValueError:
        logger.warning("Ignoring invalid TELEGRAM_TOPIC_ID (not an integer)")
        return None


def _parse_allowed_chat_ids() -> Optional[frozenset]:
    """Optional allow-list (env ``TELEGRAM_ALLOWED_CHAT_IDS``, comma separated).

    ``None`` means "not configured" (every chat is served). If the variable is
    set, invalid entries are skipped; an all-invalid list serves no chat.
    """
    # Fall back to the production bot's single-id variable so one setting
    # covers every bot.
    raw = (os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "") or os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "")).strip()
    if not raw:
        return None
    ids = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.add(int(item))
        except ValueError:
            logger.error("Ignoring invalid chat id in TELEGRAM_ALLOWED_CHAT_IDS: %r", item)
    return frozenset(ids)


ALLOWED_CHAT_IDS: Optional[frozenset] = _parse_allowed_chat_ids()


def is_chat_allowed(chat_id) -> bool:
    return ALLOWED_CHAT_IDS is None or chat_id in ALLOWED_CHAT_IDS


class RateLimiter:
    """Sliding-window per-user rate limiter with bounded memory."""

    def __init__(self, max_messages: int = RATE_LIMIT_MESSAGES,
                 window_seconds: float = RATE_LIMIT_WINDOW,
                 max_tracked_users: int = 10000,
                 clock: Callable[[], float] = time.monotonic):
        self.max_messages = max_messages
        self.window = window_seconds
        self.max_tracked_users = max_tracked_users
        self._clock = clock
        self._hits: Dict[int, deque] = {}
        self._last_prune = clock()

    def __len__(self) -> int:
        return len(self._hits)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        for uid in [u for u, q in self._hits.items() if not q or q[-1] <= cutoff]:
            del self._hits[uid]
        if len(self._hits) > self.max_tracked_users:
            # Still too many active users: forget the least recently active.
            ordered = sorted(self._hits.items(), key=lambda kv: kv[1][-1])
            for uid, _ in ordered[:len(self._hits) - self.max_tracked_users]:
                del self._hits[uid]
        self._last_prune = now

    def is_limited(self, user_id: int) -> bool:
        now = self._clock()
        if now - self._last_prune >= self.window or len(self._hits) >= self.max_tracked_users:
            self._prune(now)
        hits = self._hits.get(user_id)
        if hits is None:
            hits = self._hits[user_id] = deque()
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.max_messages:
            return True
        hits.append(now)
        return False


rate_limiter = RateLimiter()


def is_rate_limited(user_id: int) -> bool:
    """Check (and record) a message for ``user_id``; True if over the limit."""
    return rate_limiter.is_limited(user_id)


async def _safe_send_typing(message) -> None:
    try:
        await message.chat.send_action("typing")
    except Exception as e:  # cosmetic only
        logger.debug("Could not send typing action: %s", e)


async def _reply_with_fallback(message, html_text: str, plain_text: str, reply_markup=None):
    """Send ``html_text`` as HTML; on a parse error fall back to plain text."""
    try:
        return await message.reply_text(html_text, reply_markup=reply_markup,
                                        parse_mode=ParseMode.HTML)
    except BadRequest as e:
        logger.warning("HTML reply rejected by Telegram (%s); resending as plain text", e)
        return await message.reply_text(plain_text, reply_markup=reply_markup)


def build_response_texts(display_name: str, question: str, language: str,
                         answer: str) -> Tuple[str, str]:
    """Build the (html, plain) reply texts, both within Telegram's length limit."""
    question = truncate_text(question, MAX_QUESTION_ECHO_LENGTH)
    display_name = truncate_text(display_name, 64)
    head_plain = (f"👤 Question from @{display_name}:\n❓ {question}\n\n"
                  f"🤖 AI Assistant ({language}):\n")
    foot = "\n\n💡 Was this response helpful?"
    budget = TELEGRAM_MAX_MESSAGE_LENGTH - _utf16_len(head_plain) - _utf16_len(foot)
    answer = truncate_text(answer, budget)

    esc = lambda s: html.escape(s, quote=False)  # noqa: E731
    html_text = (f"👤 <b>Question from @{esc(display_name)}:</b>\n❓ {esc(question)}\n\n"
                 f"🤖 <b>AI Assistant ({esc(language)}):</b>\n{esc(answer)}{foot}")
    plain_text = head_plain + answer + foot
    return html_text, plain_text


GENERIC_MODEL_NAME = "fine-tuned language model"


def _short_model_name(name) -> str:
    return str(name).strip().rstrip("/").split("/")[-1]


def model_display_name() -> str:
    """Name of the active model for /start and /stats, e.g. "Llama-3.2-3B + LoRA".
    Read from the active model directory's adapter_config.json (the model runs in
    the AI worker).
    Falls back to a generic name; never loads the model."""
    try:
        model_dir = str(config.model_path)
        for sub in ("", "method1"):
            path = os.path.join(model_dir, sub, "adapter_config.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    base = (json.load(f) or {}).get("base_model_name_or_path")
                return f"{_short_model_name(base)} + LoRA" if base else GENERIC_MODEL_NAME
    except Exception as e:
        logger.debug(f"Could not determine the model name: {e}")
    return GENERIC_MODEL_NAME


def check_similarity(db: Session, question: str, exclude_id: Optional[int] = None) -> List[tuple]:
    """
    Check for similar questions in the database using simple text matching.

    This function performs a basic similarity check by looking for questions
    that contain similar text patterns. In production, this could be enhanced
    with other similarity algorithms.

    Args:
        db: Database session
        question: The question text to check
        exclude_id: ID of the current question. Only *earlier* questions
            (``id < exclude_id``) are considered. If omitted, the most recent
            row is excluded (legacy behaviour, racy under concurrency).

    Returns:
        List of similar questions as (id, question) tuples, oldest first
    """
    try:
        # Take the first 20 characters as a pattern to find similar questions.
        pattern = (question or "").strip()[:20]
        if not pattern:
            return []
        # Escape LIKE wildcards so user text is matched literally.
        escaped = (pattern.replace("\\", "\\\\").replace("%", "\\%")
                   .replace("_", "\\_"))
        if exclude_id is not None:
            id_clause = "id < :exclude_id"
            params = {"pattern": f"%{escaped}%", "exclude_id": int(exclude_id)}
        else:
            id_clause = "id != (SELECT MAX(id) FROM raw_data)"
            params = {"pattern": f"%{escaped}%"}

        results = db.execute(
            text(f"""
            SELECT id, question FROM raw_data
            WHERE question LIKE :pattern ESCAPE '\\'
            AND {id_clause}
            ORDER BY id
            LIMIT 5
            """),
            params
        ).fetchall()

        logger.info(f"Found {len(results)} similar questions for pattern: {pattern}")
        return results

    except Exception as e:
        logger.error(f"Error in similarity check: {e}")
        return []


# =============================================================================
# RabbitMQ
# =============================================================================

def _connection_parameters() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=config.rabbitmq_host,
        port=config.rabbitmq_port,
        heartbeat=60,
        # Never block forever when the broker raises a resource alarm
        # (e.g. disk full): fail the publish instead.
        blocked_connection_timeout=30,
    )


def send_to_rabbitmq(message_data: dict) -> bool:
    """
    Send question data to RabbitMQ for AI processing.

    Blocking (opens a short-lived connection): call it from a worker thread,
    e.g. ``await asyncio.to_thread(send_to_rabbitmq, data)``.

    Args:
        message_data: Dictionary containing question data

    Returns:
        True if message was sent successfully, False otherwise
    """
    connection = None
    try:
        connection = pika.BlockingConnection(_connection_parameters())
        channel = connection.channel()

        # Declare the questions queue
        channel.queue_declare(queue=config.questions_queue, durable=True)

        # Send message to queue
        channel.basic_publish(
            exchange='',
            routing_key=config.questions_queue,
            body=json.dumps(message_data),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )

        logger.info(f"✅ Sent question to RabbitMQ: ID {message_data.get('raw_data_id')}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to send message to RabbitMQ: {e}")
        return False
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


class AnswerDispatcher:
    """
    Routes answers from the RabbitMQ consumer thread to the waiting handler.

    Each handler registers an ``asyncio.Future`` for its ``raw_data_id``; the
    consumer thread resolves it thread-safely. Answers nobody waits for any
    more (timed-out questions) are dropped instead of accumulating forever.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Dict[int, Tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)

    def register(self, raw_data_id: int) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            self._pending[int(raw_data_id)] = (loop, future)
        return future

    def discard(self, raw_data_id: int) -> None:
        with self._lock:
            self._pending.pop(int(raw_data_id), None)

    def dispatch(self, answer_data) -> bool:
        """Deliver an answer (thread-safe). Returns True if a handler was waiting."""
        try:
            raw_data_id = int(answer_data["raw_data_id"])
        except (TypeError, KeyError, ValueError):
            logger.warning("Dropping answer without a valid raw_data_id")
            return False
        with self._lock:
            entry = self._pending.pop(raw_data_id, None)
        if entry is None:
            logger.info("Dropping answer for question %s (no handler waiting)", raw_data_id)
            return False
        loop, future = entry

        def _resolve():
            if not future.done():
                future.set_result(answer_data)

        try:
            loop.call_soon_threadsafe(_resolve)
        except RuntimeError:  # event loop already closed
            return False
        return True

    async def wait_for(self, future: asyncio.Future, raw_data_id: int,
                       timeout: float) -> Optional[dict]:
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.discard(raw_data_id)


answer_dispatcher = AnswerDispatcher()
_consumer_stop = threading.Event()


def _on_answer(ch, method, properties, body) -> None:
    """pika callback (runs in the consumer thread)."""
    try:
        answer_data = json.loads(body)
        if not isinstance(answer_data, dict):
            raise ValueError("answer payload is not a JSON object")
    except Exception as e:
        logger.error(f"❌ Discarding malformed answer message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return
    logger.info(f"📥 Received answer from AI worker: ID {answer_data.get('raw_data_id')}")
    try:
        answer_dispatcher.dispatch(answer_data)
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)


def start_answer_consumer(stop_event: Optional[threading.Event] = None,
                          initial_delay: float = 1.0,
                          max_delay: float = 60.0) -> None:
    """
    Start the RabbitMQ consumer for receiving AI-generated answers.

    Runs in a separate thread, reconnecting with exponential backoff (capped
    at ``max_delay``) and logging repeated failures sparsely, until
    ``stop_event`` is set.
    """
    stop_event = stop_event or _consumer_stop
    delay = initial_delay
    failures = 0
    while not stop_event.is_set():
        connection = None
        try:
            connection = pika.BlockingConnection(_connection_parameters())
            channel = connection.channel()
            channel.queue_declare(queue=config.answers_queue, durable=True)
            channel.basic_qos(prefetch_count=10)
            channel.basic_consume(queue=config.answers_queue, on_message_callback=_on_answer)
            if failures:
                logger.info("✅ RabbitMQ answer consumer reconnected after %d failed attempt(s)",
                            failures)
            else:
                logger.info("🔄 Started consuming answers from RabbitMQ")
            failures = 0
            delay = initial_delay
            channel.start_consuming()
        except Exception as e:
            failures += 1
            # Log the 1st, 2nd, 4th, 8th, ... consecutive failure only (no
            # tracebacks) so an outage cannot flood the logs.
            if failures & (failures - 1) == 0:
                logger.warning("❌ RabbitMQ answer consumer error (attempt %d): %s; "
                               "retrying in %.0fs", failures, e, delay)
            else:
                logger.debug("RabbitMQ answer consumer error (attempt %d): %s", failures, e)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
        if stop_event.is_set():
            break
        if failures:
            stop_event.wait(delay)
            delay = min(delay * 2, max_delay)
        else:
            # start_consuming() returned without error (consumer cancelled).
            stop_event.wait(initial_delay)


# =============================================================================
# Database helpers (short-lived sessions; never held across awaits)
# =============================================================================

def _save_question(user, message, message_text: str, language: str) -> int:
    db = SessionLocal()
    try:
        raw_data = RawData(
            telegram_id=user.id,
            username=_display_name(user),
            question=message_text,
            language=language,
            message_thread_id=message.message_thread_id
        )
        db.add(raw_data)
        db.commit()
        db.refresh(raw_data)
        raw_data_id = raw_data.id
        logger.info(f"💾 Saved question to database with ID: {raw_data_id}")

        # Check for duplicate questions (earlier questions only)
        similar_questions = check_similarity(db, message_text, exclude_id=raw_data_id)
        if similar_questions:
            raw_data.is_duplicate = True
            raw_data.duplicate_of_id = similar_questions[0][0]
            db.commit()
            logger.info(f"🔄 Marked as duplicate of question ID: {similar_questions[0][0]}")
        return raw_data_id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _store_answer(raw_data_id: int, answer: Optional[str]) -> None:
    """Store the worker's answer unless the worker already did."""
    if answer is None:
        return  # timeout: never overwrite the DB with the timeout notice
    db = SessionLocal()
    try:
        raw_data = db.get(RawData, raw_data_id)
        if raw_data is not None and not raw_data.answer:
            raw_data.answer = answer
            raw_data.answered_at = datetime.utcnow()
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _store_message_id(raw_data_id: int, telegram_message_id: int) -> None:
    db = SessionLocal()
    try:
        raw_data = db.get(RawData, raw_data_id)
        if raw_data is not None:
            raw_data.telegram_message_id = telegram_message_id
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# Telegram handlers
# =============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming messages from Telegram users.

    This function processes user messages, detects language, checks for duplicates,
    saves to database, sends to AI worker via RabbitMQ, and responds to user.

    Args:
        update: Telegram update object containing message data
        context: Telegram context for bot operations
    """
    message = update.message
    # Validate message (edited messages / channel posts have no ``message``)
    if message is None or not message.text:
        return

    if not is_chat_allowed(message.chat.id):
        logger.warning(f"Ignoring message from non-allowed chat ID: {message.chat.id}")
        return

    # Check if message is in the correct topic (if specified)
    topic_id = _topic_id()
    if topic_id and message.message_thread_id != topic_id:
        return

    # Extract user and message information
    user = update.effective_user
    message_text = message.text
    if user is None:
        return

    # Skip command messages (those starting with '/')
    if message_text.startswith('/'):
        return

    name = _display_name(user)
    logger.info(f"📩 Received message from {name}: {message_text[:50]}...")

    if is_rate_limited(user.id):
        try:
            await message.reply_text(RATE_LIMIT_MESSAGE)
        except Exception as e:
            logger.warning(f"Could not send rate-limit notice: {e}")
        return

    try:
        # Detect language of the message
        detected_language = detect_language(message_text)
        logger.info(f"🌍 Detected language: {detected_language}")

        raw_data_id = _save_question(user, message, message_text, detected_language)

        # Show typing indicator to user
        await _safe_send_typing(message)

        # Prepare message data for RabbitMQ
        message_data = {
            'raw_data_id': raw_data_id,
            'telegram_id': user.id,
            'username': name,
            'question': message_text,
            'language': detected_language,
            'message_thread_id': message.message_thread_id,
            'update_message_id': message.message_id
        }

        # Register before publishing so a fast answer cannot be missed.
        future = answer_dispatcher.register(raw_data_id)
        try:
            sent = await asyncio.to_thread(send_to_rabbitmq, message_data)
            if not sent:
                await message.reply_text(TECHNICAL_DIFFICULTY_MESSAGE)
                return
            # Wait for the AI response (no polling, no DB session held).
            answer_data = await answer_dispatcher.wait_for(future, raw_data_id,
                                                           ANSWER_TIMEOUT_SECONDS)
        finally:
            answer_dispatcher.discard(raw_data_id)
        worker_answer = None
        if answer_data is not None:
            worker_answer = answer_data.get('answer')
            worker_answer = str(worker_answer).strip() if worker_answer is not None else ""
            if not worker_answer:
                worker_answer = None
        model_response = worker_answer or TIMEOUT_MESSAGE

        _store_answer(raw_data_id, worker_answer)

        html_text, plain_text = build_response_texts(
            name, message_text, detected_language, model_response)
        sent_message = await _reply_with_fallback(
            message, html_text, plain_text, reply_markup=_feedback_keyboard(raw_data_id))

        # Save Telegram message ID for future reference
        _store_message_id(raw_data_id, sent_message.message_id)

        logger.info(f"✅ Sent response to user: {name}")

    except Exception as e:
        logger.error(f"❌ Error handling message: {e}")
        try:
            await message.reply_text("❌ An error occurred. Please try again later.")
        except Exception as send_error:
            logger.warning(f"Could not send error notice: {send_error}")


async def _answer_query(query, text_: Optional[str] = None, show_alert: bool = False) -> None:
    try:
        await query.answer(text_, show_alert=show_alert)
    except Exception as e:
        logger.debug(f"Could not answer callback query: {e}")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline button callbacks (like/dislike feedback).

    The callback query is answered exactly once (Telegram rejects a second
    answer).

    Args:
        update: Telegram update object containing callback data
        context: Telegram context for bot operations
    """
    query = update.callback_query
    if query is None:
        return

    if query.message is not None and not is_chat_allowed(query.message.chat.id):
        await _answer_query(query, "❌ Unauthorized!", show_alert=True)
        return

    parsed = parse_callback_data(query.data)
    if parsed is None:
        await _answer_query(query, "❌ Invalid action!", show_alert=True)
        return
    action, raw_data_id = parsed
    username = getattr(query.from_user, "username", None)

    reply, alert, new_markup = "❌ An error occurred!", True, None
    db = SessionLocal()
    try:
        raw_data = db.get(RawData, raw_data_id)
        if raw_data is None:
            reply = "❌ Record not found!"
        else:
            new_like = 1 if action == "like" else -1
            changed = raw_data.like != new_like
            raw_data.like = new_like
            db.commit()
            if new_like == 1:
                reply, alert = "👍 Thank you for your feedback!", False
                logger.info(f"👍 Positive feedback from {username} for question ID: {raw_data_id}")
            else:
                reply, alert = "👎 Thank you for your feedback. We'll improve!", False
                logger.info(f"👎 Negative feedback from {username} for question ID: {raw_data_id}")
            if changed:
                new_markup = _feedback_keyboard(raw_data_id, new_like)
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error handling callback: {e}")
    finally:
        db.close()

    await _answer_query(query, reply, show_alert=alert)

    if new_markup is not None:
        try:
            await query.edit_message_reply_markup(reply_markup=new_markup)
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                logger.warning(f"Could not update feedback keyboard: {e}")
        except Exception as e:
            logger.warning(f"Could not update feedback keyboard: {e}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command - welcome message and bot information.

    Args:
        update: Telegram update object
        context: Telegram context for bot operations
    """
    message = update.effective_message
    if message is None:
        return
    if not is_chat_allowed(message.chat.id):
        return

    welcome_message = (
        "🤖 <b>Çukurova University Computer Engineering AI Assistant</b>\n\n"
        f"🧠 <b>AI Engine:</b> {html.escape(model_display_name())}\n"
        "💾 <b>Database:</b> SQLite\n"
        "🔥 <b>Status:</b> Production Ready\n\n"
        "📝 Hello! I'm here to answer your questions about Computer Engineering department.\n\n"
        "💡 <b>Example questions:</b>\n"
        "• What courses are available?\n"
        "• How to do internship?\n"
        "• What are graduation requirements?\n"
        "• Is there an Erasmus program?\n\n"
        "🚀 <b>Feel free to ask your questions!</b>"
    )

    await message.reply_text(welcome_message, parse_mode=ParseMode.HTML)
    user = update.effective_user
    logger.info(f"🚀 /start command from: {_display_name(user) if user else 'unknown'}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /stats command - show bot statistics.

    Args:
        update: Telegram update object
        context: Telegram context for bot operations
    """
    message = update.effective_message
    if message is None:
        return
    if not is_chat_allowed(message.chat.id):
        return

    db = SessionLocal()
    try:
        # Gather statistics from database
        total_questions = db.query(RawData).count()
        liked_questions = db.query(RawData).filter(RawData.like == 1).count()
        disliked_questions = db.query(RawData).filter(RawData.like == -1).count()
        approved_questions = db.query(RawData).filter(RawData.admin_approved == 1).count()
        training_data_count = db.query(TrainingData).count()
        duplicate_count = db.query(RawData).filter(RawData.is_duplicate == True).count()  # noqa: E712
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        db.rollback()
        stats_text = None
    else:
        # Calculate success rate
        total_feedback = liked_questions + disliked_questions
        success_rate = "N/A"
        if total_feedback > 0:
            success_rate = f"{(liked_questions / total_feedback * 100):.1f}%"

        # Format statistics message
        stats_text = f"""📊 <b>AI Assistant Statistics</b>

🤖 <b>Model:</b> {html.escape(model_display_name())}
📝 <b>Total Questions:</b> {total_questions}
👍 <b>Liked Responses:</b> {liked_questions}
👎 <b>Disliked Responses:</b> {disliked_questions}
✅ <b>Approved by Admin:</b> {approved_questions}
🎓 <b>Training Data:</b> {training_data_count}
🔄 <b>Duplicate Questions:</b> {duplicate_count}
📈 <b>Success Rate:</b> {success_rate}

🔥 <b>Status:</b> Active and Ready"""
    finally:
        db.close()

    try:
        if stats_text is None:
            await message.reply_text("❌ Could not retrieve statistics.")
        else:
            await message.reply_text(stats_text, parse_mode=ParseMode.HTML)
            user = update.effective_user
            logger.info(f"📊 /stats command from: {_display_name(user) if user else 'unknown'}")
    except Exception as e:
        logger.error(f"❌ Could not send statistics: {e}")


def _configure_logging() -> None:
    level = getattr(logging, config.log_level, None)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level,
        force=True,
    )
    # httpx logs every Telegram request URL (it contains the bot token) at INFO;
    # pika logs ~10 ERROR lines with tracebacks per failed connect attempt (the
    # answer consumer logs its own sparse lines).
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.CRITICAL)


def main() -> None:
    """
    Main function to start the Telegram bot.

    Initializes the database, starts the RabbitMQ consumer, and runs the bot.
    """
    _configure_logging()
    logger.info("🚀 Starting CengBot Telegram Bot (legacy entrypoint)...")
    logger.warning("telegram_bot.py is a legacy entrypoint; production uses "
                   "telegram_bot_rabbitmq.py. Do not run both at the same time.")
    if ALLOWED_CHAT_IDS is None:
        logger.warning("TELEGRAM_ALLOWED_CHAT_IDS is not set: the bot answers every chat")

    # Initialize database
    init_db()
    logger.info("✅ Database initialized")

    # Start RabbitMQ answer consumer in background thread
    _consumer_stop.clear()
    consumer_thread = threading.Thread(target=start_answer_consumer, args=(_consumer_stop,),
                                       name="answer-consumer", daemon=True)
    consumer_thread.start()
    logger.info("✅ RabbitMQ answer consumer started")

    # Create Telegram bot application. Concurrent updates: each question waits
    # up to ANSWER_TIMEOUT_SECONDS, which must not serialize all users.
    application = (Application.builder()
                   .token(config.telegram_bot_token)
                   .concurrent_updates(True)
                   .build())

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))

    # Add message handler for regular text messages
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    # Add callback handler for inline button responses
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Start the bot
    logger.info("🚀 Telegram bot is ready and running!")

    # Run the bot until interrupted
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        _consumer_stop.set()


if __name__ == '__main__':
    main()
