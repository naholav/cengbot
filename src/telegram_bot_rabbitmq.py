"""
CengBot production Telegram bot (RabbitMQ edition)
==================================================

Flow: ``handle_message`` stores the question as a ``RawData`` row, publishes it
to ``config.questions_queue`` and waits (asynchronously, with a timeout) for the
answer that ``ai_model_worker.py`` publishes to ``config.answers_queue``. A
consumer thread reads the answers queue and hands every answer to the handler
waiting for that ``raw_data_id`` through :class:`AnswerRouter`.

Author: naholav
"""

import asyncio
import html
import json
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
for _path in (_SRC_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.append(_path)

import pika
from langdetect import DetectorFactory, LangDetectException, detect
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, MessageLimit, ParseMode
from telegram.error import (
    BadRequest,
    Conflict,
    InvalidToken,
    NetworkError,
    RetryAfter,
    TelegramError,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import mention_html

from config.env_loader import load_config
from database_models import (
    RawData,
    SessionLocal,
    get_vote_statistics,
    handle_user_vote,
    init_db,
    mark_duplicate_questions,
)

# langdetect is non-deterministic unless the seed is fixed.
DetectorFactory.seed = 0

# Load configuration
config = load_config()

# Bot configuration
BOT_TOKEN = config.telegram_bot_token
BOT_TOPIC_ID = config.telegram_topic_id

# SECURITY: Only allow this specific group (overridable for staging groups).
ALLOWED_CHAT_ID = config.get_int("TELEGRAM_ALLOWED_CHAT_ID", -1002630398173)

# Seconds a user waits for the worker's answer before getting the fallback reply.
ANSWER_TIMEOUT = config.get_float("BOT_ANSWER_TIMEOUT", 30.0)
if ANSWER_TIMEOUT <= 0:
    ANSWER_TIMEOUT = 30.0

FALLBACK_RESPONSE = "Sorry, I cannot generate a response at the moment. Please try again later."
RATE_LIMIT_REPLY = "⚠️ You're sending messages too quickly. Please wait a moment before sending another message."
ERROR_REPLY = "An error occurred. Please try again later."

# Telegram limit (characters after entity parsing, counted in UTF-16 code units).
MAX_MESSAGE_LENGTH = MessageLimit.MAX_TEXT_LENGTH
# When a reply has to be shortened, the echoed question is cut to this length.
QUESTION_ECHO_LIMIT = 500
MIN_QUESTION_ECHO = 100
# Upper bound for honouring Telegram flood-control "retry after" hints.
MAX_RETRY_AFTER = 60.0

ALLOWED_UPDATES = [Update.MESSAGE, Update.CALLBACK_QUERY]

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def _quiet_noisy_loggers() -> None:
    # httpx logs every request at INFO and the Telegram URL contains the bot
    # token; pika logs ~5 INFO lines per connection.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)


_quiet_noisy_loggers()


# =============================================================================
# Rate limiting
# =============================================================================

RATE_LIMIT_MESSAGES = 5  # Messages per minute
RATE_LIMIT_WINDOW = 60  # Time window in seconds
user_message_history: Dict[int, Deque[float]] = {}
_rate_limit_lock = threading.Lock()
_rate_limit_last_sweep: Optional[float] = None


def _sweep_rate_limit_history(current_time: float) -> None:
    """Drop users without a timestamp inside the window (bounds memory)."""
    stale = [
        uid for uid, history in user_message_history.items()
        if not history or current_time - history[-1] >= RATE_LIMIT_WINDOW
    ]
    for uid in stale:
        del user_message_history[uid]


def is_rate_limited(user_id: int, now: Optional[float] = None) -> bool:
    """Check if user has exceeded rate limit (sliding window, thread-safe).

    Accepted messages are recorded; rejected ones are not.
    """
    global _rate_limit_last_sweep
    current_time = time.monotonic() if now is None else now

    with _rate_limit_lock:
        if (_rate_limit_last_sweep is None
                or abs(current_time - _rate_limit_last_sweep) >= RATE_LIMIT_WINDOW):
            _sweep_rate_limit_history(current_time)
            _rate_limit_last_sweep = current_time

        history = user_message_history.get(user_id)
        if history is None:
            history = user_message_history[user_id] = deque()

        # Clean old entries
        while history and current_time - history[0] >= RATE_LIMIT_WINDOW:
            history.popleft()

        if len(history) >= RATE_LIMIT_MESSAGES:
            return True

        history.append(current_time)
        return False


# =============================================================================
# Language detection
# =============================================================================

# Letters that only occur in Turkish (and Azerbaijani) text.
_TURKISH_ONLY_CHARS = frozenset("ğĞıİşŞ")
# langdetect lazily builds a global factory; that is not thread-safe.
_langdetect_lock = threading.Lock()


def detect_language(text: str) -> str:
    """Return 'TR' or 'EN'. Never raises; defaults to 'TR' when unsure."""
    if not text or not any(ch.isalpha() for ch in text):
        return 'TR'  # emoji / numbers / punctuation only
    if any(ch in _TURKISH_ONLY_CHARS for ch in text):
        return 'TR'
    try:
        with _langdetect_lock:
            detected = detect(text)
        return 'TR' if detected == 'tr' else 'EN'
    except LangDetectException:
        return 'TR'
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Language detection failed: {e}")
        return 'TR'


# =============================================================================
# Answer routing (consumer thread -> waiting asyncio handler)
# =============================================================================

def _set_future_result(future: "asyncio.Future", data: Any) -> None:
    if not future.done():
        future.set_result(data)


class AnswerRouter:
    """Routes answers from the RabbitMQ consumer thread to the coroutine that
    waits for that ``raw_data_id``. Thread-safe; nothing is kept for questions
    nobody waits for any more (no unbounded growth)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: Dict[int, Tuple[asyncio.AbstractEventLoop, "asyncio.Future"]] = {}

    def register(self, raw_data_id: int) -> "asyncio.Future":
        """Create a waiter. Must be called from the event loop *before* publishing."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            if raw_data_id in self._waiters:
                logger.warning(f"Replacing existing answer waiter for question ID {raw_data_id}")
            self._waiters[raw_data_id] = (loop, future)
        return future

    def unregister(self, raw_data_id: int, future: Optional["asyncio.Future"] = None) -> None:
        with self._lock:
            entry = self._waiters.get(raw_data_id)
            if entry is not None and (future is None or entry[1] is future):
                del self._waiters[raw_data_id]

    def deliver(self, raw_data_id: int, data: Any) -> bool:
        """Hand an answer to its waiter (callable from any thread).

        Returns False when nobody waits for it (asker timed out, bot restarted).
        """
        with self._lock:
            entry = self._waiters.pop(raw_data_id, None)
        if entry is None:
            return False
        loop, future = entry
        try:
            loop.call_soon_threadsafe(_set_future_result, future, data)
        except RuntimeError:  # event loop already closed
            return False
        return True

    async def wait(self, raw_data_id: int, future: "asyncio.Future", timeout: float) -> Optional[Any]:
        """Wait for the answer; returns None on timeout. Always cleans up."""
        try:
            return await asyncio.wait_for(future, timeout=max(0.0, timeout))
        except asyncio.TimeoutError:
            return None
        finally:
            self.unregister(raw_data_id, future)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._waiters)


answer_router = AnswerRouter()


# =============================================================================
# RabbitMQ
# =============================================================================

def _connection_parameters() -> pika.ConnectionParameters:
    kwargs: Dict[str, Any] = dict(
        host=config.rabbitmq_host,
        port=config.rabbitmq_port,
        connection_attempts=1,
        retry_delay=0,
        socket_timeout=5,
        stack_timeout=10,
        # Broker resource alarms (e.g. full disk) block publishers; give up
        # instead of hanging forever.
        blocked_connection_timeout=30,
    )
    user = os.getenv("RABBITMQ_USER")
    if user:
        kwargs["credentials"] = pika.PlainCredentials(user, os.getenv("RABBITMQ_PASSWORD", ""))
    return pika.ConnectionParameters(**kwargs)


def _open_channel_with_queue(connection, queue_name: str):
    """Open a channel and make sure ``queue_name`` exists (durable).

    A passive declare first, so an existing queue declared by the worker with
    extra arguments is used as-is instead of failing with PRECONDITION_FAILED.
    """
    channel = connection.channel()
    try:
        channel.queue_declare(queue=queue_name, passive=True)
        return channel
    except pika.exceptions.ChannelClosedByBroker as exc:
        if exc.reply_code != 404:
            raise
    channel = connection.channel()  # the broker closed the previous one
    channel.queue_declare(queue=queue_name, durable=True)
    return channel


def _close_quietly(connection) -> None:
    if connection is None:
        return
    try:
        if connection.is_open:
            connection.close()
    except Exception:
        pass


class RabbitMQPublisher:
    """Long-lived publisher connection with reconnect-and-retry.

    pika's BlockingConnection is not thread-safe: all access is serialized by a
    lock, and the bot only calls it from one dedicated executor thread.
    """

    MAX_ATTEMPTS = 2

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection = None
        self._channel = None
        self._queue: Optional[str] = None

    def _ensure_channel(self):
        queue = config.questions_queue
        if (self._channel is not None and self._connection is not None
                and self._connection.is_open and self._channel.is_open
                and self._queue == queue):
            return self._channel, queue

        self._reset()
        connection = pika.BlockingConnection(_connection_parameters())
        try:
            channel = _open_channel_with_queue(connection, queue)
            channel.confirm_delivery()  # basic_publish raises if the broker refuses
        except Exception:
            _close_quietly(connection)
            raise
        self._connection, self._channel, self._queue = connection, channel, queue
        return channel, queue

    def _reset(self) -> None:
        _close_quietly(self._connection)
        self._connection = self._channel = self._queue = None

    def publish(self, data: dict) -> bool:
        body = json.dumps(data)
        last_error: Optional[Exception] = None
        with self._lock:
            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                try:
                    channel, queue = self._ensure_channel()
                    channel.basic_publish(
                        exchange='',
                        routing_key=queue,
                        body=body,
                        properties=pika.BasicProperties(
                            delivery_mode=2,  # make message persistent
                            content_type='application/json',
                        ),
                        mandatory=True,
                    )
                    return True
                except Exception as e:
                    last_error = e
                    self._reset()
                    if attempt < self.MAX_ATTEMPTS:
                        logger.info(f"RabbitMQ publish failed ({type(e).__name__}: {e}); reconnecting")
        logger.error(
            f"RabbitMQ send error for question {data.get('raw_data_id')}: "
            f"{type(last_error).__name__}: {last_error}"
        )
        return False

    def close(self) -> None:
        with self._lock:
            self._reset()


_publisher = RabbitMQPublisher()
# One thread owns the publisher connection; keeps blocking I/O off the event loop.
_publisher_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rabbitmq-publisher")


def send_to_rabbitmq(data) -> bool:
    """Send question to RabbitMQ queue (blocking; returns True on success)."""
    ok = _publisher.publish(data)
    if ok:
        logger.info(f"Sent question to RabbitMQ: {data.get('raw_data_id')}")
    return ok


# Consumer reconnect backoff (seconds)
CONSUMER_RECONNECT_INITIAL_DELAY = 1.0
CONSUMER_RECONNECT_MAX_DELAY = 60.0
# A connection that lived at least this long resets the backoff.
CONSUMER_STABLE_CONNECTION_SECONDS = 30.0
ANSWER_PREFETCH_COUNT = 10

_consumer_stop_event = threading.Event()
_consumer_thread: Optional[threading.Thread] = None


def _on_answer_message(ch, method, properties, body) -> None:
    """pika callback: route one answer message to its waiting handler."""
    try:
        data = json.loads(body)
        raw_data_id = int(data['raw_data_id'])
    except (ValueError, TypeError, KeyError) as e:
        # Poison message: requeueing it would loop forever.
        logger.error(f"Discarding malformed answer message ({type(e).__name__}: {e})")
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        return

    delivered = answer_router.deliver(raw_data_id, data)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if delivered:
        logger.info(f"Received answer for question ID: {raw_data_id}")
    else:
        logger.info(
            f"No pending request for answer {raw_data_id} (asker timed out or bot restarted); "
            f"dropped - the worker already stored it in the database"
        )


def consume_answers(stop_event: Optional[threading.Event] = None) -> None:
    """Consume answers from RabbitMQ (run in a separate thread).

    Reconnects with capped exponential backoff and one concise log line per
    failed attempt.
    """
    stop_event = stop_event if stop_event is not None else _consumer_stop_event
    delay = CONSUMER_RECONNECT_INITIAL_DELAY
    failures = 0

    while not stop_event.is_set():
        connection = None
        connected_at: Optional[float] = None
        try:
            connection = pika.BlockingConnection(_connection_parameters())
            queue = config.answers_queue
            channel = _open_channel_with_queue(connection, queue)
            channel.basic_qos(prefetch_count=ANSWER_PREFETCH_COUNT)
            channel.basic_consume(queue=queue, on_message_callback=_on_answer_message)
            connected_at = time.monotonic()
            if failures:
                logger.info(f"Reconnected to RabbitMQ after {failures} failed attempt(s)")
            logger.info(f"Started consuming answers from RabbitMQ queue '{queue}'")

            while not stop_event.is_set():
                connection.process_data_events(time_limit=1)
        except Exception as e:
            if stop_event.is_set():
                break
            if (connected_at is not None
                    and time.monotonic() - connected_at >= CONSUMER_STABLE_CONNECTION_SECONDS):
                failures = 0
                delay = CONSUMER_RECONNECT_INITIAL_DELAY
            failures += 1
            message = f"RabbitMQ consumer error: {type(e).__name__}: {e}; reconnecting in {delay:.0f}s"
            if failures == 1:
                logger.error(message)
            else:
                logger.warning(f"{message} (attempt {failures})")
            stop_event.wait(delay)
            delay = min(delay * 2, CONSUMER_RECONNECT_MAX_DELAY)
        finally:
            _close_quietly(connection)

    logger.info("Answer consumer stopped")


def start_answer_consumer() -> threading.Thread:
    global _consumer_thread
    if _consumer_thread is not None and _consumer_thread.is_alive():
        return _consumer_thread
    _consumer_stop_event.clear()
    _consumer_thread = threading.Thread(
        target=consume_answers, name="rabbitmq-answer-consumer", daemon=True
    )
    _consumer_thread.start()
    return _consumer_thread


def stop_answer_consumer(timeout: float = 5.0) -> None:
    _consumer_stop_event.set()
    thread = _consumer_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout)


def _shutdown_publisher() -> None:
    try:
        _publisher_executor.submit(_publisher.close).result(timeout=5)
    except Exception:
        pass
    _publisher_executor.shutdown(wait=False)


# =============================================================================
# Database helpers (run in executor threads, one short-lived session each)
# =============================================================================

# Duplicate detection is CPU-heavy (TF-IDF per stored question); keep it off
# the event loop and off the default executor used for quick DB writes.
_duplicate_check_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="duplicate-check")
_background_tasks: set = set()


def _save_question(telegram_id: int, username: Optional[str], question: str,
                   message_thread_id: Optional[int]) -> int:
    language = detect_language(question)
    db = SessionLocal()
    try:
        raw_data = RawData(
            telegram_id=telegram_id,
            username=username,
            question=question,
            language=language,
            message_thread_id=message_thread_id,
        )
        db.add(raw_data)
        db.commit()
        return raw_data.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _record_bot_message_id(raw_data_id: int, message_id: int) -> None:
    db = SessionLocal()
    try:
        db.query(RawData).filter(RawData.id == raw_data_id).update(
            {RawData.telegram_message_id: message_id}, synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _run_duplicate_check(raw_data_id: int, question: str) -> None:
    db = SessionLocal()
    try:
        mark_duplicate_questions(db, raw_data_id, question)
    except Exception as e:
        db.rollback()
        logger.warning(f"Duplicate check failed for question {raw_data_id}: {e}")
    finally:
        db.close()


def _schedule_duplicate_check(raw_data_id: int, question: str) -> None:
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(_duplicate_check_executor, _run_duplicate_check, raw_data_id, question)
    except RuntimeError as e:  # executor shut down
        logger.warning(f"Duplicate check for question {raw_data_id} not scheduled: {e}")
        return
    _background_tasks.add(future)
    future.add_done_callback(_background_tasks.discard)


def _apply_vote(raw_data_id: int, telegram_user_id: int, vote_type: int,
                bot_message_id: Optional[int]) -> Optional[dict]:
    """Record a vote. Returns None if the record does not exist or does not
    belong to the message the button was pressed on (forged callback data)."""
    db = SessionLocal()
    try:
        raw_data = db.query(RawData).filter(RawData.id == raw_data_id).first()
        if raw_data is None:
            return None
        if (raw_data.telegram_message_id is not None and bot_message_id is not None
                and raw_data.telegram_message_id != bot_message_id):
            logger.warning(
                f"Vote for question {raw_data_id} came from message {bot_message_id}, "
                f"expected {raw_data.telegram_message_id}; ignored"
            )
            return None

        # handle_user_vote also keeps the legacy RawData.like column in sync.
        result = dict(handle_user_vote(db, raw_data_id, telegram_user_id, vote_type))
        if not result.get("success"):
            db.rollback()

        result["stats"] = get_vote_statistics(db, raw_data_id)
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# Message formatting / sending
# =============================================================================

def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _cut_index(text: str, limit: int) -> int:
    """Largest index i such that text[:i] fits into ``limit`` UTF-16 units."""
    units = 0
    for i, ch in enumerate(text):
        units += 2 if ord(ch) > 0xFFFF else 1
        if units > limit:
            return i
    return len(text)


def _truncate(text: str, limit: int) -> str:
    if _utf16_len(text) <= limit:
        return text
    return text[:_cut_index(text, max(limit - 1, 0))] + "…"


def _split_once(text: str, limit: int) -> Tuple[str, str]:
    if _utf16_len(text) <= limit:
        return text, ""
    cut = max(_cut_index(text, limit), 1)
    window = text[:cut]
    for separator in ("\n", " "):
        index = window.rfind(separator)
        if index > 0 and index >= cut // 2:
            return text[:index], text[index + 1:]
    return text[:cut], text[cut:]


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Split ``text`` into Telegram-sized pieces, preferring line/word breaks.
    Pieces are stripped (Telegram trims them anyway); blank ones are dropped."""
    parts: List[str] = []
    while text:
        head, text = _split_once(text, limit)
        head = head.strip()
        if head:
            parts.append(head)
    return parts


def _escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def _response_header(asker: str, question: str) -> str:
    return f"{asker} asked:\n❓ {question}\n\n🤖 Answer:\n"


def build_response_chunks(asker_plain: str, asker_html: str, question: str, answer: str,
                          limit: int = MAX_MESSAGE_LENGTH) -> List[Tuple[str, str]]:
    """Build the reply as a list of ``(html_text, plain_text)`` messages, each
    within Telegram's length limit. User/model text is HTML-escaped."""
    if _utf16_len(_response_header(asker_plain, question) + answer) <= limit:
        q, first, rest = question, answer, ""
    else:
        budget = limit - _utf16_len(_response_header(asker_plain, "") + answer)
        if budget >= MIN_QUESTION_ECHO:
            q, first, rest = _truncate(question, budget), answer, ""
        else:
            q = _truncate(question, QUESTION_ECHO_LIMIT)
            first_budget = limit - _utf16_len(_response_header(asker_plain, q))
            first, rest = _split_once(answer, first_budget)

    chunks = [(
        _response_header(asker_html, _escape_html(q)) + _escape_html(first),
        _response_header(asker_plain, q) + first,
    )]
    for part in split_message(rest, limit):
        chunks.append((_escape_html(part), part))
    return chunks


def _format_asker(user) -> Tuple[str, str]:
    """Return (plain, html) mention of the asker."""
    if user.username:
        plain = f"@{user.username}"
        return plain, _escape_html(plain)
    name = user.first_name or "User"
    return name, mention_html(user.id, name)


def build_vote_keyboard(raw_data_id: int, likes: int, dislikes: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(f"👍 {likes}", callback_data=f"like_{raw_data_id}"),
        InlineKeyboardButton(f"👎 {dislikes}", callback_data=f"dislike_{raw_data_id}"),
    ]])


def _extract_answer(answer_data: Any) -> Optional[str]:
    if not isinstance(answer_data, dict):
        return None
    answer = answer_data.get('answer')
    if answer is None:
        return None
    answer = answer if isinstance(answer, str) else str(answer)
    return answer if answer.strip() else None


def _retry_after_seconds(exc: RetryAfter) -> float:
    value = exc.retry_after
    seconds = value.total_seconds() if hasattr(value, "total_seconds") else float(value)
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


def _thread_kwargs(message) -> dict:
    if getattr(message, "is_topic_message", None) is True and message.message_thread_id:
        return {"message_thread_id": message.message_thread_id}
    return {}


async def _send_reply(message, html_text: str, plain_text: str, reply_markup=None):
    """Reply with HTML; fall back to plain text if Telegram rejects the markup;
    honour one flood-control wait."""
    kwargs = dict(reply_markup=reply_markup, allow_sending_without_reply=True, **_thread_kwargs(message))
    for attempt in range(2):
        try:
            try:
                return await message.reply_text(html_text, parse_mode=ParseMode.HTML, **kwargs)
            except BadRequest as e:
                if "entit" not in str(e).lower() and "parse" not in str(e).lower():
                    raise
                logger.warning(f"Telegram rejected HTML formatting ({e}); resending as plain text")
                return await message.reply_text(plain_text, **kwargs)
        except RetryAfter as e:
            if attempt:
                raise
            await asyncio.sleep(_retry_after_seconds(e))


async def _reply_plain(message, text: str) -> None:
    """Best-effort plain reply (never raises)."""
    try:
        await message.reply_text(
            text,
            reply_to_message_id=message.message_id,
            allow_sending_without_reply=True,
            **_thread_kwargs(message),
        )
    except Exception as e:
        logger.warning(f"Could not send reply: {e}")


async def _send_typing(message) -> None:
    try:
        thread_id = message.message_thread_id if getattr(message, "is_topic_message", None) is True else None
        await message.chat.send_action(ChatAction.TYPING, message_thread_id=thread_id)
    except Exception as e:
        logger.debug(f"Could not send typing action: {e}")


async def _send_answer(message, user, question: str, raw_data_id: int, answer: Optional[str]):
    asker_plain, asker_html = _format_asker(user)
    if answer is None:
        chunks = build_response_chunks(asker_plain, asker_html, question, FALLBACK_RESPONSE)
        markup = None  # nothing to vote on; votes would later stick to the real answer
    else:
        chunks = build_response_chunks(asker_plain, asker_html, question, answer)
        markup = build_vote_keyboard(raw_data_id, 0, 0)

    sent = None
    for index, (html_text, plain_text) in enumerate(chunks):
        is_last = index == len(chunks) - 1
        sent = await _send_reply(message, html_text, plain_text, markup if is_last else None)
    return sent


def _publish_before_deadline(payload: dict, deadline: float) -> bool:
    # The single publisher thread may have been stuck (e.g. broker blocked);
    # don't publish questions whose asker has already been given up on.
    if time.monotonic() > deadline:
        logger.warning(f"Not publishing question {payload.get('raw_data_id')}: asker already timed out")
        return False
    return send_to_rabbitmq(payload)


async def _publish_question(payload: dict, timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _publisher_executor, _publish_before_deadline, payload, time.monotonic() + timeout
            ),
            timeout=timeout,
        )
        return bool(result)
    except asyncio.TimeoutError:
        logger.error(f"Publishing question {payload.get('raw_data_id')} timed out after {timeout:.0f}s")
        return False
    except Exception as e:
        logger.error(f"RabbitMQ send error: {e}")
        return False


# =============================================================================
# Handlers
# =============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages in group"""
    message = update.message
    if message is None or not message.text:
        return

    # SECURITY: Only process messages from the allowed group
    if message.chat.id != ALLOWED_CHAT_ID:
        logger.warning(f"Unauthorized access attempt from chat ID: {message.chat.id}")
        return

    # Check if message is in correct topic (if BOT_TOPIC_ID is set)
    if BOT_TOPIC_ID and message.message_thread_id != BOT_TOPIC_ID:
        return

    # Channel posts auto-forwarded into a linked discussion group are not questions
    if getattr(message, "is_automatic_forward", None) is True:
        return

    user = update.effective_user
    if user is None:
        return

    message_text = message.text

    # Skip if message is a command or blank
    if message_text.startswith('/') or not message_text.strip():
        return

    # Check rate limiting
    if is_rate_limited(user.id):
        await _reply_plain(message, RATE_LIMIT_REPLY)
        return

    loop = asyncio.get_running_loop()
    username = user.username or user.first_name
    raw_data_id = None
    try:
        # Save question to database before publishing so the worker can update it
        raw_data_id = await loop.run_in_executor(
            None, _save_question, user.id, username, message_text, message.message_thread_id
        )

        future = answer_router.register(raw_data_id)
        answer_data = None
        try:
            await _send_typing(message)

            started = loop.time()
            published = await _publish_question({
                'raw_data_id': raw_data_id,
                'telegram_id': user.id,
                'username': username,
                'question': message_text,
                'message_thread_id': message.message_thread_id,
                'update_message_id': message.message_id,
                # The worker echoes every question field back in its answer, so a
                # late answer can still be routed to the right chat.
                'chat_id': message.chat_id,
            }, ANSWER_TIMEOUT)

            # Duplicate marking runs in the background; it does not delay the answer.
            _schedule_duplicate_check(raw_data_id, message_text)

            if published:
                remaining = ANSWER_TIMEOUT - (loop.time() - started)
                answer_data = await answer_router.wait(raw_data_id, future, remaining)
                if answer_data is None:
                    logger.warning(f"No answer for question {raw_data_id} within {ANSWER_TIMEOUT:.0f}s")
        finally:
            answer_router.unregister(raw_data_id, future)

        sent_message = await _send_answer(
            message, user, message_text, raw_data_id, _extract_answer(answer_data)
        )

        # Update telegram_message_id with bot's response message ID
        if sent_message is not None:
            await loop.run_in_executor(None, _record_bot_message_id, raw_data_id, sent_message.message_id)

    except Exception as e:
        logger.error(f"Error handling message (question ID {raw_data_id}): {type(e).__name__}: {e}")
        await _reply_plain(message, ERROR_REPLY)


CALLBACK_DATA_RE = re.compile(r"^(like|dislike)_(\d{1,18})$")


def parse_callback_data(data: Optional[str]) -> Optional[Tuple[str, int]]:
    """Parse ``like_<id>`` / ``dislike_<id>``; None for anything else."""
    if not isinstance(data, str):
        return None
    match = CALLBACK_DATA_RE.match(data)
    if not match:
        return None
    return match.group(1), int(match.group(2))


async def _safe_answer(query, text: Optional[str] = None, show_alert: bool = False) -> None:
    try:
        await query.answer(text, show_alert=show_alert)
    except Exception as e:
        logger.debug(f"Could not answer callback query: {e}")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks with user-based voting system"""
    query = update.callback_query
    if query is None:
        return

    # SECURITY: Only process callbacks from the allowed group
    message = query.message
    if message is None or message.chat.id != ALLOWED_CHAT_ID:
        chat_id = message.chat.id if message is not None else None
        logger.warning(f"Unauthorized callback attempt from chat ID: {chat_id}")
        await _safe_answer(query, "Unauthorized access!", show_alert=True)
        return

    # Parse callback data (clients can send arbitrary data)
    parsed = parse_callback_data(query.data)
    if parsed is None:
        logger.warning(f"Ignoring malformed callback data: {str(query.data)[:64]!r}")
        await _safe_answer(query, "Invalid action!", show_alert=True)
        return
    action, raw_data_id = parsed
    vote_type = 1 if action == "like" else -1

    loop = asyncio.get_running_loop()
    try:
        vote_result = await loop.run_in_executor(
            None, _apply_vote, raw_data_id, query.from_user.id, vote_type, message.message_id
        )
    except Exception as e:
        logger.error(f"Error handling callback: {type(e).__name__}: {e}")
        await _safe_answer(query, "An error occurred!", show_alert=True)
        return

    if vote_result is None:
        await _safe_answer(query, "Record not found!", show_alert=True)
        return

    # Answer with vote result message
    await _safe_answer(query, vote_result["message"], show_alert=not vote_result["success"])

    if vote_result["success"]:
        stats = vote_result["stats"]
        try:
            await query.edit_message_reply_markup(
                reply_markup=build_vote_keyboard(raw_data_id, stats['likes'], stats['dislikes'])
            )
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                logger.warning(f"Could not update vote buttons: {e}")
        except TelegramError as e:
            logger.warning(f"Could not update vote buttons: {e}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    # SECURITY: Only respond to start command in the allowed group
    if chat.id != ALLOWED_CHAT_ID:
        logger.warning(f"Unauthorized start command from chat ID: {chat.id}")
        return

    await message.reply_text(
        "Hello! I am the Çukurova University Computer Engineering AI assistant.\n"
        "You can write your questions to the group, and I will try to help you."
    )


NETWORK_ERROR_LOG_INTERVAL = 60.0
_network_error_lock = threading.Lock()
_network_error_log_state: Dict[str, Any] = {"last": None, "suppressed": 0}


def _log_network_error(error: Exception, now: Optional[float] = None) -> bool:
    """One concise line per interval for repeated connectivity errors."""
    current = time.monotonic() if now is None else now
    with _network_error_lock:
        last = _network_error_log_state["last"]
        if last is not None and 0 <= current - last < NETWORK_ERROR_LOG_INTERVAL:
            _network_error_log_state["suppressed"] += 1
            return False
        suppressed = _network_error_log_state["suppressed"]
        _network_error_log_state.update(last=current, suppressed=0)
    extra = f" ({suppressed} similar errors suppressed)" if suppressed else ""
    logger.warning(f"Telegram connectivity problem: {type(error).__name__}: {error}{extra}")
    return True


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Without a handler PTB logs a full traceback for every failed getUpdates."""
    error = context.error
    if isinstance(error, Conflict) or (isinstance(error, NetworkError) and not isinstance(error, BadRequest)):
        _log_network_error(error)
        return
    logger.error(f"Unhandled error while processing an update: {error}", exc_info=error)


# =============================================================================
# Application
# =============================================================================

def build_application(token: Optional[str] = None) -> Application:
    """Create the PTB application with all handlers (no network I/O)."""
    application = (
        Application.builder()
        .token(token or BOT_TOKEN)
        # Handlers wait up to ANSWER_TIMEOUT for the worker; without this,
        # every other update (questions, votes) would queue behind them.
        .concurrent_updates(True)
        .build()
    )

    # Add handlers - ONLY for the specific group
    application.add_handler(CommandHandler("start", start_command, filters=filters.Chat(ALLOWED_CHAT_ID)))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.Chat(ALLOWED_CHAT_ID)
        & filters.ChatType.GROUPS & filters.UpdateType.MESSAGE,
        handle_message
    ))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_error_handler(error_handler)
    return application


def run_polling_with_retries(application_factory: Optional[Callable[[], Application]] = None,
                             sleep: Callable[[float], None] = time.sleep,
                             initial_delay: float = 5.0,
                             max_delay: float = 300.0,
                             max_attempts: Optional[int] = None) -> None:
    """Run polling; retry with backoff if Telegram is unreachable at startup
    (PTB raises from ``initialize()`` instead of retrying, e.g. on DNS errors)."""
    factory = application_factory or build_application
    delay = initial_delay
    attempt = 0
    while True:
        attempt += 1
        application = factory()
        try:
            application.run_polling(allowed_updates=ALLOWED_UPDATES, close_loop=False)
            return
        except InvalidToken:
            raise
        except NetworkError as e:
            if isinstance(e, BadRequest) or (max_attempts is not None and attempt >= max_attempts):
                raise
            logger.error(
                f"Cannot reach Telegram ({type(e).__name__}: {e}); retrying in {delay:.0f}s "
                f"(attempt {attempt})"
            )
            sleep(delay)
            delay = min(delay * 2, max_delay)


def main() -> None:
    """Start the bot"""
    level = logging.getLevelName(config.log_level)
    logging.basicConfig(format=LOG_FORMAT, level=level if isinstance(level, int) else logging.INFO, force=True)
    _quiet_noisy_loggers()

    # Initialize database
    init_db()

    # Model is now loaded in RabbitMQ worker
    logger.info("Bot starting... Model will be loaded in RabbitMQ worker.")
    logger.info(f"SECURITY: Bot configured to ONLY work in group: {ALLOWED_CHAT_ID}")
    logger.info(
        f"Queues: questions='{config.questions_queue}', answers='{config.answers_queue}', "
        f"answer timeout={ANSWER_TIMEOUT:.0f}s"
    )

    # Start answer consumer in separate thread
    start_answer_consumer()
    try:
        logger.info("Bot started!")
        run_polling_with_retries()
    finally:
        stop_answer_consumer()
        _shutdown_publisher()
        _duplicate_check_executor.shutdown(wait=False)


if __name__ == '__main__':
    main()
