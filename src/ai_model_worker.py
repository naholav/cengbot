#!/usr/bin/env python3
"""
CengBot AI model worker
=======================

Consumes questions from RabbitMQ (``config.questions_queue``), answers them with
the fine-tuned language model and publishes the answer to ``config.answers_queue``.

Robustness rules (see the 124 GB log incident):
  * The model is loaded once, *before* connecting to RabbitMQ, with exponential
    backoff (``WORKER_BACKOFF_INITIAL``..``WORKER_BACKOFF_MAX`` seconds) and one
    concise log line per failed attempt.
  * RabbitMQ (re)connects also use exponential backoff, capped at
    ``WORKER_RECONNECT_MAX`` (15 s): a reconnect is cheap, and while the worker
    waits every new question times out in the bot (``BOT_ANSWER_TIMEOUT``, 30 s).
    The first failure is logged at once, the same error repeated at most every
    5 minutes (pika's own per-attempt error lines and tracebacks are silenced).
  * Poison messages are bounded: a failed message is re-published with an
    ``x-cengbot-attempts`` header (and the original acked); a redelivered
    message (worker crashed/connection dropped mid-processing) counts as an
    attempt too. After ``WORKER_MAX_ATTEMPTS`` attempts the RawData row gets an
    apology as its answer, the user is notified and the message is acked.
    ``basic_nack(requeue=True)`` is never used.
  * Generation runs in a helper thread while the consumer thread keeps
    servicing the AMQP connection, so heartbeats are not missed during a long
    ``generate()`` call.
  * Every question goes through ``prompt_guard`` (blocked prompts never reach
    the GPU).
  * A question whose ``deliver_until`` (epoch seconds, set by the bot) has passed,
    or would pass during generation (``DEADLINE_MARGIN``), is not generated:
    nobody would see the answer, and after a backlog (worker
    restart, outage) the GPU goes to questions whose askers still wait. The row
    stays unanswered with an ``[auto]`` note, so admins can answer it. Payloads
    without the field are answered as before (``WORKER_SKIP_EXPIRED=false``
    answers everything).
  * A lost broker connection is not a failure of the question being answered:
    the error propagates, ``run_forever`` reconnects with backoff and the
    broker redelivers the unacked message. A delivery whose ``RawData`` row is
    already answered (e.g. the connection dropped between storing and
    publishing the answer) re-publishes the stored answer instead of dropping
    it; the bot ignores answers nobody waits for, so a duplicate is harmless.
  * SIGTERM while a question is being answered finishes that question first
    (answer published, message acked, consumer cancelled) and then exits; a
    second SIGTERM stops at once. If the connection is lost while finishing, the
    worker exits instead of reconnecting (the broker redelivers the question).
"""

import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _path in (_SRC_DIR, _ROOT_DIR):
    if _path not in sys.path:
        sys.path.append(_path)

import pika  # noqa: E402
from pika.exceptions import (  # noqa: E402
    AMQPConnectionError,
    AMQPError,
    ChannelClosed,
    ChannelClosedByBroker,
    ChannelWrongStateError,
)

from database_models import SessionLocal, RawData  # noqa: E402
from llama_model_handler import model_instance  # noqa: E402
from prompt_guard import guard_input, guard_output, is_fallback_answer  # noqa: E402
from config.env_loader import load_config  # noqa: E402

logger = logging.getLogger(__name__)

ATTEMPTS_HEADER = "x-cengbot-attempts"

# The connection/channel is gone (broker restart, network): not the question's fault.
CONNECTION_LOST_ERRORS = (AMQPConnectionError, ChannelClosed, ChannelWrongStateError)

# raw_data.id is an SQLite INTEGER (signed 64 bit).
MAX_RAW_DATA_ID = 2**63 - 1
_DIGITS_RE = re.compile(r"[0-9]{1,19}")

# A question is skipped once its answer could no longer reach the asker in time: this
# much before deliver_until (seconds; a v1.2 answer takes 2.3 s on average, 3.8 s at most).
DEADLINE_MARGIN = 4.0

# Broker reconnect delay cap (seconds), see the module docstring.
RECONNECT_MAX_DELAY = 15.0
# The same error repeated is logged at most this often (seconds); the first one always.
REPEATED_ERROR_LOG_INTERVAL = 300.0

FAILED_ANSWER = {
    "tr": "Üzgünüm, sorunu şu anda yanıtlayamadım. Lütfen biraz sonra tekrar dene.",
    "en": "Sorry, I couldn't answer your question right now. Please try again later.",
}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid %s=%r, using %d", name, os.getenv(name), default)
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid %s=%r, using %s", name, os.getenv(name), default)
        return default


def _utcnow() -> datetime:
    """Naive UTC timestamp (same convention as the rest of the schema)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _short(exc: BaseException, limit: int = 300) -> str:
    # pika's AMQPConnectionError for a refused/unreachable broker has an empty str(); the
    # reason is only in the repr() of its argument ("...: ConnectionRefusedError(111, ...)").
    detail = str(exc) or "; ".join(str(arg) or repr(arg) for arg in getattr(exc, "args", ()))
    text = f"{type(exc).__name__}: {detail}"
    return text if len(text) <= limit else text[:limit] + "…"


def _is_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


class ModelNotLoadedError(RuntimeError):
    """Raised instead of returning silently when the model is not available (no busy loop)."""


class _ShutdownState:
    """SIGTERM bookkeeping (only touched from the main thread: signal handlers and pika callbacks)."""

    def __init__(self):
        self.requested = False  # a stop was requested while a question was being answered
        self.busy = False       # the consumer callback is running

    def reset(self) -> None:
        self.requested = self.busy = False


_shutdown = _ShutdownState()


class Backoff:
    """Exponential backoff: initial, initial*factor, ... capped at ``maximum``."""

    def __init__(self, initial: float = 5.0, maximum: float = 300.0, factor: float = 2.0):
        self.initial = max(0.0, initial)
        self.maximum = max(self.initial, maximum)
        self.factor = max(1.0, factor)
        self.attempts = 0

    def next_delay(self) -> float:
        try:
            # the exponent is bounded: factor ** attempts overflows a float after ~1000 attempts
            delay = min(self.maximum, self.initial * (self.factor ** min(self.attempts, 64)))
        except OverflowError:
            delay = self.maximum
        self.attempts += 1
        return delay

    def reset(self) -> None:
        self.attempts = 0

    @classmethod
    def from_env(cls) -> "Backoff":
        return cls(_env_float("WORKER_BACKOFF_INITIAL", 5.0), _env_float("WORKER_BACKOFF_MAX", 300.0))


class RabbitMQWorker:
    def __init__(self, config=None, connect: bool = True):
        self.config = config or load_config()
        self.questions_queue = getattr(self.config, "questions_queue", None) or "questions"
        self.answers_queue = getattr(self.config, "answers_queue", None) or "answers"
        self.max_attempts = _env_int("WORKER_MAX_ATTEMPTS", 3, minimum=1)
        self.retry_delay = _env_float("WORKER_RETRY_DELAY", 2.0)
        self.pump_interval = _env_float("WORKER_HEARTBEAT_PUMP_INTERVAL", 1.0, minimum=0.01)
        self.skip_expired = os.getenv("WORKER_SKIP_EXPIRED", "true").strip().lower() not in (
            "0", "false", "no", "off")
        self.connection = None
        self.channel = None
        self._consumer_tag = None
        if connect:
            self.connect()

    # ------------------------------------------------------------------ AMQP
    def _connection_parameters(self) -> pika.ConnectionParameters:
        credentials = pika.PlainCredentials(
            os.getenv("RABBITMQ_USER") or "guest",
            os.getenv("RABBITMQ_PASSWORD") or "guest",
        )
        return pika.ConnectionParameters(
            host=getattr(self.config, "rabbitmq_host", "localhost"),
            port=getattr(self.config, "rabbitmq_port", 5672),
            credentials=credentials,
            heartbeat=_env_int("RABBITMQ_HEARTBEAT", 60),
            blocked_connection_timeout=_env_float("RABBITMQ_BLOCKED_TIMEOUT", 300.0),
            connection_attempts=1,
        )

    def connect(self):
        """Connect to RabbitMQ and declare the queues."""
        try:
            self.connection = pika.BlockingConnection(self._connection_parameters())
            self.channel = self.connection.channel()
            for queue in (self.questions_queue, self.answers_queue):
                self._declare_queue(queue)
            self.channel.basic_qos(prefetch_count=1)
            self.channel.confirm_delivery()
            logger.info("Connected to RabbitMQ (questions=%s, answers=%s)", self.questions_queue,
                        self.answers_queue)
        except Exception as exc:
            logger.debug("RabbitMQ connection error: %s", _short(exc))  # run_forever logs it (once)
            self.close()
            raise

    def _declare_queue(self, queue: str) -> None:
        """Passive declare first (an existing queue with extra arguments is used as-is, like the bot does),
        create it durable if it does not exist."""
        try:
            self.channel.queue_declare(queue=queue, passive=True)
            return
        except ChannelClosedByBroker as exc:
            if exc.reply_code != 404:
                raise
        self.channel = self.connection.channel()  # the broker closed the previous channel
        self.channel.queue_declare(queue=queue, durable=True)

    def close(self) -> None:
        connection, self.connection, self.channel = self.connection, None, None
        if connection is not None:
            try:
                if getattr(connection, "is_open", True):
                    connection.close()
            except Exception:  # connection already gone
                pass

    def _pump_events(self) -> None:
        """Service the AMQP connection (heartbeats) while the consumer callback is busy."""
        connection = self.connection
        if connection is None:
            return
        try:
            connection.process_data_events(time_limit=0)
        except Exception as exc:
            if not self._pump_failed:
                logger.warning("RabbitMQ connection problem during generation: %s", _short(exc))
                self._pump_failed = True

    _pump_failed = False

    def _run_keeping_connection_alive(self, func, *args, **kwargs):
        result: Dict[str, Any] = {}

        def target():
            try:
                result["value"] = func(*args, **kwargs)
            except BaseException as exc:  # re-raised in the consumer thread
                result["error"] = exc

        self._pump_failed = False
        thread = threading.Thread(target=target, name="cengbot-generation", daemon=True)
        thread.start()
        while True:
            thread.join(timeout=self.pump_interval)
            if not thread.is_alive():
                break
            self._pump_events()
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        connection = self.connection
        if connection is not None and hasattr(connection, "sleep") and getattr(connection, "is_open", True):
            connection.sleep(seconds)  # keeps the connection serviced
        else:
            # pika raises ValueError("Timeout closed before call") from sleep() on a dead connection
            time.sleep(seconds)

    # -------------------------------------------------------------- messages
    @staticmethod
    def _parse_raw_data_id(value) -> int:
        """Strict: an int (or a string of digits) in 1..2**63-1. Floats/bools are rejected,
        int() would silently turn 391.5 or true into another row's id."""
        if isinstance(value, bool):
            raise ValueError("raw_data_id must be an integer, not a boolean")
        if isinstance(value, str) and _DIGITS_RE.fullmatch(value.strip()):
            value = int(value.strip())
        if not isinstance(value, int):
            raise ValueError(f"raw_data_id must be an integer, not {type(value).__name__}")
        if not 1 <= value <= MAX_RAW_DATA_ID:
            raise ValueError("raw_data_id out of range")
        return value

    @classmethod
    def _parse_message(cls, body) -> Dict[str, Any]:
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("payload is not a JSON object")
        if "raw_data_id" not in data or "question" not in data:
            raise ValueError("payload lacks raw_data_id/question")
        data["raw_data_id"] = cls._parse_raw_data_id(data["raw_data_id"])
        if not isinstance(data["question"], str):
            data["question"] = "" if data["question"] is None else str(data["question"])
        return data

    @staticmethod
    def _get_attempts(properties) -> int:
        headers = getattr(properties, "headers", None) or {}
        try:
            return max(0, int(headers.get(ATTEMPTS_HEADER, 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _language(text: str) -> str:
        detect = getattr(model_instance, "detect_language", None)
        if callable(detect):
            try:
                return "tr" if detect(text) == "tr" else "en"
            except Exception:
                pass
        return "tr"

    def process_question(self, ch, method, properties, body):
        """Consumer callback: always acks exactly once (no infinite requeue).

        A lost connection/channel propagates (the broker redelivers the unacked
        message, ``run_forever`` reconnects). If a stop was requested while this
        question was being answered, exit once it is done.
        """
        _shutdown.busy = True
        try:
            self._process_question(ch, method, properties, body)
        finally:
            _shutdown.busy = False
        if _shutdown.requested:
            logger.info("Current question finished; stopping")
            raise SystemExit(0)

    def _ack(self, ch, delivery_tag) -> None:
        if _shutdown.requested and self._consumer_tag is not None:
            # Cancel first: with prefetch=1 the broker would otherwise deliver the next
            # question right after this ack, and it would come back marked as redelivered.
            consumer_tag, self._consumer_tag = self._consumer_tag, None
            try:
                ch.basic_cancel(consumer_tag)
            except Exception as exc:  # stopping anyway
                logger.debug("basic_cancel failed: %s", exc)
        ch.basic_ack(delivery_tag=delivery_tag)

    def _process_question(self, ch, method, properties, body):
        delivery_tag = method.delivery_tag
        try:
            data = self._parse_message(body)
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            logger.error("Dropping malformed question message (%s): %r", exc, (body or b"")[:200])
            self._ack(ch, delivery_tag)
            return

        raw_data_id = data["raw_data_id"]
        attempts = self._get_attempts(properties)
        if getattr(method, "redelivered", False):
            attempts += 1  # the previous delivery was never acked (crash / lost connection)
            if self._resend_if_answered(data):
                # The answer was generated and stored, only publishing it failed (connection lost):
                # send it now instead of making the asker wait behind the whole queue.
                self._ack(ch, delivery_tag)
                return

        if attempts >= self.max_attempts:
            self._give_up(ch, delivery_tag, data, f"gave up after {attempts} delivery attempts")
            return
        if getattr(method, "redelivered", False):
            logger.warning("Question %s was redelivered; re-queueing as attempt %d/%d", raw_data_id,
                           attempts + 1, self.max_attempts)
            self._requeue(ch, delivery_tag, body, properties, attempts, delay=False)
            return

        logger.info("Processing question %s: %r", raw_data_id, data["question"][:50])
        try:
            self._handle(data)
        except CONNECTION_LOST_ERRORS:
            raise  # not this question's fault; requeueing on a dead channel would fail anyway
        except Exception as exc:
            attempts += 1
            if attempts >= self.max_attempts:
                logger.error("Question %s failed (attempt %d/%d): %s", raw_data_id, attempts,
                             self.max_attempts, _short(exc))
                self._give_up(ch, delivery_tag, data, f"failed after {attempts} attempts: {_short(exc, 200)}")
            else:
                logger.warning("Question %s failed (attempt %d/%d), retrying: %s", raw_data_id, attempts,
                               self.max_attempts, _short(exc))
                self._requeue(ch, delivery_tag, body, properties, attempts, delay=True)
            return
        self._ack(ch, delivery_tag)

    def _requeue(self, ch, delivery_tag, body, properties, attempts: int, delay: bool) -> None:
        if delay:
            self._sleep(self.retry_delay * attempts)
        headers = dict(getattr(properties, "headers", None) or {})
        headers[ATTEMPTS_HEADER] = attempts
        ch.basic_publish(
            exchange="",
            routing_key=self.questions_queue,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type=getattr(properties, "content_type", None),
                headers=headers,
            ),
        )
        self._ack(ch, delivery_tag)

    def _give_up(self, ch, delivery_tag, data: Dict[str, Any], reason: str) -> None:
        raw_data_id = data["raw_data_id"]
        answer = FAILED_ANSWER[self._language(data.get("question", ""))]
        try:
            if self._store_answer(raw_data_id, answer, notes=f"[auto] worker: {reason}", only_if_unanswered=True):
                self._publish_answer(data, answer, status="error")
            else:
                self._resend_stored_answer(data)  # answered by an earlier attempt that died before publishing
        except CONNECTION_LOST_ERRORS:
            raise
        except Exception as exc:
            logger.error("Could not record failure for question %s: %s", raw_data_id, _short(exc))
        logger.error("Question %s dropped: %s", raw_data_id, reason)
        self._ack(ch, delivery_tag)

    # ------------------------------------------------------------- the work
    def _raw_data_state(self, raw_data_id: int) -> Optional[bool]:
        """None if the row does not exist, else True if it already has an answer."""
        for attempt in range(5):  # the bot commits right before publishing; tolerate a short lag
            db = SessionLocal()
            try:
                row = db.get(RawData, raw_data_id)
                if row is not None:
                    return row.answer is not None and row.answered_at is not None
            finally:
                db.close()
            if attempt < 4:
                time.sleep(0.1)
        return None

    def _generate(self, text: str, language: str) -> Tuple[str, Dict[str, Any]]:
        generate = getattr(model_instance, "generate_response_with_info", None)
        if callable(generate):
            return self._run_keeping_connection_alive(generate, text, raise_errors=True, language=language)
        return self._run_keeping_connection_alive(model_instance.generate_response, text), {}

    def _handle(self, data: Dict[str, Any]) -> None:
        raw_data_id = data["raw_data_id"]
        state = self._raw_data_state(raw_data_id)
        if state is None:
            logger.error("RawData %s not found; dropping question", raw_data_id)
            return
        if state:
            # Duplicate delivery, or the previous attempt stored the answer but lost the
            # connection before publishing it: re-send the stored answer, never regenerate.
            logger.info("RawData %s already answered; re-sending the stored answer", raw_data_id)
            self._resend_stored_answer(data)
            return

        overdue = self._overdue_seconds(data)
        if overdue is not None:
            asked_at = data.get("asked_at")
            age = (f"asked {time.time() - asked_at:.0f}s ago" if _is_number(asked_at)
                   else f"{overdue:.0f}s past its deadline")
            logger.info("Question %s skipped (%s): nobody waits for the answer any more", raw_data_id, age)
            self._add_note(raw_data_id, f"[auto] worker: not answered, the asker was no longer waiting ({age})")
            return

        started = time.monotonic()
        question = data["question"]
        language = self._language(question)
        guard = guard_input(question, language)
        info: Dict[str, Any] = {}
        notes = None
        if not guard.allowed:
            answer = guard.refusal or FAILED_ANSWER[language]
            status = "blocked"
            notes = f"[auto] prompt_guard: {guard.reason}"
        else:
            answer, info = self._generate(guard.text, language)
            answer = guard_output(answer, language)
            status = "ok"
            if is_fallback_answer(answer):
                # the model's answer was replaced (system-prompt leak etc.): never train on it
                notes = "[auto] output_guard: generated answer replaced by the safe fallback"

        self._store_answer(
            raw_data_id, answer,
            processing_time=time.monotonic() - started,
            info=info, notes=notes,
        )
        self._publish_answer(data, answer, status=status)
        logger.info("Answer sent for question %s (%s, %.1fs)", raw_data_id, status, time.monotonic() - started)

    def _overdue_seconds(self, data: Dict[str, Any]) -> Optional[float]:
        """Seconds past the payload's ``deliver_until`` minus the generation time
        (``DEADLINE_MARGIN``), or None (no/invalid field, still worth answering)."""
        if not getattr(self, "skip_expired", True):
            return None
        deadline = data.get("deliver_until")
        if not _is_number(deadline):
            return None
        overdue = time.time() - (deadline - DEADLINE_MARGIN)
        return overdue if overdue > 0 else None

    def _add_note(self, raw_data_id: int, note: str) -> None:
        db = SessionLocal()
        try:
            row = db.get(RawData, raw_data_id)
            if row is None:
                return
            row.admin_notes = f"{row.admin_notes}\n{note}" if row.admin_notes else note
            row.last_updated = _utcnow()
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _store_answer(self, raw_data_id: int, answer: str, processing_time: Optional[float] = None,
                      info: Optional[Dict[str, Any]] = None, notes: Optional[str] = None,
                      only_if_unanswered: bool = False) -> bool:
        info = info or {}
        db = SessionLocal()
        try:
            row = db.get(RawData, raw_data_id)
            if row is None:
                logger.warning("RawData %s disappeared before the answer could be stored", raw_data_id)
                return False
            if only_if_unanswered and row.answer is not None:
                return False
            now = _utcnow()
            row.answer = answer
            row.answered_at = now
            row.last_updated = now
            if processing_time is not None:
                row.processing_time = round(processing_time, 3)
            version = info.get("model_version")
            if version is None and getattr(model_instance, "spec", None) is not None:
                version = getattr(model_instance.spec, "version", None)
            if isinstance(version, str) and version:
                row.model_version = version[:50]
            if isinstance(info.get("input_tokens"), int):
                row.context_length = info["input_tokens"]
            if isinstance(info.get("output_tokens"), int):
                row.response_length = info["output_tokens"]
            if notes:
                row.admin_notes = f"{row.admin_notes}\n{notes}" if row.admin_notes else notes
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _resend_if_answered(self, data: Dict[str, Any]) -> bool:
        """Re-send the stored answer of an already answered row; False if unanswered/unknown."""
        try:
            if not self._raw_data_state(data["raw_data_id"]):
                return False
            logger.info("Question %s was redelivered but is already answered; re-sending the stored answer",
                        data["raw_data_id"])
            return self._resend_stored_answer(data)
        except CONNECTION_LOST_ERRORS:
            raise
        except Exception as exc:  # e.g. database locked: fall back to the normal redelivery path
            logger.warning("Could not check question %s: %s", data["raw_data_id"], _short(exc))
            return False

    def _resend_stored_answer(self, data: Dict[str, Any]) -> bool:
        db = SessionLocal()
        try:
            row = db.get(RawData, data["raw_data_id"])
            answer = row.answer if row is not None else None
        finally:
            db.close()
        if not answer:
            return False
        self._publish_answer(data, answer, status="resent")
        return True

    def _publish_answer(self, data: Dict[str, Any], answer: str, status: str = "ok") -> None:
        # Forward every field of the question payload (chat_id, telegram_id, message ids, ...)
        # so the consumer can deliver late answers; the classic keys always exist.
        answer_data = dict(data)
        for key in ("telegram_id", "message_thread_id", "username", "update_message_id", "chat_id"):
            answer_data.setdefault(key, None)
        answer_data["answer"] = answer
        answer_data["status"] = status
        self.channel.basic_publish(
            exchange="",
            routing_key=self.answers_queue,
            body=json.dumps(answer_data),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )

    # ------------------------------------------------------------- consume
    def start_consuming(self):
        """Consume until the connection drops (raises) or the consumer is stopped."""
        if not model_instance.is_loaded() and not load_model_with_backoff(max_attempts=1):
            self.close()
            raise ModelNotLoadedError(getattr(model_instance, "last_load_error", None) or "model is not loaded")
        self.channel.basic_qos(prefetch_count=1)
        self._consumer_tag = self.channel.basic_consume(queue=self.questions_queue,
                                                        on_message_callback=self.process_question)
        logger.info("Waiting for questions on '%s'...", self.questions_queue)
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            try:
                self.channel.stop_consuming()
            except Exception:
                pass
            raise
        finally:
            self.close()


def load_model_with_backoff(model=None, backoff: Optional[Backoff] = None,
                            max_attempts: Optional[int] = None) -> bool:
    """Load the model, retrying with exponential backoff and concise logs."""
    model = model or model_instance
    backoff = backoff or Backoff.from_env()
    last_error = None
    attempt = 0
    while True:
        attempt += 1
        if model.load_model():
            if attempt > 1:
                logger.info("Model loaded after %d attempts", attempt)
            return True
        error = getattr(model, "last_load_error", None) or "unknown error"
        if max_attempts is not None and attempt >= max_attempts:
            return False
        delay = backoff.next_delay()
        if error != last_error:
            logger.error("Model load failed (attempt %d): %s. Retrying in %.0fs", attempt, str(error)[:300], delay)
            last_error = error
        else:
            logger.warning("Model load failed again (attempt %d, same error). Retrying in %.0fs", attempt, delay)
        time.sleep(delay)


def _setup_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=level, force=True)
    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # pika logs ~10 ERROR lines incl. two tracebacks for every failed (re)connect
    # attempt; the worker logs one concise line per failure itself.
    logging.getLogger("pika").setLevel(logging.CRITICAL)


def _install_signal_handlers() -> None:
    _shutdown.reset()

    def _terminate(signum, frame):
        if _shutdown.busy and not _shutdown.requested:
            _shutdown.requested = True
            logger.info("Stop requested: finishing the current question first (SIGTERM again to stop now)")
            return
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _terminate)
    except ValueError:  # not in the main thread
        pass


def main():
    _setup_logging()
    _install_signal_handlers()
    try:
        run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker stopped")


def run_forever(max_cycles: Optional[int] = None) -> None:
    """Load the model (with backoff), then consume forever, reconnecting with backoff.

    The old connection is always closed *before* sleeping/retrying, and a model that
    cannot be loaded never leads to a busy loop (start_consuming raises instead of returning).
    """
    load_model_with_backoff()
    backoff = Backoff.from_env()
    reconnect_max = _env_float("WORKER_RECONNECT_MAX", RECONNECT_MAX_DELAY, minimum=1.0)
    healthy_after = _env_float("WORKER_HEALTHY_AFTER", 60.0)
    last_error = None
    last_logged = 0.0
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        worker = None
        started = time.monotonic()
        try:
            worker = RabbitMQWorker()
            worker.start_consuming()
            # start_consuming only returns if the broker cancelled the consumer.
            # Back off here too: a broker that keeps cancelling must not cause a
            # tight reconnect loop (and a log line per iteration).
            worker.close()
            if time.monotonic() - started >= healthy_after:
                backoff.reset()
            delay = backoff.next_delay()
            logger.warning("Consumer stopped by the broker; reconnecting in %.0fs", delay)
            time.sleep(delay)
        except Exception as exc:
            if worker is not None:
                worker.close()
            if _shutdown.requested:
                # SIGTERM arrived while a question was being answered and then the connection
                # was lost: stop as requested (the broker redelivers the unacked question).
                logger.info("Stop requested; not reconnecting after: %s", _short(exc))
                raise SystemExit(0) from None
            now = time.monotonic()
            if now - started >= healthy_after:
                # the previous connection was healthy for a while: start the backoff over
                backoff.reset()
                last_error = None
            delay = backoff.next_delay()
            if isinstance(exc, AMQPError):
                delay = min(delay, reconnect_max)
            summary = _short(exc)
            if summary != last_error:
                logger.error("Worker error: %s - reconnecting in %.0fs", summary, delay,
                             exc_info=not isinstance(exc, (AMQPError, RuntimeError)))
                last_error, last_logged = summary, now
            elif now - last_logged >= REPEATED_ERROR_LOG_INTERVAL:
                logger.warning("Worker error repeated (%d in a row): %s - reconnecting in %.0fs",
                               backoff.attempts, summary, delay)
                last_logged = now
            else:
                logger.debug("Worker error repeated (%d in a row) - reconnecting in %.0fs",
                             backoff.attempts, delay)
            time.sleep(delay)
            if not model_instance.is_loaded():
                load_model_with_backoff()
        finally:
            if worker is not None:
                worker.close()


if __name__ == "__main__":
    main()
