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
  * RabbitMQ (re)connects also use exponential backoff.
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
"""

import json
import logging
import os
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
from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker  # noqa: E402

from database_models import SessionLocal, RawData  # noqa: E402
from llama_model_handler import model_instance  # noqa: E402
from prompt_guard import guard_input, guard_output, is_fallback_answer  # noqa: E402
from config.env_loader import load_config  # noqa: E402

logger = logging.getLogger(__name__)

ATTEMPTS_HEADER = "x-cengbot-attempts"

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
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + "…"


class ModelNotLoadedError(RuntimeError):
    """Raised instead of returning silently when the model is not available (no busy loop)."""


class Backoff:
    """Exponential backoff: initial, initial*factor, ... capped at ``maximum``."""

    def __init__(self, initial: float = 5.0, maximum: float = 300.0, factor: float = 2.0):
        self.initial = max(0.0, initial)
        self.maximum = max(self.initial, maximum)
        self.factor = max(1.0, factor)
        self.attempts = 0

    def next_delay(self) -> float:
        delay = min(self.maximum, self.initial * (self.factor ** self.attempts))
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
        self.connection = None
        self.channel = None
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
            logger.error("RabbitMQ connection error: %s", _short(exc))
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
        if connection is not None and hasattr(connection, "sleep"):
            connection.sleep(seconds)  # keeps the connection serviced
        else:
            time.sleep(seconds)

    # -------------------------------------------------------------- messages
    @staticmethod
    def _parse_message(body) -> Dict[str, Any]:
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("payload is not a JSON object")
        if "raw_data_id" not in data or "question" not in data:
            raise ValueError("payload lacks raw_data_id/question")
        data["raw_data_id"] = int(data["raw_data_id"])
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
        """Consumer callback: always acks exactly once (no infinite requeue)."""
        delivery_tag = method.delivery_tag
        try:
            data = self._parse_message(body)
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            logger.error("Dropping malformed question message (%s): %r", exc, (body or b"")[:200])
            ch.basic_ack(delivery_tag=delivery_tag)
            return

        raw_data_id = data["raw_data_id"]
        attempts = self._get_attempts(properties)
        if getattr(method, "redelivered", False):
            attempts += 1  # the previous delivery was never acked (crash / lost connection)

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
        ch.basic_ack(delivery_tag=delivery_tag)

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
        ch.basic_ack(delivery_tag=delivery_tag)

    def _give_up(self, ch, delivery_tag, data: Dict[str, Any], reason: str) -> None:
        raw_data_id = data["raw_data_id"]
        answer = FAILED_ANSWER[self._language(data.get("question", ""))]
        try:
            if self._store_answer(raw_data_id, answer, notes=f"[auto] worker: {reason}", only_if_unanswered=True):
                self._publish_answer(data, answer, status="error")
        except Exception as exc:
            logger.error("Could not record failure for question %s: %s", raw_data_id, _short(exc))
        logger.error("Question %s dropped: %s", raw_data_id, reason)
        ch.basic_ack(delivery_tag=delivery_tag)

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
            logger.info("RawData %s already answered; skipping duplicate delivery", raw_data_id)
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
        self.channel.basic_consume(queue=self.questions_queue, on_message_callback=self.process_question)
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
    for noisy in ("pika", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _install_signal_handlers() -> None:
    def _terminate(signum, frame):
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
    healthy_after = _env_float("WORKER_HEALTHY_AFTER", 60.0)
    last_error = None
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
            if time.monotonic() - started >= healthy_after:
                # the previous connection was healthy for a while: start the backoff over
                backoff.reset()
                last_error = None
            delay = backoff.next_delay()
            summary = _short(exc)
            if summary != last_error:
                logger.error("Worker error: %s. Reconnecting in %.0fs", summary, delay,
                             exc_info=not isinstance(exc, (AMQPConnectionError, RuntimeError)))
                last_error = summary
            else:
                logger.warning("Worker error repeated (%d in a row). Reconnecting in %.0fs",
                               backoff.attempts, delay)
            time.sleep(delay)
            if not model_instance.is_loaded():
                load_model_with_backoff()
        finally:
            if worker is not None:
                worker.close()


if __name__ == "__main__":
    main()
