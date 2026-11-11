"""
Standalone Telegram bot for CengBot (no RabbitMQ / no AI worker).

ALTERNATIVE / DEVELOPMENT entrypoint: it loads the model in-process and calls
``model_instance.generate_response()`` directly. Production uses
``src/telegram_bot_rabbitmq.py`` + ``src/ai_model_worker.py``. Never run it
together with the production bot (same Telegram token -> polling conflict,
and a second copy of the model on the GPU).

Because this bot bypasses the worker, it applies ``prompt_guard`` itself
(when the module is available).

Run from the project root: ``python3 src/telegram_bot_standalone.py``
"""

import os
import sys

# ``src/`` for sibling modules, project root for ``config``.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

import asyncio
import logging
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

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

from database_models import RawData, SessionLocal, TrainingData, init_db
from config.env_loader import get_config

# Loads .env (os.getenv alone would not see TELEGRAM_BOT_TOKEN from .env).
config = get_config()

# langdetect is non-deterministic unless seeded.
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# The heavy model module (torch/transformers/peft) is imported lazily by
# get_model(), so importing this module never loads the model stack.
model_instance = None

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_QUESTION_ECHO_LENGTH = 500
RATE_LIMIT_MESSAGES = 5
RATE_LIMIT_WINDOW = 60.0

MODEL_ERROR_MESSAGE = (
    "I apologize, I'm experiencing a technical issue at the moment. Please try "
    "again later or contact the department secretary: +90 322 338 60 10"
)
RATE_LIMIT_MESSAGE = (
    "⚠️ You're sending messages too quickly. Please wait a moment before "
    "sending another message."
)
DEFAULT_REFUSALS = {
    "tr": ("Bu isteğe yardımcı olamıyorum. Lütfen Çukurova Üniversitesi "
           "Bilgisayar Mühendisliği bölümüyle ilgili bir soru sorun."),
    "en": ("I can't help with that request. Please ask a question about the "
           "Çukurova University Computer Engineering department."),
}

_TURKISH_CHARS = set("çğışöüÇĞİŞÖÜ")
_CALLBACK_RE = re.compile(r"(like|dislike)_(\d{1,18})")

_model_executor: Optional[ThreadPoolExecutor] = None
_guard_missing_warned = False


# =============================================================================
# Helpers
# =============================================================================

def detect_language(text_: str) -> str:
    """Return 'TR' or 'EN' (deterministic; defaults to 'TR')."""
    if not text_ or not text_.strip():
        return 'TR'
    if any(ch in _TURKISH_CHARS for ch in text_):
        return 'TR'
    try:
        return 'TR' if detect(text_) == 'tr' else 'EN'
    except LangDetectException:
        return 'TR'


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
        InlineKeyboardButton("👍" + (" ✓" if like == 1 else ""), callback_data=f"like_{raw_data_id}"),
        InlineKeyboardButton("👎" + (" ✓" if like == -1 else ""), callback_data=f"dislike_{raw_data_id}"),
    ]])


def build_response_text(display_name: str, question: str, answer: str) -> str:
    """Plain-text reply that always fits in one Telegram message."""
    question = truncate_text(question, MAX_QUESTION_ECHO_LENGTH)
    head = f"👤 @{truncate_text(display_name, 64)} asked:\n❓ {question}\n\n🤖 AI Answer:\n"
    return head + truncate_text(answer, TELEGRAM_MAX_MESSAGE_LENGTH - _utf16_len(head))


def _parse_allowed_chat_ids() -> Optional[frozenset]:
    """Optional allow-list (env ``TELEGRAM_ALLOWED_CHAT_IDS``, comma separated).

    ``None`` = not configured (every chat is served). Invalid entries are
    skipped; an all-invalid list serves no chat.
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
    return rate_limiter.is_limited(user_id)


def check_similarity(db: Session, question: str, exclude_id: Optional[int] = None) -> list:
    """Find earlier questions containing the first 20 chars of ``question``.

    Returns ``(id, question)`` rows, oldest first. With ``exclude_id`` only
    rows with ``id < exclude_id`` are considered (never the question itself).
    """
    try:
        pattern = (question or "").strip()[:20]
        if not pattern:
            return []
        escaped = (pattern.replace("\\", "\\\\").replace("%", "\\%")
                   .replace("_", "\\_"))
        params = {"pattern": f"%{escaped}%"}
        id_clause = ""
        if exclude_id is not None:
            id_clause = "AND id < :exclude_id"
            params["exclude_id"] = int(exclude_id)
        return db.execute(
            text(f"""
            SELECT id, question FROM raw_data
            WHERE question LIKE :pattern ESCAPE '\\'
            {id_clause}
            ORDER BY id
            LIMIT 5
            """),
            params
        ).fetchall()
    except Exception as e:
        logger.error(f"Similarity check error: {e}")
        return []


# =============================================================================
# Model + prompt guard
# =============================================================================

def get_model():
    """Return the shared model instance, importing the model stack on first use."""
    global model_instance
    if model_instance is None:
        from llama_model_handler import model_instance as _model_instance
        model_instance = _model_instance
    return model_instance


def _get_model_executor() -> ThreadPoolExecutor:
    # One thread: generation is serialized (the model is not thread-safe) and
    # never blocks the asyncio event loop.
    global _model_executor
    if _model_executor is None:
        _model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cengbot-model")
    return _model_executor


async def generate_answer(prompt: str) -> Tuple[str, bool]:
    """Run the model off the event loop. Returns ``(answer, ok)``."""
    loop = asyncio.get_running_loop()
    try:
        model = get_model()
        logger.info("🤖 Generating response with LLaMA 3.2 3B + LoRA...")
        response = await loop.run_in_executor(_get_model_executor(),
                                              model.generate_response, prompt)
    except Exception as e:
        logger.error(f"❌ Model error: {e}")
        return MODEL_ERROR_MESSAGE, False
    response = "" if response is None else str(response).strip()
    if not response:
        return MODEL_ERROR_MESSAGE, False
    logger.info("✅ AI response generated")
    return response, True


def _default_refusal(language: Optional[str]) -> str:
    return DEFAULT_REFUSALS.get((language or "tr").lower(), DEFAULT_REFUSALS["tr"])


def _get_prompt_guard():
    """Return the ``prompt_guard`` module, or None if it is not available."""
    global _guard_missing_warned
    try:
        import prompt_guard
    except ImportError as e:
        if not _guard_missing_warned:
            logger.warning("prompt_guard not available (%s); answering WITHOUT the "
                           "jailbreak guard", e)
            _guard_missing_warned = True
        return None
    return prompt_guard


def apply_input_guard(question: str, language: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """Return ``(allowed, prompt_for_model, refusal_text)``.

    Fails closed: if the guard is installed but raises, the question is not
    sent to the model.
    """
    guard = _get_prompt_guard()
    if guard is None:
        return True, question, None
    try:
        result = guard.guard_input(question, language=language)
    except Exception as e:
        logger.error(f"prompt_guard.guard_input failed ({e}); not answering")
        return False, question, MODEL_ERROR_MESSAGE
    if not getattr(result, "allowed", False):
        logger.info(f"🛡️ Prompt guard blocked a message: {getattr(result, 'reason', None)}")
        return False, question, getattr(result, "refusal", None) or _default_refusal(language)
    sanitized = getattr(result, "text", None)
    if sanitized is None:
        return True, question, None
    if not str(sanitized).strip():
        return False, question, _default_refusal(language)
    return True, str(sanitized), None


def apply_output_guard(answer: str, language: Optional[str]) -> str:
    """Filter the model answer through ``guard_output`` (fails closed)."""
    guard = _get_prompt_guard()
    if guard is None:
        return answer
    try:
        guarded = guard.guard_output(answer, language=language)
    except Exception as e:
        logger.error(f"prompt_guard.guard_output failed ({e}); withholding answer")
        return MODEL_ERROR_MESSAGE
    if guarded is None or not str(guarded).strip():
        return _default_refusal(language)
    return str(guarded)


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

        similar_questions = check_similarity(db, message_text, exclude_id=raw_data_id)
        if similar_questions:
            raw_data.is_duplicate = True
            raw_data.duplicate_of_id = similar_questions[0][0]
            db.commit()
        return raw_data_id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _update_question(raw_data_id: int, **fields) -> None:
    db = SessionLocal()
    try:
        raw_data = db.get(RawData, raw_data_id)
        if raw_data is not None:
            for key, value in fields.items():
                setattr(raw_data, key, value)
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
    """Handle incoming messages"""
    message = update.message
    if message is None or not message.text:
        return
    if not is_chat_allowed(message.chat.id):
        logger.warning(f"Ignoring message from non-allowed chat ID: {message.chat.id}")
        return

    user = update.effective_user
    message_text = message.text
    if user is None or message_text.startswith('/'):
        return

    name = _display_name(user)
    logger.info(f"📩 {name}: {message_text[:50]}")

    if is_rate_limited(user.id):
        try:
            await message.reply_text(RATE_LIMIT_MESSAGE)
        except Exception as e:
            logger.warning(f"Could not send rate-limit notice: {e}")
        return

    try:
        language = detect_language(message_text)
        raw_data_id = _save_question(user, message, message_text, language)

        try:
            await message.chat.send_action("typing")
        except Exception as e:
            logger.debug(f"Could not send typing action: {e}")

        guard_language = language.lower()
        allowed, prompt, refusal = apply_input_guard(message_text, guard_language)
        auto_note = None
        if not allowed:
            model_response = refusal
            reply_markup = None
            auto_note = "[auto] prompt_guard: blocked"
        else:
            model_response, ok = await generate_answer(prompt)
            if ok:
                model_response = apply_output_guard(model_response, guard_language)
                guard = _get_prompt_guard()
                is_fallback = getattr(guard, "is_fallback_answer", None) if guard else None
                if callable(is_fallback) and is_fallback(model_response):
                    auto_note = "[auto] output_guard: generated answer replaced by the safe fallback"
            reply_markup = _feedback_keyboard(raw_data_id)

        if model_response != MODEL_ERROR_MESSAGE:
            # Never store the technical-error notice as an answer (it would
            # end up as training data). Guard refusals/fallbacks are stored with an
            # "[auto]" note, which the admin API refuses to approve as-is.
            extra = {"admin_notes": auto_note} if auto_note else {}
            _update_question(raw_data_id, answer=model_response,
                             answered_at=datetime.utcnow(), **extra)

        sent_message = await message.reply_text(
            build_response_text(name, message_text, model_response),
            reply_markup=reply_markup
        )

        _update_question(raw_data_id, telegram_message_id=sent_message.message_id)
        logger.info(f"✅ Response sent: {name}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        try:
            await message.reply_text("An error occurred. Please try again later.")
        except Exception as send_error:
            logger.warning(f"Could not send error notice: {send_error}")


async def _answer_query(query, text_: Optional[str] = None, show_alert: bool = False) -> None:
    try:
        await query.answer(text_, show_alert=show_alert)
    except Exception as e:
        logger.debug(f"Could not answer callback query: {e}")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks (answers the query exactly once)."""
    query = update.callback_query
    if query is None:
        return
    if query.message is not None and not is_chat_allowed(query.message.chat.id):
        await _answer_query(query, "Unauthorized!", show_alert=True)
        return

    parsed = parse_callback_data(query.data)
    if parsed is None:
        await _answer_query(query, "Invalid action!", show_alert=True)
        return
    action, raw_data_id = parsed
    username = getattr(query.from_user, "username", None)

    reply, alert, new_markup = "An error occurred!", True, None
    db = SessionLocal()
    try:
        raw_data = db.get(RawData, raw_data_id)
        if raw_data is None:
            reply = "Record not found!"
        else:
            new_like = 1 if action == "like" else -1
            changed = raw_data.like != new_like
            raw_data.like = new_like
            db.commit()
            if new_like == 1:
                reply, alert = "You liked it! 👍", False
                logger.info(f"👍 {username} - Q{raw_data_id}")
            else:
                reply, alert = "You disliked it! 👎", False
                logger.info(f"👎 {username} - Q{raw_data_id}")
            if changed:
                new_markup = _feedback_keyboard(raw_data_id, new_like)
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Callback error: {e}")
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
    """Handle /start command"""
    message = update.effective_message
    if message is None or not is_chat_allowed(message.chat.id):
        return
    await message.reply_text(
        "🤖 <b>Çukurova University Computer Engineering AI Assistant</b>\n\n"
        "🧠 <b>AI Engine:</b> LLaMA 3.2 3B + LoRA Fine-tuned\n"
        "💾 <b>Database:</b> SQLite\n"
        "🔥 <b>Status:</b> Production Ready\n\n"
        "📝 Hello! I'm here to answer your questions about Computer Engineering.\n\n"
        "💡 <b>Example questions:</b>\n"
        "• What courses are available?\n"
        "• How to do internship?\n"
        "• What are the graduation requirements?\n"
        "• Is there an Erasmus program?\n\n"
        "🚀 <b>You can ask your questions!</b>",
        parse_mode=ParseMode.HTML
    )
    user = update.effective_user
    logger.info(f"🚀 Start: {_display_name(user) if user else 'unknown'}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command"""
    message = update.effective_message
    if message is None or not is_chat_allowed(message.chat.id):
        return
    db = SessionLocal()
    try:
        total = db.query(RawData).count()
        liked = db.query(RawData).filter(RawData.like == 1).count()
        disliked = db.query(RawData).filter(RawData.like == -1).count()
        approved = db.query(RawData).filter(RawData.admin_approved == 1).count()
        training_count = db.query(TrainingData).count()
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        db.rollback()
        stats_text = None
    else:
        success_rate = "N/A"
        if (liked + disliked) > 0:
            success_rate = f"{(liked / (liked + disliked) * 100):.1f}%"
        stats_text = f"""📊 <b>AI Assistant Statistics</b>

🤖 <b>Model:</b> LLaMA 3.2 3B + LoRA
📝 <b>Total Questions:</b> {total}
👍 <b>Liked:</b> {liked}
👎 <b>Disliked:</b> {disliked}
✅ <b>Approved:</b> {approved}
🎓 <b>Training Data:</b> {training_count}
📈 <b>Success Rate:</b> {success_rate}

🔥 <b>Status:</b> Active and Ready"""
    finally:
        db.close()

    try:
        if stats_text is None:
            await message.reply_text("Could not retrieve statistics.")
        else:
            await message.reply_text(stats_text, parse_mode=ParseMode.HTML)
            user = update.effective_user
            logger.info(f"📊 Stats: {_display_name(user) if user else 'unknown'}")
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


def main() -> None:
    """Start the bot"""
    global _model_executor
    _configure_logging()
    logger.info("🚀 Starting CengBot standalone bot (no RabbitMQ worker)...")
    if ALLOWED_CHAT_IDS is None:
        logger.warning("TELEGRAM_ALLOWED_CHAT_IDS is not set: the bot answers every chat")
    token = config.telegram_bot_token  # raises a clear error if missing

    # Initialize database
    init_db()
    logger.info("✅ SQLite Database ready")

    if _get_prompt_guard() is None:
        logger.warning("Jailbreak guard (prompt_guard) is NOT active for this bot")

    # Load model
    logger.info("🤖 Loading LLaMA 3.2 3B + LoRA model...")
    if not get_model().load_model():
        logger.error("❌ Model could not be loaded! Stopping bot.")
        raise SystemExit(1)

    logger.info("🎉 AI Model successfully loaded!")

    # Concurrent updates: callbacks/commands are served while the (serialized)
    # model generates in its own thread.
    application = Application.builder().token(token).concurrent_updates(True).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Start polling
    logger.info("🚀 Bot ready and running!")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        if _model_executor is not None:
            _model_executor.shutdown(wait=False)
            _model_executor = None


if __name__ == '__main__':
    main()
