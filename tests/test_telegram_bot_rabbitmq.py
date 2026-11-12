"""Tests for src/telegram_bot_rabbitmq.py (production Telegram bot).

Everything external is mocked: no Telegram API calls, no RabbitMQ connection,
no model. ``main()`` is never called.
"""

import asyncio
import html
import json
import logging
import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pika
import pytest
from langdetect import LangDetectException
from telegram.constants import ParseMode
from telegram.error import BadRequest, InvalidToken, NetworkError, RetryAfter, TimedOut

import database_models
import telegram_bot_rabbitmq as bot
from database_models import RawData, UserVotes


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_bot_state(monkeypatch):
    """Fresh router / publisher / rate limiter per test; duplicate check stubbed."""
    monkeypatch.setattr(bot, "answer_router", bot.AnswerRouter())
    monkeypatch.setattr(bot, "_publisher", bot.RabbitMQPublisher())
    monkeypatch.setattr(bot, "BOT_TOPIC_ID", None)
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 5.0)
    monkeypatch.setattr(bot, "mark_duplicate_questions", MagicMock(return_value=False))
    monkeypatch.setattr(bot, "_rate_limit_last_sweep", None)
    bot.user_message_history.clear()
    bot._network_error_log_state.update(last=None, suppressed=0)
    yield
    bot.user_message_history.clear()


def make_user(user_id=42, username="student", first_name="Ali"):
    return SimpleNamespace(id=user_id, username=username, first_name=first_name, is_bot=False)


def make_message(text="When is the midterm exam?", chat_id=None, message_id=10,
                 thread_id=None, is_topic=False):
    message = MagicMock()
    message.text = text
    message.chat.id = bot.ALLOWED_CHAT_ID if chat_id is None else chat_id
    message.chat.send_action = AsyncMock(return_value=True)
    message.message_id = message_id
    message.message_thread_id = thread_id
    message.is_topic_message = is_topic
    message.is_automatic_forward = False
    counter = iter(range(1000, 2000))
    message.reply_text = AsyncMock(side_effect=lambda *a, **k: SimpleNamespace(message_id=next(counter)))
    return message


def make_update(message, user=None):
    update = MagicMock()
    update.message = message
    update.effective_message = message
    update.effective_chat = message.chat if message is not None else None
    update.effective_user = user or make_user()
    return update


def visible_text(html_text):
    """What Telegram displays for an HTML message (used for length checks)."""
    return html.unescape(re.sub(r"<[^>]+>", "", html_text))


def fetch_raw(raw_id):
    s = database_models.SessionLocal()
    try:
        return s.query(RawData).filter(RawData.id == raw_id).first()
    finally:
        s.close()


def all_raw():
    s = database_models.SessionLocal()
    try:
        return s.query(RawData).order_by(RawData.id).all()
    finally:
        s.close()


async def drain_background():
    pending = list(bot._background_tasks)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def answering_publisher(answer_for=lambda data: f"Answer to: {data['question']}", delay=0.0):
    """Fake send_to_rabbitmq that plays the worker + consumer: answers from
    another thread through the real consumer callback."""
    published = []

    def fake_send(data):
        published.append((data, threading.current_thread()))

        def worker():
            time.sleep(delay)
            body = json.dumps({"raw_data_id": data["raw_data_id"], "answer": answer_for(data)})
            ch = MagicMock()
            bot._on_answer_message(ch, SimpleNamespace(delivery_tag=1), None, body.encode())

        threading.Thread(target=worker, daemon=True).start()
        return True

    return fake_send, published


# ---------------------------------------------------------------------------
# AnswerRouter
# ---------------------------------------------------------------------------

async def test_router_delivers_from_other_thread():
    router = bot.AnswerRouter()
    fut = router.register(7)
    t = threading.Thread(target=router.deliver, args=(7, {"answer": "x"}))
    t.start()
    result = await router.wait(7, fut, 2.0)
    t.join()
    assert result == {"answer": "x"}
    assert router.pending_count() == 0


async def test_router_routes_concurrent_answers_out_of_order():
    router = bot.AnswerRouter()
    futures = {i: router.register(i) for i in (1, 2, 3)}

    def deliver_reversed():
        for i in (3, 1, 2):
            router.deliver(i, {"raw_data_id": i})

    threading.Thread(target=deliver_reversed).start()
    results = await asyncio.gather(*(router.wait(i, futures[i], 2.0) for i in (1, 2, 3)))
    assert [r["raw_data_id"] for r in results] == [1, 2, 3]
    assert router.pending_count() == 0


async def test_router_timeout_cleans_up_and_late_answer_is_dropped():
    router = bot.AnswerRouter()
    fut = router.register(5)
    assert await router.wait(5, fut, 0.05) is None
    assert router.pending_count() == 0
    # Without a late callback the late answer is not stored anywhere (no unbounded growth)
    assert router.deliver(5, {"answer": "late"}) is None
    assert router.pending_count() == 0 and router.late_count() == 0


async def test_router_unknown_answer_is_not_kept():
    router = bot.AnswerRouter()
    for i in range(100):
        assert router.deliver(i, {"answer": "orphan"}) is None
    assert router.pending_count() == 0 and router.late_count() == 0


async def settle():
    """Let call_soon_threadsafe callbacks run, then wait for the background tasks."""
    for _ in range(5):
        await asyncio.sleep(0.01)
        await drain_background()


async def test_router_late_answer_goes_to_the_callback_once():
    router = bot.AnswerRouter()
    got = []

    async def on_late(data):
        got.append(data)

    fut = router.register(8)
    assert await router.wait(8, fut, 0.01, on_late=on_late, late_window=60) is None
    assert router.pending_count() == 0 and router.late_count() == 1
    assert router.deliver(8, {"answer": "late"}) == "late"
    assert router.deliver(8, {"answer": "late", "status": "resent"}) is None  # duplicate: dropped
    await settle()
    assert got == [{"answer": "late"}] and router.late_count() == 0


async def test_router_late_window_expires(monkeypatch):
    router = bot.AnswerRouter()
    got = []

    async def on_late(data):
        got.append(data)

    fut = router.register(9)
    assert await router.wait(9, fut, 0.01, on_late=on_late, late_window=0.05) is None
    await asyncio.sleep(0.1)
    assert router.deliver(9, {"answer": "too late"}) is None
    router.register(10)  # registering sweeps expired late entries too
    await settle()
    assert got == [] and router.late_count() == 0


async def test_router_answer_delivered_at_the_timeout_is_not_lost():
    """deliver() pops the waiter just as the wait times out: the answer is returned, not
    dropped and not sent a second time as a late reply."""
    router = bot.AnswerRouter()
    late = []

    async def on_late(data):
        late.append(data)

    fut = router.register(11)
    assert router.deliver(11, {"answer": "just in time"}) == "waiting"  # scheduled, not yet run
    assert await router.wait(11, fut, 0, on_late=on_late, late_window=60) == {"answer": "just in time"}
    await settle()
    assert late == [] and router.late_count() == 0 and router.pending_count() == 0


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_allows_five_per_window_then_blocks():
    results = [bot.is_rate_limited(1, now=100.0 + i) for i in range(6)]
    assert results == [False] * 5 + [True]
    # Other users are independent
    assert bot.is_rate_limited(2, now=105.0) is False
    # Blocked attempts are not recorded: the window opens 60s after the first message
    assert bot.is_rate_limited(1, now=159.0) is True
    assert bot.is_rate_limited(1, now=160.0) is False


def test_rate_limit_history_does_not_grow_with_idle_users():
    for uid in range(1000):
        bot.is_rate_limited(uid, now=0.0)
    assert len(bot.user_message_history) == 1000
    # One window later a single call sweeps all idle users
    bot.is_rate_limited(99999, now=bot.RATE_LIMIT_WINDOW + 1.0)
    assert set(bot.user_message_history) == {99999}


def test_rate_limit_is_thread_safe():
    barrier = threading.Barrier(20)
    results = []

    def hit():
        barrier.wait()
        results.append(bot.is_rate_limited(77, now=10.0))

    threads = [threading.Thread(target=hit) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(False) == bot.RATE_LIMIT_MESSAGES
    assert len(bot.user_message_history[77]) == bot.RATE_LIMIT_MESSAGES


async def test_handle_message_rate_limited_reply(monkeypatch, db_session):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bot, "send_to_rabbitmq", send)
    for _ in range(bot.RATE_LIMIT_MESSAGES):
        bot.is_rate_limited(42)
    msg = make_message()
    await bot.handle_message(make_update(msg), None)
    msg.reply_text.assert_awaited_once()
    assert "too quickly" in msg.reply_text.await_args.args[0]
    send.assert_not_called()
    assert all_raw() == []


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "123", "😀😀", "?!", "   ", "42 + 7"])
def test_detect_language_no_letters_defaults_to_tr(text):
    assert bot.detect_language(text) == "TR"


def test_detect_language_basic_cases():
    assert bot.detect_language("When is the midterm exam for this course?") == "EN"
    assert bot.detect_language("Vize sınavları ne zaman yapılacak?") == "TR"
    # Turkish-only letters decide even for very short text
    assert bot.detect_language("kayıt") == "TR"
    assert bot.detect_language("şu") == "TR"


def test_detect_language_is_deterministic():
    samples = ["ok", "hocam", "merhaba", "hello there", "staj ne zaman", "midterm?"]
    first = [bot.detect_language(s) for s in samples]
    for _ in range(5):
        assert [bot.detect_language(s) for s in samples] == first


def test_detect_language_exception_defaults_to_tr(monkeypatch):
    def boom(_):
        raise LangDetectException(0, "No features in text.")

    monkeypatch.setattr(bot, "detect", boom)
    assert bot.detect_language("hello world") == "TR"


# ---------------------------------------------------------------------------
# Message building / splitting
# ---------------------------------------------------------------------------

def test_split_message_respects_utf16_limit_with_emoji():
    parts = bot.split_message("😀" * 3000, 4096)
    assert len(parts) == 2
    assert all(bot._utf16_len(p) <= 4096 for p in parts)
    assert "".join(parts) == "😀" * 3000


def test_split_message_prefers_line_and_word_boundaries():
    text = ("line one is here\n" * 10) + ("word " * 50)
    parts = bot.split_message(text, 60)
    assert all(bot._utf16_len(p) <= 60 for p in parts)
    assert all(not p.endswith("wor") for p in parts)
    assert " ".join(p.strip() for p in parts).split() == text.split()


def test_split_message_skips_whitespace_only_chunks():
    assert bot.split_message("a" + " " * 50 + "\n" * 50 + "b", 10) == ["a", "b"]
    assert bot.split_message("", 10) == []


def test_build_response_chunks_escapes_html():
    chunks = bot.build_response_chunks("@a_b", "@a_b", "is <b> & </i> ok?", "use <script>x</script> & y")
    assert len(chunks) == 1
    html_text, plain_text = chunks[0]
    assert "&lt;b&gt; &amp; &lt;/i&gt;" in html_text
    assert "&lt;script&gt;x&lt;/script&gt; &amp; y" in html_text
    assert "<script>" not in html_text
    assert plain_text == "@a_b asked:\n❓ is <b> & </i> ok?\n\n🤖 Answer:\nuse <script>x</script> & y"


def test_build_response_chunks_long_question_is_truncated_to_single_message():
    question = "q" * 5000
    chunks = bot.build_response_chunks("@u", "@u", question, "short answer")
    assert len(chunks) == 1
    html_text, plain_text = chunks[0]
    assert bot._utf16_len(plain_text) <= bot.MAX_MESSAGE_LENGTH
    assert bot._utf16_len(visible_text(html_text)) <= bot.MAX_MESSAGE_LENGTH
    assert "…" in plain_text and plain_text.endswith("short answer")


def test_build_response_chunks_long_answer_is_split():
    answer = "word " * 3000
    chunks = bot.build_response_chunks("@u", "@u", "q?", answer)
    assert len(chunks) >= 4
    for html_text, plain_text in chunks:
        assert bot._utf16_len(plain_text) <= bot.MAX_MESSAGE_LENGTH
        assert bot._utf16_len(visible_text(html_text)) <= bot.MAX_MESSAGE_LENGTH
    joined = " ".join(p for _, p in chunks)
    assert joined.count("word") == 3000


def test_format_asker_escapes_and_mentions():
    plain, html_ = bot._format_asker(make_user(username=None, first_name="<Ayşe>"))
    assert plain == "<Ayşe>"
    assert html_ == '<a href="tg://user?id=42">&lt;Ayşe&gt;</a>'
    assert bot._format_asker(make_user(username="ali_v"))[0] == "@ali_v"


@pytest.mark.parametrize("data,expected", [
    ({"answer": "hi"}, "hi"),
    ({"answer": ""}, None),
    ({"answer": "   \n"}, None),
    ({"answer": None}, None),
    ({}, None),
    (None, None),
    ("text", None),
    ({"answer": 12}, "12"),
])
def test_extract_answer(data, expected):
    assert bot._extract_answer(data) == expected


# ---------------------------------------------------------------------------
# handle_message
# ---------------------------------------------------------------------------

async def test_handle_message_happy_path(monkeypatch, db_session):
    fake_send, published = answering_publisher(lambda d: "Answer <with> & markup *_[")
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    msg = make_message("When is the midterm exam?")

    await bot.handle_message(make_update(msg), None)
    await drain_background()

    # Published from the publisher thread, not the event loop thread
    assert len(published) == 1
    payload, thread = published[0]
    assert thread is not threading.main_thread()
    rows = all_raw()
    assert len(rows) == 1
    row = rows[0]
    assert payload["raw_data_id"] == row.id
    assert payload["question"] == "When is the midterm exam?"
    assert payload["update_message_id"] == 10
    assert payload["chat_id"] == msg.chat_id
    assert row.language == "EN"
    assert row.username == "student"

    msg.reply_text.assert_awaited_once()
    call = msg.reply_text.await_args
    text = call.args[0]
    assert call.kwargs["parse_mode"] == ParseMode.HTML
    assert "Answer &lt;with&gt; &amp; markup *_[" in text
    assert text.startswith("@student asked:\n❓ When is the midterm exam?")
    keyboard = call.kwargs["reply_markup"].inline_keyboard
    assert [b.callback_data for b in keyboard[0]] == [f"like_{row.id}", f"dislike_{row.id}"]
    assert [b.text for b in keyboard[0]] == ["👍 0", "👎 0"]
    msg.chat.send_action.assert_awaited_once()

    assert row.telegram_message_id == 1000
    assert bot.answer_router.pending_count() == 0
    bot.mark_duplicate_questions.assert_called_once()
    assert bot.mark_duplicate_questions.call_args.args[1:] == (row.id, "When is the midterm exam?")


async def test_concurrent_questions_get_their_own_answers(monkeypatch, db_session):
    published = []
    both_published = threading.Event()

    def fake_send(data):
        published.append(data)
        if len(published) == 2:
            both_published.set()
        return True

    def worker():
        assert both_published.wait(5)
        # Answers arrive in reverse order
        for data in reversed(list(published)):
            body = json.dumps({"raw_data_id": data["raw_data_id"], "answer": f"A:{data['question']}"})
            bot._on_answer_message(MagicMock(), SimpleNamespace(delivery_tag=1), None, body)

    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    threading.Thread(target=worker, daemon=True).start()

    m1 = make_message("first question about exams", message_id=1)
    m2 = make_message("second question about courses", message_id=2)
    await asyncio.gather(
        bot.handle_message(make_update(m1, make_user(1, "one")), None),
        bot.handle_message(make_update(m2, make_user(2, "two")), None),
    )
    await drain_background()

    assert "A:first question about exams" in m1.reply_text.await_args.args[0]
    assert "A:second question about courses" in m2.reply_text.await_args.args[0]
    assert bot.answer_router.pending_count() == 0


async def test_timeout_sends_fallback_without_keyboard(monkeypatch, db_session):
    """BOT_LATE_ANSWER_WINDOW=0: the old behaviour (fallback, late answers dropped)."""
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 0.2)
    monkeypatch.setattr(bot, "LATE_ANSWER_WINDOW", 0.0)
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bot, "send_to_rabbitmq", send)
    msg = make_message()

    started = time.monotonic()
    await bot.handle_message(make_update(msg), None)
    await drain_background()
    assert time.monotonic() - started < 3

    call = msg.reply_text.await_args
    assert bot.FALLBACK_RESPONSE in call.args[0]
    assert call.kwargs["reply_markup"] is None
    assert bot.answer_router.pending_count() == 0 and bot.answer_router.late_count() == 0
    payload = send.call_args.args[0]
    # nobody sees an answer after the timeout: the worker may skip it from then on
    assert payload["deliver_until"] == pytest.approx(payload["asked_at"] + 0.2, abs=0.01)

    row = all_raw()[0]
    assert row.answer is None  # stays unanswered; the worker may fill it later
    # A late answer is acknowledged and dropped, not queued
    ch = MagicMock()
    bot._on_answer_message(ch, SimpleNamespace(delivery_tag=9), None,
                           json.dumps({"raw_data_id": row.id, "answer": "late"}))
    ch.basic_ack.assert_called_once_with(delivery_tag=9)
    assert bot.answer_router.pending_count() == 0


def late_answer_body(raw_id, answer="Geç gelen cevap.", **extra):
    return json.dumps({"raw_data_id": raw_id, "answer": answer, **extra}).encode()


async def test_answer_after_the_timeout_is_posted_as_a_reply(monkeypatch, db_session):
    """It used to be dropped: the asker only ever saw "Sorry, I cannot generate a response"."""
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 0.2)
    monkeypatch.setattr(bot, "LATE_ANSWER_WINDOW", 600.0)
    published = []
    monkeypatch.setattr(bot, "send_to_rabbitmq", lambda data: published.append(data) or True)
    msg = make_message("Staj defteri ne zaman teslim edilir?")
    await bot.handle_message(make_update(msg), None)
    await drain_background()

    first = msg.reply_text.await_args
    assert "taking longer than usual" in first.args[0] and "10 minutes" in first.args[0]
    assert bot.FALLBACK_RESPONSE not in first.args[0] and first.kwargs["reply_markup"] is None
    row = all_raw()[0]
    assert row.telegram_message_id == 1000
    assert bot.answer_router.pending_count() == 0 and bot.answer_router.late_count() == 1

    payload = published[0]
    assert payload["deliver_until"] == pytest.approx(payload["asked_at"] + 0.2 + 600.0, abs=0.01)
    assert abs(payload["asked_at"] - time.time()) < 5

    ch = MagicMock()
    bot._on_answer_message(ch, SimpleNamespace(delivery_tag=3), None, late_answer_body(row.id))
    bot._on_answer_message(ch, SimpleNamespace(delivery_tag=4), None,
                           late_answer_body(row.id, status="resent"))  # duplicate from the worker
    await settle()
    assert msg.reply_text.await_count == 2  # the duplicate was not posted again
    late = msg.reply_text.await_args
    assert "Geç gelen cevap." in late.args[0] and late.args[0].startswith("@student asked:")
    keyboard = late.kwargs["reply_markup"].inline_keyboard
    assert [b.callback_data for b in keyboard[0]] == [f"like_{row.id}", f"dislike_{row.id}"]
    assert fetch_raw(row.id).telegram_message_id == 1001  # votes go to the answer message
    assert bot.answer_router.late_count() == 0


async def test_late_answer_never_overtakes_the_timeout_reply(monkeypatch, db_session):
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 0.1)
    monkeypatch.setattr(bot, "LATE_ANSWER_WINDOW", 600.0)
    monkeypatch.setattr(bot, "send_to_rabbitmq", MagicMock(return_value=True))
    msg = make_message()
    order = []

    async def slow_reply(text, **kwargs):
        order.append(text)
        if len(order) == 1:  # the timeout reply is slow; the answer arrives meanwhile
            raw_id = all_raw()[0].id
            bot._on_answer_message(MagicMock(), SimpleNamespace(delivery_tag=1), None, late_answer_body(raw_id))
            await asyncio.sleep(0.2)
        return SimpleNamespace(message_id=2000 + len(order))

    msg.reply_text = AsyncMock(side_effect=slow_reply)
    await bot.handle_message(make_update(msg), None)
    await settle()
    assert len(order) == 2
    assert "taking longer" in order[0] and "Geç gelen cevap." in order[1]


async def test_empty_late_answer_is_not_posted(monkeypatch, db_session):
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 0.1)
    monkeypatch.setattr(bot, "send_to_rabbitmq", MagicMock(return_value=True))
    msg = make_message()
    await bot.handle_message(make_update(msg), None)
    bot._on_answer_message(MagicMock(), SimpleNamespace(delivery_tag=1), None,
                           late_answer_body(all_raw()[0].id, answer="  "))
    await settle()
    assert msg.reply_text.await_count == 1


async def test_publish_failure_replies_immediately(monkeypatch, db_session):
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 10.0)
    monkeypatch.setattr(bot, "send_to_rabbitmq", MagicMock(return_value=False))
    msg = make_message()

    started = time.monotonic()
    await bot.handle_message(make_update(msg), None)
    await drain_background()

    assert time.monotonic() - started < 2
    call = msg.reply_text.await_args
    assert bot.FALLBACK_RESPONSE in call.args[0]
    assert call.kwargs["reply_markup"] is None
    assert len(all_raw()) == 1  # question still stored
    assert bot.answer_router.pending_count() == 0


async def test_empty_answer_sends_fallback(monkeypatch, db_session):
    fake_send, _ = answering_publisher(lambda d: "   ")
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    msg = make_message()
    await bot.handle_message(make_update(msg), None)
    await drain_background()
    call = msg.reply_text.await_args
    assert bot.FALLBACK_RESPONSE in call.args[0]
    assert call.kwargs["reply_markup"] is None


async def test_parse_error_falls_back_to_plain_text(monkeypatch, db_session):
    fake_send, _ = answering_publisher(lambda d: "a <b> answer")
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    msg = make_message("question <x>")
    msg.reply_text = AsyncMock(side_effect=[
        BadRequest("Can't parse entities: unsupported start tag"),
        SimpleNamespace(message_id=555),
    ])
    await bot.handle_message(make_update(msg), None)
    await drain_background()

    assert msg.reply_text.await_count == 2
    second = msg.reply_text.await_args_list[1]
    assert "parse_mode" not in second.kwargs
    assert second.args[0] == "@student asked:\n❓ question <x>\n\n🤖 Answer:\na <b> answer"
    assert second.kwargs["reply_markup"] is not None
    assert all_raw()[0].telegram_message_id == 555


async def test_flood_control_is_retried_once(monkeypatch, db_session):
    fake_send, _ = answering_publisher()
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    sleeps = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds, *a, **k):
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(bot.asyncio, "sleep", fake_sleep)
    msg = make_message()
    msg.reply_text = AsyncMock(side_effect=[RetryAfter(3), SimpleNamespace(message_id=77)])
    await bot.handle_message(make_update(msg), None)
    await drain_background()
    assert 3 in sleeps
    assert msg.reply_text.await_count == 2
    assert all_raw()[0].telegram_message_id == 77


async def test_long_answer_is_sent_in_chunks_with_keyboard_on_last(monkeypatch, db_session):
    fake_send, _ = answering_publisher(lambda d: "lorem ipsum " * 1200)
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    msg = make_message("q" * 4000)
    await bot.handle_message(make_update(msg), None)
    await drain_background()

    calls = msg.reply_text.await_args_list
    assert len(calls) >= 4
    for call in calls:
        assert bot._utf16_len(visible_text(call.args[0])) <= bot.MAX_MESSAGE_LENGTH
    assert all(c.kwargs["reply_markup"] is None for c in calls[:-1])
    assert calls[-1].kwargs["reply_markup"] is not None
    # telegram_message_id points at the message that carries the vote buttons
    assert all_raw()[0].telegram_message_id == 1000 + len(calls) - 1


async def test_topic_messages_keep_thread_id(monkeypatch, db_session):
    fake_send, published = answering_publisher()
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    msg = make_message(thread_id=55, is_topic=True)
    await bot.handle_message(make_update(msg), None)
    await drain_background()
    assert msg.chat.send_action.await_args.kwargs["message_thread_id"] == 55
    assert msg.reply_text.await_args.kwargs["message_thread_id"] == 55
    assert published[0][0]["message_thread_id"] == 55
    assert all_raw()[0].message_thread_id == 55


async def test_typing_failure_does_not_break_answer(monkeypatch, db_session):
    fake_send, _ = answering_publisher()
    monkeypatch.setattr(bot, "send_to_rabbitmq", fake_send)
    msg = make_message()
    msg.chat.send_action = AsyncMock(side_effect=NetworkError("boom"))
    await bot.handle_message(make_update(msg), None)
    await drain_background()
    assert "Answer to:" in msg.reply_text.await_args.args[0]


def test_stale_question_is_not_published(monkeypatch):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bot, "send_to_rabbitmq", send)
    assert bot._publish_before_deadline({"raw_data_id": 1}, time.monotonic() - 1) is False
    send.assert_not_called()
    assert bot._publish_before_deadline({"raw_data_id": 1}, time.monotonic() + 60) is True
    send.assert_called_once()


async def test_hanging_publish_times_out(monkeypatch, db_session):
    monkeypatch.setattr(bot, "ANSWER_TIMEOUT", 0.3)
    release = threading.Event()

    def hanging_send(data):
        release.wait(5)
        return True

    monkeypatch.setattr(bot, "send_to_rabbitmq", hanging_send)
    msg = make_message()
    started = time.monotonic()
    try:
        await bot.handle_message(make_update(msg), None)
    finally:
        release.set()
    await drain_background()
    assert time.monotonic() - started < 3
    assert bot.FALLBACK_RESPONSE in msg.reply_text.await_args.args[0]
    assert bot.answer_router.pending_count() == 0


async def test_db_failure_replies_error_and_does_not_publish(monkeypatch, db_session):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bot, "send_to_rabbitmq", send)

    def broken_save(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(bot, "_save_question", broken_save)
    msg = make_message()
    await bot.handle_message(make_update(msg), None)
    send.assert_not_called()
    assert msg.reply_text.await_args.args[0] == bot.ERROR_REPLY
    assert bot.answer_router.pending_count() == 0


async def test_error_reply_failure_does_not_raise(monkeypatch, db_session):
    monkeypatch.setattr(bot, "_save_question", MagicMock(side_effect=RuntimeError("x")))
    msg = make_message()
    msg.reply_text = AsyncMock(side_effect=NetworkError("down"))
    await bot.handle_message(make_update(msg), None)  # must not raise


@pytest.mark.parametrize("mutate", [
    lambda u, m: setattr(u, "message", None),                          # edited msg / other update
    lambda u, m: setattr(m, "text", None),                             # sticker / photo
    lambda u, m: setattr(m.chat, "id", -100123),                       # foreign chat
    lambda u, m: setattr(m, "text", "/start"),                         # command
    lambda u, m: setattr(m, "text", "   "),                            # blank
    lambda u, m: setattr(m, "is_automatic_forward", True),             # linked channel post
    lambda u, m: setattr(u, "effective_user", None),                   # no sender
])
async def test_ignored_updates(monkeypatch, db_session, mutate):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bot, "send_to_rabbitmq", send)
    msg = make_message()
    update = make_update(msg)
    mutate(update, msg)
    await bot.handle_message(update, None)
    send.assert_not_called()
    msg.reply_text.assert_not_awaited()
    assert all_raw() == []


async def test_topic_filter(monkeypatch, db_session):
    monkeypatch.setattr(bot, "BOT_TOPIC_ID", 99)
    send = MagicMock(return_value=False)
    monkeypatch.setattr(bot, "send_to_rabbitmq", send)

    other = make_message(thread_id=5, is_topic=True)
    await bot.handle_message(make_update(other), None)
    send.assert_not_called()

    right = make_message(thread_id=99, is_topic=True)
    await bot.handle_message(make_update(right), None)
    await drain_background()
    send.assert_called_once()


async def test_real_duplicate_check_runs_in_background(monkeypatch, db_session):
    monkeypatch.setattr(bot, "mark_duplicate_questions", database_models.mark_duplicate_questions)
    db_session.add(RawData(telegram_id=1, question="When is the final exam?"))
    db_session.commit()
    original_id = db_session.query(RawData).first().id

    monkeypatch.setattr(bot, "send_to_rabbitmq", MagicMock(return_value=False))
    msg = make_message("When is the final exam?")
    await bot.handle_message(make_update(msg), None)
    await drain_background()

    new_row = [r for r in all_raw() if r.id != original_id][0]
    assert new_row.is_duplicate is True
    assert new_row.duplicate_of_id == original_id


# ---------------------------------------------------------------------------
# RabbitMQ publisher
# ---------------------------------------------------------------------------

def fake_connection():
    conn = MagicMock()
    conn.is_open = True
    channel = MagicMock()
    channel.is_open = True
    conn.channel.return_value = channel
    return conn, channel


def test_publisher_uses_config_queue_persistent_and_reuses_connection(monkeypatch):
    monkeypatch.setenv("QUESTIONS_QUEUE", "q_test")
    conn, channel = fake_connection()
    factory = MagicMock(return_value=conn)
    monkeypatch.setattr(bot.pika, "BlockingConnection", factory)

    assert bot.send_to_rabbitmq({"raw_data_id": 1, "question": "a"}) is True
    assert bot.send_to_rabbitmq({"raw_data_id": 2, "question": "b"}) is True

    assert factory.call_count == 1  # one long-lived connection
    channel.confirm_delivery.assert_called_once()
    channel.queue_declare.assert_called_with(queue="q_test", passive=True)
    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["routing_key"] == "q_test"
    assert kwargs["exchange"] == ""
    assert kwargs["properties"].delivery_mode == 2
    assert json.loads(kwargs["body"]) == {"raw_data_id": 2, "question": "b"}


def test_publisher_default_queue_name():
    assert bot.config.questions_queue == "questions"
    assert bot.config.answers_queue == "answers"


def test_publisher_declares_durable_queue_when_missing(monkeypatch):
    conn = MagicMock()
    conn.is_open = True
    first, second = MagicMock(), MagicMock()
    first.queue_declare.side_effect = pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
    second.is_open = True
    conn.channel.side_effect = [first, second]
    monkeypatch.setattr(bot.pika, "BlockingConnection", MagicMock(return_value=conn))

    assert bot.send_to_rabbitmq({"raw_data_id": 1}) is True
    second.queue_declare.assert_called_once_with(queue="questions", durable=True)
    second.basic_publish.assert_called_once()


def test_publisher_reconnects_after_stale_connection(monkeypatch):
    conn1, ch1 = fake_connection()
    conn2, ch2 = fake_connection()
    ch1.basic_publish.side_effect = pika.exceptions.StreamLostError("lost")
    factory = MagicMock(side_effect=[conn1, conn2])
    monkeypatch.setattr(bot.pika, "BlockingConnection", factory)

    assert bot.send_to_rabbitmq({"raw_data_id": 1}) is True
    assert factory.call_count == 2
    ch2.basic_publish.assert_called_once()


def test_publisher_returns_false_when_broker_down(monkeypatch, caplog):
    factory = MagicMock(side_effect=pika.exceptions.AMQPConnectionError("refused"))
    monkeypatch.setattr(bot.pika, "BlockingConnection", factory)
    with caplog.at_level(logging.INFO, logger=bot.logger.name):
        assert bot.send_to_rabbitmq({"raw_data_id": 3}) is False
    assert factory.call_count == bot.RabbitMQPublisher.MAX_ATTEMPTS
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is None


def test_connection_parameters_from_config(monkeypatch):
    monkeypatch.setenv("RABBITMQ_HOST", "mq.local")
    monkeypatch.setenv("RABBITMQ_PORT", "5673")
    monkeypatch.setenv("RABBITMQ_USER", "bot")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "pw")
    params = bot._connection_parameters()
    assert params.host == "mq.local"
    assert params.port == 5673
    assert params.credentials.username == "bot"
    assert params.blocked_connection_timeout == 30


# ---------------------------------------------------------------------------
# RabbitMQ consumer
# ---------------------------------------------------------------------------

def test_on_answer_message_routes_and_acks():
    router = bot.answer_router
    loop = MagicMock()
    fut = MagicMock()
    with router._lock:
        router._waiters[12] = (loop, fut)
    ch = MagicMock()
    bot._on_answer_message(ch, SimpleNamespace(delivery_tag=4), None,
                           json.dumps({"raw_data_id": "12", "answer": "x"}).encode())
    ch.basic_ack.assert_called_once_with(delivery_tag=4)
    loop.call_soon_threadsafe.assert_called_once()
    assert router.pending_count() == 0


@pytest.mark.parametrize("body", [b"not json", b"[]", b'"str"', b"{}", b'{"raw_data_id": "abc"}', b"\xff\xfe"])
def test_on_answer_message_rejects_poison_without_requeue(body):
    ch = MagicMock()
    bot._on_answer_message(ch, SimpleNamespace(delivery_tag=8), None, body)
    ch.basic_reject.assert_called_once_with(delivery_tag=8, requeue=False)
    ch.basic_nack.assert_not_called()
    ch.basic_ack.assert_not_called()


class RecordingStopEvent(threading.Event):
    """Records backoff waits instead of sleeping; stops after N waits."""

    def __init__(self, stop_after):
        super().__init__()
        self.delays = []
        self.stop_after = stop_after

    def wait(self, timeout=None):
        self.delays.append(timeout)
        if len(self.delays) >= self.stop_after:
            self.set()
        return self.is_set()


class ClockedStopEvent(RecordingStopEvent):
    """RecordingStopEvent whose waits advance a fake ``time.monotonic``."""

    def __init__(self, stop_after, monkeypatch):
        super().__init__(stop_after)
        self.now = 1000.0
        monkeypatch.setattr(bot.time, "monotonic", lambda: self.now)

    def wait(self, timeout=None):
        self.now += timeout or 0
        return super().wait(timeout)


def test_consumer_backoff_is_exponential_capped_and_concise(monkeypatch, caplog):
    factory = MagicMock(side_effect=pika.exceptions.AMQPConnectionError("refused"))
    monkeypatch.setattr(bot.pika, "BlockingConnection", factory)
    stop = ClockedStopEvent(stop_after=9, monkeypatch=monkeypatch)
    with caplog.at_level(logging.INFO, logger=bot.logger.name):
        bot.consume_answers(stop_event=stop)
    # capped well below ANSWER_TIMEOUT (30 s): after a broker outage the consumer is back
    # within 15 s, so answers do not keep missing their askers for a whole minute
    assert stop.delays == [1, 2, 4, 8, 15, 15, 15, 15, 15]
    assert bot.CONSUMER_RECONNECT_MAX_DELAY < 30.0
    records = [r for r in caplog.records if "RabbitMQ consumer" in r.getMessage()]
    # the outage is visible at once; the retries within the next 5 minutes are not logged
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert "refused" in records[0].getMessage() and not records[0].exc_info


def test_long_consumer_outage_is_logged_sparsely_and_recovery_is_logged(monkeypatch, caplog):
    """15 s retries used to log one line each (240 lines per hour of outage)."""
    attempts = {"n": 0}
    stop = ClockedStopEvent(stop_after=10_000, monkeypatch=monkeypatch)

    def factory(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 242:  # ~1 hour of 15 s retries
            raise pika.exceptions.AMQPConnectionError("refused")
        conn, _ = fake_connection()
        conn.process_data_events.side_effect = lambda **k: stop.set()
        return conn

    monkeypatch.setattr(bot.pika, "BlockingConnection", factory)
    with caplog.at_level(logging.INFO, logger=bot.logger.name):
        bot.consume_answers(stop_event=stop)
    assert sum(stop.delays) >= 3500
    failures = [r for r in caplog.records if "RabbitMQ consumer error" in r.getMessage()]
    assert failures[0].levelno == logging.ERROR
    assert all(r.levelno == logging.WARNING for r in failures[1:])
    assert 10 <= len(failures) <= 14  # one per 5 minutes instead of 242
    assert any("Reconnected to RabbitMQ after 242 failed attempt(s)" in r.getMessage() for r in caplog.records)


def test_consumer_flapping_connection_keeps_backing_off(monkeypatch):
    def make_conn(*a, **k):
        conn, _ = fake_connection()
        conn.process_data_events.side_effect = pika.exceptions.StreamLostError("lost")
        return conn

    monkeypatch.setattr(bot.pika, "BlockingConnection", MagicMock(side_effect=make_conn))
    stop = RecordingStopEvent(stop_after=4)
    bot.consume_answers(stop_event=stop)
    assert stop.delays == [1, 2, 4, 8]


def test_consumer_uses_answers_queue_and_stops_cleanly(monkeypatch):
    monkeypatch.setenv("ANSWERS_QUEUE", "a_test")
    conn, channel = fake_connection()
    stop = threading.Event()
    conn.process_data_events.side_effect = lambda **k: stop.set()
    monkeypatch.setattr(bot.pika, "BlockingConnection", MagicMock(return_value=conn))

    bot.consume_answers(stop_event=stop)

    channel.queue_declare.assert_called_with(queue="a_test", passive=True)
    channel.basic_qos.assert_called_once()
    channel.basic_consume.assert_called_once_with(queue="a_test", on_message_callback=bot._on_answer_message)
    conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Callback (votes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data,expected", [
    ("like_12", ("like", 12)),
    ("dislike_7", ("dislike", 7)),
    ("like_", None),
    ("like", None),
    ("foo_1", None),
    ("like_1_2", None),
    ("like_-1", None),
    ("like_abc", None),
    ("like_1 ", None),
    ("like_" + "9" * 30, None),
    (None, None),
    ("", None),
])
def test_parse_callback_data(data, expected):
    assert bot.parse_callback_data(data) == expected


def make_query(data, chat_id=None, message_id=777, user_id=500):
    query = MagicMock()
    query.data = data
    query.message.chat.id = bot.ALLOWED_CHAT_ID if chat_id is None else chat_id
    query.message.message_id = message_id
    query.from_user.id = user_id
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return update, query


def add_answered_row(db_session, telegram_message_id=777):
    row = RawData(telegram_id=1, question="q", answer="a", telegram_message_id=telegram_message_id)
    db_session.add(row)
    db_session.commit()
    return row.id


async def test_vote_records_and_updates_buttons(db_session):
    raw_id = add_answered_row(db_session)
    update, query = make_query(f"like_{raw_id}")
    await bot.handle_callback(update, None)

    query.answer.assert_awaited_once()
    assert query.answer.await_args.kwargs["show_alert"] is False
    markup = query.edit_message_reply_markup.await_args.kwargs["reply_markup"]
    assert [b.text for b in markup.inline_keyboard[0]] == ["👍 1", "👎 0"]
    assert fetch_raw(raw_id).like == 1  # legacy column kept in sync (by handle_user_vote)

    # Second user dislikes -> tie -> like is cleared
    update2, query2 = make_query(f"dislike_{raw_id}", user_id=501)
    await bot.handle_callback(update2, None)
    markup = query2.edit_message_reply_markup.await_args.kwargs["reply_markup"]
    assert [b.text for b in markup.inline_keyboard[0]] == ["👍 1", "👎 1"]
    assert fetch_raw(raw_id).like is None


async def test_double_click_same_vote(db_session):
    raw_id = add_answered_row(db_session)
    update, query = make_query(f"like_{raw_id}")
    await bot.handle_callback(update, None)
    update, query = make_query(f"like_{raw_id}")
    await bot.handle_callback(update, None)
    assert query.answer.await_args.kwargs["show_alert"] is True
    assert "already" in query.answer.await_args.args[0]
    query.edit_message_reply_markup.assert_not_awaited()
    s = database_models.SessionLocal()
    try:
        assert s.query(UserVotes).count() == 1
    finally:
        s.close()


@pytest.mark.parametrize("data", ["garbage", "like_1_2", None, "like_x"])
async def test_malformed_callback_data_is_rejected(db_session, data):
    update, query = make_query(data)
    await bot.handle_callback(update, None)
    query.answer.assert_awaited_once()
    assert query.answer.await_args.kwargs["show_alert"] is True
    query.edit_message_reply_markup.assert_not_awaited()


async def test_callback_from_other_chat_is_rejected(db_session):
    raw_id = add_answered_row(db_session)
    update, query = make_query(f"like_{raw_id}", chat_id=-100999)
    await bot.handle_callback(update, None)
    assert query.answer.await_args.args[0] == "Unauthorized access!"
    query.edit_message_reply_markup.assert_not_awaited()


async def test_callback_without_message_is_rejected(db_session):
    update, query = make_query("like_1")
    query.message = None
    await bot.handle_callback(update, None)
    assert query.answer.await_args.args[0] == "Unauthorized access!"


async def test_callback_missing_record(db_session):
    update, query = make_query("like_424242")
    await bot.handle_callback(update, None)
    assert query.answer.await_args.args[0] == "Record not found!"


async def test_forged_callback_for_other_message_is_rejected(db_session):
    raw_id = add_answered_row(db_session, telegram_message_id=111)
    update, query = make_query(f"like_{raw_id}", message_id=222)
    await bot.handle_callback(update, None)
    assert query.answer.await_args.args[0] == "Record not found!"
    s = database_models.SessionLocal()
    try:
        assert s.query(UserVotes).count() == 0
    finally:
        s.close()


async def test_not_modified_error_is_swallowed(db_session):
    raw_id = add_answered_row(db_session)
    update, query = make_query(f"like_{raw_id}")
    query.edit_message_reply_markup = AsyncMock(side_effect=BadRequest("Message is not modified"))
    await bot.handle_callback(update, None)  # must not raise
    assert query.answer.await_count == 1


async def test_callback_db_error(monkeypatch, db_session):
    monkeypatch.setattr(bot, "_apply_vote", MagicMock(side_effect=RuntimeError("db down")))
    update, query = make_query("like_1")
    await bot.handle_callback(update, None)
    assert query.answer.await_args.args[0] == "An error occurred!"


# ---------------------------------------------------------------------------
# /start, error handler, application wiring
# ---------------------------------------------------------------------------

async def test_start_command_allowed_and_foreign_chat():
    msg = make_message()
    await bot.start_command(make_update(msg), None)
    msg.reply_text.assert_awaited_once()

    other = make_message(chat_id=-1)
    await bot.start_command(make_update(other), None)
    other.reply_text.assert_not_awaited()

    update = MagicMock()
    update.effective_message = None
    await bot.start_command(update, None)  # no crash


async def test_error_handler_throttles_network_errors(caplog):
    ctx = SimpleNamespace(error=NetworkError("httpx.ConnectError: name resolution"))
    with caplog.at_level(logging.WARNING, logger=bot.logger.name):
        for _ in range(5):
            await bot.error_handler(None, ctx)
    records = [r for r in caplog.records if "connectivity" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is None
    assert bot._network_error_log_state["suppressed"] == 4
    # After the interval a summary line is logged
    last = bot._network_error_log_state["last"]
    assert bot._log_network_error(TimedOut(), now=last + bot.NETWORK_ERROR_LOG_INTERVAL) is True


async def test_error_handler_logs_other_errors_with_traceback(caplog):
    try:
        raise ValueError("bug")
    except ValueError as e:
        err = e
    with caplog.at_level(logging.ERROR, logger=bot.logger.name):
        await bot.error_handler(None, SimpleNamespace(error=err))
    assert caplog.records[-1].exc_info is not None


def test_build_application_is_concurrent_and_wired():
    app = bot.build_application(token="123456:TEST")
    assert app.concurrent_updates > 1
    handlers = app.handlers[0]
    assert len(handlers) == 3
    assert bot.error_handler in app.error_handlers


def test_run_polling_retries_with_backoff_on_network_errors():
    apps = [MagicMock(), MagicMock(), MagicMock()]
    apps[0].run_polling.side_effect = NetworkError("httpx.ConnectError: name resolution")
    apps[1].run_polling.side_effect = TimedOut()
    sleeps = []
    bot.run_polling_with_retries(application_factory=iter(apps).__next__, sleep=sleeps.append,
                                 initial_delay=5, max_delay=300)
    assert sleeps == [5, 10]
    kwargs = apps[2].run_polling.call_args.kwargs
    assert kwargs["allowed_updates"] == ["message", "callback_query"]
    assert kwargs["close_loop"] is False


def test_run_polling_does_not_retry_invalid_token():
    app = MagicMock()
    app.run_polling.side_effect = InvalidToken()
    sleeps = []
    with pytest.raises(InvalidToken):
        bot.run_polling_with_retries(application_factory=lambda: app, sleep=sleeps.append)
    assert sleeps == []


def test_run_polling_backoff_is_capped():
    app = MagicMock()
    app.run_polling.side_effect = NetworkError("down")
    sleeps = []
    with pytest.raises(NetworkError):
        bot.run_polling_with_retries(application_factory=lambda: app, sleep=sleeps.append,
                                     initial_delay=5, max_delay=40, max_attempts=6)
    assert sleeps == [5, 10, 20, 40, 40]


def test_httpx_logger_does_not_leak_token_at_info():
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


def test_pika_connection_noise_is_silenced():
    """While the broker is down pika logs ~10 ERROR lines with two tracebacks per
    connect attempt (36 KB for a 75 s outage in the live run); the bot logs one line."""
    bot._quiet_noisy_loggers()
    for name in ("pika.adapters.utils.io_services_utils", "pika.adapters.blocking_connection",
                 "pika.adapters.utils.connection_workflow", "pika.connection"):
        assert not logging.getLogger(name).isEnabledFor(logging.ERROR)


def test_real_pika_consumer_failure_is_one_concise_line(monkeypatch, caplog):
    """With the real pika against a closed local port: one bot line, nothing from pika."""
    import socket

    with socket.socket() as probe:  # a free port nobody listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("RABBITMQ_HOST", "127.0.0.1")
    monkeypatch.setenv("RABBITMQ_PORT", str(port))
    bot._quiet_noisy_loggers()
    stop = ClockedStopEvent(stop_after=3, monkeypatch=monkeypatch)
    with caplog.at_level(logging.DEBUG):
        bot.consume_answers(stop_event=stop)
    visible = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert [r.name for r in visible] == [bot.logger.name, bot.logger.name]  # the error + "consumer stopped"
    assert visible[0].levelno == logging.ERROR and "AMQPConnectionError" in visible[0].getMessage()
    assert "Connection refused" in visible[0].getMessage()  # str() of pika's error is empty
    assert not visible[0].exc_info
    assert stop.delays == [1, 2, 4]


def test_publisher_failure_line_has_the_reason(monkeypatch, caplog):
    from pika.adapters.utils.connection_workflow import AMQPConnectorSocketConnectError

    # exactly what pika raises for a refused connection: str() is empty at both levels
    refused = pika.exceptions.AMQPConnectionError(
        AMQPConnectorSocketConnectError(ConnectionRefusedError(111, "Connection refused")))
    assert str(refused) == ""
    monkeypatch.setattr(bot.pika, "BlockingConnection", MagicMock(side_effect=refused))
    with caplog.at_level(logging.INFO, logger=bot.logger.name):
        assert bot.send_to_rabbitmq({"raw_data_id": 5}) is False
    (error,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "question 5: AMQPConnectionError: AMQPConnectorSocketConnectError: ConnectionRefusedError(111, " \
        "'Connection refused')" in error.getMessage()
