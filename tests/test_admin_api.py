"""Tests for the FastAPI admin API (src/admin_rest_api.py).

Uses the temp SQLite DB from conftest (``db_session`` fixture), never the
real university_bot.db, never the network, never a GPU.
"""

import ast
import base64
import hashlib
import logging
import os
import runpy
from datetime import datetime, timedelta
from pathlib import Path

import re

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import admin_rest_api as api
import database_models
from database_models import RawData, TrainingData, UserVotes

MODULE_PATH = Path(api.__file__).resolve()
REAL_GPU_INFO = api._gpu_info  # saved before the autouse fixture stubs it
AUTH = ("admin", "test-admin-password")

BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)

# Routes that are intentionally reachable without credentials.
PUBLIC_ROUTES = {
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/openapi.json"),
    ("GET", "/docs"),
    ("GET", "/docs/oauth2-redirect"),
    ("GET", "/redoc"),
}


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep auth env at the conftest defaults and never write the real logs."""
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", "")
    calls = []

    def fake_handle_database_error(error, operation, table=None):
        calls.append((type(error).__name__, operation, table))
        return {"success": False, "error": {"message": "Database operation failed. Please try again."}}

    monkeypatch.setattr(api, "handle_database_error", fake_handle_database_error)
    monkeypatch.setattr(api, "_gpu_info", lambda: {"available": False})
    return calls


@pytest.fixture
def client():
    # Tests that read or write data also request ``db_session``, which
    # (re)creates an empty schema on the temp DB; auth/CORS/docs tests don't
    # touch the DB and skip that (slow) reset.
    return TestClient(api.app)


def seed_raw(db, n=1, **overrides):
    rows = []
    for i in range(n):
        values = dict(
            telegram_id=1000 + i,
            telegram_message_id=500 + i,
            username=f"user{i}",
            question=f"Question number {i} about the computer engineering department?",
            answer=f"Answer number {i}",
            language="EN",
            admin_approved=0,
            is_duplicate=False,
            created_at=BASE_TIME + timedelta(minutes=i),
        )
        values.update(overrides)
        row = RawData(**values)
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows if n != 1 else rows[0]


def seed_training(db, answer, source_id=None, question="q?", **overrides):
    row = TrainingData(source_id=source_id, question=question, answer=answer, language="EN", **overrides)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def add_votes(db, raw_id, votes):
    for idx, vote in enumerate(votes):
        db.add(UserVotes(raw_data_id=raw_id, telegram_user_id=90000 + idx, current_vote=vote))
    db.commit()


def basic_header(value: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(value.encode("latin-1")).decode()}


def _route_url(path: str) -> str:
    return path.replace("{item_id}", "1")


def _iter_api_routes():
    """All APIRoutes, including those of included routers.

    Newer FastAPI versions keep included routers as a lazy ``_IncludedRouter``
    entry in ``app.routes`` instead of flattening them, so also walk the
    ``protected`` router directly (deduplicated by method/path).
    """
    seen = set()
    for route in list(api.app.routes) + list(api.protected.routes):
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods - {"HEAD"}):
            key = (method, route.path)
            if key not in seen:
                seen.add(key)
                yield key


def _protected_routes():
    return [key for key in _iter_api_routes() if key not in PUBLIC_ROUTES]


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def test_hardcoded_password_removed():
    source = MODULE_PATH.read_text(encoding="utf-8")
    # credentials must come from the environment, never from a literal in code
    # no string literal passed directly as an argument of compare_digest
    assert not re.search(r"compare_digest\(\s*b?[\"']", source)
    assert not re.search(r"compare_digest\([^,()]*(?:\([^()]*\))?[^,()]*,\s*b?[\"']", source)
    assert not re.search(r"(?i)password\s*==\s*[\"']", source)


def test_every_data_route_is_protected():
    routes = _protected_routes()
    paths = {p for _, p in routes}
    # Sanity: the data endpoints are all in the protected set.
    for expected in ("/raw-data", "/raw-data/{item_id}", "/approve/{item_id}", "/stats",
                     "/duplicates", "/training-data", "/training-data/{item_id}",
                     "/detect-duplicates", "/docs/list", "/docs/read", "/auth/login"):
        assert expected in paths


@pytest.mark.parametrize("method,path", _protected_routes())
def test_protected_route_requires_credentials(client, method, path):
    kwargs = {"params": {"path": "README.md"}} if path == "/docs/read" else {}
    if method == "PUT":
        kwargs["json"] = {"answer": "x"}
    resp = client.request(method, _route_url(path), **kwargs)
    assert resp.status_code == 401, (method, path, resp.text)
    assert resp.headers.get("www-authenticate") == "Basic"

    resp = client.request(method, _route_url(path), auth=("admin", "wrong"), **kwargs)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Basic"


def test_login_success_and_failures(client):
    ok = client.post("/auth/login", json={}, auth=AUTH)
    assert ok.status_code == 200
    assert ok.json() == {"success": True, "username": "admin"}

    for creds in [("admin", "wrong"), ("root", AUTH[1]), ("", ""), ("admin", "")]:
        resp = client.post("/auth/login", auth=creds)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Basic"


@pytest.mark.parametrize("header", [
    {"Authorization": "Bearer abc"},
    {"Authorization": "Basic !!!not-base64!!!"},
    basic_header("no-colon-here"),
    {"Authorization": ""},
])
def test_malformed_authorization_headers(client, header):
    resp = client.post("/auth/login", headers=header)
    assert resp.status_code == 401


def test_unset_password_rejects_everything(client, monkeypatch, caplog):
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", "")
    for pw in ["test-admin-password", "", "admin"]:
        assert client.post("/auth/login", auth=("admin", pw)).status_code == 401
    assert client.get("/stats", auth=AUTH).status_code == 401

    monkeypatch.delenv("ADMIN_PASSWORD")
    monkeypatch.delenv("ADMIN_PASSWORD_HASH")
    assert client.post("/auth/login", auth=AUTH).status_code == 401

    with caplog.at_level(logging.ERROR, logger=api.logger.name):
        api.log_auth_configuration()
    assert any(r.levelno == logging.ERROR and "NOT configured" in r.getMessage() for r in caplog.records)


def test_startup_lifespan_logs_auth_error(monkeypatch, caplog, db_session):
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    with caplog.at_level(logging.ERROR, logger=api.logger.name):
        with TestClient(api.app) as c:
            assert c.get("/").status_code == 200
    assert any("NOT configured" in r.getMessage() for r in caplog.records)


def test_hash_mode(client, monkeypatch):
    digest = hashlib.sha256(b"s3cret-pass").hexdigest()
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", digest)
    assert client.post("/auth/login", auth=("admin", "s3cret-pass")).status_code == 200
    # The hash takes precedence over ADMIN_PASSWORD.
    assert client.post("/auth/login", auth=AUTH).status_code == 401
    # Upper-case hex digest is accepted as well.
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", digest.upper())
    assert client.post("/auth/login", auth=("admin", "s3cret-pass")).status_code == 200


@pytest.mark.parametrize("bad_hash", [
    "your_admin_password_hash_here",
    "abc123",
    "g" * 64,
    "a" * 63,
    "a" * 65,
])
def test_malformed_hash_falls_back_to_plain_password(client, monkeypatch, bad_hash):
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", bad_hash)
    assert client.post("/auth/login", auth=AUTH).status_code == 200
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    assert client.post("/auth/login", auth=AUTH).status_code == 401


def test_hash_of_empty_password_is_not_accepted(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", hashlib.sha256(b"").hexdigest())
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    assert client.post("/auth/login", auth=("admin", "")).status_code == 401


@pytest.mark.parametrize("placeholder", ["your_admin_password_here", "changeme", "<password>", "   "])
def test_placeholder_plain_password_is_unset(client, monkeypatch, placeholder):
    monkeypatch.setenv("ADMIN_PASSWORD", placeholder)
    assert client.post("/auth/login", auth=("admin", placeholder)).status_code == 401


def test_username_from_env_and_default(client, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    assert client.post("/auth/login", auth=AUTH).status_code == 401
    resp = client.post("/auth/login", auth=("boss", AUTH[1]))
    assert resp.status_code == 200 and resp.json()["username"] == "boss"
    monkeypatch.setenv("ADMIN_USERNAME", "")
    assert client.post("/auth/login", auth=AUTH).status_code == 200


def test_env_is_read_at_request_time(client, monkeypatch):
    assert client.post("/auth/login", auth=AUTH).status_code == 200
    monkeypatch.setenv("ADMIN_PASSWORD", "rotated-password")
    assert client.post("/auth/login", auth=AUTH).status_code == 401
    assert client.post("/auth/login", auth=("admin", "rotated-password")).status_code == 200


def test_non_ascii_config_does_not_crash(client, monkeypatch):
    # secrets.compare_digest(str, str) raises TypeError for non-ASCII -> was a 500.
    monkeypatch.setenv("ADMIN_PASSWORD", "şifre-ğüç")
    assert client.post("/auth/login", auth=AUTH).status_code == 401
    monkeypatch.setenv("ADMIN_PASSWORD", AUTH[1])
    monkeypatch.setenv("ADMIN_USERNAME", "yönetici")
    assert client.post("/auth/login", auth=AUTH).status_code == 401
    # Latin-1 bytes in the header are not valid ASCII -> 401, not 500.
    assert client.post("/auth/login", headers=basic_header("admin:pässword")).status_code == 401


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #


def test_cors_settings_normalisation():
    assert api._cors_settings(["*"]) == {"allow_origins": ["*"], "allow_credentials": False}
    assert api._cors_settings(["http://a.example/", " http://a.example ", ""]) == {
        "allow_origins": ["http://a.example"],
        "allow_credentials": True,
    }
    assert api._cors_settings([])["allow_origins"] == ["http://localhost:3000"]
    mixed = api._cors_settings(["http://a.example", "*"])
    assert mixed["allow_credentials"] is False


def test_cors_middleware_uses_config():
    from starlette.middleware.cors import CORSMiddleware

    mw = [m for m in api.app.user_middleware if m.cls is CORSMiddleware]
    assert len(mw) == 1
    # Starlette renamed Middleware.options -> Middleware.kwargs
    options = getattr(mw[0], "kwargs", None) or mw[0].options
    expected = api._cors_settings(api.config.cors_origins)
    assert options["allow_origins"] == expected["allow_origins"]
    assert not ("*" in options["allow_origins"] and options["allow_credentials"])


def test_cors_preflight(client):
    allowed = api._cors["allow_origins"][0]
    if allowed == "*":  # pragma: no cover - depends on .env
        pytest.skip("CORS_ORIGINS is '*' in this environment")
    headers = {
        "Origin": allowed,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    }
    ok = client.options("/raw-data", headers=headers)
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == allowed

    evil = client.options("/raw-data", headers={**headers, "Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


# --------------------------------------------------------------------------- #
# GET /raw-data
# --------------------------------------------------------------------------- #


def test_raw_data_empty(client, db_session):
    resp = client.get("/raw-data", auth=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {
        "data": [], "total": 0, "page": 1, "page_size": 20,
        "total_pages": 0, "has_next": False, "has_prev": False,
    }


def test_raw_data_pagination_math(client, db_session):
    seed_raw(db_session, 25)
    p1 = client.get("/raw-data", params={"page": 1, "page_size": 10}, auth=AUTH).json()
    assert (p1["total"], p1["total_pages"], p1["has_next"], p1["has_prev"]) == (25, 3, True, False)
    assert len(p1["data"]) == 10
    p3 = client.get("/raw-data", params={"page": 3, "page_size": 10}, auth=AUTH).json()
    assert len(p3["data"]) == 5 and p3["has_next"] is False and p3["has_prev"] is True
    p4 = client.get("/raw-data", params={"page": 4, "page_size": 10}, auth=AUTH).json()
    assert p4["data"] == [] and p4["has_next"] is False
    exact = client.get("/raw-data", params={"page_size": 25}, auth=AUTH).json()
    assert exact["total_pages"] == 1 and exact["has_next"] is False
    # Newest first.
    created = [item["created_at"] for item in p1["data"]]
    assert created == sorted(created, reverse=True)


def test_raw_data_pages_are_stable_with_identical_timestamps(client, db_session):
    seed_raw(db_session, 9, created_at=BASE_TIME)
    seen = []
    for page in (1, 2, 3):
        seen += [d["id"] for d in client.get("/raw-data", params={"page": page, "page_size": 3}, auth=AUTH).json()["data"]]
    assert len(seen) == 9 and len(set(seen)) == 9


@pytest.mark.parametrize("params", [
    {"page": 0}, {"page": -1}, {"page": "abc"}, {"page": 10**30},
    {"page_size": 0}, {"page_size": -5}, {"page_size": 101}, {"page_size": "1;DROP TABLE raw_data"},
    {"only_unapproved": "1 OR 1=1"}, {"only_unapproved": "true'; DROP TABLE raw_data;--"},
])
def test_raw_data_invalid_params(client, db_session, params):
    seed_raw(db_session, 2)
    resp = client.get("/raw-data", params=params, auth=AUTH)
    assert resp.status_code == 422
    # Table still intact.
    assert db_session.query(RawData).count() == 2


def test_raw_data_unapproved_filter_and_shape(client, db_session):
    a, b, c = seed_raw(db_session, 3)
    b.admin_approved = 1
    c.admin_approved = None
    c.is_duplicate = None  # legacy NULLs must not break the response model
    db_session.commit()
    add_votes(db_session, a.id, [1, 1, -1, None])

    allrows = client.get("/raw-data", auth=AUTH).json()
    assert allrows["total"] == 3
    item_a = next(d for d in allrows["data"] if d["id"] == a.id)
    assert (item_a["likes"], item_a["dislikes"], item_a["total_votes"], item_a["vote_score"]) == (2, 1, 4, 1)
    item_c = next(d for d in allrows["data"] if d["id"] == c.id)
    assert item_c["admin_approved"] == 0 and item_c["is_duplicate"] is False
    expected_keys = {
        "id", "telegram_id", "telegram_message_id", "username", "question", "answer", "language",
        "like", "admin_approved", "is_duplicate", "duplicate_of_id", "similarity_score", "created_at",
        "answered_at", "message_thread_id", "likes", "dislikes", "total_votes", "vote_score",
    }
    assert set(item_a) == expected_keys

    unapproved = client.get("/raw-data", params={"only_unapproved": "true"}, auth=AUTH).json()
    assert {d["id"] for d in unapproved["data"]} == {a.id, c.id}
    assert unapproved["total"] == 2


def test_raw_data_has_no_n_plus_one(client, db_session, test_engine):
    rows = seed_raw(db_session, 40)
    for r in rows:
        add_votes(db_session, r.id, [1])
    engine = test_engine
    counter = {"n": 0}

    def _count(*_a, **_k):
        counter["n"] += 1

    event.listen(engine, "before_cursor_execute", _count)
    try:
        counter["n"] = 0
        client.get("/raw-data", params={"page_size": 5}, auth=AUTH)
        small = counter["n"]
        counter["n"] = 0
        client.get("/raw-data", params={"page_size": 40}, auth=AUTH)
        large = counter["n"]
    finally:
        event.remove(engine, "before_cursor_execute", _count)
    assert small == large <= 4


def test_raw_data_internal_error_is_generic(client, db_session, monkeypatch, _isolate):
    seed_raw(db_session, 1)

    def boom(*_a, **_k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api, "_vote_stats_for", boom)
    resp = client.get("/raw-data", auth=AUTH)
    assert resp.status_code == 500
    assert "secret internal detail" not in resp.text
    assert _isolate and _isolate[-1][1:] == ("select", "raw_data")


# --------------------------------------------------------------------------- #
# PUT /raw-data/{id}
# --------------------------------------------------------------------------- #


def test_update_answer(client, db_session):
    raw = seed_raw(db_session, answer=None)
    resp = client.put(f"/raw-data/{raw.id}", json={"answer": "  New answer  "}, auth=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True and body["training_data_updated"] is False
    db_session.expire_all()
    fresh = db_session.get(RawData, raw.id)
    assert fresh.answer == "New answer"
    assert fresh.answered_at is not None


@pytest.mark.parametrize("payload", [{"answer": ""}, {"answer": "   "}, {}, {"answer": None},
                                     {"answer": "x" * (api.MAX_ANSWER_LENGTH + 1)}])
def test_update_answer_validation(client, db_session, payload):
    raw = seed_raw(db_session)
    assert client.put(f"/raw-data/{raw.id}", json=payload, auth=AUTH).status_code == 422
    db_session.expire_all()
    assert db_session.get(RawData, raw.id).answer == raw.answer


@pytest.mark.parametrize("item_id,code", [("999", 404), ("0", 422), ("-3", 422), ("abc", 422),
                                          (str(2**64), 422), ("1 OR 1=1", 422)])
def test_update_answer_bad_ids(client, db_session, item_id, code):
    seed_raw(db_session)
    assert client.put(f"/raw-data/{item_id}", json={"answer": "x"}, auth=AUTH).status_code == code


def test_update_answer_propagates_to_training_copy(client, db_session):
    older = seed_training(db_session, "The department office is in building A, room 101.")
    raw = seed_raw(db_session, answer="The department office is in building A, room 101.")
    training = seed_training(db_session, raw.answer, source_id=raw.id,
                             is_answer_duplicate=True, duplicate_answer_of_id=older.id,
                             answer_similarity_score=1.0)
    other = seed_training(db_session, raw.answer, question="other",
                          is_answer_duplicate=True, duplicate_answer_of_id=training.id,
                          answer_similarity_score=1.0)
    resp = client.put(f"/raw-data/{raw.id}",
                      json={"answer": "Completely different text about exam schedules and dates."},
                      auth=AUTH)
    assert resp.status_code == 200 and resp.json()["training_data_updated"] is True
    db_session.expire_all()
    edited = db_session.get(TrainingData, training.id)
    assert edited.answer.startswith("Completely different")
    # Its own marking (made for the old text) was recomputed: no longer a duplicate.
    assert edited.is_answer_duplicate is False and edited.duplicate_answer_of_id is None
    # The stale duplicate marking that pointed at the edited answer is cleared.
    stale = db_session.get(TrainingData, other.id)
    assert stale.is_answer_duplicate is False and stale.duplicate_answer_of_id is None


# --------------------------------------------------------------------------- #
# POST /approve/{id}
# --------------------------------------------------------------------------- #


def test_approve_happy_path(client, db_session):
    raw = seed_raw(db_session, answer="Registration happens online through the student portal.")
    resp = client.post(f"/approve/{raw.id}", auth=AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True and isinstance(body["training_data_id"], int)
    db_session.expire_all()
    td = db_session.get(TrainingData, body["training_data_id"])
    assert td.source_id == raw.id and td.answer == raw.answer and td.question == raw.question
    assert td.review_status == "approved" and td.reviewer_id == "admin"
    assert db_session.get(RawData, raw.id).admin_approved == 1

    again = client.post(f"/approve/{raw.id}", auth=AUTH)
    assert again.status_code == 400 and again.json()["detail"] == "Already in training data"
    assert db_session.query(TrainingData).count() == 1


@pytest.mark.parametrize("answer", [None, "", "   \n "])
def test_approve_requires_answer(client, db_session, answer):
    raw = seed_raw(db_session, answer=answer)
    resp = client.post(f"/approve/{raw.id}", auth=AUTH)
    assert resp.status_code == 400
    assert db_session.query(TrainingData).count() == 0


def test_approve_not_found(client, db_session):
    assert client.post("/approve/12345", auth=AUTH).status_code == 404
    assert client.post("/approve/0", auth=AUTH).status_code == 422


def test_approve_marks_duplicate_answer(client, db_session):
    text_ = "You can find the course schedule on the department website under academics."
    existing = seed_training(db_session, text_)
    raw = seed_raw(db_session, answer=text_ + " ")
    body = client.post(f"/approve/{raw.id}", auth=AUTH).json()
    db_session.expire_all()
    td = db_session.get(TrainingData, body["training_data_id"])
    assert td.is_answer_duplicate is True
    assert td.duplicate_answer_of_id == existing.id
    assert td.answer_similarity_score == pytest.approx(1.0)


def test_approve_does_not_create_question_cycles(client, db_session):
    a = seed_raw(db_session, question="How do I register for courses this semester at the department?",
                 answer="Use the student portal to register.")
    b = seed_raw(db_session, question="How do I register for courses this semester at the department??",
                 answer="Portal.", is_duplicate=True, created_at=BASE_TIME + timedelta(hours=1))
    b.duplicate_of_id = a.id
    db_session.commit()
    assert client.post(f"/approve/{a.id}", auth=AUTH).status_code == 200
    db_session.expire_all()
    fresh_a = db_session.get(RawData, a.id)
    assert not fresh_a.is_duplicate and fresh_a.duplicate_of_id is None
    assert db_session.get(RawData, b.id).duplicate_of_id == a.id


def test_approve_integrity_race_returns_400(client, db_session, monkeypatch):
    raw = seed_raw(db_session)
    real_commit = Session.commit
    state = {"raised": False}

    def flaky_commit(self):
        if not state["raised"]:
            state["raised"] = True
            raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", flaky_commit)
    resp = client.post(f"/approve/{raw.id}", auth=AUTH)
    assert resp.status_code == 400 and resp.json()["detail"] == "Already in training data"


# --------------------------------------------------------------------------- #
# GET /stats
# --------------------------------------------------------------------------- #


def test_stats_empty(client, db_session):
    body = client.get("/stats", auth=AUTH).json()
    assert body["total_questions"] == 0 and body["approval_rate"] == "0%"
    assert body["unanswered_questions"] == 0


def test_stats_counts(client, db_session):
    rows = seed_raw(db_session, 6)
    rows[0].admin_approved = 1
    rows[1].like = 1                      # legacy column
    rows[2].like = -1
    rows[3].answer = None
    rows[4].answer = "   "
    rows[5].is_duplicate = True
    db_session.commit()
    add_votes(db_session, rows[3].id, [1, 1, -1])   # net liked via votes
    add_votes(db_session, rows[4].id, [-1])         # net disliked via votes
    add_votes(db_session, rows[5].id, [1, -1])      # net zero -> neither
    add_votes(db_session, rows[1].id, [-1, -1])     # legacy like wins
    seed_training(db_session, "some answer", source_id=rows[0].id)

    body = client.get("/stats", auth=AUTH).json()
    assert body == {
        "total_questions": 6,
        "approved_questions": 1,
        "liked_questions": 2,
        "disliked_questions": 2,
        "training_data_count": 1,
        "duplicate_count": 1,
        "unanswered_questions": 2,
        "approval_rate": "16.7%",
    }


# --------------------------------------------------------------------------- #
# GET /duplicates
# --------------------------------------------------------------------------- #


def test_duplicates_groups(client, db_session):
    assert client.get("/duplicates", auth=AUTH).json() == []
    a, b, c, d = seed_raw(db_session, 4)
    b.is_duplicate, b.duplicate_of_id, b.similarity_score = True, a.id, 0.9
    c.is_duplicate, c.duplicate_of_id = True, a.id
    db_session.commit()
    add_votes(db_session, b.id, [1])

    groups = client.get("/duplicates", auth=AUTH).json()
    assert len(groups) == 1
    assert groups[0]["original"]["id"] == a.id
    assert [x["id"] for x in groups[0]["duplicates"]] == [b.id, c.id]
    assert groups[0]["duplicates"][0]["likes"] == 1
    assert groups[0]["duplicates"][0]["similarity_score"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# GET /training-data
# --------------------------------------------------------------------------- #


def test_training_data_pagination_and_point(client, db_session):
    liked, disliked, legacy, novote = seed_raw(db_session, 4)
    legacy.like = -1
    db_session.commit()
    add_votes(db_session, liked.id, [1])
    add_votes(db_session, disliked.id, [-1, -1, 1])
    for raw in (liked, disliked, legacy, novote):
        seed_training(db_session, f"answer for {raw.id}", source_id=raw.id)
    standalone = seed_training(db_session, "standalone answer")

    body = client.get("/training-data", params={"page_size": 10}, auth=AUTH).json()
    assert body["total"] == 5 and body["total_pages"] == 1
    points = {d["source_id"]: d["point"] for d in body["data"]}
    assert points == {liked.id: 1, disliked.id: -1, legacy.id: -1, novote.id: None, None: None}
    assert {d["id"] for d in body["data"]} >= {standalone.id}
    assert set(body["data"][0]) == {
        "id", "source_id", "question", "answer", "language", "created_at", "point",
        "is_answer_duplicate", "duplicate_answer_of_id", "answer_similarity_score",
    }

    p = client.get("/training-data", params={"page": 2, "page_size": 2}, auth=AUTH).json()
    assert len(p["data"]) == 2 and p["total_pages"] == 3 and p["has_next"] and p["has_prev"]


@pytest.mark.parametrize("params", [{"page": 0}, {"page_size": 0}, {"page_size": 1000}, {"page": "1'--"}])
def test_training_data_invalid_params(client, params):
    assert client.get("/training-data", params=params, auth=AUTH).status_code == 422


# --------------------------------------------------------------------------- #
# DELETE endpoints
# --------------------------------------------------------------------------- #


def test_delete_raw_data_cleans_up(client, db_session):
    a, b, c = seed_raw(db_session, 3)
    b.is_duplicate, b.duplicate_of_id, b.similarity_score = True, a.id, 0.8
    db_session.commit()
    add_votes(db_session, a.id, [1, -1])
    add_votes(db_session, c.id, [1])
    td = seed_training(db_session, "answer a", source_id=a.id)
    other_td = seed_training(db_session, "answer a", is_answer_duplicate=True, duplicate_answer_of_id=td.id)

    a_id, b_id, c_id, td_id, other_id = a.id, b.id, c.id, td.id, other_td.id

    resp = client.delete(f"/raw-data/{a_id}", auth=AUTH)
    assert resp.status_code == 200 and resp.json()["success"] is True
    db_session.expunge_all()
    assert db_session.get(RawData, a_id) is None
    assert db_session.get(TrainingData, td_id) is None
    assert db_session.query(UserVotes).filter(UserVotes.raw_data_id == a_id).count() == 0
    assert db_session.query(UserVotes).filter(UserVotes.raw_data_id == c_id).count() == 1
    fresh_b = db_session.get(RawData, b_id)
    assert fresh_b.is_duplicate is False and fresh_b.duplicate_of_id is None
    assert db_session.get(TrainingData, other_id).duplicate_answer_of_id is None

    assert client.delete(f"/raw-data/{a_id}", auth=AUTH).status_code == 404
    assert client.delete("/raw-data/abc", auth=AUTH).status_code == 422


def test_delete_training_data(client, db_session):
    raw = seed_raw(db_session, admin_approved=1)
    td = seed_training(db_session, "x answer", source_id=raw.id)
    dup = seed_training(db_session, "x answer", is_answer_duplicate=True, duplicate_answer_of_id=td.id)
    standalone = seed_training(db_session, "standalone")

    raw_id, td_id, dup_id, standalone_id = raw.id, td.id, dup.id, standalone.id

    assert client.delete(f"/training-data/{td_id}", auth=AUTH).status_code == 200
    db_session.expunge_all()
    assert db_session.get(TrainingData, td_id) is None
    assert db_session.get(RawData, raw_id).admin_approved == 0
    assert db_session.get(TrainingData, dup_id).duplicate_answer_of_id is None

    assert client.delete(f"/training-data/{standalone_id}", auth=AUTH).status_code == 200
    assert client.delete(f"/training-data/{td_id}", auth=AUTH).status_code == 404

    # The raw entry can be approved again afterwards.
    assert client.post(f"/approve/{raw_id}", auth=AUTH).status_code == 200


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_similarity(monkeypatch):
    """Deterministic similarity: texts are 'group:tag'; same group -> score."""
    scores = {}

    def sim(a, b):
        return scores.get(frozenset((a, b)), 1.0 if a == b else 0.0)

    monkeypatch.setattr(database_models, "calculate_cosine_similarity", sim)
    return scores


def test_find_question_duplicates_flattens_groups(fake_similarity):
    a, b, c, d = ("question A long enough", "question B long enough",
                  "question C long enough", "question D long enough")
    fake_similarity[frozenset((a, b))] = 0.9
    fake_similarity[frozenset((b, c))] = 0.6   # C only matches B ...
    fake_similarity[frozenset((a, d))] = 0.3   # below the 0.45 threshold
    result = api.find_question_duplicates([(3, c), (1, a), (2, b), (4, d)])
    # ... and is linked to B's original A: no chains, no cycles.
    assert result == {2: (1, 0.9), 3: (1, 0.6)}


def test_find_question_duplicates_oldest_match_wins(fake_similarity):
    a, b, c = "first question text", "second question text", "third question text"
    fake_similarity[frozenset((a, c))] = 0.5
    fake_similarity[frozenset((b, c))] = 0.99
    assert api.find_question_duplicates([(1, a), (2, b), (3, c)]) == {3: (1, 0.5)}


def test_find_question_duplicates_window_and_short_texts(fake_similarity):
    first, last = "the very first question", "the very last question"
    fake_similarity[frozenset((first, last))] = 1.0
    filler = [(i, f"unrelated filler question {i}") for i in range(2, 2 + api.QUESTION_WINDOW)]
    rows = [(1, first)] + filler + [(2 + api.QUESTION_WINDOW, last)]
    # 'first' is outside the window of the last question (same as the bot).
    assert api.find_question_duplicates(rows) == {}
    short = "hi there"  # <= 10 chars: never marked, but may be an original
    fake_similarity[frozenset((short, last))] = 1.0
    assert api.find_question_duplicates([(1, short), (2, short), (3, last)]) == {3: (1, 1.0)}
    assert api.find_question_duplicates([]) == {}


def test_find_answer_duplicates_most_similar_older(fake_similarity):
    a, b, c = "answer text number one", "answer text number two", "answer text number three"
    fake_similarity[frozenset((a, c))] = 0.86
    fake_similarity[frozenset((b, c))] = 0.95
    fake_similarity[frozenset((a, b))] = 0.5
    assert api.find_answer_duplicates([(1, a), (2, b), (3, c)]) == {3: (2, 0.95)}
    fake_similarity[frozenset((b, c))] = 0.86  # tie -> oldest
    assert api.find_answer_duplicates([(1, a), (2, b), (3, c)]) == {3: (1, 0.86)}


def test_detect_duplicates_endpoint(client, db_session, monkeypatch):
    q = "How can I apply for the Erasmus exchange program?"
    a = seed_raw(db_session, question=q)
    b = seed_raw(db_session, question="Where is the computer engineering building located?",
                 created_at=BASE_TIME + timedelta(minutes=1))
    c = seed_raw(db_session, question=q + "?", created_at=BASE_TIME + timedelta(minutes=2))
    short = seed_raw(db_session, question="hi", created_at=BASE_TIME + timedelta(minutes=3))
    # Stale / cyclic markings from the old implementation must be cleared.
    a.is_duplicate, a.duplicate_of_id = True, c.id
    b.is_duplicate, b.duplicate_of_id = True, a.id
    db_session.commit()

    ans = "Applications are submitted through the international office website every spring."
    t1 = seed_training(db_session, ans)
    t2 = seed_training(db_session, "The cafeteria is open from eight in the morning until five.")
    t3 = seed_training(db_session, ans)

    commits = []
    real_commit = Session.commit

    def counting_commit(self):
        commits.append(1)
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", counting_commit)
    resp = client.post("/detect-duplicates", auth=AUTH)
    monkeypatch.setattr(Session, "commit", real_commit)
    assert resp.status_code == 200, resp.text
    assert len(commits) == 1  # one transaction, no per-row commits
    body = resp.json()
    assert body["success"] is True
    assert (body["question_duplicates"], body["answer_duplicates"]) == (1, 1)
    assert set(body) == {"success", "message", "question_duplicates", "answer_duplicates"}

    db_session.expire_all()
    marks = {r.id: (bool(r.is_duplicate), r.duplicate_of_id) for r in db_session.query(RawData)}
    assert marks == {a.id: (False, None), b.id: (False, None), c.id: (True, a.id), short.id: (False, None)}
    tmarks = {t.id: (bool(t.is_answer_duplicate), t.duplicate_answer_of_id) for t in db_session.query(TrainingData)}
    assert tmarks == {t1.id: (False, None), t2.id: (False, None), t3.id: (True, t1.id)}

    # Idempotent.
    assert client.post("/detect-duplicates", auth=AUTH).json()["question_duplicates"] == 1


def test_detect_duplicates_is_atomic(client, db_session, monkeypatch):
    a, b = seed_raw(db_session, 2)
    b.is_duplicate, b.duplicate_of_id = True, a.id
    db_session.commit()

    def boom(*_a, **_k):
        raise RuntimeError("fail mid-way")

    monkeypatch.setattr(api, "find_answer_duplicates", boom)
    resp = client.post("/detect-duplicates", auth=AUTH)
    assert resp.status_code == 500 and "fail mid-way" not in resp.text
    db_session.expire_all()
    assert db_session.get(RawData, b.id).duplicate_of_id == a.id


def test_detect_duplicates_scales(client, db_session):
    topics = ["erasmus", "library", "cafeteria", "internship", "graduation", "dormitory"]
    rows = [
        RawData(telegram_id=i, question=f"Question about {topics[i % 6]} rules number {i} please help",
                answer="a", created_at=BASE_TIME + timedelta(seconds=i))
        for i in range(600)
    ]
    db_session.add_all(rows)
    db_session.commit()
    import time
    start = time.monotonic()
    resp = client.post("/detect-duplicates", auth=AUTH)
    assert resp.status_code == 200
    assert time.monotonic() - start < 20
    db_session.expire_all()
    # No duplicate points to another duplicate.
    dup_targets = {r.duplicate_of_id for r in db_session.query(RawData).filter(RawData.is_duplicate.is_(True))}
    assert not any(db_session.get(RawData, t).is_duplicate for t in dup_targets)


# --------------------------------------------------------------------------- #
# /health and /
# --------------------------------------------------------------------------- #


def test_root_public(client):
    assert client.get("/").json() == {"message": "University Bot Admin API"}


def test_health_public_view_leaks_nothing(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy" and body["database"] == {"healthy": True}
    assert set(body) == {"status", "timestamp", "database", "model"}
    # Wrong / malformed credentials still get the public view (no 401).
    assert set(client.get("/health", auth=("admin", "bad")).json()) == set(body)
    assert client.get("/health", headers={"Authorization": "Basic %%%"}).status_code == 200


def test_health_authenticated_view(client, monkeypatch):
    monkeypatch.setattr(api, "_gpu_info", lambda: {"available": True, "device_count": 1,
                                                   "current_device": 0, "device_name": "Fake GPU"})
    body = client.get("/health", auth=AUTH).json()
    assert set(body["system"]) == {"cpu_usage_percent", "memory_usage_percent", "memory_available_gb",
                                   "disk_usage_percent", "disk_free_gb"}
    assert body["gpu"]["device_name"] == "Fake GPU"


def test_health_db_down_returns_503(client, monkeypatch):
    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "SessionLocal", broken)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy" and "db down" not in resp.text


def test_module_does_not_import_torch():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "torch" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch"


def test_gpu_info_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(api, "_gpu_cache", {"ts": 0.0, "value": None})
    monkeypatch.setattr(api.shutil, "which", lambda _name: None)
    assert REAL_GPU_INFO() == {"available": False}


def test_gpu_info_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(api, "_gpu_cache", {"ts": 0.0, "value": None})
    monkeypatch.setattr(api.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    class Done:
        stdout = "0, NVIDIA GeForce RTX 4090, 1234, 24564, 7\n"

    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: Done())
    info = REAL_GPU_INFO()
    assert info["available"] is True and info["device_count"] == 1
    assert info["device_name"] == "NVIDIA GeForce RTX 4090"
    assert info["devices"][0]["memory_total_mb"] == 24564.0


# --------------------------------------------------------------------------- #
# /docs/list and /docs/read
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "scripts").mkdir()
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / "README.md").write_text("# Project readme\n")
    (root / "docs" / "API.md").write_text("api docs")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=supersecret\n")
    (root / "university_bot.db").write_bytes(b"SQLite format 3\x00")
    (root / "logs" / "small.log").write_text("line1\nline2\n")
    (root / "logs" / "notes.txt").write_text("not a log")
    (root / "scripts" / "start.sh").write_text("#!/bin/bash\necho hi\n")
    (root / "scripts" / "helper.py").write_text("print('x')")
    (root / "node_modules" / "pkg" / "README.md").write_text("npm junk")
    (root / ".venv" / "lib" / "README.md").write_text("venv junk")
    outside = tmp_path / "outside_secret.md"
    outside.write_text("outside secret")
    (root / "docs" / "link.md").symlink_to(outside)
    # 64 MiB sparse log whose last line is known.
    big = root / "logs" / "worker.log"
    with open(big, "wb") as f:
        f.truncate(64 * 1024 * 1024)
        f.seek(64 * 1024 * 1024)
        f.write(b"\nLAST LINE OF LOG\n")
    big_md = root / "docs" / "BIG.md"
    big_md.write_text("START\n" + "x" * (2 * 1024 * 1024))
    monkeypatch.setattr(api, "PROJECT_ROOT", str(root))
    return root


def test_docs_list(client, fake_project):
    resp = client.get("/docs/list", auth=AUTH)
    assert resp.status_code == 200
    by_path = {d["path"]: d for d in resp.json()}
    assert set(by_path) == {"README.md", "docs/API.md", "docs/BIG.md",
                            "logs/small.log", "logs/worker.log", "scripts/start.sh"}
    assert by_path["logs/worker.log"]["type"] == "log"
    assert by_path["scripts/start.sh"]["type"] == "script"
    assert by_path["README.md"] == {"path": "README.md", "name": "README.md",
                                    "type": "documentation", "size": len("# Project readme\n")}


def test_docs_read_ok(client, fake_project):
    resp = client.get("/docs/read", params={"path": "README.md"}, auth=AUTH)
    assert resp.status_code == 200 and resp.text == "# Project readme\n"
    assert resp.headers["x-content-truncated"] == "false"
    assert client.get("/docs/read", params={"path": "scripts/start.sh"}, auth=AUTH).status_code == 200


@pytest.mark.parametrize("path,code", [
    (".env", 403),
    ("university_bot.db", 403),
    ("logs/notes.txt", 403),
    ("scripts/helper.py", 403),
    ("node_modules/pkg/README.md", 403),
    (".venv/lib/README.md", 403),
    ("docs", 403),
    ("../outside_secret.md", 400),
    ("docs/../../outside_secret.md", 400),
    ("/etc/passwd", 400),
    ("docs/link.md", 400),          # symlink escaping the project root
    ("docs/missing.md", 404),
    ("bad\x00name.md", 400),
    ("", 422),
])
def test_docs_read_rejections(client, fake_project, path, code):
    resp = client.get("/docs/read", params={"path": path}, auth=AUTH)
    assert resp.status_code == code, (path, resp.text)
    assert "supersecret" not in resp.text and "outside secret" not in resp.text


def test_docs_read_huge_log_is_tailed(client, fake_project):
    resp = client.get("/docs/read", params={"path": "logs/worker.log"}, auth=AUTH)
    assert resp.status_code == 200
    assert resp.headers["x-content-truncated"] == "true"
    assert len(resp.content) <= api.DOCS_MAX_BYTES + 200
    assert resp.text.rstrip().endswith("LAST LINE OF LOG")
    assert resp.text.startswith("[... file is 64.0 MiB")


def test_docs_read_big_doc_is_head_truncated(client, fake_project):
    resp = client.get("/docs/read", params={"path": "docs/BIG.md"}, auth=AUTH)
    assert resp.status_code == 200
    assert resp.text.startswith("START\n")
    assert len(resp.content) <= api.DOCS_MAX_BYTES + 200
    assert "showing first" in resp.text[-200:]


def test_docs_endpoints_use_project_root_not_hardcoded_path():
    assert "/home/ceng/cu_ceng_bot" not in MODULE_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# __main__ uses config.api_host / api_port, reload off by default
# --------------------------------------------------------------------------- #


def _run_main(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    runpy.run_path(str(MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    return calls[0]


def test_main_uses_config_without_reload(monkeypatch):
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8123")
    monkeypatch.delenv("API_RELOAD", raising=False)
    args, kwargs = _run_main(monkeypatch)
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8123
    assert not kwargs.get("reload")
    assert not isinstance(args[0], str)  # the app object itself


def test_main_reload_uses_import_string(monkeypatch):
    monkeypatch.setenv("API_RELOAD", "true")
    args, kwargs = _run_main(monkeypatch)
    assert args[0] == "admin_rest_api:app" and kwargs["reload"] is True
    assert os.path.isdir(kwargs["app_dir"])


# --- worker "[auto]" answers must not be approved as-is (lead addition) ------
def test_auto_answer_cannot_be_approved_until_edited(client, db_session):
    row = seed_raw(
        db_session,
        answer="Üzgünüz, şu anda cevap veremiyoruz.",
        admin_notes="[auto] worker: model failed 3 times",
    )
    r = client.post(f"/approve/{row.id}", auth=AUTH)
    assert r.status_code == 400
    assert "automatic" in r.json()["detail"]
    assert db_session.query(TrainingData).count() == 0

    r = client.put(f"/raw-data/{row.id}", json={"answer": "Staj zorunludur, iki staj yapılır."}, auth=AUTH)
    assert r.status_code == 200
    db_session.expire_all()
    assert "[auto]" not in db_session.get(RawData, row.id).admin_notes

    r = client.post(f"/approve/{row.id}", auth=AUTH)
    assert r.status_code == 200, r.text
    assert db_session.query(TrainingData).count() == 1


def test_auto_marker_appended_to_existing_notes_is_detected(client, db_session):
    row = seed_raw(db_session, admin_notes="checked by admin\n[auto] prompt_guard: injection")
    r = client.post(f"/approve/{row.id}", auth=AUTH)
    assert r.status_code == 400
