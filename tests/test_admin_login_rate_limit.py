"""Failed-login rate limiting of the admin API (src/admin_rest_api.py).

No database access is needed: wrong passwords never reach a route body.
"""

import logging
import threading

import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import admin_rest_api as api

GOOD = ("admin", "test-admin-password")
BAD = ("admin", "wrong-password")
LIMIT_VARS = ("ADMIN_LOGIN_MAX_FAILURES", "ADMIN_LOGIN_WINDOW_SECONDS", "ADMIN_LOGIN_LOCKOUT_SECONDS")


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", "")
    monkeypatch.setattr(api, "_gpu_info", lambda: {"available": False})
    for name in LIMIT_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(monkeypatch, clock):
    lim = api.LoginRateLimiter(max_failures=3, window_seconds=60, lockout_seconds=300, clock=clock)
    monkeypatch.setattr(api, "login_rate_limiter", lim)
    monkeypatch.setattr(api, "health_credentials_limiter",
                        api.LoginRateLimiter(max_failures=3, window_seconds=60, lockout_seconds=300,
                                             clock=clock))
    return lim


def client_for(host: str) -> TestClient:
    return TestClient(api.app, client=(host, 40000))


def login(client, auth):
    return client.post("/auth/login", auth=auth)


def test_lockout_after_max_failures_with_retry_after(limiter, clock):
    c = client_for("203.0.113.5")
    assert [login(c, BAD).status_code for _ in range(2)] == [401, 401]
    third = login(c, BAD)
    assert third.status_code == 429
    assert third.headers["Retry-After"] == "300"
    assert "www-authenticate" not in third.headers  # no browser login pop-up
    assert "Too many failed login attempts" in third.json()["detail"]

    clock.now += 100
    blocked = login(c, GOOD)  # even the right password is refused while locked out
    assert blocked.status_code == 429 and blocked.headers["Retry-After"] == "200"
    assert c.get("/stats", auth=GOOD).status_code == 429  # every protected route

    clock.now += 200
    assert login(c, GOOD).status_code == 200


def test_failures_outside_the_window_are_forgotten(limiter, clock):
    c = client_for("203.0.113.6")
    login(c, BAD)
    login(c, BAD)
    clock.now += 61
    assert login(c, BAD).status_code == 401  # only 1 failure in the last 60 s
    assert login(c, BAD).status_code == 401
    assert login(c, BAD).status_code == 429


def test_clients_are_independent_and_success_resets_only_itself(limiter):
    a, b = client_for("198.51.100.1"), client_for("198.51.100.2")
    for c in (a, b):
        login(c, BAD)
        login(c, BAD)
    assert login(a, GOOD).status_code == 200  # resets a only
    assert login(a, BAD).status_code == 401
    assert login(a, BAD).status_code == 401
    assert login(b, BAD).status_code == 429  # b still had its 2 failures
    assert login(a, GOOD).status_code == 200  # a is not affected by b's lockout


def test_spoofed_forwarded_for_does_not_help(limiter):
    """The header is only honoured by uvicorn for FORWARDED_ALLOW_IPS peers;
    the app itself never reads it."""
    c = client_for("203.0.113.7")
    codes = [c.post("/auth/login", auth=BAD, headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
             for i in range(4)]
    assert codes == [401, 401, 429, 429]


def test_ipv6_clients_are_grouped_per_64(limiter):
    for i in range(3):
        status = login(client_for(f"2001:db8:1:2::{i + 1}"), BAD).status_code
    assert status == 429
    assert login(client_for("2001:db8:1:2:ffff::9"), GOOD).status_code == 429
    assert login(client_for("2001:db8:1:3::1"), GOOD).status_code == 200


def test_each_failed_request_counts_once(limiter):
    """/approve declares verify_credentials on the route and the router; the
    dependency runs once per request, so one wrong request is one failure."""
    c = client_for("203.0.113.13")
    assert c.post("/approve/1", auth=BAD).status_code == 401
    assert c.post("/approve/1", auth=BAD).status_code == 401
    assert c.post("/approve/1", auth=BAD).status_code == 429


def test_missing_credentials_do_not_count(limiter):
    c = client_for("203.0.113.8")
    for _ in range(5):
        assert c.post("/auth/login").status_code == 401
    assert login(c, GOOD).status_code == 200


def test_health_credentials_have_one_limit_for_all_clients(limiter, clock):
    """/health with credentials would otherwise be an unlimited password oracle;
    its failures are counted for all clients together, so changing the address
    does not help, and they never lock anybody out of the API."""
    for i in range(3):
        body = client_for(f"203.0.113.{20 + i}").get("/health", auth=BAD).json()
        assert "system" not in body
    for host in ("203.0.113.20", "203.0.113.99"):
        resp = client_for(host).get("/health", auth=GOOD)
        assert resp.status_code == 200  # the public health check keeps working...
        assert "system" not in resp.json()  # ...but the admin view is locked for everyone
    assert login(client_for("203.0.113.20"), GOOD).status_code == 200  # the API is not
    assert api.login_rate_limiter.tracked_clients == 0
    clock.now += 300
    assert "system" in client_for("203.0.113.20").get("/health", auth=GOOD).json()


def test_api_lockout_does_not_hide_health_and_vice_versa(limiter):
    c = client_for("203.0.113.30")
    for _ in range(3):
        login(c, BAD)
    assert login(c, GOOD).status_code == 429
    assert "system" in c.get("/health", auth=GOOD).json()


def proxied_client() -> TestClient:
    """The app as uvicorn.run() serves it (proxy headers trusted from 127.0.0.1),
    reached through a local reverse proxy."""
    app = ProxyHeadersMiddleware(api.app, trusted_hosts="127.0.0.1,::1")
    return TestClient(app, client=("127.0.0.1", 40000))


def test_forged_forwarded_for_on_health_is_harmless(limiter):
    """Regression: a proxy route that passes the client's own X-Forwarded-For
    (nginx 'location /health' without proxy_set_header) made the limiter key
    attacker-chosen: unlimited guessing by rotating the header, and forged
    failures 'from' the admin's address locked the admin out of the API."""
    c = proxied_client()
    for i in range(20):  # rotating addresses do not reset the /health limit
        c.get("/health", auth=("admin", f"guess{i}"), headers={"X-Forwarded-For": f"10.9.{i}.1"})
    right = c.get("/health", auth=GOOD, headers={"X-Forwarded-For": "10.9.99.1"})
    assert "system" not in right.json()  # the oracle is closed

    admin_ip = "198.51.100.7"
    for _ in range(10):
        c.get("/health", auth=BAD, headers={"X-Forwarded-For": admin_ip})
    # the admin's real API request ('X-Forwarded-For: <client>' appended by nginx)
    assert c.post("/auth/login", auth=GOOD, headers={"X-Forwarded-For": admin_ip}).status_code == 200
    assert api.login_rate_limiter.tracked_clients == 0


def test_non_ip_client_hosts_share_one_key(limiter):
    """A forged, non-address X-Forwarded-For cannot create keys (bounded memory)."""
    c = proxied_client()
    codes = [c.post("/auth/login", auth=BAD, headers={"X-Forwarded-For": f"garbage-{i}"}).status_code
             for i in range(4)]
    assert codes == [401, 401, 429, 429]
    assert api.login_rate_limiter.tracked_clients == 1
    assert api.login_rate_limiter.retry_after(api.INVALID_CLIENT_KEY) > 0
    # real addresses behind the same proxy are not affected
    assert c.post("/auth/login", auth=GOOD, headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200


def test_lockout_is_logged_once(limiter, caplog):
    c = client_for("203.0.113.11")
    with caplog.at_level(logging.WARNING, logger="admin_rest_api"):
        for _ in range(6):
            login(c, BAD)
    messages = [r.getMessage() for r in caplog.records if "failed attempts" in r.getMessage()]
    assert len(messages) == 1 and "203.0.113.11" in messages[0]


def test_memory_is_bounded(clock):
    lim = api.LoginRateLimiter(max_failures=3, window_seconds=60, lockout_seconds=60,
                               max_clients=5, clock=clock)
    for i in range(50):
        lim.record_failure(f"10.0.0.{i}")
    assert lim.tracked_clients == 5
    lim.record_failure("10.0.0.49")
    lim.record_failure("10.0.0.49")
    assert lim.retry_after("10.0.0.49") == 60  # most recent clients are the ones kept


def test_disabled_limiter_never_blocks(monkeypatch):
    monkeypatch.setenv("ADMIN_LOGIN_MAX_FAILURES", "0")
    lim = api.LoginRateLimiter.from_env()
    monkeypatch.setattr(api, "login_rate_limiter", lim)
    c = client_for("203.0.113.12")
    assert {login(c, BAD).status_code for _ in range(20)} == {401}
    assert lim.tracked_clients == 0


def test_env_configuration(monkeypatch, caplog):
    monkeypatch.setenv("ADMIN_LOGIN_MAX_FAILURES", "4")
    monkeypatch.setenv("ADMIN_LOGIN_WINDOW_SECONDS", "30")
    monkeypatch.setenv("ADMIN_LOGIN_LOCKOUT_SECONDS", "120.5")
    lim = api.LoginRateLimiter.from_env()
    assert (lim.max_failures, lim.window_seconds, lim.lockout_seconds) == (4, 30.0, 120.5)

    monkeypatch.setenv("ADMIN_LOGIN_MAX_FAILURES", "lots")
    monkeypatch.setenv("ADMIN_LOGIN_WINDOW_SECONDS", "-1")
    monkeypatch.setenv("ADMIN_LOGIN_LOCKOUT_SECONDS", "nan")
    with caplog.at_level(logging.WARNING, logger="admin_rest_api"):
        lim = api.LoginRateLimiter.from_env()
    assert (lim.max_failures, lim.window_seconds, lim.lockout_seconds) == (
        api.DEFAULT_LOGIN_MAX_FAILURES, api.DEFAULT_LOGIN_WINDOW_SECONDS, api.DEFAULT_LOGIN_LOCKOUT_SECONDS)
    assert all(name in caplog.text for name in LIMIT_VARS)


def test_defaults():
    lim = api.LoginRateLimiter.from_env()
    assert (lim.max_failures, lim.window_seconds, lim.lockout_seconds) == (10, 900.0, 900.0)
    assert lim.enabled


@pytest.mark.parametrize("host,key", [
    ("192.0.2.1", "192.0.2.1"),
    ("::ffff:192.0.2.1", "192.0.2.1"),
    ("2001:db8::1", "2001:db8::/64"),
    ("testclient", "invalid"),
    ("1.2.3.4, 5.6.7.8", "invalid"),
    ("<script>", "invalid"),
])
def test_client_key(host, key):
    class Req:
        client = type("C", (), {"host": host})()

    assert api.client_key(Req()) == key


def test_concurrent_failures_are_counted_exactly(clock):
    lim = api.LoginRateLimiter(max_failures=1000, window_seconds=60, lockout_seconds=60, clock=clock)

    def hammer():
        for _ in range(100):
            lim.record_failure("10.1.1.1")

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert lim.retry_after("10.1.1.1") == 0  # 800 < 1000
    for _ in range(199):
        assert lim.record_failure("10.1.1.1") == 0
    assert lim.record_failure("10.1.1.1") == 60  # exactly the 1000th
