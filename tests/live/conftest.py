"""Registers the ``live`` marker for the live integration tests in this directory.

``live`` tests talk to the real RabbitMQ broker (unique, throw-away queues), start
the real worker (GPU + local model weights) and a real uvicorn server, all on a
copy of the database. They are skipped unless ``RUN_LIVE_TESTS=1``.
"""

import os

import pytest

LIVE_ENV = "RUN_LIVE_TESTS"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: real RabbitMQ / GPU worker / uvicorn integration test; skipped unless RUN_LIVE_TESTS=1",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get(LIVE_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"live integration test: set {LIVE_ENV}=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
