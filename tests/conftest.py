# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in test groups.

``long_context`` tests (``tests/test_long_context.py``) run single calls of more than a million
tokens on one H100 and need tens of GB of GPU memory; the default suite skips them. Run them with
``pytest tests/test_long_context.py --run-long-context``.

``slow`` tests are extra cases of GPU kernel tests that take minutes to compile and run (for example
the whole-op cases of ``tests/test_vector_gate_bwd_schedule.py``); the default suite skips them. Run
them with ``--run-slow``.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-long-context", action="store_true", default=False,
                     help="run the long_context tests (1M+ token single calls, tens of GB of GPU memory)")
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run the slow tests (extra GPU kernel cases that take minutes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "long_context: 1M+ token single-call tests; opt in with --run-long-context")
    config.addinivalue_line("markers", "slow: extra GPU kernel cases that take minutes; opt in with --run-slow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-long-context"):
        skip = pytest.mark.skip(reason="long_context test: pass --run-long-context to run it")
        for item in items:
            if "long_context" in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="slow test: pass --run-slow to run it")
        for item in items:
            if item.get_closest_marker("slow") is not None:
                item.add_marker(skip)
