# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Transport retries back off in real time; tests that fail the transport on purpose should not wait."""
    from kbmine import pulse

    monkeypatch.setattr(pulse, "_RETRY_BACKOFF_SEC", 0.0)
