# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every create of a runner sandbox carries the source commit and the runner cache manifest pin.

The runner's start step verifies its cache entry against ``HYPERLOOM_SOURCE_SHA`` and
``HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256`` and fails in seconds without them, so a create that dropped them on a retry
would look like a broken cache. This drives the pool the hosted toolbox builds, over the real E2B provider, through
every way a test's sandbox is created again and reads the environment off each request the platform received.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure import sandboxes
from hyperloom.measure.runner.cache_layout import MANIFEST_PIN_ENV
from hyperloom.measure.sandboxes import AttemptTag
from hyperloom.sandbox.socket import IDEMPOTENCY_KEY, SandboxCreateFailed
from hyperloom.sandbox.tests.test_sandbox_e2b import (
    CREATE_FAILED_BODY,
    GATEWAY_CUTS,
    KEY_SPENT_BODY,
    SANDBOX,
    _listed,
    _Service,
    service,  # noqa: F401 (the canned control plane the creates go to)
)
from hyperloom_multibranch.toolbox.hosted_toolbox import SOURCE_SHA_ENV
from hyperloom_multibranch.toolbox.tests.test_hosted_toolbox import _MANIFEST, _SOURCE_SHA, _toolbox

_QUOTA_FULL = (429, {"code": 429, "message": "grant_quota_exceeded: 4 of 4 running"})


def test_every_create_of_a_runner_sandbox_carries_the_source_and_the_manifest_pin(
    service: _Service,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandboxes, "_QUOTA_RETRY_SEC", 0.05)
    service.reply(
        "POST",
        "/v2/sandboxes",
        GATEWAY_CUTS[0],  # attempt 1: the gateway cuts the create, which the provider replays under the same key
        (500, CREATE_FAILED_BODY),  # and the platform's sandbox failed before it ran: the attempt ends
        (410, KEY_SPENT_BODY),  # attempt 2, the toolbox's one retry: its key was spent before, so the pool takes :r1
        _QUOTA_FULL,  # the grant is full: the pool waits for room and creates again under :r1
        (201, SANDBOX),
    )
    service.reply("GET", "/v2/sandboxes", (200, []))
    service.reply("GET", "/sandboxes/sbx-1", (200, _listed("sbx-1", {})))
    toolbox = _toolbox(
        None, tmp_path, api_url=service.url, sandbox_url=service.url, runner_cache_manifest_sha256=_MANIFEST
    )
    pool = toolbox.backend.pool
    with pytest.raises(SandboxCreateFailed):
        pool.acquire(1, AttemptTag("req-1", 1), timeout_sec=100)
    assert pool.acquire(1, AttemptTag("req-1", 2), timeout_sec=100).sandbox_id == "sbx-1"

    creates = service.calls("POST", "/v2/sandboxes")
    keys = [call.body["metadata"][IDEMPOTENCY_KEY] for call in creates]
    assert keys == ["run-1:req-1:1", "run-1:req-1:1", "run-1:req-1:2", "run-1:req-1:2:r1", "run-1:req-1:2:r1"]
    pinned = {SOURCE_SHA_ENV: _SOURCE_SHA, MANIFEST_PIN_ENV: _MANIFEST}
    assert SOURCE_SHA_ENV == "HYPERLOOM_SOURCE_SHA", "the name the runner's start step reads"
    assert [call.body["envVars"] for call in creates] == [pinned] * len(creates)
