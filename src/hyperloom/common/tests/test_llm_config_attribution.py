# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the untagged-call warning in ``llm_config``.

An empty ``component`` is how a call site that nobody instrumented looks from
inside the tagging helpers, and for a long time it was skipped in silence. The
spend still reached the gateway, it simply arrived naming no producer, and
nothing in the logs pointed at the code that made it. These tests pin that the
skip is now audible, that it names the caller rather than the helper it passed
through, and that it stays quiet for a deployment that emits no attribution.
"""

from __future__ import annotations

import logging

import pytest

from hyperloom.common import llm_config
from hyperloom.common.llm_attribution import ATTRIBUTION_ENV, CLAW_SESSION_ID_ENV


@pytest.fixture(autouse=True)
def _forget_warned_sites() -> None:
    """The dedupe set is module state; a leak would silence the next test."""
    llm_config._UNTAGGED_SITES.clear()


@pytest.fixture
def attributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select a gateway, so an untagged call is a defect worth reporting."""
    monkeypatch.setenv(ATTRIBUTION_ENV, "litellm")
    monkeypatch.setenv(CLAW_SESSION_ID_ENV, "claw-abc")


class TestNamedComponent:
    """A call site that names itself is tagged and says nothing."""

    def test_headers_are_rendered(self, attributed: None) -> None:
        headers = llm_config._headers_for("critic", "review_commit")
        assert "component=critic" in headers["x-litellm-tags"]

    def test_nothing_is_logged(self, attributed: None, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=llm_config.log.name):
            llm_config._headers_for("critic", "review_commit")
        assert caplog.records == []


class TestUnnamedComponent:
    """The silent skip that let uninstrumented call sites reach production."""

    def test_the_skip_is_reported(self, attributed: None, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=llm_config.log.name):
            assert llm_config._headers_for("", "") == {}
        assert len(caplog.records) == 1
        assert "names no attribution component" in caplog.records[0].getMessage()

    def test_the_report_names_the_caller_not_the_helper(
        self,
        attributed: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Routed through _tag_request so there is a helper frame to skip past.
        with caplog.at_level(logging.WARNING, logger=llm_config.log.name):
            llm_config._tag_request({}, "")
        assert "llm_config.py" not in caplog.records[0].getMessage()
        assert __file__.rsplit("/", 1)[-1] in caplog.records[0].getMessage()

    def test_one_call_site_is_reported_once(
        self,
        attributed: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=llm_config.log.name):
            for _ in range(5):
                llm_config._headers_for("", "")
        assert len(caplog.records) == 1

    def test_an_unattributed_deployment_stays_quiet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv(ATTRIBUTION_ENV, raising=False)
        with caplog.at_level(logging.WARNING, logger=llm_config.log.name):
            assert llm_config._headers_for("", "") == {}
        assert caplog.records == []
