# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pre-rename opt-out variable must not fail silently.

``FORGE_DISABLE_COMPILED_FELLOWS`` was renamed to
``FORGE_DISABLE_COMPILED_KERNEL_BACKENDS``. Deleting the old name outright is not
enough: ``FORGE_`` is on env_safety's dotenv prefix allowlist, so an operator's
stale value is still forwarded into the run and then ignored -- and the thing it
used to switch off (the compiled kernel backends) comes back on with no signal.
That is the failure mode this module pins.
"""

from __future__ import annotations

import logging

import pytest

from hyperloom.agents.kernel.tools.backends import forge_submit


@pytest.fixture(autouse=True)
def _reset_warn_latch():
    """The warning latches once per process; tests need it un-fired."""
    forge_submit._retired_opt_out_warned = False
    yield
    forge_submit._retired_opt_out_warned = False


def test_retired_name_warns_and_is_not_honoured(monkeypatch, caplog):
    monkeypatch.delenv("FORGE_DISABLE_COMPILED_KERNEL_BACKENDS", raising=False)
    monkeypatch.setenv("FORGE_DISABLE_COMPILED_FELLOWS", "1")

    with caplog.at_level(logging.WARNING, logger=forge_submit.__name__):
        resolved = forge_submit._kernel_backend_for_source_type("ck")

    # Not honoured: the compiled mapping still resolves.
    assert resolved == "ck"
    assert "FORGE_DISABLE_COMPILED_FELLOWS" in caplog.text
    assert "FORGE_DISABLE_COMPILED_KERNEL_BACKENDS" in caplog.text


def test_the_warning_fires_once_not_per_kernel(monkeypatch, caplog):
    monkeypatch.setenv("FORGE_DISABLE_COMPILED_FELLOWS", "1")

    with caplog.at_level(logging.WARNING, logger=forge_submit.__name__):
        for _ in range(5):
            forge_submit._kernel_backend_for_source_type("ck")

    assert caplog.text.count("FORGE_DISABLE_COMPILED_FELLOWS is set") == 1


def test_new_name_still_disables_compiled_backends(monkeypatch):
    monkeypatch.delenv("FORGE_DISABLE_COMPILED_FELLOWS", raising=False)
    monkeypatch.setenv("FORGE_DISABLE_COMPILED_KERNEL_BACKENDS", "1")

    assert forge_submit._kernel_backend_for_source_type("ck") is None
    # Triton is not a compiled backend, so the opt-out must not touch it.
    assert forge_submit._kernel_backend_for_source_type("triton") == "triton"


def test_no_warning_when_the_retired_name_is_unset(monkeypatch, caplog):
    monkeypatch.delenv("FORGE_DISABLE_COMPILED_FELLOWS", raising=False)

    with caplog.at_level(logging.WARNING, logger=forge_submit.__name__):
        forge_submit._kernel_backend_for_source_type("ck")

    assert "FORGE_DISABLE_COMPILED_FELLOWS" not in caplog.text
