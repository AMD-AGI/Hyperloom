# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a leg freezes in a session on shared storage is the group's to read, as the next leg may be another account of
the group: ``state.json`` and ``base_launch.json`` are ``664``, and owner-only when they would hold a credential."""

from __future__ import annotations

import stat
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.env_safety import holds_credentials
from hyperloom.common.session_files import SERVER_ARGS_ENV
from hyperloom.inference_optimizer.session.base_launch import base_launch_path, write_base_launch
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import freeze_base
from hyperloom.orchestrator.state.shared_state import SharedState

TOKEN = "hf_" + "a" * 34


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A session directory on shared storage, ``2775``, and no pins exported by this run."""

    monkeypatch.delenv("INFERENCE_OPTIMIZER_EXTRA_ENV", raising=False)
    monkeypatch.delenv(SERVER_ARGS_ENV, raising=False)
    tmp_path.chmod(0o2775)
    directory = tmp_path / "session"
    directory.mkdir()
    directory.chmod(0o2775)
    return directory


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_state_and_base_a_session_freezes_are_the_groups_to_read(session: Path) -> None:
    freeze_base(session, base_launch())

    assert (_mode(SharedState.state_path(session)), _mode(base_launch_path(session))) == (0o664, 0o664)


@pytest.mark.parametrize(
    "pins",
    [{"operator_extra_env": {"HF_TOKEN": TOKEN}}, {"operator_server_args": f"--api-key {TOKEN}"}],
    ids=["extra-env", "server-args"],
)
def test_a_state_whose_operator_pins_hold_a_credential_stays_owner_only(session: Path, pins: dict) -> None:
    state = SharedState()
    for name, value in pins.items():
        setattr(state, name, value)

    state.save(session)

    assert _mode(SharedState.state_path(session)) == 0o600


def test_a_state_of_a_run_that_exports_a_credential_pin_stays_owner_only(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SERVER_ARGS_ENV, f"--api-key {TOKEN}")

    SharedState().save(session)

    assert _mode(SharedState.state_path(session)) == 0o600


def test_a_base_launch_that_holds_a_credential_stays_owner_only(session: Path) -> None:
    base = base_launch()

    write_base_launch(session, replace(base, args=(*base.args, "--api-key", TOKEN)))

    assert _mode(base_launch_path(session)) == 0o600


@pytest.mark.parametrize(
    ("document", "credential"),
    [
        ({"envs": {"HF_TOKEN": TOKEN}}, True),
        ({"extra_envs": {"HF_TOKEN": "x"}}, True),
        ({"args": ["--hf-token", "x"]}, True),
        ({"url": "https://user:secret@example.com/repo.git"}, True),
        ({"authored_framework_levers": [{"note": "authored"}]}, False),
        ({"envs": {"MAX_NUM_BATCHED_TOKENS": "8192"}, "args": ["--tp-size", "8"]}, False),
        ({"HF_TOKEN": ""}, False),
    ],
)
def test_holds_credentials_judges_names_only_inside_environment_mappings(document: dict, credential: bool) -> None:
    assert holds_credentials(document) is credential
