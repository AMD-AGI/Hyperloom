# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestrator reads the session's base launch only through ``session_server_base``, which checks its digest."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session.base_launch import BASE_LAUNCH_FILE
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.state.base_launch import BaseLaunchMismatch, ServerBase, session_server_base
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite.contracts import Redacted

from ._session_base import freeze_session_base, session_base

_SRC = Path(__file__).resolve().parents[4]
_REFERENCE_FIELDS = {"reference_server_args", "reference_envs", "reference_launch_controls", "reference_model"}


def _modules():
    for path in sorted(_SRC.rglob("*.py")):
        relative = path.relative_to(_SRC)
        if "tests" not in relative.parts:
            yield relative, ast.parse(path.read_text(encoding="utf-8"))


def _attribute_reads(tree: ast.AST, names: set[str]) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names:
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        ):
            lines.append(node.lineno)
    return lines


def _users(name: str) -> set[Path]:
    users = set()
    for relative, tree in _modules():
        for node in ast.walk(tree):
            imported = isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names)
            if imported or (isinstance(node, ast.Attribute) and node.attr == name):
                users.add(relative)
    return users


def test_no_module_but_the_state_reads_the_reference_fields():
    offenders = [
        f"{relative}:{line}"
        for relative, tree in _modules()
        if relative != Path("hyperloom/orchestrator/state/shared_state.py")
        for line in _attribute_reads(tree, _REFERENCE_FIELDS)
    ]
    assert offenders == []


def test_only_the_state_reader_reads_the_frozen_file():
    assert _users("read_base_launch") == {Path("hyperloom/orchestrator/state/base_launch.py")}


def test_the_tuple_shaped_accessor_is_left_to_its_last_caller():
    assert _users("resolve_reference_launch") == {Path("hyperloom/orchestrator/kernel/_forge_gemm_probe.py")}


@pytest.fixture
def session(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(tmp_path))
    return tmp_path


def test_a_session_that_froze_no_base_starts_from_nothing(session):
    SharedState(session_id="scriptable").save(session)

    assert session_server_base() == ServerBase()


def test_the_server_base_carries_the_argv_and_the_env_a_launch_may_set(session):
    hidden = Redacted("a" * 64)
    freeze_session_base(
        session,
        args=("--kv-cache-dtype", "fp8_e4m3", "--speculative-config", '{"method":"mtp"}'),
        env={"SGLANG_USE_AITER": "1", "HF_TOKEN_FILE": hidden, "MODEL_PATH": "/elsewhere"},
    )

    base = session_server_base()

    assert base.args == '--kv-cache-dtype fp8_e4m3 --speculative-config {"method":"mtp"}'
    assert dict(base.env) == {"SGLANG_USE_AITER": "1"}


def test_a_base_file_the_session_did_not_record_is_refused(session):
    freeze_session_base(session, args=("--kv-cache-dtype", "fp8_e4m3"))
    other = session_base(args=("--kv-cache-dtype", "auto"))
    (session / BASE_LAUNCH_FILE).write_text(json.dumps(other.to_document()), encoding="utf-8")

    with pytest.raises(BaseLaunchMismatch, match=other.digest()):
        session_server_base()


def test_a_recorded_base_whose_file_is_gone_is_refused(session):
    freeze_session_base(session)
    (session / BASE_LAUNCH_FILE).unlink()

    with pytest.raises(BaseLaunchMismatch, match="no file"):
        session_server_base()


def test_a_base_file_in_a_session_that_recorded_none_is_refused(session):
    freeze_session_base(session)
    state = SharedState.load_or_init(session)
    state.base_launch_digest = ""
    state.save(session)

    with pytest.raises(BaseLaunchMismatch, match="recorded \\(none\\)"):
        session_server_base()
