# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestrator reads the session's base launch only through ``frozen_session_base``, which checks its digest."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.inference_optimizer.session.base_launch import BASE_LAUNCH_FILE
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.state.base_launch import (
    BaseLaunchMismatch,
    FrozenBase,
    ServerBase,
    frozen_base,
    frozen_session_base,
    session_server_base,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite import INFERENCEX_CLIENT_COMMIT
from inference_suite.contracts import ClientLaunch, GoldenAcceptance, GoldenCurve, Redacted

from ._session_base import freeze_base, freeze_session_base, session_base

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


def test_no_module_reads_the_reference_fields():
    offenders = [
        f"{relative}:{line}" for relative, tree in _modules() for line in _attribute_reads(tree, _REFERENCE_FIELDS)
    ]
    assert offenders == []


def test_only_the_state_reader_reads_the_frozen_file_besides_the_banner_printing_its_warnings():
    assert _users("read_base_launch") == {
        Path("hyperloom/orchestrator/state/base_launch.py"),
        Path("hyperloom/inference_optimizer/cli/lineup_banners.py"),
    }


@pytest.fixture
def session(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(tmp_path))
    return tmp_path


def test_a_session_that_froze_no_base_starts_from_nothing(session):
    SharedState(session_id="scriptable").save(session)

    assert session_server_base() == ServerBase()
    assert frozen_session_base() is None


def test_the_frozen_base_names_its_file_beside_the_launch_it_holds(session):
    base = freeze_session_base(session, args=("--kv-cache-dtype", "fp8_e4m3"))

    assert frozen_session_base() == FrozenBase(session / BASE_LAUNCH_FILE, base)


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


def test_the_golden_layer_reaches_the_server_base_above_the_base(session):
    golden = GoldenAcceptance(
        value=3.48,
        method="match-expected",
        token_mode="real-draft-token",
        spec_method="EAGLE",
        draft_length=3,
        curve=GoldenCurve(repo="https://github.com/org/curves", commit="e" * 40, path="al.json", sha256="f" * 64),
        args=("--speculative-algorithm", "EAGLE"),
        env={"SGLANG_SIMULATE_ACC_LEN": "3.48"},
    )
    freeze_session_base(session, args=("--kv-cache-dtype", "fp8_e4m3"), env={"SGLANG_USE_AITER": "1"}, golden=golden)

    base = session_server_base()

    assert base.args == "--kv-cache-dtype fp8_e4m3 --speculative-algorithm EAGLE"
    assert dict(base.env) == {"SGLANG_USE_AITER": "1", "SGLANG_SIMULATE_ACC_LEN": "3.48"}


@pytest.mark.parametrize(
    "name",
    [
        # Credentials.
        "SOME_SECRET",
        "HF_TOKEN",
        "TOKENIZER_API_KEY",
        "MY_TOKENIZER_SECRET",
        # Interpreter hijacks and traffic redirects.
        "PYTHONUSERBASE",
        "NODE_OPTIONS",
        "PERL5OPT",
        "GIT_SSH_COMMAND",
        "HTTPS_PROXY",
        "HF_ENDPOINT",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PATH",
        # Device selection.
        *sorted(GPU_MASK_ENV_NAMES),
        # Workload keys the optimizer owns.
        "TP",
        "MODEL",
        "CONC",
        "RUN_EVAL",
        "RESULT_DIR",
    ],
)
def test_a_launch_file_export_a_launch_may_not_set_never_reaches_the_server(session, name):
    tuning = {"VLLM_ROCM_USE_AITER": "1", "TOKENIZERS_PARALLELISM": "false", "VLLM_CACHE_ROOT": "/workspace/cache"}
    freeze_session_base(session, env={**tuning, name: "value"})

    assert dict(session_server_base().env) == tuning


def test_a_base_file_the_session_did_not_record_is_refused(session):
    freeze_session_base(session, args=("--kv-cache-dtype", "fp8_e4m3"))
    other = session_base(args=("--kv-cache-dtype", "auto"))
    (session / BASE_LAUNCH_FILE).write_text(json.dumps(other.to_document()), encoding="utf-8")

    with pytest.raises(BaseLaunchMismatch, match=other.digest()):
        session_server_base()
    with pytest.raises(BaseLaunchMismatch, match=other.digest()):
        frozen_session_base()


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


def test_a_base_whose_client_is_the_pinned_one_is_accepted(session):
    base = freeze_session_base(session)

    assert base.client.commit == INFERENCEX_CLIENT_COMMIT
    assert frozen_base(session) == FrozenBase(session / BASE_LAUNCH_FILE, base)


def test_a_base_whose_client_is_off_the_pin_is_refused_naming_both_commits(session):
    drifted = "e" * 40
    freeze_base(session, dataclasses.replace(session_base(), client=ClientLaunch(tool="aiperf", commit=drifted)))

    with pytest.raises(BaseLaunchMismatch, match=f"{drifted}.*{INFERENCEX_CLIENT_COMMIT}.*new session"):
        frozen_base(session)
