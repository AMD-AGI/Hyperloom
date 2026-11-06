# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The component keys an Arbor checkout declares decide what MB writes for it.

A key that changes what Arbor runs, which the pinned Arbor lacks, refuses its start before it runs, naming the Arbor
commit to pin; an input it can run without is left out, and its branch notes why (an Arbor pin
without ``component.profile`` once refused three starts in a row).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist.agent_onboarding import make_ready
from hyperloom_multibranch.specialist.arbor_adapter import (
    BASE_LAUNCH_FILE,
    BASE_LAUNCH_SCHEMA,
    BRANCH_RUN,
    DOSSIER_PASS,
    KEY_SINCE,
    ArborCheckout,
    ArborRefused,
    arbor_config,
    arbor_manifest,
    declared_keys,
    fit_keys,
    read_launch,
    refusals,
)
from hyperloom_multibranch.tests.agent_rig import (
    CHECKOUT_FILES,
    COMPONENT_AGENT,
    CONFIG_FILE,
    GATED_COMPONENT,
    MAP,
    Component,
    Process,
    Sandbox,
    _committed,
    _setup,
    _spec,
    _write,
)

#: An Arbor of 10-28, before the dossier pass and the root's profile (Arbor 15a46e6).
BEFORE_PROFILE = (
    'BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",\n'
    '              "root", "plan")\n'
)
PINNED = "the pinned Arbor"


def _checkout(tmp_path: Path, component: str) -> Path:
    return _write(tmp_path / "checkout", {**CHECKOUT_FILES, "arbor/component.py": component})


def test_the_keys_are_read_from_the_checkout_s_block_keys_and_none_when_it_declares_none(tmp_path: Path) -> None:
    assert declared_keys(_checkout(tmp_path / "a", GATED_COMPONENT)) >= {"plan", "stop_at", "profile", "review"}
    assert declared_keys(_checkout(tmp_path / "b", BEFORE_PROFILE)) == {
        *("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes", "root", "plan")
    }
    assert declared_keys(_checkout(tmp_path / "c", "# component mode\n")) is None


def test_a_dossier_pass_on_an_arbor_without_stop_at_is_refused_naming_the_commit_to_pin(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path, BEFORE_PROFILE)
    config = arbor_config(_spec(tmp_path), checkout, DOSSIER_PASS)

    said = [reason for reason in refusals(config, declared_keys(checkout)) if PINNED in reason]

    assert said == [f"{PINNED} has no component.stop_at: pin Arbor {KEY_SINCE['stop_at']} or later"]


def test_a_branch_on_an_arbor_that_has_its_keys_is_not_refused_for_them(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path, GATED_COMPONENT)
    plan = _write(tmp_path / "shares", {"share-1.json": json.dumps(MAP)}) / "share-1.json"
    config = arbor_config(_spec(tmp_path, plan=str(plan)), checkout, BRANCH_RUN)

    assert not [reason for reason in refusals(config, declared_keys(checkout)) if PINNED in reason]
    assert not [reason for reason in refusals(config, None) if PINNED in reason]


def test_the_root_s_profile_is_left_out_with_a_note_for_an_arbor_without_it(tmp_path: Path) -> None:
    config = {"pack": "serving", "component": {"host": "hyperloom-mb", "profile": "/s/op_shares.json"}}

    older, notes = fit_keys(config, _checkout(tmp_path / "a", BEFORE_PROFILE))
    current, none = fit_keys(config, _checkout(tmp_path / "b", GATED_COMPONENT))
    unread, unknown = fit_keys(config, _checkout(tmp_path / "c", "# component mode\n"))

    assert older == {"pack": "serving", "component": {"host": "hyperloom-mb"}}
    assert notes == (f"{PINNED} has no component.profile (Arbor {KEY_SINCE['profile']}): started without it",)
    assert current == unread == config and none == unknown == ()


async def test_a_branch_on_an_arbor_without_the_profile_starts_without_it_and_says_so(tmp_path: Path) -> None:
    fake = {"component.py": COMPONENT_AGENT, "fake-plan.json": json.dumps({**asdict(Component()), "refuse": []})}
    files = {**CHECKOUT_FILES, "arbor/component.py": BEFORE_PROFILE, **fake}
    repo = _committed(_write(tmp_path / "arbor-repo", files))
    share = _write(tmp_path / "shares", {"share-1.json": json.dumps(MAP)}) / "share-1.json"
    spec = _spec(tmp_path, plan=str(share))
    spec = replace(spec, root_shares=OpShares(spec.root, "prelude-1", "p1", {"gemm": 41.5, "attention": 22.0}))

    ready = await make_ready(str(repo), spec, tmp_path / "agent", _setup(Sandbox([Process(None)]), None))

    component = json.loads((ready.source_dir / CONFIG_FILE).read_text())["component"]
    assert "profile" not in component and component["plan"]
    assert ready.notes == (f"{PINNED} has no component.profile (Arbor {KEY_SINCE['profile']}): started without it",)


# -- the session's base launch record --------------------------------------------------------------------------------

#: A record as MB freezes it: vLLM's own entrypoint, the model as a positional, runner-owned flags in both spellings.
BASE = {
    "schema": BASE_LAUNCH_SCHEMA,
    "framework": "vllm",
    "entrypoint": ["vllm", "serve"],
    "model": {"id": "amd/Qwen3-8B", "path": "/models/q"},
    "args": [
        "/models/q",
        "--tensor-parallel-size",
        "2",
        "--max-num-seqs",
        "128",
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 3}',
        "--port",
        "30000",
        "--host=0.0.0.0",
        "--chat-template",
        "a b",
    ],
    "env": {"VLLM_ROCM_USE_AITER": "1", "HIP_VISIBLE_DEVICES": "0"},
    "runner_owned": {"args": ["--host", "--port"], "env": ["HIP_VISIBLE_DEVICES"]},
    "digest": "sha256:base",
}


def _base(tmp_path: Path, **changes: object) -> Path:
    return _write(tmp_path / "record", {"base_launch.json": json.dumps({**BASE, **changes})}) / "base_launch.json"


def test_a_base_launch_record_is_read_less_what_the_runner_owns_and_the_model(tmp_path: Path) -> None:
    client = _spec(tmp_path).client
    record = _base(tmp_path, client={"command": client})

    launch = read_launch(record)

    assert launch.server_flags == (
        '--tensor-parallel-size 2 --max-num-seqs 128 --speculative-config {"method":"mtp","num_speculative_tokens":3}'
        " --chat-template 'a b'"
    )
    assert launch.server_env == {"VLLM_ROCM_USE_AITER": "1"}
    assert (launch.model_name, launch.client, launch.record) == ("amd/Qwen3-8B", client, str(record))
    assert (launch.digest, launch.entrypoint) == ("sha256:base", ("vllm", "serve"))


def test_a_branch_served_a_record_carries_it_for_arbor_s_start_check(tmp_path: Path) -> None:
    record = _base(tmp_path)
    spec, checkout = _spec(tmp_path, launch=str(record)), _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)

    carried = checkout.absolute() / BASE_LAUNCH_FILE
    component = manifest.config["component"]
    assert component["base_launch"] == {"path": str(carried), "digest": "sha256:base"}
    assert json.loads(carried.read_text()) == BASE
    assert "entrypoint" not in component  # vLLM's own command
    assert manifest.config["model_name"] == "amd/Qwen3-8B"


def test_an_arbor_without_the_check_runs_the_record_unchecked_but_not_another_entrypoint(tmp_path: Path) -> None:
    record = _base(tmp_path, entrypoint=["python3", "-m", "vllm.entrypoints.openai.api_server"])
    checkout = _checkout(tmp_path, GATED_COMPONENT)
    config = arbor_config(_spec(tmp_path, launch=str(record)), checkout, BRANCH_RUN)

    said = [reason for reason in refusals(config, declared_keys(checkout)) if PINNED in reason]
    fitted, notes = fit_keys(config, checkout)

    assert config["component"]["entrypoint"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert said == [f"{PINNED} has no component.entrypoint: pin Arbor {KEY_SINCE['entrypoint']} or later"]
    assert "base_launch" not in fitted["component"]
    assert notes == (f"{PINNED} has no component.base_launch (Arbor {KEY_SINCE['base_launch']}): started without it",)


def test_a_record_that_cannot_be_used_is_refused_with_every_reason(tmp_path: Path) -> None:
    record = _base(tmp_path, args="--tp 1", env={"A": 1}, digest="")

    with pytest.raises(ArborRefused) as refused:
        read_launch(record)

    assert [reason.split(": ", 1)[1] for reason in refused.value.reasons] == [
        "its args are not a list of strings",
        "its env is not an object of strings",
        "it has no digest",
    ]
