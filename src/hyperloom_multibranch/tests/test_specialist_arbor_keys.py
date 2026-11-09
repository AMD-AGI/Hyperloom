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

from hyperloom.common.perf_metric import GRADED_INTVTY_P50
from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist.agent_onboarding import make_ready
from hyperloom_multibranch.specialist.arbor_adapter import (
    BASE_LAUNCH_FILE,
    BRANCH_RUN,
    DOSSIER_PASS,
    KEY_SINCE,
    ArborCheckout,
    arbor_config,
    arbor_manifest,
    declared_keys,
    fit_keys,
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
from hyperloom_multibranch.tests.base_records import base_record, write_record

#: An Arbor of 10-28, before the dossier pass and the root's profile (Arbor 15a46e6).
BEFORE_PROFILE = (
    'BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",\n'
    '              "root", "plan")\n'
)
PINNED = "the pinned Arbor"
VENDORED_ARBOR = Path(__file__).parents[2] / "hyperloom_arbor"


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


#: Arbor 6a62def's arbor/component.py, as far as the adapter reads it: component.bench_read without require.
BENCH_READ_6A62DEF = (
    'BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",\n'
    '              "root", "plan", "stop_at", "profile", "review", "base_launch", "entrypoint", "bench_read")\n'
    "def bench_read_errors(raw, command):\n"
    '    unknown = set(raw) - {"file", "keys"}\n'
)


def test_an_agentx_branch_measured_by_mb_s_round_on_an_arbor_without_bench_read_require_is_refused_naming_the_commit(
    tmp_path: Path,
) -> None:
    client = "python -m inference_suite run --config /s/mb-bench.yaml --run-dir /s/mb-bench"
    spec = _spec(tmp_path, workload={"kind": "agentx", "conc": 8}, metric=GRADED_INTVTY_P50, client=client)
    gated, read = _checkout(tmp_path / "gated", GATED_COMPONENT), _checkout(tmp_path / "read", BENCH_READ_6A62DEF)
    config = arbor_config(spec, gated, BRANCH_RUN)

    said = [reason for reason in refusals(config, declared_keys(gated)) if PINNED in reason]
    unrequired = [reason for reason in refusals(config, declared_keys(read)) if PINNED in reason]

    assert said == [
        f"{PINNED} has no component.bench_read: pin Arbor 6a62def or later",
        f"{PINNED} has no component.bench_read.require: pin Arbor 1c6eef1 or later",
    ]
    assert unrequired == [f"{PINNED} has no component.bench_read.require: pin Arbor 1c6eef1 or later"]
    assert "bench_read" in fit_keys(config, gated)[0]["component"]
    assert {"bench_read", "bench_read.require"} <= declared_keys(VENDORED_ARBOR)


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


def test_a_branch_served_a_record_carries_it_for_arbor_s_start_check(tmp_path: Path) -> None:
    record = write_record(tmp_path / "record" / "base_launch.json")
    spec, checkout = _spec(tmp_path, launch=str(record)), _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)

    carried = checkout.absolute() / BASE_LAUNCH_FILE
    component = manifest.config["component"]
    assert component["base_launch"] == {"path": str(carried), "digest": base_record().digest()}
    assert carried.read_bytes() == record.read_bytes()
    assert "entrypoint" not in component  # vLLM's own command


def test_an_arbor_without_the_check_runs_the_record_unchecked_but_not_another_entrypoint(tmp_path: Path) -> None:
    entrypoint = ("python3", "-m", "vllm.entrypoints.openai.api_server")
    record = write_record(tmp_path / "record" / "base_launch.json", base_record(entrypoint=entrypoint))
    checkout = _checkout(tmp_path, GATED_COMPONENT)
    config = arbor_config(_spec(tmp_path, launch=str(record)), checkout, BRANCH_RUN)

    said = [reason for reason in refusals(config, declared_keys(checkout)) if PINNED in reason]
    fitted, notes = fit_keys(config, checkout)

    assert config["component"]["entrypoint"] == list(entrypoint)
    assert said == [f"{PINNED} has no component.entrypoint: pin Arbor {KEY_SINCE['entrypoint']} or later"]
    assert "base_launch" not in fitted["component"]
    assert notes == (f"{PINNED} has no component.base_launch (Arbor {KEY_SINCE['base_launch']}): started without it",)
