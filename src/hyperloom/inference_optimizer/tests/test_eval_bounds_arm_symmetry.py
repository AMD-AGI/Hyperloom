# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Both arms must truncate at the same place, and say so in the fingerprint.

The accuracy gate is differential: it subtracts the candidate's score from the
baseline's. That subtraction is only meaningful when both arms evaluated under
the same generation bounds, so two gaps left by the bounds work are pinned here.

* The baseline arm verified that the bounds shim installed and failed loudly
  when it could not; the grid arm called the same installer, discarded the
  result, and only called it at all when ``$INFERENCEX_PATH`` happened to be
  set -- while the baseline arm also accepts env discovery. A bounded baseline
  could therefore be compared against an unbounded candidate.
* The bounds knobs decide where each answer is cut off, but did not participate
  in the eval-contract fingerprint, so changing one left the contract looking
  identical to a run that scored under different rules.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _accuracy_gate
from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    eval_contract_fingerprint,
    materialized_run_eval_disabled,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    _run_magpie,
    run_grid,
)
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    EVAL_PROBE_UNPATCHABLE_RETURNCODE,
)

_PATCHER = "hyperloom.orchestrator.actions.executors._grid_runner"


def _write_config(path: Path, **envs) -> Path:
    """Write a materialized benchmark YAML carrying ``envs`` in benchmark.envs."""
    base_envs = {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256}
    base_envs.update(envs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/path/models/Qwen-Qwen3-8B",
                    "precision": "bf16",
                    "run_mode": "local",
                    "benchmark_script": "sglang.sh",
                    "envs": base_envs,
                    "timeout_seconds": 600,
                    "profiler": {
                        "torch_profiler": {"enabled": False},
                        "system_profiler": {"enabled": False},
                        "tracelens": {"enabled": False},
                    },
                    "gpu_selection": {"auto": False},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_workspace(slot: Path, *, tput: float = 1500.0) -> Path:
    ws = slot / "benchmark_sglang_20260812_010101"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "benchmark_report.json").write_text(
        yaml.safe_dump(
            {
                "success": True,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
            }
        ),
        encoding="utf-8",
    )
    return ws


# ---------------------------------------------------------------------------
# Eval-contract fingerprint: the bounds knobs are part of the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("knob", "changed"),
    [
        ("HYPERLOOM_EVAL_MAX_TOKENS", "1024"),
        ("HYPERLOOM_EVAL_DERIVE_STOP", "0"),
        ("HYPERLOOM_EVAL_STOP_STRINGS", "<|im_end|>"),
    ],
)
def test_a_bounds_knob_change_changes_the_eval_contract_fingerprint(tmp_path, knob, changed):
    """Two runs that truncate differently must not claim the same eval contract.

    The enablement re-run path compares fingerprints to decide whether a
    recorded accuracy still applies. A knob that moves the truncation point but
    leaves the digest alone would let a score taken under one bound be reused to
    satisfy a different one.
    """
    before = _write_config(tmp_path / "before.yaml", **{knob: "4096"})
    after = _write_config(tmp_path / "after.yaml", **{knob: changed})

    fp_before = eval_contract_fingerprint(config_path=before)
    fp_after = eval_contract_fingerprint(config_path=after)

    assert fp_before and fp_after
    assert fp_before != fp_after


def test_an_absent_bounds_knob_matches_itself(tmp_path):
    """Omitting the knobs entirely stays stable: absence is a contract too."""
    a = _write_config(tmp_path / "a.yaml")
    b = _write_config(tmp_path / "b.yaml")
    assert eval_contract_fingerprint(config_path=a) == eval_contract_fingerprint(config_path=b)


def test_a_tunable_server_arg_stays_out_of_the_fingerprint(tmp_path):
    """The digest must survive the very thing the optimizer is allowed to change.

    Server args are what a candidate tunes; folding them in would make every
    candidate look like a different eval contract and defeat drift detection.
    """
    before = _write_config(tmp_path / "before.yaml", EXTRA_SGLANG_ARGS="--chunked-prefill-size 2048")
    after = _write_config(tmp_path / "after.yaml", EXTRA_SGLANG_ARGS="--chunked-prefill-size 8192")
    assert eval_contract_fingerprint(config_path=before) == eval_contract_fingerprint(config_path=after)


# ---------------------------------------------------------------------------
# Grid arm: assert the bounds landed, on the same terms as the baseline arm
# ---------------------------------------------------------------------------


def test_the_grid_arm_asserts_bounds_even_when_inferencex_path_is_unset(tmp_path, monkeypatch):
    """The install must be attempted via env discovery, as the baseline arm does.

    Gating it on ``$INFERENCEX_PATH`` is what let a session whose checkout is
    only reachable through ``$MAGPIE_PATH/InferenceX`` bench candidates with no
    bounds while its baseline had them.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    calls: list[object] = []

    def recording_ensure(root=None):
        calls.append(root)
        return True

    with (
        patch(f"{_PATCHER}.ensure_eval_probe_patched", side_effect=recording_ensure),
        patch(
            f"{_PATCHER}.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
        ),
    ):
        rc, _out, _err = _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=_write_config(tmp_path / "config.yaml"),
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )

    assert calls == [None], "bounds install must be attempted with env discovery"
    assert rc == 0


def test_a_variant_fails_when_the_bounds_target_is_present_but_unpatchable(tmp_path, monkeypatch):
    """Present-and-unpatchable is a broken contract, so nothing may be benched.

    Same verdict the baseline arm already reaches. Benching anyway would score
    this variant against a baseline that stopped generating somewhere else.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    launched: list[object] = []

    with (
        patch(f"{_PATCHER}.ensure_eval_probe_patched", return_value=False),
        patch(f"{_PATCHER}.eval_probe_targets_exist", return_value=True),
        patch(
            f"{_PATCHER}.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: launched.append(cmd) or subprocess.CompletedProcess(cmd, 0, "ok", ""),
        ),
    ):
        rc, _out, err = _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=_write_config(tmp_path / "config.yaml"),
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )

    assert rc == EVAL_PROBE_UNPATCHABLE_RETURNCODE
    assert launched == [], "no benchmark may run without the bounds it will be graded under"
    assert "bounds" in err


def test_a_variant_still_runs_when_no_bounds_target_exists(tmp_path, monkeypatch):
    """Target absent is an unrecognized layout, not a broken contract.

    The baseline arm warns rather than failing here, so the grid arm must too --
    otherwise an InferenceX laid out somewhere unrecognized fails every variant.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")

    with (
        patch(f"{_PATCHER}.ensure_eval_probe_patched", return_value=False),
        patch(f"{_PATCHER}.eval_probe_targets_exist", return_value=False),
        patch(
            f"{_PATCHER}.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
        ),
    ):
        rc, _out, _err = _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=_write_config(tmp_path / "config.yaml"),
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )

    assert rc == 0


def test_a_variant_that_runs_no_eval_is_not_failed_by_the_bounds_check(tmp_path, monkeypatch):
    """No eval this round means no eval contract to keep symmetric.

    A throughput-only round is graded on throughput; failing it over an eval
    shim it never loads would drop candidates for an irrelevant reason.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")

    with (
        patch(f"{_PATCHER}.ensure_eval_probe_patched", return_value=False),
        patch(f"{_PATCHER}.eval_probe_targets_exist", return_value=True),
        patch(
            f"{_PATCHER}.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
        ),
    ):
        rc, _out, _err = _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=_write_config(tmp_path / "config.yaml", RUN_EVAL="false"),
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )

    assert rc == 0


@pytest.mark.asyncio
async def test_run_grid_labels_the_bounds_gap_instead_of_a_missing_workspace(tmp_path):
    """The ledger must name the cause, using the baseline arm's own class.

    Nothing launches, so the generic path would find no ``benchmark_*`` dir and
    file this as ``no_benchmark_workspace`` -- a missing-directory message
    hiding an eval-contract gap.
    """
    base = _write_config(tmp_path / "base.yaml")

    with (
        patch(f"{_PATCHER}.ensure_eval_probe_patched", return_value=False),
        patch(f"{_PATCHER}.eval_probe_targets_exist", return_value=True),
        patch(
            f"{_PATCHER}.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
        ),
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("vA")],
            output_root=tmp_path / "out",
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_class == "eval_probe_unpatchable"
    assert results[0].returncode == EVAL_PROBE_UNPATCHABLE_RETURNCODE


# ---------------------------------------------------------------------------
# The shared RUN_EVAL reader both arms now key off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["false", "0", "no", "off", ""])
def test_run_eval_disabled_recognizes_every_falsey_spelling(tmp_path, spelling):
    cfg = _write_config(tmp_path / f"cfg_{spelling or 'empty'}.yaml", RUN_EVAL=spelling)
    assert materialized_run_eval_disabled(cfg) is True


def test_run_eval_absent_reads_as_enabled(tmp_path):
    """Matches the materialize default, which is "true" when the key is absent."""
    assert materialized_run_eval_disabled(_write_config(tmp_path / "cfg.yaml")) is False


def test_an_unreadable_config_reads_as_eval_enabled(tmp_path):
    """Fail closed: an unreadable config must not silently skip the eval guards.

    Reading "disabled" from a config that could not be parsed would turn the
    bounds check off precisely when least is known about the run.
    """
    assert materialized_run_eval_disabled(tmp_path / "does_not_exist.yaml") is False

    broken = tmp_path / "broken.yaml"
    broken.write_text("benchmark: [unclosed\n", encoding="utf-8")
    assert materialized_run_eval_disabled(broken) is False


def test_the_shared_reader_lives_in_a_module_every_arm_can_import():
    """All three arms reach one reader, none of them through an import cycle.

    ``_workload_envs`` imports ``_grid_runner`` at module scope, so hosting the
    reader there forced the grid arm to import it from inside a function. It sits
    in ``_accuracy_gate`` instead, whose leaf property — it imports no executor
    sibling — is what lets the grid, the baseline and the env materializer all
    import it at module scope. Keep that property or the cycle comes back.
    """
    import ast
    from pathlib import Path as _Path

    executors = _Path(_accuracy_gate.__file__).parent
    tree = ast.parse((executors / "_accuracy_gate.py").read_text(encoding="utf-8"))
    siblings = {
        node.module.lstrip(".")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
    }
    assert siblings == set(), f"_accuracy_gate must stay a leaf, but it imports {sorted(siblings)}"

    from hyperloom.orchestrator.actions.executors import _grid_runner, _workload_envs, baseline

    for arm in (_grid_runner, baseline):
        assert arm.materialized_run_eval_disabled is materialized_run_eval_disabled, arm.__name__
    # The env materializer decides the same question from raw envs rather than a
    # written config, so it shares the spellings instead of the reader.
    assert _workload_envs._RUN_EVAL_FALSE_VALUES is _accuracy_gate._RUN_EVAL_FALSE_VALUES
