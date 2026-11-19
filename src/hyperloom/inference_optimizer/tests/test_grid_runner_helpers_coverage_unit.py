# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplemental coverage for _grid_runner pure helpers: compatibility filter model-class drop, runtime override env branches, report parsing, and per-variant yaml env injection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
    removals_absent_from_args,
)
from hyperloom.orchestrator.actions.executors.grid import _grid_runner as gr
from hyperloom.orchestrator.actions.executors.grid import _capability_probe as cp
from hyperloom.orchestrator.actions.executors.grid import _round_report as rr
from hyperloom.orchestrator.actions.executors.grid._grid_variant_filter import ServingStack
from hyperloom.orchestrator.actions.executors import _benchmark_interpreter as bi
from hyperloom.orchestrator.actions.executors.workload._benchmark_runtime import apply_runtime_benchmark_overrides

# Patch compatibility-filter helpers in the ``_grid_variant_filter`` sibling, where apply_compatibility_filter
# resolves them (not via the re-export).
from hyperloom.orchestrator.actions.executors.grid import _grid_variant_filter as vf


def _variant(name: str, args: str = "", envs: dict | None = None) -> gr.GridVariant:
    return gr.GridVariant(name=name, extra_server_args=args, extra_envs=envs or {})


# -- apply_compatibility_filter -------------------------------------------
def test_compatibility_filter_drops_on_model_class(monkeypatch) -> None:
    monkeypatch.setattr(vf, "_detect_model_class", lambda mp: (False, False))
    grid = [_variant("mla", "--enable-flashinfer-mla"), _variant("plain", "")]
    kept, dropped = vf.apply_compatibility_filter(grid, serving=ServingStack("sglang", "meta-llama-3-8b"))
    assert [v.name for v in kept] == ["plain"]
    assert len(dropped) == 1
    assert dropped[0]["source"] == "compatibility_filter"
    assert "MLA" in dropped[0]["reason"]


def test_compatibility_filter_keeps_a_flag_the_model_class_supports(monkeypatch) -> None:
    # Whether the installed server accepts the flag is left to its own argparse at launch.
    monkeypatch.setattr(vf, "_detect_model_class", lambda mp: (True, True))
    grid = [_variant("moe", "--enable-ep-moe")]
    kept, dropped = vf.apply_compatibility_filter(grid, serving=ServingStack("sglang", "deepseek-moe"))
    assert [v.name for v in kept] == ["moe"] and dropped == []


def test_compatibility_filter_no_model_path_assumes_compatible() -> None:
    grid = [_variant("moe", "--enable-ep-moe")]
    kept, dropped = vf.apply_compatibility_filter(grid, serving=ServingStack("sglang", ""))
    assert [v.name for v in kept] == ["moe"] and dropped == []


# -- ungrounded_env_file_reason (env-value file grounding) ----------------
def test_ungrounded_env_file_reason_drops_missing_config_path(tmp_path) -> None:
    # A proposal may point a config env at an artifact no stage authored; the server would read it
    # unconditionally and die at config load, so the variant must be dropped with a reason.
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": str(tmp_path / "nope.csv")})
    reason = cp.ungrounded_env_file_reason(v.extra_envs)
    assert reason and "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE" in reason
    assert str(tmp_path / "nope.csv") in reason


def test_ungrounded_env_file_reason_drops_colon_joined_missing_element(tmp_path) -> None:
    a = tmp_path / "t.csv"
    a.write_text("gfx,M,N,K\ngfx950,1,2,3\n", encoding="utf-8")
    missing = str(tmp_path / "never_authored.csv")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": f"{a}:{missing}"})
    reason = cp.ungrounded_env_file_reason(v.extra_envs)
    assert reason and missing in reason and str(a) not in reason.split("element(s)")[1].split(" do not exist")[0]


def test_ungrounded_env_file_reason_treats_an_unreadable_path_as_missing(monkeypatch) -> None:
    # An unreadable parent or an overlong name makes Path.is_file() raise; the variant is dropped, the grid runs on.
    def raise_eacces(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(gr.Path, "is_file", raise_eacces)
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": "/root/private/tuned.csv"})
    reason = cp.ungrounded_env_file_reason(v.extra_envs)
    assert reason and "/root/private/tuned.csv" in reason


def test_run_grid_fast_fail_drops_a_variant_whose_config_file_is_missing(tmp_path, monkeypatch) -> None:
    # The backstop of the admission check: run_grid's fast-fail drops it before boot, whatever the framework.
    monkeypatch.setenv("FRAMEWORK", "sglang")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": str(tmp_path / "never_written.csv")})
    assert "AITER_CONFIG_GEMM_BF16" in (cp.unsupported_capability_reason(v) or "")


def test_run_grid_fast_fail_drops_a_list_whose_other_table_is_written(tmp_path, monkeypatch) -> None:
    # One element on disk beside one nothing wrote: the server would still boot on the thinner merge, but the arm
    # would be benched against a table that is only partly there, so run_grid asks the same every-element question
    # admission asks and names the element that is missing.
    monkeypatch.setenv("FRAMEWORK", "sglang")
    written = tmp_path / "tuned.csv"
    written.write_text("gfx,M,N,K\ngfx950,1,2,3\n", encoding="utf-8")
    missing = str(tmp_path / "never_written.csv")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": f"{written}:{missing}"})
    named_missing = (cp.unsupported_capability_reason(v) or "").split("element(s)")[-1].split(" do not exist")[0]
    assert missing in named_missing and str(written) not in named_missing


def test_ungrounded_env_file_reason_noop_without_config_envs(tmp_path) -> None:
    # Non-file envs (even under aiter-ish prefixes like the JSON-payload AITER_CONFIG) must stay untouched.
    v = _variant(
        "plain",
        envs={
            "AITER_CONFIG": '{"path": "/tmp/never_scanned.json"}',
            "SGLANG_USE_AITER": "1",
            "AITER_CONFIG_GEMM_A8W8": "",
        },
    )
    assert cp.ungrounded_env_file_reason(v.extra_envs) is None


# -- unsupported_capability_reason (env-flag build probe) -----------------
def _clear_cap_cache() -> None:
    cp._CAP_PROBE_CACHE.clear()


@pytest.mark.parametrize(
    ("framework", "envs"),
    [
        ("vllm", {}),
        ("sglang", {"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1"}),
        ("vllm", {"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0"}),
    ],
    ids=["flag-absent", "non-vllm-framework", "falsey-flag"],
)
def test_capability_reason_does_not_probe_without_an_enabled_vllm_flag(monkeypatch, framework, envs) -> None:
    monkeypatch.setenv("FRAMEWORK", framework)
    monkeypatch.setattr(
        cp,
        "_probe_vllm_aiter_shared_expert_unsupported",
        lambda: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    assert cp.unsupported_capability_reason(_variant("se", envs=envs)) is None


def test_capability_reason_drops_when_module_missing(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    _clear_cap_cache()
    monkeypatch.setattr(
        cp,
        "_probe_vllm_aiter_shared_expert_unsupported",
        lambda: "missing module(s): vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe",
    )
    v = _variant("se", envs={"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1"})
    reason = cp.unsupported_capability_reason(v)
    assert reason is not None and "rocm_aiter_fused_moe" in reason


_SHARED_EXPERTS_ON = {"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1"}


def _record_probe_runs(monkeypatch, status: str) -> list[list[str]]:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setattr(cp, "_CAP_PROBE_CACHE", {})
    runs: list[list[str]] = []

    def fake_run(cmd, *_a, **_k):
        runs.append(list(cmd))
        payload = {"status": status, "missing": ["vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe"]}
        return SimpleNamespace(stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    return runs


@pytest.mark.parametrize(
    ("status", "drops", "probes"),
    [("ok", False, 1), ("unsupported", True, 1), ("unknown", False, 2)],
    ids=["ok", "unsupported", "unknown"],
)
def test_the_build_probe_answer_is_reused_across_variants_unless_unknown(monkeypatch, status, drops, probes) -> None:
    runs = _record_probe_runs(monkeypatch, status)
    reasons = [cp.unsupported_capability_reason(_variant(name, envs=_SHARED_EXPERTS_ON)) for name in ("se-a", "se-b")]
    assert [reason is not None for reason in reasons] == [drops, drops]
    assert len(runs) == probes


def test_probe_swallows_subprocess_error(monkeypatch) -> None:
    _clear_cap_cache()

    def boom(*_a, **_k):
        raise OSError("no python3")

    monkeypatch.setattr(cp.subprocess, "run", boom)
    assert cp._probe_vllm_aiter_shared_expert_unsupported() is None


# -- _resolve_probe_python / probe interpreter selection ------------------
@pytest.mark.parametrize(("image", "expected"), [(True, "/opt/venv/bin/python"), (False, "/usr/bin/python3")])
def test_resolve_probe_python_without_vllm_takes_the_images_interpreter_else_python3(
    monkeypatch, image, expected
) -> None:
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(bi.shutil, "which", lambda name, **_k: "/usr/bin/python3" if name == "python3" else None)
    monkeypatch.setattr(bi.Path, "is_file", lambda self: image and str(self) == "/opt/venv/bin/python")
    assert bi._resolve_probe_python("sglang") == expected


def test_the_build_probe_runs_under_the_vllm_venv_interpreter(monkeypatch, tmp_path) -> None:
    venv_python = tmp_path / "bin" / "python"
    venv_python.parent.mkdir()
    venv_python.touch(mode=0o755)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(tmp_path))
    runs = _record_probe_runs(monkeypatch, "ok")
    cp.unsupported_capability_reason(_variant("se", envs=_SHARED_EXPERTS_ON))
    assert [cmd[0] for cmd in runs] == [str(venv_python)]


# -- apply_runtime_benchmark_overrides ------------------------------------
def test_runtime_overrides_model_precision_and_gpu_no_framework_agent(monkeypatch) -> None:
    monkeypatch.setenv("PRECISION", "fp8")
    for k in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench: dict = {}
    apply_runtime_benchmark_overrides(
        bench,
        model_path="/models/x",
        gpu_type="mi355x",
    )
    assert bench["model"] == "/models/x"
    assert bench["precision"] == "fp8"
    assert "runner_type" not in bench
    assert "benchmark_script" not in bench


def test_runtime_overrides_env_ints_and_rocr_autofill(monkeypatch) -> None:
    for k in ("PRECISION",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ISL", "128")
    monkeypatch.setenv("TP", "2")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    bench: dict = {}
    envs = apply_runtime_benchmark_overrides(bench)
    assert envs["ISL"] == 128
    assert envs["TP"] == 2
    # TP>1 with no explicit ROCR -> auto-filled device list
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1"


# -- _parse_report ---------------------------------------------------------
def test_parse_report_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text("{bad", encoding="utf-8")
    assert rr._parse_report(tmp_path) is None


def test_parse_report_non_dict(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text("[1,2,3]", encoding="utf-8")
    assert rr._parse_report(tmp_path) is None


# -- _build_variant_yaml ---------------------------------------------------
def test_build_variant_yaml_dedupes_repeated_flags(tmp_path: Path) -> None:
    """When base YAML + base_extra_args + variant all set the same flag, the materialized YAML must contain each flag
    only once (last wins).
    """
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "envs": {"EXTRA_VLLM_ARGS": "--attention-backend ROCM_ATTN"},
                },
            }
        ),
        encoding="utf-8",
    )
    variant = _variant("v1", "--attention-backend ROCM_AITER_FA")
    out = gr._build_variant_yaml(
        gr.GridRecipe(base_yaml_path=base),
        variant,
        output_subdir=tmp_path / "v1",
    )
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    args = cfg["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert args.count("--attention-backend") == 1, f"duplicate flag: {args}"
    assert "ROCM_AITER_FA" in args, "last-wins should keep variant value"


def test_build_variant_yaml_repeated_multi_value_flag_keeps_last(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"EXTRA_SGLANG_ARGS": "--cuda-graph-bs 1 2 4"}}}),
        encoding="utf-8",
    )
    out = gr._build_variant_yaml(
        gr.GridRecipe(base_yaml_path=base), _variant("v1", "--cuda-graph-bs 8 16"), output_subdir=tmp_path / "v1"
    )
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] == (
        "--cuda-graph-bs 8 16"
    )


def test_build_variant_yaml_prepends_legit_overlay(tmp_path: Path) -> None:
    # SWSPLAT-42358: a legitimate overlay is a single existing directory; it is prepended to PYTHONPATH unchanged.
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    variant = _variant("v1")
    variant.overlay_pythonpath = str(overlay)  # type: ignore[attr-defined]
    out = gr._build_variant_yaml(gr.GridRecipe(base_yaml_path=base), variant, output_subdir=tmp_path / "v1")
    envs = yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert envs.get("PYTHONPATH", "").startswith(str(overlay))


@pytest.mark.parametrize(
    ("config_envs", "tail"),
    [({}, ["/inh/a", "/inh/b"]), ({"PYTHONPATH": "/cfg"}, ["/cfg"])],
    ids=["inherited", "config-authoritative"],
)
def test_build_variant_yaml_overlay_builds_on_the_launch_pythonpath(tmp_path, monkeypatch, config_envs, tail) -> None:
    # The launch overlays benchmark.envs on os.environ, so a config without PYTHONPATH carries the inherited one.
    # The overlay is relative so the path carries no drive colon on any host.
    import os

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/inh/a", "/inh/b"]))
    (tmp_path / "overlay").mkdir()
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": config_envs}}), encoding="utf-8")
    variant = _variant("v1")
    variant.overlay_pythonpath = "overlay"  # type: ignore[attr-defined]
    out = gr._build_variant_yaml(gr.GridRecipe(base_yaml_path=base), variant, output_subdir=tmp_path / "v1")
    envs = yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert envs["PYTHONPATH"] == os.pathsep.join(["overlay", *tail])


def test_build_variant_yaml_drops_unsafe_overlay(tmp_path: Path) -> None:
    # SWSPLAT-42358: a ``:``-joined / traversal / non-existent overlay is dropped (would otherwise smuggle extra
    # PYTHONPATH entries or escape).
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    for bad in (f"{overlay}:/etc", f"{overlay}/../evil", str(tmp_path / "missing")):
        variant = _variant("v1")
        variant.overlay_pythonpath = bad  # type: ignore[attr-defined]
        out = gr._build_variant_yaml(gr.GridRecipe(base_yaml_path=base), variant, output_subdir=tmp_path / "v1")
        envs = yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["envs"]
        assert "PYTHONPATH" not in envs or bad not in envs.get("PYTHONPATH", ""), bad


def test_compose_server_args_replace_still_applies_remove_args() -> None:
    out = compose_server_args(
        inherited_args="--bad-base 1",
        base_extra_args="--also-bad 2",
        variant_extra_args="--bad-base 3 --keep 4",
        remove_args=["--bad-base"],
        args_mode="replace",
    )
    assert out == "--also-bad 2 --keep 4"


def test_compose_server_args_keeps_the_harness_flag_in_replace_mode() -> None:
    out = compose_server_args(
        inherited_args="--ignored",
        base_extra_args="--no-enable-prefix-caching --max-num-seqs 256",
        variant_extra_args="--kv-cache-dtype fp8",
        args_mode="replace",
    )
    assert out == "--no-enable-prefix-caching --max-num-seqs 256 --kv-cache-dtype fp8"


def test_removals_absent_from_args_drops_only_what_the_args_assert() -> None:
    kept = removals_absent_from_args(
        "--tp 8 --speculative-num-steps 3",
        ["--speculative-num-steps --enable-hierarchical-cache", "--hicache-ratio"],
    )
    assert kept == ["--enable-hierarchical-cache", "--hicache-ratio"]


def test_removals_absent_from_args_keeps_a_removal_scoped_to_another_value() -> None:
    """``--tp 4`` deletes nothing from args that run ``--tp 8``, so it stays a delta for the recipe."""
    assert removals_absent_from_args("--tp 8", ["--tp 4"]) == ["--tp 4"]
    assert removals_absent_from_args("--tp 8", ["--tp=8"]) == []
