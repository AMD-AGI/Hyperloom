# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplemental coverage for _grid_runner pure helpers: compatibility filter model-class drop, runtime override env branches, report parsing, and per-variant yaml env injection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from hyperloom.inference_optimizer.grid_server_args import compose_server_args, remove_server_args
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._grid_variant_filter import ServingStack
from hyperloom.orchestrator.actions.executors import _benchmark_interpreter as bi
from hyperloom.orchestrator.actions.executors._benchmark_runtime import apply_runtime_benchmark_overrides

# Patch compatibility-filter helpers in the ``_grid_variant_filter`` sibling, where apply_compatibility_filter
# resolves them (not via the re-export).
from hyperloom.orchestrator.actions.executors import _grid_variant_filter as vf


def _variant(name: str, args: str = "", envs: dict | None = None) -> gr.GridVariant:
    return gr.GridVariant(name=name, extra_server_args=args, extra_envs=envs or {})


def test_validate_magpie_python_override_requires_python(tmp_path):
    py = tmp_path / "python3"
    py.write_text("", encoding="utf-8")
    bash = tmp_path / "bash"
    bash.write_text("", encoding="utf-8")
    assert gr._validate_magpie_python_override(str(py)) == str(py)
    with pytest.raises(ValueError, match="Python interpreter"):
        gr._validate_magpie_python_override(str(bash))
    assert gr._validate_magpie_python_override(sys.executable).endswith(Path(sys.executable).name)


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
    reason = gr.ungrounded_env_file_reason(v.extra_envs)
    assert reason and "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE" in reason
    assert str(tmp_path / "nope.csv") in reason


def test_ungrounded_env_file_reason_passes_authored_file(tmp_path) -> None:
    csv = tmp_path / "tuned.csv"
    csv.write_text("gfx,M,N,K,splitK\ngfx950,64,5120,5120,2\n", encoding="utf-8")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": str(csv)})
    assert gr.ungrounded_env_file_reason(v.extra_envs) is None


def test_ungrounded_env_file_reason_accepts_colon_joined_existing_files(tmp_path) -> None:
    # A set AITER_CONFIG_* is aiter's SET form: a `:`-joined list, each element loaded (csvs_aiter_will_load).
    # Both files on disk is a valid variant and must not be dropped.
    a = tmp_path / "t.csv"
    a.write_text("gfx,M,N,K\ngfx950,1,2,3\n", encoding="utf-8")
    b = tmp_path / "t2.csv"
    b.write_text("gfx,M,N,K\ngfx950,4,5,6\n", encoding="utf-8")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": f"{a}:{b}"})
    assert gr.ungrounded_env_file_reason(v.extra_envs) is None


def test_ungrounded_env_file_reason_drops_colon_joined_missing_element(tmp_path) -> None:
    a = tmp_path / "t.csv"
    a.write_text("gfx,M,N,K\ngfx950,1,2,3\n", encoding="utf-8")
    missing = str(tmp_path / "never_authored.csv")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": f"{a}:{missing}"})
    reason = gr.ungrounded_env_file_reason(v.extra_envs)
    assert reason and missing in reason and str(a) not in reason.split("element(s)")[1].split(" do not exist")[0]


def test_ungrounded_env_file_reason_treats_an_unreadable_path_as_missing(monkeypatch) -> None:
    # An unreadable parent or an overlong name makes Path.is_file() raise; the variant is dropped, the grid runs on.
    def raise_eacces(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(gr.Path, "is_file", raise_eacces)
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": "/root/private/tuned.csv"})
    reason = gr.ungrounded_env_file_reason(v.extra_envs)
    assert reason and "/root/private/tuned.csv" in reason


def test_run_grid_fast_fail_drops_a_variant_whose_config_file_is_missing(tmp_path, monkeypatch) -> None:
    # The backstop of the admission check: run_grid's fast-fail drops it before boot, whatever the framework.
    monkeypatch.setenv("FRAMEWORK", "sglang")
    v = _variant("gemm", envs={"AITER_CONFIG_GEMM_BF16": str(tmp_path / "never_written.csv")})
    assert "AITER_CONFIG_GEMM_BF16" in (gr.unsupported_capability_reason(v) or "")


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
    assert gr.ungrounded_env_file_reason(v.extra_envs) is None


# -- unsupported_capability_reason (env-flag build probe) -----------------
def _clear_cap_cache() -> None:
    gr._CAP_PROBE_CACHE.clear()


def test_capability_reason_noop_when_flag_absent(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    _clear_cap_cache()
    # Probe must NOT even run when the flag isn't set on the variant.
    monkeypatch.setattr(
        gr,
        "_probe_vllm_aiter_shared_expert_unsupported",
        lambda: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    assert gr.unsupported_capability_reason(_variant("plain")) is None


def test_capability_reason_noop_for_non_vllm_framework(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "sglang")
    _clear_cap_cache()
    v = _variant("se", envs={"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1"})
    assert gr.unsupported_capability_reason(v) is None


def test_capability_reason_drops_when_module_missing(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    _clear_cap_cache()
    monkeypatch.setattr(
        gr,
        "_probe_vllm_aiter_shared_expert_unsupported",
        lambda: "missing module(s): vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe",
    )
    v = _variant("se", envs={"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1"})
    reason = gr.unsupported_capability_reason(v)
    assert reason is not None and "rocm_aiter_fused_moe" in reason


def test_capability_reason_falsey_flag_not_probed(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    _clear_cap_cache()
    monkeypatch.setattr(
        gr,
        "_probe_vllm_aiter_shared_expert_unsupported",
        lambda: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    v = _variant("se", envs={"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0"})
    assert gr.unsupported_capability_reason(v) is None


def test_probe_caches_ok_and_unsupported(monkeypatch) -> None:
    _clear_cap_cache()
    calls = {"n": 0}

    class _Proc:
        def __init__(self, out: str) -> None:
            self.stdout = out
            self.stderr = ""

    def fake_run(*_a, **_k):
        calls["n"] += 1
        return _Proc(json.dumps({"status": "ok"}))

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    assert gr._probe_vllm_aiter_shared_expert_unsupported() is None
    # Second call must hit the cache (no second subprocess).
    assert gr._probe_vllm_aiter_shared_expert_unsupported() is None
    assert calls["n"] == 1


def test_probe_unknown_not_cached(monkeypatch) -> None:
    _clear_cap_cache()

    class _Proc:
        stdout = json.dumps({"status": "unknown"})
        stderr = ""

    monkeypatch.setattr(gr.subprocess, "run", lambda *_a, **_k: _Proc())
    assert gr._probe_vllm_aiter_shared_expert_unsupported() is None
    assert "vllm" not in gr._CAP_PROBE_CACHE


def test_probe_swallows_subprocess_error(monkeypatch) -> None:
    _clear_cap_cache()

    def boom(*_a, **_k):
        raise OSError("no python3")

    monkeypatch.setattr(gr.subprocess, "run", boom)
    assert gr._probe_vllm_aiter_shared_expert_unsupported() is None


# -- _resolve_probe_python / probe interpreter selection ------------------
def test_resolve_probe_python_prefers_magpie_interpreter(monkeypatch) -> None:
    # The harness interpreter is used directly; no vllm-exe resolution is tried.
    monkeypatch.setattr(bi, "_resolve_magpie_python", lambda _env=None: "/srv/venv/bin/python")
    monkeypatch.setattr(
        bi.shutil, "which", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("which() should not be called"))
    )
    assert bi._resolve_probe_python() == "/srv/venv/bin/python"


def test_resolve_probe_python_falls_back_to_vllm_venv(monkeypatch) -> None:
    # magpie_python is the canonical default -> pin the venv that backs ``vllm serve``.
    monkeypatch.setattr(bi, "_resolve_magpie_python", lambda _env=None: "/opt/venv/bin/python")
    monkeypatch.setattr(bi.shutil, "which", lambda name, **_k: "/other/venv/bin/vllm" if name == "vllm" else None)
    monkeypatch.setattr(bi.os.path, "exists", lambda p: p == "/other/venv/bin/python")
    assert bi._resolve_probe_python() == "/other/venv/bin/python"


def test_resolve_probe_python_no_bare_python3_fallback(monkeypatch) -> None:
    # With no resolvable vllm exe, fall back to the canonical magpie default — never a bare "python3".
    monkeypatch.setattr(bi, "_resolve_magpie_python", lambda _env=None: "/opt/venv/bin/python")
    monkeypatch.setattr(bi.shutil, "which", lambda *_a, **_k: None)
    assert bi._resolve_probe_python() == "/opt/venv/bin/python"


def test_probe_invokes_resolved_interpreter(monkeypatch) -> None:
    # The capability probe must run under the resolved interpreter, not "python3".
    _clear_cap_cache()
    monkeypatch.setattr(gr, "_resolve_probe_python", lambda: "/srv/venv/bin/python")
    seen: dict = {}

    class _Proc:
        stdout = json.dumps({"status": "ok"})
        stderr = ""

    def fake_run(cmd, *_a, **_k):
        seen["cmd"] = list(cmd)
        return _Proc()

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    gr._probe_vllm_aiter_shared_expert_unsupported()
    assert seen["cmd"][0] == "/srv/venv/bin/python"
    assert seen["cmd"][0] != "python3"


# -- apply_runtime_benchmark_overrides ------------------------------------
def test_runtime_overrides_model_precision_and_gpu_no_framework_agent(monkeypatch) -> None:
    monkeypatch.setenv("PRECISION", "fp8")
    for k in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench: dict = {}  # no framework -> benchmark_script popped
    apply_runtime_benchmark_overrides(
        bench,
        model_path="/models/x",
        gpu_type="mi355x",
    )
    assert bench["model"] == "/models/x"
    assert bench["precision"] == "fp8"
    assert bench["runner_type"] == "mi355x"
    assert "benchmark_script" not in bench


def test_runtime_overrides_framework_pins_generic_script(monkeypatch) -> None:
    for k in ("PRECISION", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench = {"framework": "sglang"}
    apply_runtime_benchmark_overrides(bench, gpu_type="mi300x")
    assert bench["benchmark_script"] == "sglang_mi300x.sh"


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


def test_runtime_overrides_explicit_benchmark_script_wins(monkeypatch) -> None:
    for k in ("PRECISION", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench = {"framework": "vllm"}
    apply_runtime_benchmark_overrides(
        bench,
        gpu_type="mi300x",
        benchmark_script="custom.sh",
    )
    assert bench["benchmark_script"] == "custom.sh"


# -- _parse_report ---------------------------------------------------------
def test_parse_report_missing(tmp_path: Path) -> None:
    assert gr._parse_report(tmp_path) is None


def test_parse_report_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text("{bad", encoding="utf-8")
    assert gr._parse_report(tmp_path) is None


def test_parse_report_non_dict(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text("[1,2,3]", encoding="utf-8")
    assert gr._parse_report(tmp_path) is None


def test_parse_report_valid(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text(
        json.dumps({"output_throughput": 100.0}),
        encoding="utf-8",
    )
    assert gr._parse_report(tmp_path) == {"output_throughput": 100.0}


# -- _build_variant_yaml ---------------------------------------------------
def test_build_variant_yaml_injects_extra_envs(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}),
        encoding="utf-8",
    )
    variant = _variant("v1", "--foo 1", envs={"USE_AITER": "1"})
    out = gr._build_variant_yaml(
        gr.GridRecipe(base_yaml_path=base),
        variant,
        output_subdir=tmp_path / "v1",
    )
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    envs = cfg["benchmark"]["envs"]
    assert envs["USE_AITER"] == "1"
    # variant + base server args merged into the framework's args env
    arg_key = server_args_env_name("sglang")
    assert "--foo 1" in envs[arg_key]


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


def test_compose_server_args_launches_with_the_harness_flags_the_baseline_ran_with() -> None:
    out = compose_server_args(
        inherited_args="--no-enable-prefix-caching --block-size 128",
        base_extra_args="--no-enable-prefix-caching",
        variant_extra_args="--kv-cache-dtype fp8 --no-enable-prefix-caching",
    )
    assert "--no-enable-prefix-caching" in out
    assert "--block-size 128" in out
    assert "--kv-cache-dtype fp8" in out


def test_compose_server_args_keeps_the_harness_flag_in_replace_mode() -> None:
    out = compose_server_args(
        inherited_args="--ignored",
        base_extra_args="--no-enable-prefix-caching --max-num-seqs 256",
        variant_extra_args="--kv-cache-dtype fp8",
        args_mode="replace",
    )
    assert out == "--no-enable-prefix-caching --max-num-seqs 256 --kv-cache-dtype fp8"


def test_launch_keeps_what_the_persisted_recipe_strips() -> None:
    from hyperloom.inference_optimizer.grid_server_args import strip_benchmark_harness_flags

    launched = compose_server_args(base_extra_args="--no-enable-prefix-caching --max-num-seqs 256")
    assert "--no-enable-prefix-caching" in launched
    assert strip_benchmark_harness_flags(launched) == "--max-num-seqs 256"


def test_remove_server_args_accepts_multi_flag_string() -> None:
    out = remove_server_args(
        "--flag-a --flag-b --flag-c 3 --keep 4",
        "--flag-a --flag-b --flag-c",
    )
    assert out == "--keep 4"


def test_remove_server_args_keeps_a_sibling_json_value_parseable() -> None:
    """Removing one flag must not corrupt a JSON-valued sibling."""
    args = (
        '--online_quant_config {"global_quant_config":"ptpc_fp8",'
        '"exclude_layer":["*.mlp.gate","*expert*"]} '
        "--no-enable-prefix-caching --tp 8"
    )
    out = remove_server_args(args, ["--no-enable-prefix-caching"])
    tokens = out.split(" ")
    assert tokens[0] == "--online_quant_config"
    assert json.loads(tokens[1]) == {
        "global_quant_config": "ptpc_fp8",
        "exclude_layer": ["*.mlp.gate", "*expert*"],
    }
    assert tokens[2:] == ["--tp", "8"]


def test_remove_server_args_unbalanced_brace_drops_only_its_own_flag() -> None:
    """A stray ``}`` is not a JSON blob and must not take the tail with it."""
    out = remove_server_args("--foo a} --tp 8 --max-num-seqs 64", ["--foo"])
    assert out == "--tp 8 --max-num-seqs 64"
