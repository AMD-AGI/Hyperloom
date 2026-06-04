# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the small helpers inside ``kernel_request_handlers``."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.shared_state import SharedState


def _ensure_torch_module(monkeypatch):
    try:
        import torch
    except ModuleNotFoundError:
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda: 0),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# _coerce_runtime_value
class TestCoerceRuntimeValue:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("42", 42),
            ("  17  ", 17),
            ("3.14", pytest.approx(3.14)),
            ("not-a-number", "not-a-number"),
            ("3.14.invalid", "3.14.invalid"),
            (5, 5),
            (3.5, 3.5),
            (None, None),
        ],
    )
    def test_roundtrips(self, value, expected):
        assert krh._coerce_runtime_value(value) == expected


# _backend_order
class TestBackendOrder:
    def test_documented_kernel_opt_backends_env_is_honored(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("KERNEL_OPT_BACKENDS", "geak")

        assert krh._backend_order({}) == ["geak"]


# _backend_order
class TestBackendOrder:
    def test_documented_kernel_opt_backends_env_is_honored(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("KERNEL_OPT_BACKENDS", "geak")

        assert krh._backend_order({}) == ["geak"]


# _candidate_env_allowed
class TestCandidateEnvAllowed:
    @pytest.mark.parametrize("name", ["AWS_SECRET_ACCESS_KEY", "ANTHROPIC_API_KEY"])
    def test_sensitive_env_blocked(self, name):
        assert krh._candidate_env_allowed(name) is False

    def test_known_prefix_allowed(self):
        # Probe one prefix without depending on the product-internal allowlist.
        prefixes = krh._CANDIDATE_ENV_PREFIXES
        assert prefixes  # registry not empty
        sample = next(iter(prefixes))
        assert krh._candidate_env_allowed(sample + "FOO") is True

    def test_explicit_allowlisted_key(self):
        keys = krh._CANDIDATE_ENV_KEYS
        if not keys:
            pytest.skip("no explicit allowlist entries in build")
        sample = next(iter(keys))
        assert krh._candidate_env_allowed(sample) is True


# _is_runtime_generated_kernel
class TestRuntimeGeneratedKernel:
    def test_runtime_generated_path_treats_as_generated(self):
        markers = krh._RUNTIME_GENERATED_SOURCE_MARKERS
        if not markers:
            pytest.skip("no runtime markers in build")
        marker = next(iter(markers))
        assert (
            krh._is_runtime_generated_kernel("kernel", f"/tmp/{marker}_x.py")
            is True
        )

    def test_reusable_source_root_overrides_compile_marker(self):
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        roots = krh._reusable_source_roots()
        if not markers or not roots:
            pytest.skip("required tables empty in build")
        marker = next(iter(markers))
        reusable_root = next(iter(roots))
        # Name matches but source lives under a reusable root → False.
        assert (
            krh._is_runtime_generated_kernel(marker, f"{reusable_root}/foo.py")
            is False
        )


# _split_server_args
class TestSplitServerArgs:
    def test_empty_returns_empty(self):
        assert krh._split_server_args("") == []

    def test_split_uses_shlex(self):
        argv = krh._split_server_args("--foo 1 --bar 'x y'")
        assert argv == ["--foo", "1", "--bar", "x y"]

    def test_unterminated_quote_returns_empty(self):
        # shlex.split raises ValueError on bad input; helper returns [].
        argv = krh._split_server_args('--foo "unterminated')
        assert argv == []


# _load_candidate_metadata
class TestLoadCandidateMetadata:
    def test_uses_inline_candidate(self):
        out = krh._load_candidate_metadata({"candidate": {"kernel_id": "x"}})
        assert out == {"kernel_id": "x"}

    def test_returns_empty_when_no_kernel_id(self):
        assert krh._load_candidate_metadata({}) == {}
        assert krh._load_candidate_metadata({"candidates_path": "x"}) == {}

    def test_reads_kernel_from_disk(self, tmp_path):
        candidates = tmp_path / "hot.json"
        candidates.write_text(json.dumps({
            "hot_kernels": [
                {"kernel_id": "k0", "name": "first"},
                {"kernel_id": "k1", "name": "second"},
            ],
        }))
        out = krh._load_candidate_metadata({
            "candidates_path": str(candidates),
            "kernel_id": "k1",
        })
        assert out["name"] == "second"

    def test_returns_empty_on_missing_kernel(self, tmp_path):
        candidates = tmp_path / "hot.json"
        candidates.write_text(json.dumps({"hot_kernels": []}))
        assert krh._load_candidate_metadata({
            "candidates_path": str(candidates),
            "kernel_id": "missing",
        }) == {}

    def test_returns_empty_on_bad_json(self, tmp_path):
        candidates = tmp_path / "hot.json"
        candidates.write_text("{not json")
        assert krh._load_candidate_metadata({
            "candidates_path": str(candidates),
            "kernel_id": "x",
        }) == {}


# _load_materialized_workload_metadata
class TestLoadMaterializedWorkloadMetadata:
    def test_empty_when_no_path(self):
        assert krh._load_materialized_workload_metadata("") == {}

    def test_empty_when_path_missing(self, tmp_path):
        assert krh._load_materialized_workload_metadata(str(tmp_path / "no.yaml")) == {}

    def test_parses_sglang_metadata(self, tmp_path):
        cfg = tmp_path / "magpie.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: sglang\n"
            "  model: /weights/m\n"
            "  precision: bf16\n"
            "  envs:\n"
            "    TP: 1\n"
            "    CONC: 16\n"
            "    ISL: 1024\n"
            "    OSL: 512\n"
            "    EXTRA_SGLANG_ARGS: '--foo 1'\n"
        )
        out = krh._load_materialized_workload_metadata(str(cfg))
        runtime = out["runtime_args"]
        assert runtime["framework"] == "sglang"
        assert runtime["server_args"] == "--foo 1"
        assert runtime["server_args_argv"] == ["--foo", "1"]
        workload = runtime["workload"]
        assert workload["tp"] == 1
        assert workload["conc"] == 16
        assert "TP" in out["env_vars"]

    @pytest.mark.parametrize(
        "framework,env_name,expected_args",
        [
            ("sglang", "EXTRA_SGLANG_ARGS", "--mem-fraction-static=0.8"),
            ("vllm",   "EXTRA_VLLM_ARGS",   "--gpu-memory-utilization 0.9"),
            ("atom",   "EXTRA_ATOM_ARGS",
             "--trust-remote-code --level 2 --enable-expert-parallel"),
        ],
    )
    def test_server_args_read_from_per_framework_env_key(
        self, tmp_path, framework, env_name, expected_args,
    ):
        """The handler reads the per-framework ``EXTRA_<FRAMEWORK>_ARGS`` slot, not always ``EXTRA_SGLANG_ARGS``."""
        cfg = tmp_path / f"magpie_{framework}.yaml"
        cfg.write_text(
            "benchmark:\n"
            f"  framework: {framework}\n"
            "  model: /weights/m\n"
            "  precision: bf16\n"
            "  envs:\n"
            "    TP: 4\n"
            "    CONC: 32\n"
            "    ISL: 1024\n"
            "    OSL: 1024\n"
            f"    {env_name}: '{expected_args}'\n"
        )
        out = krh._load_materialized_workload_metadata(str(cfg))
        runtime = out["runtime_args"]
        assert runtime["framework"] == framework
        assert runtime["server_args"] == expected_args, (
            f"framework={framework!r} expected server_args="
            f"{expected_args!r}; got {runtime['server_args']!r}."
        )

    def test_atom_server_args_not_read_from_extra_sglang_args(self, tmp_path):
        """When an atom YAML carries both EXTRA_ATOM_ARGS and a stray EXTRA_SGLANG_ARGS, the atom slot wins."""
        cfg = tmp_path / "magpie_atom_mixed.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: atom\n"
            "  model: /weights/m\n"
            "  precision: fp8\n"
            "  envs:\n"
            "    TP: 4\n"
            "    CONC: 32\n"
            "    ISL: 1024\n"
            "    OSL: 1024\n"
            "    EXTRA_SGLANG_ARGS: '--should-be-ignored'\n"
            "    EXTRA_ATOM_ARGS: '--trust-remote-code --level 2'\n"
        )
        out = krh._load_materialized_workload_metadata(str(cfg))
        runtime = out["runtime_args"]
        assert runtime["framework"] == "atom"
        assert runtime["server_args"] == "--trust-remote-code --level 2"
        assert "--should-be-ignored" not in runtime["server_args"]


# enrichment helpers
class TestEnrichCandidate:
    def test_enrich_candidate_runtime_metadata_setdefault_semantics(self):
        candidates = [{"kernel_id": "k", "env_vars": {"TP": "8"}}]
        metadata = {"env_vars": {"TP": "1", "CONC": "16"}, "runtime_args": {"framework": "sglang"}}
        krh._enrich_candidate_runtime_metadata(candidates, metadata)
        assert candidates[0]["env_vars"] == {"TP": "8", "CONC": "16"}
        assert candidates[0]["runtime_args"]["framework"] == "sglang"

    def test_enrich_candidate_runtime_metadata_ignores_non_dict_items(self):
        candidates = ["not a dict", {"kernel_id": "x"}]
        krh._enrich_candidate_runtime_metadata(candidates, {"env_vars": {"A": "B"}})
        assert candidates[1].get("env_vars") == {"A": "B"}

    def test_enrich_candidate_trace_report_skips_blank_path(self):
        candidates = [{"kernel_id": "k"}]
        krh._enrich_candidate_trace_report(candidates, "")
        assert "trace_report_path" not in candidates[0]

    def test_enrich_candidates_artifact_noop_when_missing_path(self):
        krh._enrich_candidates_artifact("", {"env_vars": {}}, trace_report_path="")


# atom-aware reusable kernel detection
class TestReusableSourceRootsAtom:
    """atom layout prefixes participate in cross-task kernel reuse
    alongside aiter/sglang/vllm."""

    def test_includes_atom_editable_path(self):
        # The matcher lowercases its source-file input, so the stored prefix is
        # lowercase ``/app/atom/atom/`` even though the real path is ``/app/ATOM/atom/``.
        assert any(
            "/app/atom/atom/" in r.lower()
            for r in krh._reusable_source_roots()
        )

    def test_includes_atom_site_packages_python_3_10(self):
        assert any(
            "/opt/venv/lib/python3.10/site-packages/atom/" in r
            for r in krh._reusable_source_roots()
        )

    def test_includes_atom_site_packages_python_3_12(self):
        assert any(
            "/opt/venv/lib/python3.12/site-packages/atom/" in r
            for r in krh._reusable_source_roots()
        )

    def test_atom_path_classified_as_reusable(self):
        """An atom-owned kernel source under /app/ATOM/atom/ is NOT runtime-generated even if its name matches a compile marker."""
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        if not markers:
            pytest.skip("compile markers empty in build")
        marker = next(iter(markers))
        result = krh._is_runtime_generated_kernel(
            marker, "/app/ATOM/atom/model_engine/model_runner.py",
        )
        assert result is False

    def test_non_framework_path_under_app_is_not_reusable(self):
        """A non-atom path under /app/ must NOT match the atom reusable-source-root prefix."""
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        if not markers:
            pytest.skip("compile markers empty in build")
        marker = next(iter(markers))
        # Under /app/ but not /app/ATOM/atom/ → runtime-generated (not reusable).
        result = krh._is_runtime_generated_kernel(
            marker, "/app/session_dir/runs/baseline/foo.py",
        )
        assert result is True


# run_gemm_tuning_handler
class TestRunGemmTuningHandler:
    def test_skips_non_fp8_without_kernel_agent_root(self, tmp_path):
        state = SharedState(precision="bf16", framework="sglang")
        state.save(tmp_path)

        result = asyncio.run(krh.run_gemm_tuning_handler({}, session_dir=tmp_path))

        assert result["status"] == "skipped"
        assert result["error_class"] == "fp8_only_action"

    def test_builds_task_file_input_not_task_argv(self, tmp_path, monkeypatch):
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))

        state = SharedState(
            precision="fp8",
            framework="sglang",
            model_path="/models/qwen-fp8",
            gpu_type="mi355x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            baseline_tput=4479.0,
        )
        state.save(tmp_path)
        captured: dict[str, object] = {}

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            captured["cmd"] = cmd
            captured["timeout_sec"] = timeout_sec
            input_path = cmd[cmd.index("--input-json") + 1]
            data = json.loads(Path(input_path).read_text())
            assert data["framework"] == "sglang"
            assert data["precision"] == "fp8"
            return 0, json.dumps({
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "tuned_file": "/tmp/a8w8_blockscale_tuned_gemm.csv",
            }), ""

        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(krh.run_gemm_tuning_handler(
            {
                "benchmark_script": "/workspace/run_sglang_test.sh",
                "dry_run": True,
                "task_id": "t1",
            },
            session_dir=tmp_path,
        ))

        assert result["status"] == "ok"
        cmd_text = " ".join(captured["cmd"])  # type: ignore[arg-type]
        assert "run_sglang_test" not in cmd_text
        assert "gemm_a8w8_blockscale_tune" not in cmd_text
        assert "--input-json" in captured["cmd"]  # type: ignore[operator]

    def test_generates_isolated_benchmark_script_when_missing(self, tmp_path, monkeypatch):
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))

        state = SharedState(
            precision="fp8",
            framework="sglang",
            model_path="/models/qwen-fp8",
            gpu_type="mi355x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            baseline_tput=4479.0,
        )
        state.save(tmp_path)

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = cmd[cmd.index("--input-json") + 1]
            data = json.loads(Path(input_path).read_text())
            bench = Path(data["benchmark_script"])
            text = bench.read_text()
            assert bench.name == "geak_gemm_benchmark.sh"
            assert "PORT=\"${PORT:-18888}\"" in text
            assert "pgrep" not in text
            assert data["benchmark_script"].endswith("geak_gemm_benchmark.sh")
            return 0, json.dumps({
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.1,
                "tuned_file": "/tmp/tuned.csv",
            }), ""

        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(krh.run_gemm_tuning_handler(
            {"dry_run": True, "task_id": "auto"},
            session_dir=tmp_path,
        ))

        assert result["status"] == "ok"


# _default_geak_budget_minutes / _geak_budget_minutes — orchestrator-side mirror
# of the kernel-agent default (PR #301); the legacy 90 forced quick-mode timing.
class TestDefaultGeakBudgetMinutes:
    @pytest.mark.parametrize(
        "geak_run_mode, expected",
        [
            (None, 130.0),       # unset -> full default
            ("", 130.0),         # empty -> full default
            ("full", 130.0),
            ("FULL", 130.0),     # case-insensitive
            ("  full  ", 130.0), # whitespace tolerated
            ("garbage", 130.0),  # unknown values fall back to full
            ("quick", 70.0),
            ("QUICK", 70.0),
            ("  quick  ", 70.0),
        ],
    )
    def test_tracks_geak_run_mode(self, monkeypatch, geak_run_mode, expected):
        if geak_run_mode is None:
            monkeypatch.delenv("GEAK_RUN_MODE", raising=False)
        else:
            monkeypatch.setenv("GEAK_RUN_MODE", geak_run_mode)
        assert krh._default_geak_budget_minutes() == expected


class TestGeakBudgetMinutes:
    def test_payload_override_wins(self, monkeypatch):
        monkeypatch.setenv("GEAK_RUN_MODE", "quick")
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "500")
        assert krh._geak_budget_minutes({"geak_budget_min": 100}) == 100.0

    def test_env_override_beats_default(self, monkeypatch):
        monkeypatch.setenv("GEAK_RUN_MODE", "quick")
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "115")
        assert krh._geak_budget_minutes({}) == 115.0

    @pytest.mark.parametrize("geak_run_mode, expected", [
        ("full", 130.0),
        ("quick", 70.0),
    ])
    def test_falls_through_to_helper_when_no_overrides(
        self, monkeypatch, geak_run_mode, expected,
    ):
        monkeypatch.delenv("HYPERLOOM_GEAK_BUDGET_MIN", raising=False)
        monkeypatch.setenv("GEAK_RUN_MODE", geak_run_mode)
        assert krh._geak_budget_minutes({}) == expected

    def test_empty_env_value_falls_through_to_helper(self, monkeypatch):
        # Pre-fix code would let "" propagate into ``float("")`` and raise.
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "")
        monkeypatch.delenv("GEAK_RUN_MODE", raising=False)
        assert krh._geak_budget_minutes({}) == 130.0


# _default_kernel_batch_parallel — adaptive batch fanout; the legacy 8
# over-admitted on smaller pods (4-GPU labs, partial-node CI shards).
class TestDefaultKernelBatchParallel:
    @pytest.fixture
    def patch_torch(self, monkeypatch):
        """Returns a setter that overrides ``torch.cuda.device_count`` and
        ``$KERNEL_AGENT_NUM_GPUS`` for the helper under test."""
        torch = _ensure_torch_module(monkeypatch)

        def _set(n_gpus, per_task=None):
            monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)
            if per_task is None:
                monkeypatch.delenv("KERNEL_AGENT_NUM_GPUS", raising=False)
            else:
                monkeypatch.setenv("KERNEL_AGENT_NUM_GPUS", str(per_task))

        return _set

    @pytest.mark.parametrize(
        "n_gpus, per_task, expected",
        [
            # Exact full-node match (8 GPU, 1 GPU/task) -> cap kicks in at 8.
            (8, 1, 8),
            # Partial node -> floor at the visible-GPU count.
            (4, 1, 4),
            # 8-GPU node with 4-GPU GEAK reservations -> 2 concurrent.
            (8, 4, 2),
            # 4-GPU pod with 2-GPU per task -> 2 concurrent.
            (4, 2, 2),
            # Larger-than-cap node -> cap still kicks in.
            (16, 1, 8),
            # Per-task larger than visible -> floor at 1 (don't stall the
            # batch with semaphore=0).
            (1, 4, 1),
        ],
    )
    def test_scales_with_visible_gpus(
        self, patch_torch, n_gpus, per_task, expected,
    ):
        patch_torch(n_gpus, per_task=per_task)
        assert krh._default_kernel_batch_parallel() == expected

    def test_per_task_unset_defaults_to_one(self, patch_torch):
        patch_torch(4, per_task=None)
        assert krh._default_kernel_batch_parallel() == 4

    def test_per_task_invalid_falls_back_to_one(self, patch_torch):
        patch_torch(4, per_task="not-an-int")
        assert krh._default_kernel_batch_parallel() == 4

    def test_zero_visible_gpus_returns_legacy_fallback(self, patch_torch):
        patch_torch(0)
        assert (
            krh._default_kernel_batch_parallel()
            == krh._DEFAULT_KERNEL_BATCH_PARALLEL
        )

    def test_torch_failure_returns_legacy_fallback(self, monkeypatch):
        torch = _ensure_torch_module(monkeypatch)

        def _boom():
            raise RuntimeError("driver init failed")

        monkeypatch.setattr(torch.cuda, "device_count", _boom)
        monkeypatch.delenv("KERNEL_AGENT_NUM_GPUS", raising=False)
        assert (
            krh._default_kernel_batch_parallel()
            == krh._DEFAULT_KERNEL_BATCH_PARALLEL
        )


# ---------------------------------------------------------------------------
# _should_parallelize_backends
#
# GPU-rich mode: race GEAK against the OOB ladder per kernel whenever the
# node can fit ONE kernel's GEAK + OOB ladder side-by-side
# (``visible_gpus >= 2 * per_task``). The decision is independent of
# ``num_candidates`` -- batch width is throttled separately by the batch
# handler's concurrency cap. Below ``2 * per_task`` there is no room for a
# second ladder, so keep the sequential GEAK-first / OOB-fallback ladder.
# Operators / tests can force the decision via payload or env.
# ---------------------------------------------------------------------------

class TestShouldParallelizeBackends:
    @pytest.fixture
    def patch_torch(self, monkeypatch):
        """Override ``torch.cuda.device_count`` + ``$KERNEL_AGENT_NUM_GPUS``
        and clear the env override so the GPU-aware math is exercised."""
        torch = _ensure_torch_module(monkeypatch)

        def _set(n_gpus, per_task=None):
            monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)
            if per_task is None:
                monkeypatch.delenv("KERNEL_AGENT_NUM_GPUS", raising=False)
            else:
                monkeypatch.setenv("KERNEL_AGENT_NUM_GPUS", str(per_task))
            monkeypatch.delenv("KERNEL_OPT_PARALLEL_BACKENDS", raising=False)

        return _set

    @pytest.mark.parametrize(
        "n_gpus, per_task, num_candidates, expected",
        [
            # 1 GPU/task: need room for both ladders -> visible_gpus >= 2.
            # The kernel count is irrelevant (batch width is capped elsewhere).
            (8, 1, 3, True),     # 8 >= 2
            (8, 1, 7, True),     # 8 >= 2 (kernel count no longer gates)
            (8, 1, 100, True),   # 8 >= 2 even when candidates >> gpus
            (2, 1, 1, True),     # 2 >= 2 boundary
            (1, 1, 1, False),    # 1 < 2 -> no room for a second ladder
            # Multi-GPU reservations: need room for TWO per_task backends.
            (8, 4, 1, True),     # 8 >= 8 boundary
            (8, 4, 5, True),     # 8 >= 8 (candidate count irrelevant)
            (8, 8, 1, False),    # 8 < 16 -> can't fit a 2nd 8-GPU backend
            (16, 8, 1, True),    # 16 >= 16
        ],
    )
    def test_gpu_aware_threshold(
        self, patch_torch, n_gpus, per_task, num_candidates, expected,
    ):
        patch_torch(n_gpus, per_task=per_task)
        assert krh._should_parallelize_backends({}, num_candidates) is expected

    def test_non_positive_candidates_is_false(self, patch_torch):
        patch_torch(64, per_task=1)  # plenty of GPUs
        assert krh._should_parallelize_backends({}, 0) is False
        assert krh._should_parallelize_backends({}, -1) is False

    def test_zero_visible_gpus_is_false(self, patch_torch):
        patch_torch(0, per_task=1)
        assert krh._should_parallelize_backends({}, 1) is False

    def test_torch_unknown_is_false(self, monkeypatch):
        torch = _ensure_torch_module(monkeypatch)

        def _boom():
            raise RuntimeError("driver init failed")

        monkeypatch.setattr(torch.cuda, "device_count", _boom)
        monkeypatch.delenv("KERNEL_OPT_PARALLEL_BACKENDS", raising=False)
        assert krh._should_parallelize_backends({}, 1) is False

    def test_payload_override_enables_below_threshold(self, patch_torch):
        patch_torch(1, per_task=1)  # GPU-aware math is False (1 < 2*1)
        assert krh._should_parallelize_backends(
            {"parallel_backends": True}, 5,
        ) is True
        assert krh._should_parallelize_backends(
            {"parallel_backends": "on"}, 5,
        ) is True

    def test_payload_override_disables_above_threshold(self, patch_torch):
        patch_torch(64, per_task=1)  # GPU-aware math would say True
        assert krh._should_parallelize_backends(
            {"parallel_backends": False}, 1,
        ) is False
        assert krh._should_parallelize_backends(
            {"parallel_backends": "no"}, 1,
        ) is False

    def test_env_override(self, patch_torch, monkeypatch):
        patch_torch(1, per_task=1)  # GPU-aware math is False (1 < 2*1)
        monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "1")
        assert krh._should_parallelize_backends({}, 5) is True
        monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "0")
        assert krh._should_parallelize_backends({}, 1) is False


# ---------------------------------------------------------------------------
# _run_optimization_batch concurrency cap (parallel-backends mode)
#
# Each parallel kernel launches TWO before_kernel_opt rocprof subprocesses
# (GEAK + OOB) *before* entering Ray, bypassing the Ray GPU lease. The batch
# caps concurrent kernels to ``visible_gpus // (2 * per_task)`` so those
# pre-Ray profilers (and the Ray tasks that follow) stay within the real GPU
# budget even when ``max_parallel`` is set higher.
# ---------------------------------------------------------------------------

class TestBatchParallelConcurrencyCap:
    def test_caps_concurrency_to_gpu_budget(self, tmp_path, monkeypatch):
        # 8 visible GPUs, 1 GPU/task -> safe concurrency = 8 // (2*1) = 4.
        monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 8)
        monkeypatch.setenv("KERNEL_AGENT_NUM_GPUS", "1")  # per_task = 1
        monkeypatch.delenv("KERNEL_OPT_PARALLEL_BACKENDS", raising=False)

        state = {"in_flight": 0, "peak": 0}

        async def fake_sequence(
            base_payload, candidate, *, session_dir, parallel_backends=False,
        ):
            assert parallel_backends is True
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            try:
                await asyncio.sleep(0.02)  # hold the slot so siblings overlap
            finally:
                state["in_flight"] -= 1
            return {
                "status": "ok",
                "kernel_id": candidate["kernel_id"],
                "source_file": candidate.get("source_file"),
                "proposal": {"decision": "REVERT"},
                "verification": {"micro_speedup": 1.0},
            }

        monkeypatch.setattr(krh, "_run_kernel_backend_sequence", fake_sequence)
        candidates = [
            {"kernel_id": f"k{i}", "source_file": f"/p/{i}.py",
             "reusable_native_kernel": True}
            for i in range(10)
        ]
        out = asyncio.run(krh._run_optimization_batch(
            payload={"candidates_path": "/dummy", "max_parallel": 10},
            candidates=candidates,
            session_dir=tmp_path,
        ))

        assert out["parallel_backends"] is True
        # max_parallel echoes the capped value (10 -> 4).
        assert out["max_parallel"] == 4
        # Cap binds: 4 kernels * 2 ladders = 8 pre-Ray profilers == 8 GPUs.
        assert state["peak"] == 4, state["peak"]

    def test_no_cap_when_gpu_count_unknown(self, tmp_path, monkeypatch):
        # torch can't report a count (None) -> cap math is skipped so the
        # operator-supplied max_parallel is preserved (matches CI / mocks).
        monkeypatch.setattr(krh, "_visible_gpu_count", lambda: None)
        monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "1")  # force parallel

        async def fake_sequence(
            base_payload, candidate, *, session_dir, parallel_backends=False,
        ):
            return {
                "status": "ok",
                "kernel_id": candidate["kernel_id"],
                "source_file": candidate.get("source_file"),
                "proposal": {"decision": "REVERT"},
                "verification": {"micro_speedup": 1.0},
            }

        monkeypatch.setattr(krh, "_run_kernel_backend_sequence", fake_sequence)
        candidates = [
            {"kernel_id": f"k{i}", "source_file": f"/p/{i}.py",
             "reusable_native_kernel": True}
            for i in range(3)
        ]
        out = asyncio.run(krh._run_optimization_batch(
            payload={"candidates_path": "/dummy", "max_parallel": 7},
            candidates=candidates,
            session_dir=tmp_path,
        ))

        assert out["parallel_backends"] is True
        assert out["max_parallel"] == 7  # uncapped (visible GPU count unknown)


# ---------------------------------------------------------------------------
# _reconcile_kernel_id
class TestReconcileKernelId:
    CANDS = [
        {"kernel_id": "k001", "name": "aten::mm"},
        {"kernel_id": "k010", "name": "aiter::rmsnorm"},
    ]

    def test_exact_id_kept(self):
        assert krh._reconcile_kernel_id("k010", self.CANDS) == "k010"

    def test_name_match_kept(self):
        # An exact operator-name match is canonicalized to the candidate id so
        # downstream lifecycle/results are keyed by the stable k00x id.
        assert krh._reconcile_kernel_id("aten::mm", self.CANDS) == "k001"

    def test_normalized_prefix_resolves_to_real_id(self):
        assert krh._reconcile_kernel_id("kn001", self.CANDS) == "k001"
        assert krh._reconcile_kernel_id("rn010", self.CANDS) == "k010"

    def test_missing_id_falls_back_to_first(self):
        assert krh._reconcile_kernel_id("", self.CANDS) == "k001"
        assert krh._reconcile_kernel_id(None, self.CANDS) == "k001"

    def test_hallucinated_id_is_left_for_guard_or_cli_skip(self):
        # Non-empty ids are never guessed. A pure hallucination should flow to
        # the reusable-native guard / CLI skip path rather than being mapped to
        # an unrelated candidate.
        assert (
            krh._reconcile_kernel_id("aiter.silu_and_mul", self.CANDS)
            == "aiter.silu_and_mul"
        )
        assert (
            krh._reconcile_kernel_id(
                "framework_sglang_silu_and_mul_m64", self.CANDS
            )
            == "framework_sglang_silu_and_mul_m64"
        )


# _resolve_candidate_id / _all_kernel_candidates — canonicalizes an aliased id
# against the full hot ∪ skipped set (no fallback) so the reusable-native guard
# rejects the real k00x rather than the raw hallucinated alias.
class TestResolveCandidateId:
    SKIPPED = [
        {"kernel_id": "k001", "name": "aten::mm",
         "reusable_native_kernel": False, "source_file": ""},
        {"kernel_id": "k003", "name": "aten::mm",
         "reusable_native_kernel": False, "source_file": ""},
        {"kernel_id": "k010", "name": "aiter::rmsnorm",
         "reusable_native_kernel": False, "source_file": ""},
    ]

    def test_exact_id(self):
        assert krh._resolve_candidate_id("k003", self.SKIPPED) == "k003"

    def test_kn_prefix_alias_canonicalized(self):
        assert krh._resolve_candidate_id("kn001", self.SKIPPED) == "k001"
        assert krh._resolve_candidate_id("rn010", self.SKIPPED) == "k010"

    def test_non_unique_or_nonroutable_name_not_resolved(self):
        # ``aten::mm`` is non-unique and non-routable -> cannot disambiguate,
        # so leave it untouched (returns "") rather than guess a k00x.
        assert krh._resolve_candidate_id("aten::mm", self.SKIPPED) == ""

    def test_pure_hallucination_returns_empty(self):
        assert krh._resolve_candidate_id("aiter.silu_and_mul", self.SKIPPED) == ""

    def test_empty_request_returns_empty(self):
        assert krh._resolve_candidate_id("", self.SKIPPED) == ""
        assert krh._resolve_candidate_id(None, self.SKIPPED) == ""


class TestAllKernelCandidates:
    def test_union_of_hot_and_skipped(self, tmp_path):
        cp = tmp_path / "kc.json"
        cp.write_text(json.dumps({
            "hot_kernels": [{"kernel_id": "k005", "name": "moe"}],
            "skipped_kernels": [{"kernel_id": "k001", "name": "aten::mm"}],
        }), encoding="utf-8")
        out = krh._all_kernel_candidates({"candidates_path": str(cp)})
        assert {c["kernel_id"] for c in out} == {"k005", "k001"}

    def test_missing_path_returns_empty(self):
        assert krh._all_kernel_candidates({}) == []
