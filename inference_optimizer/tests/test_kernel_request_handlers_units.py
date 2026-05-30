"""Unit tests for the small helpers inside ``kernel_request_handlers``.

The handler entry points are exercised by full Coordinator e2e tests, but
several module-level helpers (env coercion, candidate enrichment,
materialized-config parsing) have private branches that current tests
miss. We target those here so the helper contracts stay locked.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh


def _ensure_torch_module(monkeypatch):
    try:
        import torch
    except ModuleNotFoundError:
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda: 0),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ---------------------------------------------------------------------------
# _coerce_runtime_value
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _candidate_env_allowed
# ---------------------------------------------------------------------------

class TestCandidateEnvAllowed:
    @pytest.mark.parametrize("name", ["AWS_SECRET_ACCESS_KEY", "ANTHROPIC_API_KEY"])
    def test_sensitive_env_blocked(self, name):
        assert krh._candidate_env_allowed(name) is False

    def test_known_prefix_allowed(self):
        # Probe at least one well-known prefix without depending on the
        # full allowlist contents (which are product-internal).
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


# ---------------------------------------------------------------------------
# _is_runtime_generated_kernel
# ---------------------------------------------------------------------------

class TestRuntimeGeneratedKernel:
    def test_runtime_generated_path_treats_as_generated(self):
        # Pick whichever marker the production list publishes and ensure
        # the source-file check fires.
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
        roots = krh._REUSABLE_SOURCE_ROOTS
        if not markers or not roots:
            pytest.skip("required tables empty in build")
        marker = next(iter(markers))
        reusable_root = next(iter(roots))
        # Name matches but source lives under a reusable root → False.
        assert (
            krh._is_runtime_generated_kernel(marker, f"{reusable_root}/foo.py")
            is False
        )


# ---------------------------------------------------------------------------
# _split_server_args
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _load_candidate_metadata
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _load_materialized_workload_metadata
# ---------------------------------------------------------------------------

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
        # ARGV split is preserved.
        assert runtime["server_args_argv"] == ["--foo", "1"]
        workload = runtime["workload"]
        assert workload["tp"] == 1
        assert workload["conc"] == 16
        # Env vars passed through the allowlist guard.
        assert "TP" in out["env_vars"]


# ---------------------------------------------------------------------------
# enrichment helpers
# ---------------------------------------------------------------------------

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
        # Should not raise even though path does not exist.
        krh._enrich_candidates_artifact("", {"env_vars": {}}, trace_report_path="")


# ---------------------------------------------------------------------------
# _default_geak_budget_minutes / _geak_budget_minutes
#
# Orchestrator-side mirror of the kernel-agent default (PR #301). The
# legacy hard-coded 90 forced quick-mode timing on the orchestrator path
# even when ``install.sh`` had set ``GEAK_RUN_MODE=full``.
# ---------------------------------------------------------------------------

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
        # payload wins over both env signals
        assert krh._geak_budget_minutes({"geak_budget_min": 100}) == 100.0

    def test_env_override_beats_default(self, monkeypatch):
        monkeypatch.setenv("GEAK_RUN_MODE", "quick")
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "115")
        # env wins over the GEAK_RUN_MODE-derived default
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


# ---------------------------------------------------------------------------
# _default_kernel_batch_parallel
#
# Adaptive batch fanout. The legacy hard-coded 8 over-admitted on smaller
# pods (4-GPU labs, partial-node CI shards), letting the asyncio
# semaphore queue up siblings that Ray could not actually schedule.
# ---------------------------------------------------------------------------

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
