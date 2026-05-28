"""Unit tests for the small helpers inside ``kernel_request_handlers``.

The handler entry points are exercised by full Coordinator e2e tests, but
several module-level helpers (env coercion, candidate enrichment,
materialized-config parsing) have private branches that current tests
miss. We target those here so the helper contracts stay locked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh


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
# Phase 2.5: atom-aware reusable kernel detection
# ---------------------------------------------------------------------------

class TestReusableSourceRootsAtom:
    """Phase 2.5 — atom layout prefixes participate in cross-task
    kernel reuse alongside aiter/sglang/vllm."""

    def test_includes_atom_editable_path(self):
        # The matcher (``_is_runtime_generated_kernel``) lowercases its
        # source-file input before substring matching, so the stored
        # prefix is lowercase ``/app/atom/atom/`` even though the real
        # filesystem path is ``/app/ATOM/atom/``. PolicyGate uses a
        # case-sensitive ``startswith`` and keeps the canonical case in
        # ``framework_paths._DEFAULT_SOURCE_ROOTS`` separately.
        assert any(
            "/app/atom/atom/" in r.lower() for r in krh._REUSABLE_SOURCE_ROOTS
        )

    def test_includes_atom_site_packages_python_3_10(self):
        assert any(
            "/opt/venv/lib/python3.10/site-packages/atom/" in r
            for r in krh._REUSABLE_SOURCE_ROOTS
        )

    def test_includes_atom_site_packages_python_3_12(self):
        assert any(
            "/opt/venv/lib/python3.12/site-packages/atom/" in r
            for r in krh._REUSABLE_SOURCE_ROOTS
        )

    def test_atom_path_classified_as_reusable(self):
        """A representative atom-owned kernel source (model_runner.py) at
        /app/ATOM/atom/ must NOT be flagged as runtime-generated even
        when its kernel name matches an inductor / triton compile
        marker. This is the exact condition the reusable-roots check
        guards against in ``_is_runtime_generated_kernel``."""
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        if not markers:
            pytest.skip("compile markers empty in build")
        marker = next(iter(markers))
        result = krh._is_runtime_generated_kernel(
            marker, "/app/ATOM/atom/model_engine/model_runner.py",
        )
        # Same logic as the existing sglang/vllm test (line 84): the
        # name marker would normally classify as runtime-generated, but
        # the source path lives under a reusable root so the kernel is
        # treated as patchable framework code.
        assert result is False

    def test_non_framework_path_under_app_is_not_reusable(self):
        """A non-atom path under /app/ (e.g. /app/session_dir/runs/...)
        must NOT match the atom reusable-source-root prefix — only
        /app/ATOM/atom/ specifically."""
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        if not markers:
            pytest.skip("compile markers empty in build")
        marker = next(iter(markers))
        # Path under /app/ but NOT /app/ATOM/atom/ — must classify as
        # runtime-generated (i.e. not reusable).
        result = krh._is_runtime_generated_kernel(
            marker, "/app/session_dir/runs/baseline/foo.py",
        )
        assert result is True
