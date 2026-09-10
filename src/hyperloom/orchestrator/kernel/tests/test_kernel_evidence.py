# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A server log is only a shape source if it dispatched through aiter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel import kernel_evidence as ke

# A real aiter hit line, copied from a fleet server.log.
HIT = (
    "shape is M:{m}, N:512, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' "
    "bias=False, scaleAB=False, bpreshuffle=False found padded_M: 16384, N:512, "
    "K:4096 is tuned on cu_num = 256 in /tmp/aiter_configs/bf16_tuned_gemm.csv, "
    "libtype is opus, kernel name is opus_gemm\n"
)
MISS = "shape is M:{m}, N:512, K:4096 not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, using default\n"
QUIET = "INFO server started on 0.0.0.0:8000\nINFO warmup complete\n"
# A MoE dispatch line: no dense "shape is M:" anywhere, but the log is fully informative for the routing decisions
# kernelforge makes off the same path.
MOE = "[aiter] [fused_moe] using ck_moe_2stages for ('bf16', 'bf16', 128, 8, 1, 0, 0)\n"


class _State:
    def __init__(self, current_best=None, last_baseline=None):
        self.current_best = current_best or {}
        self.last_baseline = last_baseline or {}


def _log(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestEvidenceDetection:
    def test_a_hit_line_is_evidence(self, tmp_path):
        assert ke._log_has_aiter_evidence(_log(tmp_path / "a.log", HIT.format(m=128)))

    def test_a_miss_line_is_also_evidence(self, tmp_path):
        # A miss still proves the process routed a GEMM through aiter, which is what makes the log a usable shape
        # source.
        assert ke._log_has_aiter_evidence(_log(tmp_path / "a.log", MISS.format(m=128)))

    def test_a_quiet_log_is_not_evidence(self, tmp_path):
        assert not ke._log_has_aiter_evidence(_log(tmp_path / "a.log", QUIET))

    def test_a_missing_file_is_not_evidence(self, tmp_path):
        assert not ke._log_has_aiter_evidence(tmp_path / "nope.log")

    def test_a_marker_straddling_a_chunk_boundary_is_still_found(self, tmp_path, monkeypatch):
        # Fleet logs are ~17MB, so the scan is chunked; the overlap must cover a marker split across two reads.
        monkeypatch.setattr(ke, "_LOG_SCAN_CHUNK", 16)
        text = "x" * 10 + HIT.format(m=128)
        assert ke._log_has_aiter_evidence(_log(tmp_path / "a.log", text))

    def test_evidence_late_in_a_large_log_is_found(self, tmp_path):
        text = ("noise line\n" * 200_000) + HIT.format(m=99)
        assert ke._log_has_aiter_evidence(_log(tmp_path / "a.log", text))

    def test_a_moe_only_log_is_evidence_too(self, tmp_path):
        # The resolved log is not only a dense-shape source: kernelforge's router reads it for MoE stage coverage and
        # 1-stage ASM detection, which parse [fused_moe] lines.
        assert ke._log_has_aiter_evidence(_log(tmp_path / "a.log", MOE))
        assert ke._log_has_aiter_evidence(_log(tmp_path / "b.log", "Mxfp4 MoE backend selected\n"))

    def test_a_moe_only_log_yields_no_dense_tokens(self, tmp_path):
        # ...and it must not invent any: --tokens comes from dense M only.
        assert ke.tokens_from_serving_log(_log(tmp_path / "a.log", MOE)) == ""

    def test_a_dispatch_line_at_m_zero_is_still_evidence(self, tmp_path):
        # The M counter skips 0, so an evidence check derived from its output read this log as silent.
        assert ke._log_has_aiter_evidence(_log(tmp_path / "a.log", HIT.format(m=0)))


class TestSelection:
    def test_a_quiet_current_best_no_longer_ends_the_search(self, tmp_path):
        ws = tmp_path / "runs" / "explore" / "h1" / "measure_round" / "b1"
        _log(ws / "server.log", QUIET)
        good = _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", HIT.format(m=256))

        picked = ke.resolve_forge_server_log(_State(current_best={"workspace": str(ws)}), tmp_path)

        assert picked == str(good)

    def test_a_current_best_with_evidence_still_wins(self, tmp_path):
        ws = tmp_path / "runs" / "explore" / "h1" / "measure_round" / "b1"
        mine = _log(ws / "server.log", HIT.format(m=256))
        _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", HIT.format(m=1))

        assert ke.resolve_forge_server_log(_State(current_best={"workspace": str(ws)}), tmp_path) == str(mine)

    def test_the_warmup_sibling_search_skips_quiet_logs(self, tmp_path):
        # server.log lives in warmup_round while current_best points at measure_round; several warmup benchmark dirs
        # can exist and only some of them dispatched.
        run = tmp_path / "runs" / "explore" / "h1"
        ws = run / "measure_round" / "b1"
        ws.mkdir(parents=True)
        newest = _log(run / "warmup_round" / "z_quiet" / "server.log", QUIET)
        older = _log(run / "warmup_round" / "a_real" / "server.log", HIT.format(m=77))
        import os

        os.utime(older, (1, 1))  # make the quiet one strictly newer

        assert newest.stat().st_mtime > older.stat().st_mtime
        assert ke.resolve_forge_server_log(_State(current_best={"workspace": str(ws)}), tmp_path) == str(older)

    def test_the_runs_fallback_skips_quiet_logs(self, tmp_path):
        import os

        good = _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", MISS.format(m=8))
        newer_quiet = _log(tmp_path / "runs" / "gemm_tuning" / "h9" / "warmup_round" / "b9" / "server.log", QUIET)
        os.utime(good, (1, 1))

        assert newer_quiet.stat().st_mtime > good.stat().st_mtime
        assert ke.resolve_forge_server_log(_State(), tmp_path) == str(good)

    def test_no_candidate_with_evidence_returns_empty_and_says_why(self, tmp_path, caplog):
        _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", QUIET)

        with caplog.at_level("WARNING"):
            assert ke.resolve_forge_server_log(_State(), tmp_path) == ""

        # "no log at all" and "logs exist but are silent" are different problems; only the second is actionable.
        assert "AITER_LOG_TUNED_CONFIG" in caplog.text

    def test_no_logs_at_all_is_quiet(self, tmp_path, caplog):
        (tmp_path / "runs").mkdir()
        with caplog.at_level("WARNING"):
            assert ke.resolve_forge_server_log(_State(), tmp_path) == ""
        assert "AITER_LOG_TUNED_CONFIG" not in caplog.text

    def test_baseline_is_consulted_when_current_best_is_quiet(self, tmp_path):
        quiet = tmp_path / "runs" / "explore" / "h1" / "measure_round" / "b1"
        _log(quiet / "server.log", QUIET)
        base = tmp_path / "runs" / "baseline" / "h0" / "measure_round" / "b0"
        base.mkdir(parents=True)
        real = _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", HIT.format(m=5))

        state = _State(current_best={"workspace": str(quiet)}, last_baseline={"workspace": str(base)})

        assert ke.resolve_forge_server_log(state, tmp_path) == str(real)


class TestTokensFromServingLog:
    def test_the_observed_m_values_come_back_sorted(self, tmp_path):
        text = "".join(HIT.format(m=m) for m in (512, 128, 15842))
        assert ke.tokens_from_serving_log(_log(tmp_path / "a.log", text)) == "128,512,15842"

    def test_a_miss_line_counts_too(self, tmp_path):
        text = MISS.format(m=64) + HIT.format(m=32)
        assert ke.tokens_from_serving_log(_log(tmp_path / "a.log", text)) == "32,64"

    def test_the_most_frequent_m_values_win_the_budget(self, tmp_path):
        # 1 appears once, 2..4 appear three times each; with a budget of 3 the rare one is dropped.
        text = HIT.format(m=1) + "".join(HIT.format(m=m) * 3 for m in (2, 3, 4))
        assert ke.tokens_from_serving_log(_log(tmp_path / "a.log", text), limit=3) == "2,3,4"

    def test_uniform_counts_do_not_starve_the_prefill_end(self, tmp_path):
        # Regression: a serving warmup sweeps every M about equally often, so the counts come out uniform and a plain
        # frequency ranking degenerates into its tie-break -- which kept the smallest M and dropped the large prefill
        # shapes the runtime then missed.
        ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16384, 24576, 32768]
        text = "".join(HIT.format(m=m) * 4 for m in ms)
        got = ke.tokens_from_serving_log(_log(tmp_path / "a.log", text), limit=8)
        picked = {int(t) for t in got.split(",")}
        assert {24576, 32768} <= picked, got
        assert len(picked) == 8, got
        # ...and the decode end still gets the majority of the budget.
        assert len([m for m in picked if m <= 256]) >= 4, got

    def test_a_quiet_log_yields_nothing(self, tmp_path):
        assert ke.tokens_from_serving_log(_log(tmp_path / "a.log", QUIET)) == ""

    def test_a_missing_log_yields_nothing(self, tmp_path):
        assert ke.tokens_from_serving_log(tmp_path / "nope.log") == ""

    def test_the_result_is_what_forge_accepts(self, tmp_path):
        # forge parses --tokens as int(t) for t in value.split(","); round-trip through the normaliser must not change
        # it. Pinned here rather than in the lane's own suite because it is the seam between the two: the scanner is
        # free to change how it picks, not what shape it hands over.
        from hyperloom.orchestrator.kernel.request_handlers import _normalize_tokens

        text = "".join(HIT.format(m=m) for m in (7, 4096))
        raw = ke.tokens_from_serving_log(_log(tmp_path / "a.log", text))
        assert _normalize_tokens(raw) == raw
        assert [int(t) for t in raw.split(",")] == [7, 4096]

    def test_m_values_are_read_across_chunk_boundaries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ke, "_LOG_SCAN_CHUNK", 24)
        text = "".join(HIT.format(m=m) for m in (11, 22, 33))
        assert ke.tokens_from_serving_log(_log(tmp_path / "a.log", text)) == "11,22,33"


class TestShapeManifestResolution:
    def test_the_newest_manifest_wins(self, tmp_path):
        import os

        old = tmp_path / "runs" / "baseline" / "bypass" / "trace_shape_manifest.json"
        new = tmp_path / "runs" / "explore" / "bypass" / "trace_shape_manifest.json"
        for path in (old, new):
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
        os.utime(old, (1, 1))

        assert ke.resolve_trace_shape_manifest(_State(), tmp_path) == str(new)

    def test_no_manifest_is_an_empty_string_not_an_error(self, tmp_path):
        assert ke.resolve_trace_shape_manifest(_State(), tmp_path) == ""

    def test_a_missing_session_dir_is_survivable(self, tmp_path):
        assert ke.resolve_trace_shape_manifest(_State(), tmp_path / "gone") == ""


class TestCampaignRepositoryDiscovery:
    """Which repositories a rewrite could name, before anything is sealed."""

    _GIT_IDENTITY = {
        "GIT_AUTHOR_NAME": "evidence-test",
        "GIT_AUTHOR_EMAIL": "evidence-test@local",
        "GIT_COMMITTER_NAME": "evidence-test",
        "GIT_COMMITTER_EMAIL": "evidence-test@local",
    }

    @classmethod
    def _git(cls, repo: Path, *args: str) -> str:
        import os
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            env={**os.environ, **cls._GIT_IDENTITY},
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    @classmethod
    def _repo(cls, tmp_path: Path, name: str = "framework") -> Path:
        repo = tmp_path / name
        repo.mkdir()
        cls._git(repo, "init")
        (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
        cls._git(repo, "add", ".")
        cls._git(repo, "commit", "-m", "upstream")
        return repo

    @pytest.fixture(autouse=True)
    def _no_runtime_discovery(self, monkeypatch):
        """Keep these tests off whatever framework the host has installed.

        Stubbed at ``find_spec`` rather than at ``_package_repository`` so the
        resolver itself still runs for the tests that are about it.
        """
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    def test_the_framework_being_served_is_found_without_configuration(self, tmp_path, monkeypatch):
        """All three configured sources were empty in the GLM-5.2 session."""
        repo = self._repo(tmp_path, "sglang-checkout")
        package = repo / "python" / "sglang"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        monkeypatch.setattr(
            ke,
            "_package_repository",
            lambda name: repo.resolve() if name == "sglang" else None,
        )

        assert ke.campaign_repositories(SimpleNamespace(framework_repo_path="")) == (repo.resolve(),)

    def test_a_configured_root_is_added_to_what_the_runtime_found(self, tmp_path, monkeypatch):
        """An operator pointing at a fourth checkout is honoured, not overridden."""
        served = self._repo(tmp_path, "served")
        extra = self._repo(tmp_path, "extra")
        monkeypatch.setattr(
            ke,
            "_package_repository",
            lambda name: served.resolve() if name == "aiter" else None,
        )

        roots = ke.campaign_repositories(SimpleNamespace(framework_repo_path=str(extra)))

        assert set(roots) == {served.resolve(), extra.resolve()}

    def test_a_wheel_installed_framework_is_no_repository(self, monkeypatch):
        """A wheel carries no source to rewrite, so it has no base to pin."""
        import importlib.util

        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: SimpleNamespace(origin="/opt/venv/lib/python3.10/site-packages/vllm/__init__.py"),
        )

        assert ke._package_repository("vllm") is None

    def test_a_configured_file_resolves_to_the_repository_holding_it(self, tmp_path):
        """A launch recipe or a source file names its repository just as well."""
        repo = self._repo(tmp_path)
        inside = repo / "nested" / "config.yaml"
        inside.parent.mkdir()
        inside.write_text("{}\n", encoding="utf-8")

        assert ke.campaign_repositories(SimpleNamespace(framework_repo_path=str(inside))) == (repo.resolve(),)

    def test_a_configured_path_in_no_repository_is_dropped(self, tmp_path):
        loose = tmp_path / "loose"
        loose.mkdir()

        assert ke.campaign_repositories(SimpleNamespace(framework_repo_path=str(loose))) == ()

    def test_a_package_that_cannot_be_imported_names_no_repository(self, monkeypatch):
        """A broken install is not a repository, and must not raise on the way out."""
        import importlib.util

        def _raise(_name):
            raise ImportError("boom")

        monkeypatch.setattr(importlib.util, "find_spec", _raise)
        assert ke._package_repository("sglang") is None

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        assert ke._package_repository("sglang") is None

    def test_a_namespace_package_with_no_origin_names_no_repository(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: SimpleNamespace(origin=None))

        assert ke._package_repository("sglang") is None


class TestFusionDecodeTraceResolution:
    def test_the_payload_wins_and_a_directory_resolves_to_its_newest_trace(self, tmp_path):
        state_dir = tmp_path / "state_trace"
        payload_dir = tmp_path / "payload_trace"
        state_dir.mkdir()
        payload_dir.mkdir()
        state_trace = state_dir / "old.trace.json.gz"
        payload_old = payload_dir / "old.trace.json.gz"
        payload_new = payload_dir / "new.trace.json"
        state_trace.write_text("state", encoding="utf-8")
        payload_old.write_text("old", encoding="utf-8")
        payload_new.write_text("new", encoding="utf-8")
        import os

        os.utime(payload_old, (1, 1))
        os.utime(payload_new, (10, 10))

        class _TraceState:
            last_profile_trace = str(state_dir)

        state = _TraceState()

        assert ke.resolve_fusion_decode_trace(state, {"trace_path": str(payload_dir)}) == str(payload_new)
        assert ke.resolve_fusion_decode_trace(state, {}) == str(state_trace)
        # An unusable explicit path falls back to the session's own trace rather than failing.
        assert ke.resolve_fusion_decode_trace(state, {"trace_path": "/missing"}) == str(state_trace)


class TestFp8QuantTypeResolution:
    @staticmethod
    def _write_cfg(model_dir: Path, cfg: dict) -> str:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return str(model_dir)

    def test_resolve_fp8_quant_type(self, tmp_path):
        block = self._write_cfg(
            tmp_path / "block",
            {"hidden_size": 7168, "quantization_config": {"weight_block_size": [128, 128]}},
        )
        method_block = self._write_cfg(
            tmp_path / "mblock",
            {"quantization_config": {"quant_method": "fp8_block"}},
        )
        plain = self._write_cfg(tmp_path / "plain", {"hidden_size": 2048})
        assert ke.resolve_fp8_quant_type(block) == "blockscale"
        assert ke.resolve_fp8_quant_type(method_block) == "blockscale"
        assert ke.resolve_fp8_quant_type(plain) == "per_token"
        # Multimodal: quantization_config nested under text_config is detected.
        nested_block = self._write_cfg(
            tmp_path / "nblock",
            {"text_config": {"quantization_config": {"weight_block_size": [128, 128]}}},
        )
        assert ke.resolve_fp8_quant_type(nested_block) == "blockscale"
        # Unreadable / missing config -> auto.
        assert ke.resolve_fp8_quant_type(str(tmp_path / "missing")) == "auto"
        assert ke.resolve_fp8_quant_type("") == "auto"


class TestUntunedCsvResolution:
    @staticmethod
    def _write_aiter_csv(session_dir: Path, hash_id: str, fname: str, rows: str) -> Path:
        cfg = session_dir / "runs" / "specialist" / hash_id / "worktree" / "aiter" / "configs"
        cfg.mkdir(parents=True, exist_ok=True)
        path = cfg / fname
        path.write_text(rows, encoding="utf-8")
        return path

    @staticmethod
    def _write_model_config(model_dir: Path, hidden_size: int) -> str:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps({"hidden_size": hidden_size}), encoding="utf-8")
        return str(model_dir)

    def test_resolve_forge_untuned_csv_fp8_blockscale(self, tmp_path):
        expected = self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "auto") == str(expected)
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(expected)

    def test_resolve_forge_untuned_csv_per_token(self, tmp_path):
        expected = self._write_aiter_csv(
            tmp_path, "abc", "a8w8_untuned_gemm.csv", "M,N,K,q_dtype_w\n16,1536,7168,fp8\n"
        )
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "per_token") == str(expected)

    def test_resolve_forge_untuned_csv_skips_header_only(self, tmp_path):
        # Header-only / empty files are not a valid shape source.
        self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n")
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == ""

    def test_resolve_forge_untuned_csv_picks_newest_nonempty(self, tmp_path):
        old = self._write_aiter_csv(tmp_path, "old", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n1,2,3\n")
        new = self._write_aiter_csv(tmp_path, "new", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n4,5,6\n")
        import os

        os.utime(old, (1, 1))
        os.utime(new, (10_000_000, 10_000_000))
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(new)

    def test_resolve_forge_untuned_csv_bf16_returns_empty(self, tmp_path):
        # bf16 dense derives shapes from config.json; no CSV needed.
        self._write_aiter_csv(tmp_path, "abc", "bf16_untuned_gemm.csv", "M,N,K\n1,2,3\n")
        assert ke.resolve_forge_untuned_csv(tmp_path, "bf16", "none") == ""

    def test_resolve_forge_untuned_csv_no_specialist_dir(self, tmp_path):
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == ""

    @pytest.mark.parametrize(
        "quant_type",
        [
            "blockscale_bpreshuffle",
            "a8w8_blockscale_bpreshuffle",
            "blockscale+bpreshuffle",
        ],
    )
    def test_resolve_forge_untuned_csv_blockscale_bpreshuffle(self, tmp_path, quant_type):
        expected = self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
            "M,N,K\n16,1536,7168\n",
        )
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", quant_type) == str(expected)

    def test_resolve_forge_untuned_csv_rejects_unknown_fp8_quant(self, tmp_path):
        self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_untuned_gemm.csv",
            "M,N,K\n16,1536,7168\n",
        )

        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "misspelled_quant_type") == ""

    def test_resolve_forge_untuned_csv_rejects_model_mismatch(self, tmp_path):
        # CSV carries K=7168 shapes but the model has hidden_size=2048: reject it so forge derives per-model shapes
        # from config.json.
        self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        model_path = self._write_model_config(tmp_path / "model", hidden_size=2048)
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", model_path) == ""

    def test_resolve_forge_untuned_csv_accepts_model_match(self, tmp_path):
        # A CSV whose K column includes the model hidden_size is accepted.
        expected = self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_untuned_gemm.csv",
            "M,N,K\n16,6144,2048\n16,2048,8192\n",
        )
        model_path = self._write_model_config(tmp_path / "model", hidden_size=2048)
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", model_path) == str(expected)

    def test_resolve_forge_untuned_csv_no_model_path_keeps_legacy(self, tmp_path):
        # Without a model_path the resolver cannot validate; returns newest non-empty CSV.
        expected = self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(expected)

    def test_resolve_forge_untuned_csv_unreadable_config_keeps_csv(self, tmp_path):
        # Missing/unreadable config.json: cannot validate, so keep the CSV.
        expected = self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        assert ke.resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", str(tmp_path / "no_such_model")) == str(
            expected
        )

    def test_csv_matches_model_helpers(self, tmp_path):
        csv_mismatch = self._write_aiter_csv(
            tmp_path, "h1", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        csv_match = self._write_aiter_csv(tmp_path, "h2", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,6144,2048\n")
        model_path = self._write_model_config(tmp_path / "m", hidden_size=2048)
        assert ke._model_hidden_size(model_path) == 2048
        assert ke._csv_k_values(csv_mismatch) == {7168}
        assert ke._csv_k_values(csv_match) == {2048}
        assert ke._csv_matches_model(csv_mismatch, model_path) is False
        assert ke._csv_matches_model(csv_match, model_path) is True
        # No model_path / unreadable config -> cannot validate -> accept.
        assert ke._csv_matches_model(csv_mismatch, "") is True
