"""A server log is only a shape source if it dispatched through aiter.

``_resolve_forge_server_log`` used to pick the first log that *existed*. A
``current_best`` workspace whose server never routed a GEMM through aiter
therefore ended the search, and the ``runs/`` fallback -- which might hold a log
that did -- was unreachable. The tuner then read shapes from a file with none.

The same scan feeds ``--tokens``: the M values in those lines are the only
record of what the model actually ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as rh

# A real aiter hit line, copied from a fleet server.log.
HIT = (
    "shape is M:{m}, N:512, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' "
    "bias=False, scaleAB=False, bpreshuffle=False found padded_M: 16384, N:512, "
    "K:4096 is tuned on cu_num = 256 in /tmp/aiter_configs/bf16_tuned_gemm.csv, "
    "libtype is opus, kernel name is opus_gemm\n"
)
MISS = "shape is M:{m}, N:512, K:4096 not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, using default\n"
QUIET = "INFO server started on 0.0.0.0:8000\nINFO warmup complete\n"
# A MoE dispatch line: no dense "shape is M:" anywhere, but the log is fully
# informative for the routing decisions kernelforge makes off the same path.
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
        assert rh._log_has_aiter_evidence(_log(tmp_path / "a.log", HIT.format(m=128)))

    def test_a_miss_line_is_also_evidence(self, tmp_path):
        # A miss still proves the process routed a GEMM through aiter, which is
        # what makes the log a usable shape source.
        assert rh._log_has_aiter_evidence(_log(tmp_path / "a.log", MISS.format(m=128)))

    def test_a_quiet_log_is_not_evidence(self, tmp_path):
        assert not rh._log_has_aiter_evidence(_log(tmp_path / "a.log", QUIET))

    def test_a_missing_file_is_not_evidence(self, tmp_path):
        assert not rh._log_has_aiter_evidence(tmp_path / "nope.log")

    def test_a_marker_straddling_a_chunk_boundary_is_still_found(self, tmp_path, monkeypatch):
        # Fleet logs are ~17MB, so the scan is chunked; the overlap must cover a
        # marker split across two reads.
        monkeypatch.setattr(rh, "_LOG_SCAN_CHUNK", 16)
        text = "x" * 10 + HIT.format(m=128)
        assert rh._log_has_aiter_evidence(_log(tmp_path / "a.log", text))

    def test_evidence_late_in_a_large_log_is_found(self, tmp_path):
        text = ("noise line\n" * 200_000) + HIT.format(m=99)
        assert rh._log_has_aiter_evidence(_log(tmp_path / "a.log", text))

    def test_a_moe_only_log_is_evidence_too(self, tmp_path):
        # The resolved log is not only a dense-shape source: kernelforge's router
        # reads it for MoE stage coverage and 1-stage ASM detection, which parse
        # [fused_moe] lines. Rejecting a log that has those but no dense
        # "shape is M:" line would blind the router on exactly the MoE models
        # this lane runs against.
        assert rh._log_has_aiter_evidence(_log(tmp_path / "a.log", MOE))
        assert rh._log_has_aiter_evidence(_log(tmp_path / "b.log", "Mxfp4 MoE backend selected\n"))

    def test_a_moe_only_log_yields_no_dense_tokens(self, tmp_path):
        # ...and it must not invent any: --tokens comes from dense M only.
        assert rh._tokens_from_serving_log(_log(tmp_path / "a.log", MOE)) == ""

    def test_a_dispatch_line_at_m_zero_is_still_evidence(self, tmp_path):
        # The M counter skips 0, so an evidence check derived from its output
        # read this log as silent.
        assert rh._log_has_aiter_evidence(_log(tmp_path / "a.log", HIT.format(m=0)))


class TestSelection:
    def test_a_quiet_current_best_no_longer_ends_the_search(self, tmp_path):
        ws = tmp_path / "runs" / "explore" / "h1" / "measure_round" / "b1"
        _log(ws / "server.log", QUIET)
        good = _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", HIT.format(m=256))

        picked = rh._resolve_forge_server_log(_State(current_best={"workspace": str(ws)}), tmp_path)

        assert picked == str(good)

    def test_a_current_best_with_evidence_still_wins(self, tmp_path):
        ws = tmp_path / "runs" / "explore" / "h1" / "measure_round" / "b1"
        mine = _log(ws / "server.log", HIT.format(m=256))
        _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", HIT.format(m=1))

        assert rh._resolve_forge_server_log(_State(current_best={"workspace": str(ws)}), tmp_path) == str(mine)

    def test_the_warmup_sibling_search_skips_quiet_logs(self, tmp_path):
        # server.log lives in warmup_round while current_best points at
        # measure_round; several warmup benchmark dirs can exist and only some
        # of them dispatched.
        run = tmp_path / "runs" / "explore" / "h1"
        ws = run / "measure_round" / "b1"
        ws.mkdir(parents=True)
        newest = _log(run / "warmup_round" / "z_quiet" / "server.log", QUIET)
        older = _log(run / "warmup_round" / "a_real" / "server.log", HIT.format(m=77))
        import os

        os.utime(older, (1, 1))  # make the quiet one strictly newer

        assert newest.stat().st_mtime > older.stat().st_mtime
        assert rh._resolve_forge_server_log(_State(current_best={"workspace": str(ws)}), tmp_path) == str(older)

    def test_the_runs_fallback_skips_quiet_logs(self, tmp_path):
        import os

        good = _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", MISS.format(m=8))
        newer_quiet = _log(tmp_path / "runs" / "gemm_tuning" / "h9" / "warmup_round" / "b9" / "server.log", QUIET)
        os.utime(good, (1, 1))

        assert newer_quiet.stat().st_mtime > good.stat().st_mtime
        assert rh._resolve_forge_server_log(_State(), tmp_path) == str(good)

    def test_no_candidate_with_evidence_returns_empty_and_says_why(self, tmp_path, caplog):
        _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", QUIET)

        with caplog.at_level("WARNING"):
            assert rh._resolve_forge_server_log(_State(), tmp_path) == ""

        # "no log at all" and "logs exist but are silent" are different
        # problems; only the second is actionable.
        assert "AITER_LOG_TUNED_CONFIG" in caplog.text

    def test_no_logs_at_all_is_quiet(self, tmp_path, caplog):
        (tmp_path / "runs").mkdir()
        with caplog.at_level("WARNING"):
            assert rh._resolve_forge_server_log(_State(), tmp_path) == ""
        assert "AITER_LOG_TUNED_CONFIG" not in caplog.text

    def test_baseline_is_consulted_when_current_best_is_quiet(self, tmp_path):
        quiet = tmp_path / "runs" / "explore" / "h1" / "measure_round" / "b1"
        _log(quiet / "server.log", QUIET)
        base = tmp_path / "runs" / "baseline" / "h0" / "measure_round" / "b0"
        base.mkdir(parents=True)
        real = _log(tmp_path / "runs" / "baseline" / "h0" / "warmup_round" / "b0" / "server.log", HIT.format(m=5))

        state = _State(current_best={"workspace": str(quiet)}, last_baseline={"workspace": str(base)})

        assert rh._resolve_forge_server_log(state, tmp_path) == str(real)


class TestTokensFromServingLog:
    def test_the_observed_m_values_come_back_sorted(self, tmp_path):
        text = "".join(HIT.format(m=m) for m in (512, 128, 15842))
        assert rh._tokens_from_serving_log(_log(tmp_path / "a.log", text)) == "128,512,15842"

    def test_a_miss_line_counts_too(self, tmp_path):
        text = MISS.format(m=64) + HIT.format(m=32)
        assert rh._tokens_from_serving_log(_log(tmp_path / "a.log", text)) == "32,64"

    def test_the_most_frequent_m_values_win_the_budget(self, tmp_path):
        # 1 appears once, 2..4 appear three times each; with a budget of 3 the
        # rare one is dropped. Tuning where the model spends its time beats
        # tuning the largest M it ever reached.
        text = HIT.format(m=1) + "".join(HIT.format(m=m) * 3 for m in (2, 3, 4))
        assert rh._tokens_from_serving_log(_log(tmp_path / "a.log", text), limit=3) == "2,3,4"

    def test_uniform_counts_do_not_starve_the_prefill_end(self, tmp_path):
        # Regression: a serving warmup sweeps every M about equally often, so
        # the counts come out uniform and a plain frequency ranking degenerates
        # into its tie-break -- which kept the smallest M and dropped the large
        # prefill shapes the runtime then missed. Both fleet sessions we
        # replayed looked exactly like this (17 distinct M x4, 44 distinct
        # M x40), and both missed on 16384/24576/32768 and 57344/65536.
        ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16384, 24576, 32768]
        text = "".join(HIT.format(m=m) * 4 for m in ms)
        got = rh._tokens_from_serving_log(_log(tmp_path / "a.log", text), limit=8)
        picked = {int(t) for t in got.split(",")}
        assert {24576, 32768} <= picked, got
        assert len(picked) == 8, got
        # ...and the decode end still gets the majority of the budget.
        assert len([m for m in picked if m <= 256]) >= 4, got

    def test_a_quiet_log_yields_nothing(self, tmp_path):
        assert rh._tokens_from_serving_log(_log(tmp_path / "a.log", QUIET)) == ""

    def test_a_missing_log_yields_nothing(self, tmp_path):
        assert rh._tokens_from_serving_log(tmp_path / "nope.log") == ""

    def test_the_result_is_what_forge_accepts(self, tmp_path):
        # forge parses --tokens as int(t) for t in value.split(","); round-trip
        # through the normaliser must not change it.
        text = "".join(HIT.format(m=m) for m in (7, 4096))
        raw = rh._tokens_from_serving_log(_log(tmp_path / "a.log", text))
        assert rh._normalize_tokens(raw) == raw
        assert [int(t) for t in raw.split(",")] == [7, 4096]

    def test_m_values_are_read_across_chunk_boundaries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rh, "_LOG_SCAN_CHUNK", 24)
        text = "".join(HIT.format(m=m) for m in (11, 22, 33))
        assert rh._tokens_from_serving_log(_log(tmp_path / "a.log", text)) == "11,22,33"


class TestShapeManifestResolution:
    def test_the_newest_manifest_wins(self, tmp_path):
        import os

        old = tmp_path / "runs" / "baseline" / "bypass" / "trace_shape_manifest.json"
        new = tmp_path / "runs" / "explore" / "bypass" / "trace_shape_manifest.json"
        for path in (old, new):
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
        os.utime(old, (1, 1))

        assert rh._resolve_trace_shape_manifest(_State(), tmp_path) == str(new)

    def test_no_manifest_is_an_empty_string_not_an_error(self, tmp_path):
        assert rh._resolve_trace_shape_manifest(_State(), tmp_path) == ""

    def test_a_missing_session_dir_is_survivable(self, tmp_path):
        assert rh._resolve_trace_shape_manifest(_State(), tmp_path / "gone") == ""


class TestTheWrapperForwardsTheNewInputs:
    @pytest.mark.parametrize(
        "key,flag",
        [
            ("shapes_manifest", "--shapes-manifest"),
            ("demand_json", "--demand"),
        ],
    )
    def test_a_populated_field_reaches_the_forge_cli(self, key, flag):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_fgt",
            Path(rh.__file__).parents[2] / "agents" / "kernel" / "tools" / "forge_gemm_tuning.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        base = {
            "model_path": "/m",
            "framework": "sglang",
            "precision": "bf16",
            "output_dir": "/o",
        }
        assert flag not in mod._build_cmd(base)

        cmd = mod._build_cmd({**base, key: "/tmp/x.json"})
        assert cmd[cmd.index(flag) + 1] == "/tmp/x.json"


class _StopHere(Exception):
    """Raised from the MoE CSV writer to end the run at the point under test."""


class TestTokensAreDerivedBeforeTheMoeCsvIsBuilt:
    """Both lanes must see the observed M sweep, not just the dense one.

    ``_write_fmoe_untuned_csv_from_log`` consumes ``tokens`` directly and its
    fallback for an empty one is ``[1]``. While the serving-log derivation sat
    below that call, the dense lane got the full sweep and the MoE lane got a
    one-row table -- which then missed on every prefill and large-batch lookup
    and was reverted as ``no_shape_key_matched``. That is the exact failure this
    change set exists to remove, so an ordering regression here would leave the
    MoE half of it in place.
    """

    @pytest.mark.asyncio
    async def test_the_moe_writer_receives_the_serving_log_sweep(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        log_path = _log(tmp_path / "server.log", "".join(HIT.format(m=m) * 4 for m in (1, 64, 2048, 16384)))

        seen: dict[str, str] = {}

        def _writer(_log_path, tokens, _workspace):
            seen["tokens"] = tokens
            raise _StopHere

        from hyperloom.common import model_paths
        from hyperloom.inference_optimizer import model_config_utils
        from hyperloom.orchestrator.policy import gate

        monkeypatch.setattr(rh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(rh, "_resolve_forge_precision_and_quant", lambda *_a, **_k: ("bf16", ""))
        monkeypatch.setattr(model_paths, "resolve_serving_model_path", lambda p: str(p))
        monkeypatch.setattr(model_config_utils, "resolve_local_model_dir", lambda _p: model_dir)
        monkeypatch.setattr(gate, "detect_gpu_count", lambda: 8)
        monkeypatch.setattr(rh, "_resolve_forge_shapes", lambda *_a, **_k: "")
        monkeypatch.setattr(rh, "_resolve_forge_untuned_csv", lambda *_a, **_k: "")
        monkeypatch.setattr(rh, "_write_fmoe_untuned_csv_from_log", _writer)

        payload = {
            "model_path": str(model_dir),
            "framework": "sglang",
            "kernel_signature_log": str(log_path),
        }
        with pytest.raises(_StopHere):
            await rh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        assert seen["tokens"] == rh._tokens_from_serving_log(log_path)
        # Not the ``[1]`` fallback, and not a single value of any kind.
        assert len(seen["tokens"].split(",")) > 1

    @pytest.mark.asyncio
    async def test_an_explicit_payload_tokens_value_still_wins(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        log_path = _log(tmp_path / "server.log", HIT.format(m=999))

        seen: dict[str, str] = {}

        def _writer(_log_path, tokens, _workspace):
            seen["tokens"] = tokens
            raise _StopHere

        from hyperloom.common import model_paths
        from hyperloom.inference_optimizer import model_config_utils
        from hyperloom.orchestrator.policy import gate

        monkeypatch.setattr(rh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(rh, "_resolve_forge_precision_and_quant", lambda *_a, **_k: ("bf16", ""))
        monkeypatch.setattr(model_paths, "resolve_serving_model_path", lambda p: str(p))
        monkeypatch.setattr(model_config_utils, "resolve_local_model_dir", lambda _p: model_dir)
        monkeypatch.setattr(gate, "detect_gpu_count", lambda: 8)
        monkeypatch.setattr(rh, "_resolve_forge_shapes", lambda *_a, **_k: "")
        monkeypatch.setattr(rh, "_resolve_forge_untuned_csv", lambda *_a, **_k: "")
        monkeypatch.setattr(rh, "_write_fmoe_untuned_csv_from_log", _writer)

        payload = {
            "model_path": str(model_dir),
            "framework": "sglang",
            "kernel_signature_log": str(log_path),
            "tokens": "1,2,4",
        }
        with pytest.raises(_StopHere):
            await rh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        assert seen["tokens"] == "1,2,4"
