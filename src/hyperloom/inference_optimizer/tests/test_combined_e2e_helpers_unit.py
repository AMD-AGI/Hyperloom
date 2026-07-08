# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the combined-E2E helpers in ``kernel_request_handlers``.

These cover the pure logic of the autonomous combined-E2E step (pair collection,
serving-config resolution, model-path resolution, and the opt-in/backend/model/GPU
guards in ``_run_combined_e2e_sync``) without launching a server or GPU work.
"""

from __future__ import annotations

from hyperloom.orchestrator.kernel import request_handlers as krh


# ---- _collect_combined_e2e_pairs ----


def test_collect_pairs_picks_best_patch_and_skips_patchless():
    results = [
        # has a best patch -> kept
        {
            "kernel_id": "k007",
            "source_file": "/sgl/aiter/csrc/kernels/quant_kernels.cu",
            "attempts": [{"backend_paths": {"geak_per_task_best_patch": "/p/quant.patch"}}],
        },
        # no patch -> skipped
        {
            "kernel_id": "k001",
            "source_file": "/sgl/aiter/csrc/kernels/attention_ragged.cu",
            "attempts": [{"backend_paths": {}}],
        },
        # no source_file -> skipped
        {
            "kernel_id": "k009",
            "source_file": "",
            "attempts": [{"backend_paths": {"geak_per_task_best_patch": "/p/x.patch"}}],
        },
    ]
    pairs = krh._collect_combined_e2e_pairs(results)
    assert pairs == [("/p/quant.patch", "/sgl/aiter/csrc/kernels/quant_kernels.cu")]


def test_collect_pairs_handles_empty_and_malformed():
    assert krh._collect_combined_e2e_pairs([]) == []
    assert krh._collect_combined_e2e_pairs(None) == []
    # non-dict entries are ignored, not crashed on
    assert krh._collect_combined_e2e_pairs(["not-a-dict", 42]) == []


# ---- _combined_e2e_serving_config ----


def test_serving_config_from_explicit_payload():
    cfg = krh._combined_e2e_serving_config(
        {"serving_config": {"tp": 1, "isl": 8192, "osl": 1024, "conc": 64, "num_prompts": 320, "framework": "sglang"}}
    )
    assert cfg == {"tp": 1, "isl": 8192, "osl": 1024, "conc": 64, "num_prompts": 320, "framework": "sglang"}


def test_serving_config_framework_fallback_from_payload():
    # no serving_config dict, but top-level framework present
    cfg = krh._combined_e2e_serving_config({"framework": "VLLM"})
    assert cfg.get("framework") == "vllm"


def test_serving_config_empty_when_nothing_provided():
    assert krh._combined_e2e_serving_config({}) == {}


def test_serving_config_falls_back_to_materialized_metadata(monkeypatch):
    # config_path present + no explicit serving_config -> pull numeric knobs from the
    # materialized workload metadata resolver (unset keys only).
    monkeypatch.setattr(
        krh,
        "_load_materialized_workload_metadata",
        lambda _p: {"runtime_args": {"workload": {"tp": 4, "isl": 2048, "osl": 256}}},
    )
    cfg = krh._combined_e2e_serving_config({"config_path": "/some/materialized.yaml", "framework": "vllm"})
    assert cfg["framework"] == "vllm"
    assert cfg["tp"] == 4 and cfg["isl"] == 2048 and cfg["osl"] == 256


# ---- _resolve_local_model_path ----


def test_resolve_model_path_passthrough_for_dir(tmp_path):
    # an existing directory is returned unchanged
    d = tmp_path / "some-model"
    d.mkdir()
    assert krh._resolve_local_model_path(str(d)) == str(d)


def test_resolve_model_path_empty_and_unresolvable():
    assert krh._resolve_local_model_path("") == ""
    # a bare id that resolves to nothing falls back to the raw value (serve layer decides)
    assert krh._resolve_local_model_path("definitely-not-a-real-model-xyz") == "definitely-not-a-real-model-xyz"


# ---- _run_combined_e2e_sync guards (return None without serving) ----


def test_combined_e2e_skips_when_flag_off():
    # no combined_e2e flag -> None (no-op), even with everything else present
    assert krh._run_combined_e2e_sync([], {"backend_order": "geak", "model_path": "m"}, _P()) is None


def test_combined_e2e_skips_non_geak_backend():
    assert (
        krh._run_combined_e2e_sync([], {"combined_e2e": True, "backend_order": "claude", "model_path": "m"}, _P())
        is None
    )


def test_combined_e2e_skips_without_model():
    assert krh._run_combined_e2e_sync([], {"combined_e2e": True, "backend_order": "geak"}, _P()) is None


class _P:
    """Minimal stand-in for a session_dir Path arg (the guards return before using it)."""

    def __truediv__(self, _other):
        return self

    def __str__(self):
        return "/tmp/_combined_e2e_test"


# ---- _run_combined_e2e_sync body (past the guards), with GPU + tool mocked ----


def _passing_payload():
    return {
        "combined_e2e": True,
        "backend_order": "geak",
        "model_path": "/tmp/m",
        "serving_config": {"tp": 1, "isl": 1024, "osl": 1024, "framework": "sglang"},
    }


def _one_pair_results():
    return [
        {
            "kernel_id": "k1",
            "source_file": "/sgl/x.cu",
            "attempts": [{"backend_paths": {"geak_per_task_best_patch": "/p/x.patch"}}],
        }
    ]


def test_combined_e2e_skips_when_no_pairs(monkeypatch, tmp_path):
    # Past flag/backend/model/GPU guards, but no kernel produced a patch -> None.
    monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(krh, "_resolve_local_model_path", lambda m: m)
    results = [{"kernel_id": "k1", "source_file": "/s/x.cu", "attempts": [{"backend_paths": {}}]}]
    assert krh._run_combined_e2e_sync(results, _passing_payload(), tmp_path) is None


def test_combined_e2e_skips_without_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(krh, "_resolve_local_model_path", lambda m: m)
    monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 0)
    assert krh._run_combined_e2e_sync(_one_pair_results(), _passing_payload(), tmp_path) is None


def test_combined_e2e_tool_not_found(monkeypatch, tmp_path):
    # Past all guards, but the kernel-agent tool path can't resolve -> error dict (not None, not raise).
    monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(krh, "_resolve_local_model_path", lambda m: m)
    monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda _n: (_ for _ in ()).throw(RuntimeError("no root")))
    out = krh._run_combined_e2e_sync(_one_pair_results(), _passing_payload(), tmp_path)
    assert out["status"] == "error"
    assert "apply_and_bench tool not found" in out["error"]


def test_combined_e2e_invokes_apply_and_bench(monkeypatch, tmp_path):
    # Drive the full body: GPU present, tool resolves to a tiny fake module exposing
    # apply_and_bench(**kwargs); assert the result is threaded back and kwargs are sane.
    monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(krh, "_resolve_local_model_path", lambda m: m)
    fake = tmp_path / "apply_and_bench.py"
    fake.write_text(
        "def apply_and_bench(**kw):\n"
        "    return {'status': 'ok', 'delta_pct': 1.23, 'pairs': len(kw['pairs']),\n"
        "            'model': kw['model'], 'backend': kw.get('backend'), 'tp': kw.get('tp')}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda _n: fake)
    out = krh._run_combined_e2e_sync(_one_pair_results(), _passing_payload(), tmp_path)
    assert out["status"] == "ok"
    assert out["delta_pct"] == 1.23
    assert out["pairs"] == 1
    assert out["backend"] == "sglang"
    assert out["tp"] == 1


def test_combined_e2e_apply_raises_is_caught(monkeypatch, tmp_path):
    # apply_and_bench raising must become an {status:error} dict, never propagate.
    monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(krh, "_resolve_local_model_path", lambda m: m)
    fake = tmp_path / "apply_and_bench.py"
    fake.write_text("def apply_and_bench(**kw):\n    raise RuntimeError('serve boom')\n", encoding="utf-8")
    monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda _n: fake)
    out = krh._run_combined_e2e_sync(_one_pair_results(), _passing_payload(), tmp_path)
    assert out["status"] == "error"
    assert "apply_and_bench raised" in out["error"]


def test_maybe_run_combined_e2e_async_wrapper(monkeypatch, tmp_path):
    # The async wrapper runs the sync fn in an executor and returns its result.
    # Use asyncio.run() (fresh loop) — get_event_loop() raises under py3.11 when no
    # loop is set on the thread, which is exactly the bug the wrapper now avoids too.
    import asyncio

    monkeypatch.setattr(krh, "_run_combined_e2e_sync", lambda r, p, s: {"status": "ok", "via": "wrapper"})
    out = asyncio.run(krh._maybe_run_combined_e2e([], {}, tmp_path))
    assert out == {"status": "ok", "via": "wrapper"}
