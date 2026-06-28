# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Offline tests for the quark_quantizer shell (no real agent / SDK / GPU).

The shell adds one thing: a parameter-gated trigger around the original
``quantization_agent.quantize_via_prompt``. These tests inject a fake
``quantize_via_prompt`` so nothing real runs, and assert:

* ``enabled=False`` is a no-op (the wrapped agent is never called),
* ``enabled=True`` forwards the prompt + plumbing and normalizes the result.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from quark_quantizer import runner as rn
from quark_quantizer.runner import QuarkRunResult


def _fake_agent_result(status: str, qdir: str | None, *, final="ok", eval_gap=0.01):
    """Build a stand-in for quantization_agent.QuantSkillRunResult."""
    return SimpleNamespace(
        status=status,
        quantized_model_dir=Path(qdir) if qdir else None,
        assessment=SimpleNamespace(final=final, eval_gap=eval_gap),
    )


def _capturing_qvp(result, calls: list):
    async def _qvp(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return result

    return _qvp


# ---------------------------------------------------------------------------
# the parameter gate
# ---------------------------------------------------------------------------


def test_disabled_is_noop_and_never_calls_agent(tmp_path):
    async def _must_not_run(*a, **k):  # pragma: no cover - asserts non-call
        raise AssertionError("wrapped agent must not run when disabled")

    res = asyncio.run(
        rn.quantize("fp8", enabled=False, workspace=tmp_path, quantize_via_prompt=_must_not_run)
    )
    assert res.status == "skipped"
    assert res.output_dir is None


def test_disabled_by_default(tmp_path):
    async def _must_not_run(*a, **k):  # pragma: no cover - asserts non-call
        raise AssertionError("default must be disabled")

    res = asyncio.run(rn.quantize("fp8", workspace=tmp_path, quantize_via_prompt=_must_not_run))
    assert res.status == "skipped"


# ---------------------------------------------------------------------------
# delegation + result normalization
# ---------------------------------------------------------------------------


def test_enabled_forwards_prompt_and_plumbing(tmp_path):
    calls: list = []
    qvp = _capturing_qvp(_fake_agent_result("success", str(tmp_path / "q")), calls)
    res = asyncio.run(
        rn.quantize(
            "fp8 exclude lm_head",
            enabled=True,
            workspace=tmp_path,
            quark_root="/qr",
            acceptable_eval_gap=0.05,
            quantize_via_prompt=qvp,
        )
    )
    assert res.status == "success"
    assert res.output_dir == str(tmp_path / "q")
    assert res.eval_gap == 0.01
    assert res.final == "ok"
    # The NL prompt is forwarded verbatim; plumbing is passed through.
    assert calls[0]["prompt"] == "fp8 exclude lm_head"
    assert calls[0]["workspace"] == tmp_path
    assert calls[0]["quark_root"] == "/qr"
    assert calls[0]["acceptable_eval_gap"] == 0.05
    assert calls[0]["interactive"] is False


def test_partial_with_model_is_normalized(tmp_path):
    calls: list = []
    qvp = _capturing_qvp(_fake_agent_result("partial", str(tmp_path / "q"), final="eval_gap_exceeded"), calls)
    res = asyncio.run(rn.quantize("fp8", enabled=True, workspace=tmp_path, quantize_via_prompt=qvp))
    assert res.status == "partial"
    assert res.output_dir == str(tmp_path / "q")
    assert res.final == "eval_gap_exceeded"


def test_failed_has_no_output_dir(tmp_path):
    calls: list = []
    qvp = _capturing_qvp(_fake_agent_result("failed", None, final="exec_model_load_failed"), calls)
    res = asyncio.run(rn.quantize("fp8", enabled=True, workspace=tmp_path, quantize_via_prompt=qvp))
    assert res.status == "failed"
    assert res.output_dir is None
    assert res.final == "exec_model_load_failed"


def test_quantize_sync_wrapper(tmp_path):
    calls: list = []
    qvp = _capturing_qvp(_fake_agent_result("success", str(tmp_path / "q")), calls)
    res = rn.quantize_sync("fp8", enabled=True, workspace=tmp_path, quantize_via_prompt=qvp)
    assert res.status == "success"
    assert isinstance(res, QuarkRunResult)


# ---------------------------------------------------------------------------
# cli (param-driven trigger)
# ---------------------------------------------------------------------------


def test_cli_defaults_to_disabled():
    from quark_quantizer import cli

    args = cli._parse_args(["--workspace", "/w"])
    assert args.enabled is False


def test_cli_enabled_flag_and_run(monkeypatch, capsys):
    from quark_quantizer import cli

    seen: dict = {}

    async def _fake_quantize(prompt, *, enabled, **kwargs):
        seen["enabled"] = enabled
        seen["prompt"] = prompt
        return QuarkRunResult(status="skipped")

    monkeypatch.setattr(cli, "quantize", _fake_quantize)
    rc = cli.main(["--enabled", "--workspace", "/w", "--prompt", "fp8"])
    import json

    assert rc == 0
    assert seen["enabled"] is True
    assert seen["prompt"] == "fp8"
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"


def test_cli_failed_status_exit_code(monkeypatch):
    from quark_quantizer import cli

    async def _fake_quantize(prompt, *, enabled, **kwargs):
        return QuarkRunResult(status="failed", error="boom")

    monkeypatch.setattr(cli, "quantize", _fake_quantize)
    rc = cli.main(["--enabled", "--workspace", "/w"])
    assert rc == 1
