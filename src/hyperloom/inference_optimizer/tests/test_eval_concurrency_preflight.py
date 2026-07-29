"""Preflight's eval-concurrency compat pass.

InferenceX's ``run_eval`` rejects ``--concurrent-requests`` and takes concurrency
via ``EVAL_CONCURRENT_REQUESTS`` instead, but the Magpie copy that actually
executes is the one preflight installs -- not any source checkout. If the flag
survives there, every ``RUN_EVAL=true`` baseline dies with
``Unknown parameter: --concurrent-requests`` and the run halts at
``baseline_accuracy_failed``. These tests pin the three outcomes of that pass so
a regression surfaces here rather than four minutes into a GPU run.
"""

from __future__ import annotations

import hyperloom.inference_optimizer.cli.preflight as preflight_mod
import hyperloom.orchestrator.actions.executors._magpie_patcher as patcher_mod


def test_clean_trees_report_success(monkeypatch, capsys):
    seen: dict[str, object] = {}

    def fake(magpie, inferencex):
        seen["args"] = (magpie, inferencex)
        return True

    monkeypatch.setattr(patcher_mod, "ensure_eval_concurrency_compat", fake)

    assert preflight_mod._ensure_eval_concurrency_compat("/m", "/ix") is True
    assert seen["args"] == ("/m", "/ix")
    assert "WARNING" not in capsys.readouterr().out


def test_absent_magpie_path_is_passed_through_as_none(monkeypatch):
    """An unset MAGPIE_PATH must reach the patcher as None, not "" ."""
    seen: dict[str, object] = {}

    def fake(magpie, inferencex):
        seen["args"] = (magpie, inferencex)
        return True

    monkeypatch.setattr(patcher_mod, "ensure_eval_concurrency_compat", fake)

    assert preflight_mod._ensure_eval_concurrency_compat("", "/ix") is True
    assert seen["args"] == (None, "/ix")


def test_unpatchable_script_warns_and_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        patcher_mod, "ensure_eval_concurrency_compat", lambda *_a: False
    )

    assert preflight_mod._ensure_eval_concurrency_compat("/m", "/ix") is False

    out = capsys.readouterr().out
    assert "WARNING" in out
    # The operator needs the actual failure mode, not just "something failed".
    assert "--concurrent-requests" in out
    assert "EVAL_CONCURRENT_REQUESTS" in out


def test_patcher_explosion_never_aborts_preflight(monkeypatch, capsys):
    """Preflight must survive a broken patcher: it is a compat pass, not a gate."""

    def boom(*_args):
        raise RuntimeError("patcher exploded")

    monkeypatch.setattr(patcher_mod, "ensure_eval_concurrency_compat", boom)

    assert preflight_mod._ensure_eval_concurrency_compat("/m", "/ix") is False

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "patcher exploded" in out
