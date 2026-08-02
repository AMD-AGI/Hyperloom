"""Run-time assertion of the SGLang custom-tokenizer trust patch.

``MAGPIE_TRUST_REMOTE_CODE=1`` is set for every run, but it is inert unless
Magpie's SGLang scripts were patched to forward it: upstream never passes the
``trust`` argument, so ``benchmark_serving.py`` keeps its ``--trust-remote-code``
default of False and transformers refuses to execute a model's custom tokenizer
code. ``install.sh`` applies that patch; preflight pip-installs Magpie on its
own and used to skip it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.preflight import _ensure_client_trust_compat
from hyperloom.orchestrator.actions.executors._magpie_patcher import (
    _LOCAL_TRUST_SENTINEL,
    ensure_client_trust_compat,
)

_UPSTREAM_SGLANG_MI300X_SH = """\
#!/usr/bin/env bash
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct || exit $?
  else
    run_benchmark_serving --model "$MODEL" || exit $?
  fi
fi
if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?
fi
"""

_UPSTREAM_SGLANG_MI355X_SH = """\
#!/usr/bin/env bash
SERVER_MONITOR_ARGS=()
if [[ -n "${SERVER_PID:-}" ]]; then
  SERVER_MONITOR_ARGS+=(--server-pid "$SERVER_PID")
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct || exit $?
  else
    run_benchmark_serving \\
        --model "$MODEL" \\
        --result-filename "$RESULT_FILENAME" \\
        "${SERVER_MONITOR_ARGS[@]}" \\
        --result-dir ${RESULT_DIR:-/workspace/} || exit $?
  fi
fi
"""


def _write_scripts(root: Path, *, mi300x: str | None = None, mi355x: str | None = None) -> Path:
    """Materialise Magpie's benchmark script dir and return it."""
    script_dir = root / "Magpie" / "scripts" / "benchmark"
    script_dir.mkdir(parents=True, exist_ok=True)
    if mi300x is not None:
        (script_dir / "sglang_mi300x.sh").write_text(mi300x, encoding="utf-8")
    if mi355x is not None:
        (script_dir / "sglang_mi355x.sh").write_text(mi355x, encoding="utf-8")
    return script_dir


@pytest.fixture
def upstream_magpie(tmp_path: Path) -> Path:
    """A pristine, never-patched Magpie tree — what preflight's pip install leaves."""
    _write_scripts(
        tmp_path,
        mi300x=_UPSTREAM_SGLANG_MI300X_SH,
        mi355x=_UPSTREAM_SGLANG_MI355X_SH,
    )
    return tmp_path


def test_upstream_tree_has_no_trust_gate(upstream_magpie: Path):
    """Sanity: the fixture really is the inert shape the fix targets."""
    for name in ("sglang_mi300x.sh", "sglang_mi355x.sh"):
        text = (upstream_magpie / "Magpie" / "scripts" / "benchmark" / name).read_text(
            encoding="utf-8"
        )
        assert "MAGPIE_TRUST_REMOTE_CODE" not in text


def test_remote_and_local_clients_get_trust_gating(upstream_magpie: Path):
    """Both SGLang scripts end up forwarding the env gate to the client."""
    assert ensure_client_trust_compat(upstream_magpie) is True

    script_dir = upstream_magpie / "Magpie" / "scripts" / "benchmark"
    mi300x = (script_dir / "sglang_mi300x.sh").read_text(encoding="utf-8")
    mi355x = (script_dir / "sglang_mi355x.sh").read_text(encoding="utf-8")

    # Remote-direct path (the multi-node client) on both scripts.
    for text in (mi300x, mi355x):
        assert "magpie_run_benchmark_serving_remote_direct trust" in text
        assert '"${MAGPIE_TRUST_REMOTE_CODE:-0}" == "1"' in text

    # Local-server client path exists only on the mi355x fixture.
    assert _LOCAL_TRUST_SENTINEL in mi355x
    assert "CLIENT_TRUST_ARGS+=(--trust-remote-code)" in mi355x


def test_is_idempotent_without_rewriting(upstream_magpie: Path):
    """A second call must be a pure no-op — preflight runs on every optimize."""
    assert ensure_client_trust_compat(upstream_magpie) is True
    script_dir = upstream_magpie / "Magpie" / "scripts" / "benchmark"
    first = {p.name: p.read_text(encoding="utf-8") for p in script_dir.glob("*.sh")}

    assert ensure_client_trust_compat(upstream_magpie) is True
    after = {p.name: p.read_text(encoding="utf-8") for p in script_dir.glob("*.sh")}
    assert after == first


def test_still_applies_after_the_eval_concurrency_strip(tmp_path: Path):
    """Order independence from the eval-concurrency fix.

    The legacy MI300X patcher needs the still-flagged ``run_eval`` line and
    bails out entirely once the strip removed it, discarding the trust rewrite
    it had already computed. The client-trust patcher must not share that
    coupling, otherwise a tree that saw the strip first is stuck unpatchable.
    """
    stripped = _UPSTREAM_SGLANG_MI300X_SH.replace(" --concurrent-requests $CONC", "")
    _write_scripts(tmp_path, mi300x=stripped)
    assert "--concurrent-requests" not in stripped

    assert ensure_client_trust_compat(tmp_path) is True
    text = (tmp_path / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh").read_text(
        encoding="utf-8"
    )
    assert "magpie_run_benchmark_serving_remote_direct trust" in text


def test_absent_sglang_scripts_are_not_applicable(tmp_path: Path):
    """A non-SGLang / reduced Magpie layout is not a failure."""
    assert ensure_client_trust_compat(tmp_path) is True


def test_drifted_script_is_reported(tmp_path: Path):
    """An unrecognisable client block must be surfaced, not silently accepted."""
    _write_scripts(tmp_path, mi300x="#!/usr/bin/env bash\necho hand-edited\n")
    assert ensure_client_trust_compat(tmp_path) is False


def test_one_drifted_script_does_not_starve_its_sibling(tmp_path: Path):
    """A drifted script must not short-circuit the healthy one."""
    _write_scripts(
        tmp_path,
        mi300x="#!/usr/bin/env bash\necho hand-edited\n",
        mi355x=_UPSTREAM_SGLANG_MI355X_SH,
    )
    assert ensure_client_trust_compat(tmp_path) is False
    mi355x = (tmp_path / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(
        encoding="utf-8"
    )
    assert "magpie_run_benchmark_serving_remote_direct trust" in mi355x


# --- preflight wrapper: multi-node gate + warn-only contract ----------------


def test_single_node_leaves_the_tree_untouched(upstream_magpie: Path, monkeypatch):
    """Single-node keeps whatever install.sh left; the fix is multi-node scoped."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    before = {
        p.name: p.read_text(encoding="utf-8")
        for p in (upstream_magpie / "Magpie" / "scripts" / "benchmark").glob("*.sh")
    }

    assert _ensure_client_trust_compat(str(upstream_magpie)) is True

    after = {
        p.name: p.read_text(encoding="utf-8")
        for p in (upstream_magpie / "Magpie" / "scripts" / "benchmark").glob("*.sh")
    }
    assert after == before
    assert "MAGPIE_TRUST_REMOTE_CODE" not in after["sglang_mi300x.sh"]


def test_multi_node_patches_through_the_wrapper(upstream_magpie: Path, monkeypatch):
    """The multi-node path is the one that actually gets the gate."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    assert _ensure_client_trust_compat(str(upstream_magpie)) is True
    text = (upstream_magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh").read_text(
        encoding="utf-8"
    )
    assert "magpie_run_benchmark_serving_remote_direct trust" in text


def test_drift_warns_but_never_raises(tmp_path: Path, monkeypatch, capsys):
    """Preflight must degrade, not abort: most models need no remote code."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    _write_scripts(tmp_path, mi300x="#!/usr/bin/env bash\necho hand-edited\n")

    assert _ensure_client_trust_compat(str(tmp_path)) is False

    out = capsys.readouterr().out
    assert "custom-tokenizer trust patch" in out
    assert "MAGPIE_TRUST_REMOTE_CODE=1" in out
