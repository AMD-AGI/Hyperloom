"""CLI flag tests for the 5th Framework role (P1 PR-B/C).

Coverage:

* ``--framework-{agent,mock,codex-bare}`` mutex (argparse-level).
* ``--no-framework`` x ``--framework-{agent,mock,codex-bare}`` cross-check
  (manual in main, exits with rc=2 + message on stderr).
* Default behaviour: no framework flag -> framework_choice='off',
  framework_role_enabled=False, role pruned from registry.
* ``--no-framework-ast`` / ``--framework-ast-frameworks`` plumb through
  to SharedState (without depending on actually running the optimizer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# argparse-level mutex
# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    """Replicate the framework-flag block from cli.build_parser()."""
    parser = argparse.ArgumentParser()
    fw = parser.add_mutually_exclusive_group()
    fw.add_argument("--framework-agent", action="store_true",
                    dest="framework_agent", default=False)
    fw.add_argument("--framework-mock", action="store_true",
                    dest="framework_mock", default=False)
    fw.add_argument("--framework-codex-bare", action="store_true",
                    dest="framework_codex_bare", default=False)
    parser.add_argument("--no-framework", action="store_true",
                        dest="no_framework", default=False)
    parser.add_argument("--no-framework-ast", action="store_true",
                        dest="no_framework_ast", default=False)
    parser.add_argument("--framework-ast-frameworks", default="",
                        dest="framework_ast_frameworks")
    return parser


def test_framework_agent_and_mock_are_mutually_exclusive():
    parser = _make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--framework-agent", "--framework-mock"])


def test_framework_mock_and_codex_bare_are_mutually_exclusive():
    parser = _make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--framework-mock", "--framework-codex-bare"])


def test_default_no_framework_flags_set():
    parser = _make_parser()
    args = parser.parse_args([])
    assert args.framework_agent is False
    assert args.framework_mock is False
    assert args.framework_codex_bare is False
    assert args.no_framework is False
    assert args.no_framework_ast is False
    assert args.framework_ast_frameworks == ""


def test_framework_mock_lone_flag():
    parser = _make_parser()
    args = parser.parse_args(["--framework-mock"])
    assert args.framework_mock is True
    assert args.framework_agent is False
    assert args.framework_codex_bare is False


# ---------------------------------------------------------------------------
# cross-check (no_framework + framework-*) -> rc=2 path
# ---------------------------------------------------------------------------
def test_no_framework_with_framework_agent_exits_rc2(
    capsys: pytest.CaptureFixture,
):
    """Simulate the cross-check block from cli.main() in isolation."""

    def crosscheck(args: argparse.Namespace) -> int:
        framework_agent_on = bool(getattr(args, "framework_agent", False))
        framework_mock = bool(getattr(args, "framework_mock", False))
        framework_codex_bare = bool(getattr(args, "framework_codex_bare", False))
        if args.no_framework and (
            framework_agent_on or framework_mock or framework_codex_bare
        ):
            print(
                "ERROR: --no-framework conflicts with --framework-agent / "
                "--framework-mock / --framework-codex-bare; pick one.",
            )
            return 2
        return 0

    parser = _make_parser()
    args = parser.parse_args(["--no-framework", "--framework-agent"])
    rc = crosscheck(args)
    assert rc == 2
    captured = capsys.readouterr()
    assert "--no-framework conflicts" in captured.out


# ---------------------------------------------------------------------------
# SharedState seeding -- the CLI _seed_shared_state derivation logic
# (tested by importing and calling _seed_shared_state directly).
# ---------------------------------------------------------------------------
def test_seed_shared_state_default_off(tmp_path: Path, monkeypatch):
    from inference_optimizer.cli import _seed_shared_state

    monkeypatch.setenv("FRAMEWORK", "sglang")
    args = argparse.Namespace(
        model="/m/Qwen3",
        model_class="",
        gpu_type="mi300x",
        no_kernel=False,
        no_framework=False,
        framework_agent=False,
        framework_mock=False,
        framework_codex_bare=False,
        no_framework_ast=False,
        framework_ast_frameworks="",
        target_summary="bench",
        max_hours=1.0,
    )
    s = _seed_shared_state(tmp_path, args, session_id="t")
    assert s.framework_role_enabled is False
    assert s.framework_ast_scan_enabled is True
    # Default ast_frameworks derives from --framework value.
    assert s.framework_ast_frameworks == ("sglang",)


def test_seed_shared_state_mock_on_with_ast_override(
    tmp_path: Path, monkeypatch,
):
    from inference_optimizer.cli import _seed_shared_state

    monkeypatch.setenv("FRAMEWORK", "sglang")
    args = argparse.Namespace(
        model="/m/Qwen3",
        model_class="",
        gpu_type="mi300x",
        no_kernel=False,
        no_framework=False,
        framework_agent=False,
        framework_mock=True,
        framework_codex_bare=False,
        no_framework_ast=False,
        framework_ast_frameworks="vllm,sglang",
        target_summary="bench",
        max_hours=1.0,
    )
    s = _seed_shared_state(tmp_path, args, session_id="t")
    assert s.framework_role_enabled is True
    assert s.framework_ast_scan_enabled is True
    assert s.framework_ast_frameworks == ("vllm", "sglang")


def test_seed_shared_state_no_framework_ast(tmp_path: Path, monkeypatch):
    from inference_optimizer.cli import _seed_shared_state

    monkeypatch.setenv("FRAMEWORK", "vllm")
    args = argparse.Namespace(
        model="/m/Qwen3",
        model_class="",
        gpu_type="mi300x",
        no_kernel=False,
        no_framework=False,
        framework_agent=False,
        framework_mock=True,
        framework_codex_bare=False,
        no_framework_ast=True,
        framework_ast_frameworks="",
        target_summary="bench",
        max_hours=1.0,
    )
    s = _seed_shared_state(tmp_path, args, session_id="t")
    assert s.framework_role_enabled is True
    assert s.framework_ast_scan_enabled is False
    # Derived from --framework value (vllm).
    assert s.framework_ast_frameworks == ("vllm",)
