"""IR-8 tests: --framework atom validates multi-node guard but no
longer auto-flips any kernel / framework phase knobs.

Targets ``_apply_atom_auto_tighten`` in inference_optimizer.cli.

History:
* pre-Magpie-atom-PROFILE-wiring, ``--no-enable-roofline`` was
  auto-flipped here. Once Magpie's ``atom_mi*x.sh`` learned to bridge
  ``PROFILE=1`` to atom's ``--torch-profiler-dir``, profile / roofline /
  TraceLens started working on atom natively and the roofline auto-
  disable was removed.
* atom_plan/phase2_open_kernel_agent removed the ``--no-kernel`` auto-
  flip (kernel-agent now wired for atom — source roots, reusable-
  kernel ledger, server-flag pre-flight probe).
* atom_plan/phase3_open_framework_agent removed the ``--no-framework``
  auto-flip (framework-agent now wired for atom — repo URL points at
  https://github.com/ROCm/ATOM.git; the policy gate for
  ``framework_atom_action_unsupported`` was also fully removed).

After Phase 3 the only remaining behaviour is the ``--nodes >= 2``
fail-fast guard. Multi-node TP wiring on atom is deferred; the guard
saves operators a ~6-min cold start on a doomed run.
"""

from __future__ import annotations

import argparse
import inspect

import pytest

from inference_optimizer import cli as optimizer_cli


def _fresh_args(**overrides) -> argparse.Namespace:
    """Mint a Namespace with the same default surface ``_run_optimize``
    would see for an atom invocation."""
    base = dict(
        no_kernel=False,
        no_framework=False,
        enable_roofline=True,
        nodes=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_atom_auto_tighten_only_guards_multi_node(capsys):
    """Vanilla ``--framework atom`` (no other phase flags) must NOT
    auto-flip any phase knobs after Phase 3. kernel-agent + framework-
    agent + profile / roofline / TraceLens are all wired for atom; the
    function's only remaining purpose is the ``--nodes >= 2`` fail-
    fast guard. The returned auto-disabled list is empty."""
    args = _fresh_args()
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    # No flags auto-flipped any more.
    assert args.no_kernel is False
    assert args.no_framework is False
    assert args.enable_roofline is True
    assert disabled == []
    out = capsys.readouterr().out
    # Operator-readable log line still emitted so the operator can grep
    # for the atom-tighten signal in launch logs.
    assert "framework=atom" in out
    assert "no auto-disable applied" in out
    # Regression guards: none of the historical flip targets remain.
    assert "--no-kernel" not in out
    assert "--no-framework" not in out
    assert "--no-enable-roofline" not in out


def test_atom_no_framework_flag_preserved_when_user_passes_it(capsys):
    """Explicit ``--no-framework --framework atom`` keeps
    ``args.no_framework`` True; auto-tighten does NOT fight an explicit
    operator choice (this was implicit in the previous behaviour; pin
    it here so a future re-introduction of an auto-flip remembers to
    respect the operator's value)."""
    args = _fresh_args(no_framework=True)
    optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_framework is True


def test_atom_no_kernel_flag_preserved_when_user_passes_it():
    """Same regression guard for ``--no-kernel``."""
    args = _fresh_args(no_kernel=True)
    optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is True


def test_atom_auto_tighten_does_not_touch_enable_roofline(capsys):
    """Regression guard: the historical auto-disable of roofline was
    removed when atom's profiler wiring landed. Explicitly verify that
    enable_roofline survives the auto-tighten unchanged at both True
    and False inputs."""
    for initial in (True, False):
        args = _fresh_args(enable_roofline=initial)
        optimizer_cli._apply_atom_auto_tighten(args)
        assert args.enable_roofline is initial, (
            f"enable_roofline={initial} must not be flipped by atom auto-tighten"
        )


def test_atom_auto_tighten_rejects_multi_node():
    """--framework atom + --nodes 2 must SystemExit(2) — atom has no
    multi-node TP wiring, so the run would burn a 6-min cold start
    before failing in the Magpie wrapper."""
    args = _fresh_args(nodes=2)
    with pytest.raises(SystemExit) as exc:
        optimizer_cli._apply_atom_auto_tighten(args)
    assert exc.value.code == 2


def test_atom_auto_tighten_accepts_single_node_explicitly():
    """--nodes 1 is the only allowed value; verifying we don't trip the
    >=2 guard on the explicit default."""
    args = _fresh_args(nodes=1)
    # Should not raise.
    optimizer_cli._apply_atom_auto_tighten(args)


def test_framework_choices_include_atom():
    """Parser-level: --framework atom must be accepted by argparse so
    the auto-tighten path is reachable in the first place."""
    parser = optimizer_cli._build_parser()
    parsed = parser.parse_args([
        "optimize", "--model", "/tmp/m", "--framework", "atom",
    ])
    assert parsed.framework == "atom"


def test_framework_choices_reject_unknown_value():
    """Regression guard: extending the whitelist to atom must not silently
    accept other strings (e.g. 'tensorrt')."""
    parser = optimizer_cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/tmp/m", "--framework", "tensorrt",
        ])


# ---------------------------------------------------------------------------
# Phase 3.5 G3 cross-cutting static guard:
# _apply_atom_auto_tighten purpose narrowed to multi-node guard only.
# ---------------------------------------------------------------------------
def test_atom_auto_tighten_only_purpose_is_multi_node_guard():
    """Source-level guard: the function body must not mention any of the
    historical flip targets (no_kernel / no_framework / enable_roofline)
    so a future edit that re-introduces an auto-flip has to be intentional.
    ``nodes`` must remain as the multi-node guard signal."""
    src = inspect.getsource(optimizer_cli._apply_atom_auto_tighten)
    # Strip the docstring before checking — the docstring is allowed to
    # reference the historical flips for context.
    body_only = src.split('"""', 2)[-1] if '"""' in src else src
    assert "args.no_kernel = True" not in body_only, (
        "auto-tighten body must not flip no_kernel after Phase 2"
    )
    assert "args.no_framework = True" not in body_only, (
        "auto-tighten body must not flip no_framework after Phase 3"
    )
    assert "args.enable_roofline" not in body_only, (
        "auto-tighten body must not touch enable_roofline"
    )
    # The multi-node guard literal stays.
    assert "nodes" in body_only


def test_atom_auto_tighten_log_line_is_single_line(capsys):
    """Operator-readability gate: emit exactly ONE atom-context log
    line so a `grep framework=atom kernel-agent.env.sh` returns a
    single record."""
    args = _fresh_args()
    optimizer_cli._apply_atom_auto_tighten(args)
    out = capsys.readouterr().out
    atom_lines = [l for l in out.splitlines() if "framework=atom" in l]
    assert len(atom_lines) == 1, (
        f"expected exactly one atom-context line, got "
        f"{len(atom_lines)}: {atom_lines!r}"
    )


# ---------------------------------------------------------------------------
# Gap G9 — forward-looking alias for the post-Phase-3 behaviour name.
# ---------------------------------------------------------------------------
def test_assert_atom_single_node_alias_resolves_to_same_callable():
    """`_assert_atom_single_node` is a forward-looking alias for
    `_apply_atom_auto_tighten` (atom_gap1.md G9). After Phase 3 the
    function no longer auto-tightens anything; the new name reflects
    the actual contract. Old name kept for git-blame continuity +
    test back-compat.

    Both names must resolve to the SAME callable object so a
    monkeypatch / mock against either name affects every call site.
    """
    assert hasattr(optimizer_cli, "_assert_atom_single_node")
    assert (
        optimizer_cli._assert_atom_single_node
        is optimizer_cli._apply_atom_auto_tighten
    )


def test_assert_atom_single_node_alias_exits_on_multi_node(capsys):
    """The alias must inherit the multi-node fail-fast behaviour."""
    args = _fresh_args(nodes=2)
    with pytest.raises(SystemExit) as excinfo:
        optimizer_cli._assert_atom_single_node(args)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--framework atom does not support multi-node" in err
