"""IR-8 tests: --framework atom auto-tightens incompatible phases and
fails fast on multi-node.

Targets ``_apply_atom_auto_tighten`` in inference_optimizer.cli.

History: pre-Magpie-atom-PROFILE-wiring, ``--no-enable-roofline`` was
also auto-flipped here. Once Magpie's ``atom_mi*x.sh`` learned to
bridge ``PROFILE=1`` to atom's ``--torch-profiler-dir``, profile /
roofline / TraceLens started working on atom natively and the
roofline auto-disable was removed. The tests now assert that
``--enable-roofline`` is *preserved* at its default for atom.
"""

from __future__ import annotations

import argparse

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


def test_atom_auto_tighten_flips_framework_only_when_at_defaults(capsys):
    """Vanilla ``--framework atom`` (no other phase flags) must auto-flip
    no_framework only — kernel-agent is wired for atom in Phase 2 of the
    atom_plan/ lift, so ``--no-kernel`` is preserved at its False
    default. ``enable_roofline`` also stays at True (profile / roofline
    / TraceLens work on atom)."""
    args = _fresh_args()
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    # Kernel-agent now works on atom (atom_plan/phase2_open_kernel_agent).
    assert args.no_kernel is False
    assert args.no_framework is True
    assert args.enable_roofline is True
    assert "--no-kernel" not in disabled
    assert "--no-framework" in disabled
    assert "--no-enable-roofline" not in disabled
    out = capsys.readouterr().out
    assert "framework=atom" in out
    assert "--no-framework" in out
    # Regression guard: the log line must not mention --no-kernel either
    # as an auto-disable target or as a "still disabled" leftover.
    assert "--no-kernel" not in out


def test_atom_auto_tighten_log_line_mentions_framework_only(capsys):
    """The single auto-disabling log line names only ``--no-framework``
    after Phase 2. Operator readability gate."""
    args = _fresh_args()
    optimizer_cli._apply_atom_auto_tighten(args)
    out = capsys.readouterr().out
    auto_lines = [l for l in out.splitlines() if "auto-disabling" in l]
    # Exactly one auto-disable line for the framework knob.
    assert len(auto_lines) == 1, (
        f"expected exactly one auto-disabling line; got {auto_lines!r}"
    )
    assert "--no-framework" in auto_lines[0]
    assert "--no-kernel" not in auto_lines[0]


def test_atom_auto_tighten_preserves_explicit_no_kernel():
    """Explicit ``--no-kernel`` from the operator must not be reverted.
    Auto-tighten only flips defaults; it never overrides an
    operator-supplied value, regardless of direction."""
    args = _fresh_args(no_kernel=True)
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is True
    # --no-kernel was already set; it should NOT appear in the
    # auto-disabled list (which is "what we flipped").
    assert "--no-kernel" not in disabled


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


def test_atom_auto_tighten_idempotent_when_flags_already_set(capsys):
    """If the operator already passed --no-kernel / --no-framework,
    no flip happens and no log line is emitted."""
    args = _fresh_args(no_kernel=True, no_framework=True)
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert disabled == []
    out = capsys.readouterr().out
    assert "auto-disabling" not in out


def test_atom_auto_tighten_preserves_explicit_partial_override(capsys):
    """If the operator passed --no-kernel but left framework at the
    enabled default, we still flip framework — no_kernel stays at its
    operator-supplied True."""
    args = _fresh_args(no_kernel=True)
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is True
    assert args.no_framework is True
    assert "--no-kernel" not in disabled  # already set, not auto-flipped
    assert "--no-framework" in disabled


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
# Phase 2.6 G4 cross-cutting guard: auto-tighten log is operator-readable
# ---------------------------------------------------------------------------
def test_atom_auto_tighten_log_line_is_single_line(capsys):
    """Operator-readability gate: the auto-disable log must emit
    exactly ONE line so a `grep auto-disabling kernel-agent.env.sh`
    returns a single record."""
    args = _fresh_args()
    optimizer_cli._apply_atom_auto_tighten(args)
    out = capsys.readouterr().out
    auto_disable_lines = [
        l for l in out.splitlines() if "auto-disabling" in l
    ]
    assert len(auto_disable_lines) == 1, (
        f"expected exactly one auto-disabling line, got "
        f"{len(auto_disable_lines)}: {auto_disable_lines!r}"
    )
