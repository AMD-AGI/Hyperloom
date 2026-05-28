"""B3 tests: --framework atom auto-tightens incompatible phases and
fails fast on multi-node.

Targets ``_apply_atom_auto_tighten`` in inference_optimizer.cli.
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


def test_atom_auto_tighten_flips_all_three_phases_when_at_defaults(capsys):
    """Vanilla ``--framework atom`` (no other phase flags) must auto-flip
    no_kernel, no_framework, and disable enable_roofline. The single log
    line should name all three flags so the operator sees what changed."""
    args = _fresh_args()
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is True
    assert args.no_framework is True
    assert args.enable_roofline is False
    assert "--no-kernel" in disabled
    assert "--no-framework" in disabled
    assert "--no-enable-roofline" in disabled
    out = capsys.readouterr().out
    assert "framework=atom" in out
    assert "--no-kernel" in out


def test_atom_auto_tighten_idempotent_when_flags_already_set(capsys):
    """If the operator already passed --no-kernel / --no-framework /
    --no-enable-roofline, no flip happens and no log line is emitted."""
    args = _fresh_args(no_kernel=True, no_framework=True, enable_roofline=False)
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert disabled == []
    out = capsys.readouterr().out
    assert "auto-disabling" not in out


def test_atom_auto_tighten_preserves_explicit_partial_override(capsys):
    """If the operator passed --no-kernel but left framework_pr /
    roofline at defaults, we still flip the remaining two — no_kernel
    stays at its operator-supplied True."""
    args = _fresh_args(no_kernel=True)
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is True
    assert args.no_framework is True
    assert args.enable_roofline is False
    assert "--no-kernel" not in disabled  # already set, not auto-flipped
    assert "--no-framework" in disabled
    assert "--no-enable-roofline" in disabled


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
