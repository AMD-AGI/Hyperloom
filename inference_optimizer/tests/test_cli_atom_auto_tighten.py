# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""IR-8 tests: --framework atom only enforces the ``--nodes >= 2`` fail-fast guard and does not auto-flip phase knobs."""

from __future__ import annotations

import argparse
import inspect

import pytest

from inference_optimizer import cli as optimizer_cli


def _fresh_args(**overrides) -> argparse.Namespace:
    """Mint a Namespace matching the atom-invocation default surface."""
    base = dict(
        no_kernel=False,
        no_framework=False,
        enable_roofline=True,
        nodes=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_atom_auto_tighten_only_guards_multi_node(capsys):
    """Vanilla ``--framework atom`` must NOT auto-flip any phase knobs; the auto-disabled list is empty."""
    args = _fresh_args()
    disabled = optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is False
    assert args.no_framework is False
    assert args.enable_roofline is True
    assert disabled == []
    out = capsys.readouterr().out
    assert "framework=atom" in out
    assert "no auto-disable applied" in out
    assert "--no-kernel" not in out
    assert "--no-framework" not in out
    assert "--no-enable-roofline" not in out


def test_atom_no_framework_flag_preserved_when_user_passes_it(capsys):
    """Explicit ``--no-framework`` keeps ``args.no_framework`` True; auto-tighten respects the operator choice."""
    args = _fresh_args(no_framework=True)
    optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_framework is True


def test_atom_no_kernel_flag_preserved_when_user_passes_it():
    """Same regression guard for ``--no-kernel``."""
    args = _fresh_args(no_kernel=True)
    optimizer_cli._apply_atom_auto_tighten(args)
    assert args.no_kernel is True


def test_atom_auto_tighten_does_not_touch_enable_roofline(capsys):
    """Regression guard: enable_roofline survives auto-tighten unchanged at both True and False."""
    for initial in (True, False):
        args = _fresh_args(enable_roofline=initial)
        optimizer_cli._apply_atom_auto_tighten(args)
        assert args.enable_roofline is initial, (
            f"enable_roofline={initial} must not be flipped by atom auto-tighten"
        )


def test_atom_auto_tighten_rejects_multi_node():
    """--framework atom + --nodes 2 must SystemExit(2): atom has no multi-node TP wiring."""
    args = _fresh_args(nodes=2)
    with pytest.raises(SystemExit) as exc:
        optimizer_cli._apply_atom_auto_tighten(args)
    assert exc.value.code == 2


def test_atom_auto_tighten_accepts_single_node_explicitly():
    """--nodes 1 does not trip the >=2 guard."""
    args = _fresh_args(nodes=1)
    optimizer_cli._apply_atom_auto_tighten(args)


def test_framework_choices_include_atom():
    """Parser-level: --framework atom is accepted by argparse."""
    parser = optimizer_cli._build_parser()
    parsed = parser.parse_args([
        "optimize", "--model", "/tmp/m", "--framework", "atom",
    ])
    assert parsed.framework == "atom"


def test_framework_choices_reject_unknown_value():
    """Regression guard: the whitelist must not silently accept unknown frameworks."""
    parser = optimizer_cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/tmp/m", "--framework", "tensorrt",
        ])


# Cross-cutting static guard: purpose narrowed to multi-node guard only.
def test_atom_auto_tighten_only_purpose_is_multi_node_guard():
    """Source-level guard: the function body must not flip historical targets; ``nodes`` stays as the guard signal."""
    src = inspect.getsource(optimizer_cli._apply_atom_auto_tighten)
    # Strip the docstring before checking; it may reference historical flips.
    body_only = src.split('"""', 2)[-1] if '"""' in src else src
    assert "args.no_kernel = True" not in body_only, (
        "auto-tighten body must not flip no_kernel"
    )
    assert "args.no_framework = True" not in body_only, (
        "auto-tighten body must not flip no_framework"
    )
    assert "args.enable_roofline" not in body_only, (
        "auto-tighten body must not touch enable_roofline"
    )
    assert "nodes" in body_only


def test_atom_auto_tighten_log_line_is_single_line(capsys):
    """Operator-readability gate: emit exactly ONE atom-context log line."""
    args = _fresh_args()
    optimizer_cli._apply_atom_auto_tighten(args)
    out = capsys.readouterr().out
    atom_lines = [l for l in out.splitlines() if "framework=atom" in l]
    assert len(atom_lines) == 1, (
        f"expected exactly one atom-context line, got "
        f"{len(atom_lines)}: {atom_lines!r}"
    )


# Forward-looking alias for the multi-node-guard-only behaviour.
def test_assert_atom_single_node_alias_resolves_to_same_callable():
    """`_assert_atom_single_node` is a forward-looking alias for `_apply_atom_auto_tighten`; both resolve to the same callable."""
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
