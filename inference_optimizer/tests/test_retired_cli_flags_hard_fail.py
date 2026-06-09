# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Blocker #4 — retired CLI flags must hard-fail with a migration hint."""

from __future__ import annotations

import pytest


_RETIRED_FLAGS = (
    "--use-roofline-composite",
    "--no-use-roofline-composite",
    "--deny-direct-profile",
    "--no-deny-direct-profile",
    "--force-roofline-after-baseline",
    "--no-force-roofline-after-baseline",
)


def _minimal_optimize_args() -> list[str]:
    """The smallest ``optimize`` argv that lets the retired-flag action fire."""
    return ["optimize", "--model", "/tmp/no-such-model"]


@pytest.mark.parametrize("flag", _RETIRED_FLAGS)
def test_retired_flag_exits_with_enable_roofline_hint(
    capsys: pytest.CaptureFixture[str], flag: str,
):
    """Each retired spelling raises ``SystemExit`` and names ``--enable-roofline``."""
    from inference_optimizer.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(_minimal_optimize_args() + [flag])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert flag in err, (
        f"retired-flag error must name {flag!r} so operators can grep "
        f"their scripts; got: {err!r}"
    )
    assert "--enable-roofline" in err, (
        f"retired-flag error for {flag!r} must point operators at "
        f"--enable-roofline; got: {err!r}"
    )


def test_enable_roofline_still_works():
    """Sanity guard: the surviving ``--enable-roofline`` mode-select still parses."""
    from inference_optimizer.cli import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(_minimal_optimize_args() + ["--no-enable-roofline"])
    assert ns.enable_roofline is False
    ns = parser.parse_args(_minimal_optimize_args() + ["--enable-roofline"])
    assert ns.enable_roofline is True
