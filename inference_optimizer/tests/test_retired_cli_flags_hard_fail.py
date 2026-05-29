"""Blocker #4 — retired CLI flags must hard-fail with a migration hint.

After the single-path roofline/profile refactor the following flags no
longer have any meaning, and operator scripts must be updated to use
``--enable-roofline`` / ``--no-enable-roofline``:

  * ``--use-roofline-composite`` (and ``--no-`` variant)
  * ``--deny-direct-profile``    (and ``--no-`` variant)
  * ``--force-roofline-after-baseline`` (and ``--no-`` variant) —
    PRELUDE-initial analysis is now unconditional, no replacement.

The contract is **hard-fail, not silent-alias**: silent aliases hide the
fact that the underlying semantics changed (composite/direct-profile
split is gone; PRELUDE-initial roofline is no longer optional). Any
script that still passes one of these flags must learn about the change
the next time it runs.
"""

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
    """The smallest ``optimize`` argv that gets past required arguments
    before the retired flag is processed. Argparse evaluates positional
    / required arguments after option actions, so the retired-flag
    action fires regardless of whether the rest is valid."""
    return ["optimize", "--model", "/tmp/no-such-model"]


@pytest.mark.parametrize("flag", _RETIRED_FLAGS)
def test_retired_flag_exits_with_enable_roofline_hint(
    capsys: pytest.CaptureFixture[str], flag: str,
):
    """Each retired spelling must:

    * raise ``SystemExit`` (argparse ``parser.error`` exits 2);
    * emit a message that names ``--enable-roofline`` so operators have
      a one-step migration path.
    """
    from inference_optimizer.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(_minimal_optimize_args() + [flag])
    # argparse uses exit code 2 for parse errors.
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
    """Sanity guard: the surviving mode-select must still parse so the
    retired-flag plumbing did not accidentally collide with it."""
    from inference_optimizer.cli import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(_minimal_optimize_args() + ["--no-enable-roofline"])
    assert ns.enable_roofline is False
    ns = parser.parse_args(_minimal_optimize_args() + ["--enable-roofline"])
    assert ns.enable_roofline is True
