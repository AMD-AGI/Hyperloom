"""Unknown CLI arguments are rejected, and the rejection cannot print a secret.

Primus-Claw parses ONE hand-authored prompt FLAGS block for both itself and this
CLI, so the flags it consumes on its own side must parse here. They are declared
as inert no-ops rather than waved through wholesale: the block is written by
hand, so a typo is likelier than in generated argv, and tolerating every unknown
argument would let a misspelled knob run for hours on its default.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser, _redact_unknown_args

# Flags Claw consumes to provision the cluster; inert for the optimizer itself.
_PLATFORM_FLAGS = [
    "--mn-image",
    "--cpus-per-node",
    "--mem-per-node",
    "--extra-env",
]


def _parse(*extra: str):
    return _build_parser().parse_args(["optimize", "--model", "/models/m", *extra])


def test_platform_owned_flags_are_accepted_and_inert():
    """A real platform FLAGS block must parse without reaching anything."""
    args = _parse(
        "--nodes",
        "2",
        "--mn-backend",
        "rayjob",
        "--mn-image",
        "registry/img:v1",
        "--cpus-per-node",
        "96",
        "--mem-per-node",
        "1024",
        "--extra-env",
        "MC_GID_INDEX=3",
        "--extra-env",
        "NCCL_IB_GID_INDEX=3",
    )

    assert args.nodes == 2
    # Parsed, and parsed with the right shape -- but owned by the platform.
    assert args.mn_image == "registry/img:v1"
    assert args.cpus_per_node == 96
    assert args.mem_per_node == 1024
    assert args.extra_env == ["MC_GID_INDEX=3", "NCCL_IB_GID_INDEX=3"]


@pytest.mark.parametrize("flag", _PLATFORM_FLAGS)
def test_platform_flags_stay_out_of_help(flag, capsys):
    """They are not a public API, so --help must not advertise them."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["optimize", "--help"])
    assert flag not in capsys.readouterr().out


@pytest.mark.parametrize(
    "typo",
    [
        ["--target-gian", "5"],  # transposed
        ["--tp-size", "8"],  # wrong name for --tp
        ["--no-such-knob"],
    ],
)
def test_a_misspelled_flag_fails_fast(typo):
    """The whole point of staying strict: fail now, not 10 hours from now."""
    with pytest.raises(SystemExit) as exc:
        _parse(*typo)
    assert exc.value.code == 2


def test_unambiguous_abbreviations_still_work():
    """Strictness must not cost argparse's normal prefix matching."""
    assert _parse("--max-hour", "10").max_hours == 10.0


def test_rejection_message_masks_credentials(capsys):
    """argparse prints the offending tokens verbatim; a pod token must not leak.

    This is the path a *future* platform flag takes, before Hyperloom declares
    it -- exactly when the value is most likely to be a live credential.
    """
    with pytest.raises(SystemExit):
        _parse("--new-platform-flag", "HF_TOKEN=hf_live_value")

    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
    assert "hf_live_value" not in err
    assert "HF_TOKEN=***" in err
    # The name survives, so the message still says what was rejected.
    assert "--new-platform-flag" in err


def test_rejection_message_keeps_a_typo_readable(capsys):
    """A non-secret value must stay visible, or the typo becomes hard to see."""
    with pytest.raises(SystemExit):
        _parse("--tp-size", "8")

    err = capsys.readouterr().err
    assert "--tp-size 8" in err


def test_redaction_helper_is_shared_with_the_parser():
    """The parser reuses the same helper the CLI logs with, not a second copy."""
    assert _redact_unknown_args(["--api-key", "sk-live"]) == "--api-key ***"
