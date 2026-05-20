"""DiscoveredFlag normalisation + dedup tests (P2 PR-E)."""

from __future__ import annotations

from pathlib import Path

from framework_agent.agent.flag_discovery import (
    DiscoveredFlag,
    cli_flag_names,
    dedup_and_rank,
    path_to_module,
    to_json_records,
)


def _flag(name: str, *, surface: str = "cli", via: str = "argparse",
          line: int = 1, source_path: str = "/a/b.py") -> DiscoveredFlag:
    return DiscoveredFlag(
        flag_name=name,
        module="m",
        source_path=source_path,
        line=line,
        via=via,  # type: ignore[arg-type]
        type_hint="int",
        default_repr="None",
        help_text="",
        surface=surface,  # type: ignore[arg-type]
    )


def test_dedup_removes_identical_fingerprints():
    a = _flag("--x", line=10)
    b = _flag("--x", line=10)
    c = _flag("--x", line=20)
    out = dedup_and_rank([a, b, c])
    # a and b share fingerprint (same name+surface+path+line); c keeps.
    assert len(out) == 2


def test_dedup_keeps_distinct_surfaces_of_same_flag():
    """argparse cli + dataclass config -- two surfaces of one tunable."""
    cli = _flag("kv_cache_dtype", surface="cli", line=10)
    cfg = _flag("kv_cache_dtype", surface="config", line=20)
    out = dedup_and_rank([cli, cfg])
    assert len(out) == 2


def test_dedup_orders_cli_before_config():
    cli = _flag("--alpha", surface="cli", line=10)
    cfg = _flag("beta", surface="config", line=20)
    out = dedup_and_rank([cfg, cli])
    assert out[0].surface == "cli"
    assert out[1].surface == "config"


def test_dedup_orders_argparse_before_grep_variant():
    real = _flag("--alpha", via="argparse", line=10)
    grep = _flag("--alpha", via="argparse_grep", line=20)
    out = dedup_and_rank([grep, real])
    assert out[0].via == "argparse"
    assert out[1].via == "argparse_grep"


def test_dedup_alphabetical_within_via_rank():
    a = _flag("--zeta", line=10)
    b = _flag("--alpha", line=20)
    out = dedup_and_rank([a, b])
    assert out[0].flag_name == "--alpha"
    assert out[1].flag_name == "--zeta"


def test_cli_flag_names_pick_cli_surface_only():
    flags = [
        _flag("--x", surface="cli"),
        _flag("y", surface="config"),
        _flag("--z", surface="cli"),
    ]
    assert cli_flag_names(flags) == ["--x", "--z"]


def test_to_json_records_returns_plain_dicts():
    flags = [_flag("--x")]
    records = to_json_records(flags)
    assert isinstance(records, list)
    assert isinstance(records[0], dict)
    assert records[0]["flag_name"] == "--x"
    assert records[0]["via"] == "argparse"


def test_path_to_module_relative_dotted():
    root = Path("/sgl-workspace/vllm")
    path = Path("/sgl-workspace/vllm/vllm/engine/arg_utils.py")
    assert path_to_module(path, root) == "vllm.engine.arg_utils"


def test_path_to_module_outside_root_falls_back_to_stem():
    root = Path("/elsewhere")
    path = Path("/tmp/foo.py")
    assert path_to_module(path, root) == "foo"
