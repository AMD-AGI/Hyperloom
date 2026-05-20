"""libcst AST scanner tests (P2 PR-E)."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework_agent.agent.ast_scanner import (
    AstScanResult,
    scan_framework_args,
    scan_module,
)
from framework_agent.agent.flag_discovery import DiscoveredFlag


_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# scan_module per-file
# ---------------------------------------------------------------------------
def test_scan_module_argparse_pattern_returns_flags():
    """Fixture's build_parser() has 5 add_argument calls."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/server_args.py"
    flags, err = scan_module(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    assert err is None
    argparse_flags = [f for f in flags if f.via == "argparse"]
    assert len(argparse_flags) == 5
    names = {f.flag_name for f in argparse_flags}
    assert "--max-running-requests" in names
    assert "--cuda-graph-max-bs" in names
    assert "--chunked-prefill-size" in names
    assert "--disable-radix-cache" in names
    assert "--kv-cache-dtype" in names


def test_scan_module_dataclass_pattern_returns_fields():
    """Fixture's ServerArgs @dataclass has 7 fields."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/server_args.py"
    flags, err = scan_module(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    assert err is None
    dc_flags = [f for f in flags if f.via == "dataclass"]
    names = {f.flag_name for f in dc_flags}
    assert "model_path" in names
    assert "max_num_seqs" in names
    assert "schedule_conservativeness" in names
    assert "kv_cache_dtype" in names


def test_scan_module_pydantic_pattern_returns_fields():
    """Fixture's SchedulerConfig(BaseModel) has 5 fields."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/configs/scheduler.py"
    flags, err = scan_module(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    assert err is None
    pyd_flags = [f for f in flags if f.via == "pydantic"]
    names = {f.flag_name for f in pyd_flags}
    assert "max_num_batched_tokens" in names
    assert "max_prefill_tokens" in names
    assert "enable_dp_attention" in names


def test_scan_module_skips_private_fields():
    """`_private` / `__dunder` fields must not appear."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/server_args.py"
    flags, err = scan_module(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    assert err is None
    for f in flags:
        assert not f.flag_name.startswith("_"), f.flag_name


def test_scan_module_parse_error_returns_err_string():
    """broken.py with malformed def must surface libcst_parse_failed."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/configs/broken.py"
    flags, err = scan_module(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    assert flags == []
    assert err is not None
    assert "libcst_parse_failed" in err


def test_scan_module_records_line_numbers():
    """Every flag must have line > 0 (PositionProvider working)."""
    path = _FIXTURES / "mini_vllm/vllm/engine/arg_utils.py"
    flags, err = scan_module(path, framework="vllm", source_root=_FIXTURES / "mini_vllm")
    assert err is None
    assert all(f.line > 0 for f in flags)


# ---------------------------------------------------------------------------
# scan_framework_args aggregate
# ---------------------------------------------------------------------------
def test_scan_framework_args_sglang_fixture_meets_minimum_count():
    """Design §13 P2.10 verification: real vllm tree expects >=10 flags.
    Our mini fixture has 5 argparse + 7 dataclass + 5 pydantic = 17, even
    with broken.py triggering fallback we're well above 10."""
    result = scan_framework_args("sglang", _FIXTURES / "mini_sglang")
    assert isinstance(result, AstScanResult)
    assert len(result.flags) >= 10
    # ServerArgs.max_num_seqs (dataclass) + nothing on the CLI side --
    # confirm both surfaces are represented.
    surfaces = {f.surface for f in result.flags}
    assert surfaces == {"cli", "config"}


def test_scan_framework_args_broken_file_triggers_grep_fallback():
    """broken.py forces fallback for that one file. With 3 files total
    in the sglang fixture, 1/3 = 33% > 10% threshold -> aggregate mode
    flips to grep_fallback."""
    result = scan_framework_args("sglang", _FIXTURES / "mini_sglang")
    assert result.parse_failures >= 1
    # Aggregate mode reflects the >=10% fallback ratio.
    assert result.mode == "grep_fallback"
    # Failed file is reported.
    failed_paths = [Path(p).name for p, _ in result.failed_files]
    assert "broken.py" in failed_paths


def test_scan_framework_args_vllm_fixture_libcst_only():
    """No broken files in the vllm fixture -> mode='libcst'."""
    result = scan_framework_args("vllm", _FIXTURES / "mini_vllm")
    assert result.mode == "libcst"
    assert result.parse_failures == 0
    # arg_utils.py: 5 argparse + 6 dataclass
    assert len(result.flags) >= 10
    cli_names = {f.flag_name for f in result.flags if f.surface == "cli"}
    assert "--max-model-len" in cli_names
    assert "--gpu-memory-utilization" in cli_names


def test_scan_framework_args_results_are_deduped_and_ranked():
    result = scan_framework_args("vllm", _FIXTURES / "mini_vllm")
    # CLI surface ranks before config surface.
    surface_order = [f.surface for f in result.flags]
    if "cli" in surface_order and "config" in surface_order:
        first_config = surface_order.index("config")
        last_cli = max(
            i for i, s in enumerate(surface_order) if s == "cli"
        )
        assert last_cli < first_config
    # Fingerprint uniqueness.
    fps = [f.fingerprint() for f in result.flags]
    assert len(fps) == len(set(fps))


def test_scan_framework_args_flag_metadata_complete():
    """Every flag has module / source_path / line / type_hint set."""
    result = scan_framework_args("vllm", _FIXTURES / "mini_vllm")
    for f in result.flags:
        assert f.flag_name
        assert f.module
        assert f.source_path
        assert f.line > 0
        assert f.type_hint
        assert f.via in (
            "argparse", "dataclass", "pydantic",
            "argparse_grep", "dataclass_grep",
        )
