# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Round-trip test binding the pod GPU sampler's real CSV header to the orchestrator's header predicates.

The sampler (``multi_node/scripts/launch_infera_node.py``) runs on a remote Infera pod; the consumer
(``orchestrator/actions/executors/benchmark_result.py``) reads what it wrote, later and elsewhere. Nothing connects
them but the CSV header, and a predicate that stops matching drops a field silently -- the gpu_monitor samples
quietly become empty and nothing raises. So the sampler's own argument tuple and verbatim ``--csv`` output from both
ends of Hyperloom's supported ROCm window are the input here, and every field is asserted to resolve to a *named*
column and to no neighbouring column that merely resembles it.

The fixtures in ``fixtures/amd_smi_sampler/`` are unedited tool output captured on a gfx1151 host (AMD Radeon 8060S
Graphics). Set ``HYPERLOOM_GPU_SAMPLER_LIVE=1`` to additionally re-capture from the local ``amd-smi`` and fail if it
has drifted from them; that invokes the GPU and must be run under the host's GPU lock.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - optional, opt-in amd-smi re-capture.
import sys
from pathlib import Path

import pytest

import hyperloom.orchestrator.actions.executors.benchmark_result as br

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "amd_smi_sampler"

#: The sampler flags these fixtures were captured with. Pinned separately from the fixtures so that changing the
#: sampler's query fails this test instead of silently testing a header nothing produces any more.
_PINNED_SAMPLER_ARGS: tuple[str, ...] = ("metric", "--usage", "--mem-usage", "--power", "--temperature")

#: ``amd-smi --csv`` captures, by tool version. Hyperloom supports ROCm 7.2.x and 10.0 simultaneously, so every
#: major here has to parse. The spread is wider than the column counts suggest: 26.2.2 (ROCm 7.2.3, the oldest
#: SUPPORTED stack) emits 20 columns with ``usage`` as a single column, 26.5.0 emits 45 and expands ``usage`` into
#: ``gfx_activity`` while dropping ``hotspot`` entirely, and 27.1.0 emits 55 and brings ``hotspot`` back. The
#: progression is not monotonic, which is the reason these are captures and not hand-written headers.
_AMD_SMI_FIXTURES: dict[str, str] = {
    "26.2.2": "amd-smi-26.2.2-rocm-7.2.3.csv",
    "26.5.0": "amd-smi-26.5.0-rocm-7.15.0.csv",
    "27.1.0": "amd-smi-27.1.0-rocm-10.2.0.csv",
}

#: Which named column each gpu_monitor field must be read from, given a row in which every column carries a value.
#: Spelled out rather than derived from the consumer's own preference lists, so reordering those is a test failure
#: and not a no-op. ``hotspot`` outranks ``edge`` wherever the part reports it -- on MI300 it will.
_EXPECTED_COLUMN: dict[str, dict[str, str]] = {
    "26.2.2": {"temperature_c": "hotspot", "power_w": "socket_power", "gpu_util_pct": "usage"},
    "26.5.0": {"temperature_c": "edge", "power_w": "socket_power", "gpu_util_pct": "gfx_activity"},
    "27.1.0": {"temperature_c": "hotspot", "power_w": "socket_power", "gpu_util_pct": "gfx_activity"},
}

#: Columns sitting next to the ones above that must never be mistaken for them. ``total_gtt`` is 48 GiB beside a
#: 2 GiB ``total_vram`` on an APU; ``mem_voltage`` is millivolts beside watts; the ``apu_average_*`` entries are
#: list cells whose first element a regex-based cell parser would happily return as a scalar.
_FORBIDDEN_COLUMNS = frozenset(
    {
        "gpu",
        "ts",
        "free_vram",
        "total_visible_vram",
        "used_visible_vram",
        "free_visible_vram",
        "total_gtt",
        "used_gtt",
        "free_gtt",
        "gfx_voltage",
        "soc_voltage",
        "mem_voltage",
        "throttle_status",
        "power_management",
        "ubb_power",
        "umc_activity",
        "mm_activity",
        "vcn_activity",
        "jpeg_activity",
        "apu_average_dram_reads",
        "apu_average_dram_writes",
        "apu_average_ipu_reads",
        "apu_average_ipu_writes",
        "apu_temperature_skin",
        "hbm_stacks",
        "mid",
        "aid",
        "mem",
    }
)

#: VRAM figures substituted into the all-columns-valued row, so ``vram_pct`` is checkable as a percentage rather
#: than as a column index. 179 of 2048 MiB is what this host actually reports idle.
_VRAM_USED_MIB = 179.0
_VRAM_TOTAL_MIB = 2048.0

#: Fields the consumer must fill for the gpu_monitor block to be worth injecting at all.
_REQUIRED_FIELDS = ("temperature_c", "power_w", "gpu_util_pct", "vram_pct")

#: Of those, the ones the REAL idle capture can actually yield on each stack. 26.2.2 (ROCm 7.2.3) reports both
#: ``usage`` and ``socket_power`` as ``"N/A"`` on this part, so power and utilization are legitimately absent there
#: -- the data is missing, not the parse. Whether an Instinct part on 7.2.3 populates them is unverified here.
_REAL_ROW_FIELDS: dict[str, tuple[str, ...]] = {
    "26.2.2": ("temperature_c", "vram_pct"),
}


def _load_launcher():
    """Import the pod-side launcher by path; it ships standalone and cannot be imported as a package.

    Returns:
        Any: The executed ``launch_infera_node`` module.
    """
    multi_node = Path(br.__file__).resolve().parents[3] / "inference_optimizer" / "multi_node"
    scripts = multi_node / "scripts"
    spec = importlib.util.spec_from_file_location("_phase_c_launch_infera_node", scripts / "launch_infera_node.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # The pod receives the script and its dependencies flattened into one directory, so its sibling imports are
    # top-level; reproduce that here rather than reshaping the script for the test's benefit.
    added = [str(scripts), str(multi_node / "_internal")]
    sys.path[:0] = added
    try:
        spec.loader.exec_module(mod)
    finally:
        for entry in added:
            sys.path.remove(entry)
    return mod


_LAUNCHER = _load_launcher()


def _capture(name: str) -> tuple[list[str], list[str]]:
    """Read a captured ``--csv`` fixture, prefixing the ``ts`` column the sampler's ``sed`` adds to every data row.

    Args:
        name: File name under ``fixtures/amd_smi_sampler/``.

    Returns:
        tuple[list[str], list[str]]: The header and the single data row, as the consumer's ``csv.reader`` sees them.
    """
    with (_FIXTURES / name).open(encoding="utf-8", newline="") as f:
        header, row = list(csv.reader(f))[:2]
    return ["ts", *header], ["1760000000", *row]


def _valued_row(header: list[str]) -> list[str]:
    """Give every column a distinct value, so a returned number identifies the column it came from."""
    row = [str(float(i + 1)) for i in range(len(header))]
    for name, value in (("used_vram", _VRAM_USED_MIB), ("total_vram", _VRAM_TOTAL_MIB)):
        if name in header:
            row[header.index(name)] = str(value)
    return row


def _source_columns(header: list[str], sample: dict[str, object]) -> dict[str, str]:
    """Invert a sample built from :func:`_valued_row` back to the column each field was read from."""
    by_value = {float(i + 1): header[i] for i in range(len(header))}
    return {k: by_value[v] for k, v in sample.items() if isinstance(v, float) and v in by_value}


def test_sampler_args_match_the_captured_fixtures() -> None:
    """The fixtures describe the query the sampler actually issues."""
    assert tuple(_LAUNCHER.GPU_SAMPLER_AMD_SMI_ARGS) == _PINNED_SAMPLER_ARGS


@pytest.mark.parametrize("version", sorted(_AMD_SMI_FIXTURES))
def test_amd_smi_header_resolves_every_field_to_its_named_column(version: str) -> None:
    """Every required field resolves, and resolves to the column it is supposed to."""
    header, _ = _capture(_AMD_SMI_FIXTURES[version])
    sample = br._row_to_gpu_sample(header, _valued_row(header))

    missing = [f for f in _REQUIRED_FIELDS if f not in sample]
    assert not missing, f"amd-smi {version} header matched no column for {missing}: {header}"
    assert _source_columns(header, sample) == _EXPECTED_COLUMN[version]
    assert sample["vram_pct"] == pytest.approx(_VRAM_USED_MIB / _VRAM_TOTAL_MIB * 100.0, abs=0.01)


@pytest.mark.parametrize("version", sorted(_AMD_SMI_FIXTURES))
def test_amd_smi_header_reads_no_forbidden_column(version: str) -> None:
    """No field is read from a neighbouring column that merely looks like the right one."""
    header, _ = _capture(_AMD_SMI_FIXTURES[version])
    sample = br._row_to_gpu_sample(header, _valued_row(header))

    assert not set(_source_columns(header, sample).values()) & _FORBIDDEN_COLUMNS


@pytest.mark.parametrize("version", sorted(_AMD_SMI_FIXTURES))
def test_unsupported_readings_fall_through_rather_than_landing_as_values(version: str) -> None:
    """The real idle row: fields the part does not expose read as ``N/A`` and must not reach a sample.

    gfx1151 reports no ``hotspot`` value, so temperature has to fall through to ``edge`` -- the number in the
    capture -- rather than record the string. On 26.5.0 there is no ``hotspot`` column at all, which is the same
    fall-through reached a different way. On 26.2.2 the same is true of power and utilization, which is why the
    expected set is per-version: a missing reading must leave the field out, never land as a string or a zero.
    """
    header, row = _capture(_AMD_SMI_FIXTURES[version])
    sample = br._row_to_gpu_sample(header, row)
    expected = _REAL_ROW_FIELDS.get(version, _REQUIRED_FIELDS)

    assert sorted(sample) == sorted(expected), f"amd-smi {version} resolved {sorted(sample)}"
    assert sample["temperature_c"] == float(row[header.index("edge")])
    if "power_w" in expected:
        assert sample["power_w"] == float(row[header.index("socket_power")])
    if "gpu_util_pct" in expected:
        assert sample["gpu_util_pct"] == float(row[header.index("gfx_activity")])
    # 179 of 2048 MiB, not the 2048 a ``"vram" in column`` predicate would have stored in a field read as a percent.
    assert sample["vram_pct"] == pytest.approx(8.74, abs=0.01)


@pytest.mark.parametrize("version", sorted(_AMD_SMI_FIXTURES))
@pytest.mark.parametrize("field", _REQUIRED_FIELDS)
def test_renaming_the_matched_column_moves_or_drops_the_field(version: str, field: str) -> None:
    """Proof the column assertions bite: rename the matched column and the field cannot come from it any more."""
    header, _ = _capture(_AMD_SMI_FIXTURES[version])
    expected = dict(_EXPECTED_COLUMN[version], vram_pct="used_vram")[field]
    renamed = [f"{c}_renamed" if c == expected else c for c in header]

    sample = br._row_to_gpu_sample(renamed, _valued_row(header))

    assert _source_columns(renamed, sample).get(field) != expected
    if field == "vram_pct":
        assert field not in sample


@pytest.mark.parametrize("version", sorted(_AMD_SMI_FIXTURES))
@pytest.mark.parametrize(
    ("field", "columns"),
    [
        ("temperature_c", br._AMD_SMI_TEMPERATURE_COLS),
        ("power_w", br._AMD_SMI_POWER_COLS),
        ("gpu_util_pct", br._AMD_SMI_UTIL_COLS),
        ("vram_pct", (br._AMD_SMI_VRAM_USED_COL, br._AMD_SMI_VRAM_TOTAL_COL)),
    ],
)
def test_renaming_every_declared_column_drops_the_field(version: str, field: str, columns: tuple[str, ...]) -> None:
    """Rename every spelling a field accepts and it must vanish, not resolve to something else that looked close."""
    header, _ = _capture(_AMD_SMI_FIXTURES[version])
    renamed = [f"{c}_renamed" if c in columns else c for c in header]

    sample = br._row_to_gpu_sample(renamed, _valued_row(header))

    assert field not in sample, f"{field} survived renaming {columns}: {_source_columns(renamed, sample)}"


def test_sampler_shell_runs_amd_smi_with_the_declared_args(tmp_path, monkeypatch) -> None:
    """The detached sampler string is built from the same argument tuple the fixtures are pinned against."""
    captured: dict[str, object] = {}

    class _Proc:
        pid = 4242

    def _popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/amd-smi")
    monkeypatch.setattr(_LAUNCHER.subprocess, "Popen", _popen)
    _LAUNCHER._start_gpu_sampler(tmp_path / "gpu_metrics_h.csv", tmp_path / "s.pid", 5)

    script = captured["cmd"][-1]
    assert " ".join(_LAUNCHER.GPU_SAMPLER_AMD_SMI_ARGS) in script
    assert "/usr/bin/amd-smi" in script
    # The ROCm 10 pip stack has no /opt/rocm and ships no rocm-smi, so neither may survive anywhere in the string.
    assert "rocm-smi" not in script
    assert "/opt/rocm" not in script


def test_sampler_skips_when_amd_smi_is_absent(tmp_path, monkeypatch) -> None:
    """No amd-smi on PATH means no sampler, rather than a hardcoded path that does not exist on ROCm 10."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(_LAUNCHER.subprocess, "Popen", lambda *a, **k: pytest.fail("sampler launched without amd-smi"))

    _LAUNCHER._start_gpu_sampler(tmp_path / "gpu_metrics_h.csv", tmp_path / "s.pid", 5)

    assert not (tmp_path / "s.pid").exists()


def _write_pod_csv(shared: Path, host: str, header: list[str], rows: list[list[str]]) -> None:
    """Write a pod-side sampler CSV the way the detached shell loop does, through a real CSV writer."""
    with (shared / f"gpu_metrics_{host}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.mark.parametrize("version", sorted(_AMD_SMI_FIXTURES))
def test_harvest_injects_gpu_monitor_from_a_real_sampler_csv(version: str, monkeypatch, tmp_path) -> None:
    """End to end: a verbatim amd-smi capture on shared storage becomes a gpu_monitor sample in the report."""
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", dict)
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    header, row = _capture(_AMD_SMI_FIXTURES[version])
    # Second row: the timestamped blank line amd-smi --csv ends each invocation with, as a sampler that does not
    # strip it writes it. It must be dropped, not merged as a one-cell sample.
    _write_pod_csv(shared, "10.0.0.1", header, [row, [row[0]]])
    dest = tmp_path / "ws"
    dest.mkdir()
    (dest / "benchmark_report.json").write_text("{}", encoding="utf-8")

    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=float(row[0]) - 1.0)

    assert out["rows"] == 1
    assert out["gpu_monitor_samples"] == 1
    report = json.loads((dest / "benchmark_report.json").read_text(encoding="utf-8"))
    assert set(report["gpu_monitor"][0]) >= set(_REAL_ROW_FIELDS.get(version, _REQUIRED_FIELDS))


def test_harvest_reports_a_header_it_cannot_read(monkeypatch, tmp_path, caplog) -> None:
    """An unrecognised header is logged at error level rather than injecting nothing in silence."""
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", dict)
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    _write_pod_csv(shared, "10.0.0.1", ["ts", "something", "else"], [["1760000000", "1", "2"]])
    dest = tmp_path / "ws"
    dest.mkdir()

    with caplog.at_level(logging.ERROR):
        br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=1759999999.0)

    assert "matched no gpu_monitor field" in caplog.text


@pytest.mark.skipif(
    os.environ.get("HYPERLOOM_GPU_SAMPLER_LIVE", "") != "1" or shutil.which("amd-smi") is None,
    reason="opt-in: invokes amd-smi, so it must be run under the host GPU lock",
)
def test_live_amd_smi_header_has_not_drifted() -> None:
    """Re-capture from the local amd-smi and fail if its header is not one of the pinned ones."""
    proc = subprocess.run(  # nosec B603 B607 - fixed read-only amd-smi query.
        ["amd-smi", *_LAUNCHER.GPU_SAMPLER_AMD_SMI_ARGS, "--csv"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    live = proc.stdout.splitlines()[0].strip()
    pinned = {",".join(_capture(name)[0][1:]) for name in _AMD_SMI_FIXTURES.values()}

    assert live in pinned, f"amd-smi header drifted from every pinned capture: {live}"
