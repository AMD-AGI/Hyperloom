# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The staged AgentX client is patched to keep the output token count of a reply it found nothing in.

What is pinned: both files of the installed client change and nothing else; the owning ``RECORD`` names the patched
bytes and is otherwise byte for byte what pip wrote; a client that is not the anchored one is changed in neither file
and the refusal says which; and, when the real installed client is given, the real parser exports the count the server
reported, the success line and every other failure are untouched, and the reader of ``measure.agentx_requests`` counts
empty and short exactly from the export it wrote.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import RequestClasses
from hyperloom.measure.agentx_requests import read_request_classes
from inference_suite.aiperf_patch import ERROR_OSL_METRIC, AiperfPatchError, patch_aiperf
from inference_suite.tests import aiperf_source as source

PARSER, METRIC = source.PARSER_FILE, source.METRIC_FILE
SITE_ENV = "INFERENCE_SUITE_TEST_AIPERF_SITE"
PYTHON_ENV = "INFERENCE_SUITE_TEST_AIPERF_PYTHON"
DRIVER = Path(__file__).with_name("aiperf_driver.py")
ARTIFACTS = "aiperf_artifacts"


def _both(site: Path) -> tuple[str, str]:
    return (site / PARSER).read_text(encoding="utf-8"), (site / METRIC).read_text(encoding="utf-8")


def _record_rows(record: Path) -> dict[str, str]:
    return {line.split(",", 1)[0]: line for line in record.read_bytes().decode().split("\r\n") if line}


def test_the_parser_keeps_the_token_counts_of_an_invalid_record_and_a_metric_puts_the_output_one_on_its_line(
    tmp_path: Path,
) -> None:
    source.write_site(tmp_path)
    patch_aiperf(tmp_path)
    parser, metric = _both(tmp_path)
    assert "token_counts=record.token_counts or TokenCounts()," in parser
    assert "input=record.token_counts.input" not in parser
    assert metric.startswith(source.OUTPUT_LENGTH_METRIC), "the metric module only gains a class at its end"
    added = metric.removeprefix(source.OUTPUT_LENGTH_METRIC)
    assert "class ErrorOutputSequenceLengthMetric(OutputSequenceLengthMetric):" in added
    assert f'tag = "{ERROR_OSL_METRIC}"' in added and "MetricFlags.ERROR_ONLY" in added
    changed = [line for line in parser.splitlines() if line not in source.PARSER.splitlines()]
    assert 3 <= len(changed) <= 6, "the parser changes in the one branch and nowhere else"


def test_the_record_names_the_patched_bytes_and_keeps_every_other_row_as_pip_wrote_it(tmp_path: Path) -> None:
    record = source.write_site(tmp_path)
    before = _record_rows(record)
    patch_aiperf(tmp_path)
    after = _record_rows(record)

    assert after.keys() == before.keys()
    for relative in (PARSER, METRIC):
        data = (tmp_path / relative).read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        assert after[relative] == f"{relative},sha256={digest},{len(data)}"
    assert after[f"{source.DIST_INFO}/RECORD"] == before[f"{source.DIST_INFO}/RECORD"]
    assert record.read_bytes().endswith(b"\r\n"), "pip's line ends are kept"


def test_a_parser_that_is_not_the_anchored_one_leaves_both_files_and_the_record_as_they_were(tmp_path: Path) -> None:
    moved = source.PARSER.replace("# Parsed record was invalid, return as error record", "# the branch moved")
    record = source.write_site(tmp_path, parser=moved)
    rows = record.read_bytes()
    with pytest.raises(AiperfPatchError, match="inference_result_parser.py does not hold the branch"):
        patch_aiperf(tmp_path)
    assert _both(tmp_path) == (moved, source.OUTPUT_LENGTH_METRIC), "the metric must not be added without the parser"
    assert record.read_bytes() == rows


def test_a_metric_module_without_the_output_length_metric_is_refused_and_the_parser_is_left_alone(
    tmp_path: Path,
) -> None:
    other = source.OUTPUT_LENGTH_METRIC.replace("class OutputSequenceLengthMetric(", "class Renamed(")
    source.write_site(tmp_path, metric=other)
    with pytest.raises(AiperfPatchError, match="output_sequence_length_metric.py does not define"):
        patch_aiperf(tmp_path)
    assert _both(tmp_path) == (source.PARSER, other)


def test_a_client_no_installed_distribution_records_is_refused(tmp_path: Path) -> None:
    record = source.write_site(tmp_path)
    record.unlink()
    with pytest.raises(AiperfPatchError, match="no distribution in .* records both"):
        patch_aiperf(tmp_path)
    assert _both(tmp_path) == (source.PARSER, source.OUTPUT_LENGTH_METRIC)


def test_a_site_without_the_client_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AiperfPatchError, match="no distribution"):
        patch_aiperf(tmp_path)


# --- against the real client ----------------------------------------------------------------------------------------


def _drive(python: str, site: Path, artifacts: Path) -> dict[str, dict]:
    """The real parser, metric processor and export writer of the aiperf in ``site`` on one reply of each kind."""
    artifacts.mkdir(parents=True)
    done = subprocess.run(
        [python, "-I", "-B", str(DRIVER), str(site), str(artifacts)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=artifacts,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = dict(line.split("\t", 1) for line in done.stdout.splitlines() if "\t" in line)
    module = Path(json.loads(lines.pop("module")))
    assert module.is_relative_to(site), "the driver ran another copy of the client"
    return {name: json.loads(raw) for name, raw in lines.items()}


def _copy_client(site: Path) -> Path:
    """The installed client's ``aiperf`` package and its distribution's metadata, copied to ``site``: aiperf finds its
    plugins through that metadata's entry points."""
    real = Path(os.environ[SITE_ENV])
    for name in ("aiperf", source.DIST_INFO):
        shutil.copytree(real / name, site / name, ignore=shutil.ignore_patterns("__pycache__"))
    return site


def _classes(written: Path, keep: set[str]) -> RequestClasses:
    """What the reader counts from the lines of the export in ``written`` made for the scenarios in ``keep``."""
    lines = (written / ARTIFACTS / "profile_export.jsonl").read_text(encoding="utf-8").splitlines()
    chosen = [line for line in lines if json.loads(line)["metadata"]["x_request_id"] in keep]
    subset = written / f"subset-{len(keep)}"
    (subset / ARTIFACTS).mkdir(parents=True)
    (subset / ARTIFACTS / "profile_export.jsonl").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return read_request_classes(subset, subset, 0, 8)


needs_client = pytest.mark.skipif(
    not (os.environ.get(SITE_ENV) and os.environ.get(PYTHON_ENV)),
    reason=f"needs {SITE_ENV} (a staged client's site-packages) and {PYTHON_ENV} (that client's interpreter)",
)


@needs_client
def test_the_real_client_exports_the_count_the_server_reported_and_changes_nothing_else(tmp_path: Path) -> None:
    site = _copy_client(tmp_path / "site")
    python = os.environ[PYTHON_ENV]

    before = _drive(python, site, tmp_path / "before" / ARTIFACTS)
    patch_aiperf(site)
    after = _drive(python, site, tmp_path / "after" / ARTIFACTS)

    assert not [name for name, line in before.items() if ERROR_OSL_METRIC in line["metrics"]], "the client as released"
    counted = {
        name: line["metrics"][ERROR_OSL_METRIC] for name, line in after.items() if ERROR_OSL_METRIC in line["metrics"]
    }
    assert counted == {
        "no_content_usage_zero": {"value": 0, "unit": "tokens"},
        "no_content_usage_three": {"value": 3, "unit": "tokens"},
    }, "only a reply the client found nothing in, with a count from the server, gains the field"
    for name, line in after.items():
        line["metrics"].pop(ERROR_OSL_METRIC, None)
        assert line == before[name], f"{name}: the line is what it was, the added field aside"


@needs_client
def test_the_reader_counts_empty_and_short_exactly_from_the_patched_clients_export_and_not_from_the_released_ones(
    tmp_path: Path,
) -> None:
    site = _copy_client(tmp_path / "site")
    python = os.environ[PYTHON_ENV]
    counted = {"success_with_text", "no_content_usage_zero", "no_content_usage_three", "http_error", "unreachable"}

    _drive(python, site, tmp_path / "before" / ARTIFACTS)
    patch_aiperf(site)
    _drive(python, site, tmp_path / "after" / ARTIFACTS)

    released = _classes(tmp_path / "before", counted)
    patched = _classes(tmp_path / "after", counted)
    assert (released.requests_total, released.failed_transport, released.failed_no_content) == (5, 2, 2)
    assert (released.failed_empty, released.failed_short) == (None, None), (
        "no count in the export of the released client"
    )
    assert (patched.requests_total, patched.failed_transport, patched.failed_no_content) == (5, 2, 2)
    assert (patched.failed_empty, patched.failed_short, patched.reason) == (1, 1, "")
    unsplit = _classes(tmp_path / "after", counted | {"no_content_no_usage", "no_responses_at_all"})
    assert (unsplit.failed_no_content, unsplit.failed_empty, unsplit.failed_short) == (4, None, None)
    assert "2 of the 4" in unsplit.reason, "a reply that gave no count keeps the split unknown"
