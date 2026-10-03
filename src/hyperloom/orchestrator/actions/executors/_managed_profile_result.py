# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate Magpie diagnostic evidence without accepting a performance measurement."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def _capture_rounds(capture: Mapping[str, Any], workspace: Path, expected_ranks: int) -> list[dict[str, Any]]:
    capture_id = capture.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id or Path(capture_id).name != capture_id:
        raise ValueError("AgentX profile capture has no valid capture_id")
    root = (workspace / "torch_trace" / capture_id).resolve()
    artifact = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    if artifact != capture:
        raise ValueError("AgentX profile report disagrees with its capture manifest")
    if type(capture.get("expected_ranks")) is not int or capture["expected_ranks"] != expected_ranks:
        raise ValueError("AgentX profile rank count disagrees with the accepted topology")
    rounds = capture.get("profiles")
    if rounds is None:
        rounds = [dict(capture, profile_index=1, trace_dir=str(root))]
    elif not isinstance(rounds, list) or not rounds:
        raise ValueError("AgentX profile series has no completed captures")
    if "completed_profiles" in capture and (
        capture["completed_profiles"] != len(rounds) or capture.get("effective_profiles") != len(rounds)
    ):
        raise ValueError("AgentX profile series completion accounting is inconsistent")
    return [_validated_round(row, index, root, capture) for index, row in enumerate(rounds, 1)]


def _validated_round(row: Any, index: int, root: Path, capture: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("status") != "complete" or row.get("profile_index") != index:
        raise ValueError("AgentX profile series contains an incomplete or misordered capture")
    for key in ("capture_id", "framework", "num_steps", "expected_ranks"):
        if type(row.get(key)) is not type(capture[key]) or row[key] != capture[key]:
            raise ValueError(f"AgentX profile round {key} disagrees with its capture")
    directory = Path(row["trace_dir"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("AgentX profile round is outside this capture directory")
    files = [Path(path).resolve() for path in row.get("trace_files") or []]
    if not files or any(not path.is_relative_to(directory) or not path.is_file() for path in files):
        raise ValueError("AgentX profile trace is missing or outside its capture round")
    ranks = row.get("rank_trace_files")
    expected = row.get("expected_ranks")
    if type(expected) is not int or expected <= 0 or not isinstance(ranks, dict):
        raise ValueError("AgentX profile capture has no verified rank manifest")
    if set(ranks) != {str(rank) for rank in range(expected)} or any(not paths for paths in ranks.values()):
        raise ValueError("AgentX profile capture is missing verified worker ranks")
    rank_files = {rank: [str(Path(path).resolve()) for path in paths] for rank, paths in ranks.items()}
    if sorted(path for paths in rank_files.values() for path in paths) != sorted(map(str, files)):
        raise ValueError("AgentX profile rank manifest disagrees with the trace inventory")
    return {**row, "trace_dir": str(directory), "trace_files": list(map(str, files)), "rank_trace_files": rank_files}


def diagnostic_profile_result(
    report: Mapping[str, Any] | None,
    *,
    workspace: Path,
    config_path: Path,
    returncode: int,
    subprocess_started_unix: float,
) -> dict[str, Any]:
    """Require runtime, receipt and complete traces, while keeping KEEP gates false."""
    result: dict[str, Any] = {
        "status": "failed",
        "diagnostic_only": True,
        "valid_measurement": False,
        "benchmark_valid": False,
        "publishable": False,
        "submission_valid": False,
        "trace_input_ready": False,
    }
    try:
        metrics, rounds = _validate_diagnostic_report(
            report, workspace, config_path, returncode, subprocess_started_unix
        )
        capture = metrics["profile_capture"]
        evidence = metrics["server_launch"]
        selected = rounds[-1]
        main_trace = max(selected["rank_trace_files"]["0"], key=lambda path: (Path(path).stat().st_size, path))
        result.update(
            status="succeeded",
            reported_success=True,
            trace_input_ready=True,
            agentx_server_launch=evidence,
            profile_capture=capture,
            profile_analyses=metrics.get("profile_analyses") or [],
            profile_rounds=rounds,
            selected_profile_index=selected["profile_index"],
            trace_capture=capture,
            trace_capture_status="complete",
            trace_dir=selected["trace_dir"],
            trace_files=selected["trace_files"],
            rank_trace_paths=selected["rank_trace_files"],
            main_trace_path=main_trace,
            primary_rank=0,
            profile_trace_selection_reason="latest_complete_profile_rank_0",
            trace_manifest_path=str(workspace / "torch_trace" / capture["capture_id"] / "capture.json"),
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        result.update(error_class="invalid_profile_diagnostic", error=str(exc))
    return result


def _validate_diagnostic_report(
    report: Mapping[str, Any] | None,
    workspace: Path,
    config_path: Path,
    returncode: int,
    subprocess_started_unix: float,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    from hyperloom.inference_optimizer.agentx.identity import validate_server_launch

    if returncode != 0:
        raise ValueError(f"Magpie diagnostic process exited with status {returncode}")
    if not report or report.get("success") is not True or report.get("errors"):
        raise ValueError(f"Magpie diagnostic runtime failed: {(report or {}).get('errors') or 'no successful report'}")
    if (workspace / "benchmark_report.json").stat().st_mtime + 1 < subprocess_started_unix:
        raise ValueError("Magpie diagnostic report predates this run")
    metrics = report.get("agentx_metrics") or {}
    capture = metrics.get("profile_capture") or {}
    if metrics.get("diagnostic_only") is not True or capture.get("status") != "complete":
        raise ValueError(f"Magpie diagnostic capture is incomplete: {capture.get('error') or capture.get('status')}")
    benchmark = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})["benchmark"]
    evidence = metrics.get("server_launch")
    errors = validate_server_launch(benchmark, evidence, workspace=workspace)
    if errors:
        raise ValueError("Magpie diagnostic launch is not verified: " + "; ".join(errors))
    receipt = json.loads((workspace / "agentx_server_launch.json").read_text(encoding="utf-8"))
    if receipt != evidence:
        raise ValueError("Magpie diagnostic report disagrees with its server launch receipt")
    return metrics, _bound_capture_rounds(capture, evidence, benchmark, workspace)


def _bound_capture_rounds(
    capture: Mapping[str, Any], evidence: Mapping[str, Any], benchmark: Mapping[str, Any], workspace: Path
) -> list[dict[str, Any]]:
    """Bind trace inventory to the current launch and its accepted worker topology."""
    profiling = evidence.get("torch_profiler") or {}
    if not capture.get("capture_id") or capture["capture_id"] != profiling.get("capture_id"):
        raise ValueError("Magpie diagnostic capture ID disagrees with its server launch receipt")
    directory = (workspace / "torch_trace" / capture["capture_id"]).resolve()
    if not profiling.get("trace_dir") or Path(profiling["trace_dir"]).resolve() != directory:
        raise ValueError("Magpie diagnostic capture directory disagrees with its server launch receipt")
    if capture.get("framework") != evidence.get("framework"):
        raise ValueError("Magpie diagnostic capture framework disagrees with its server launch receipt")
    if type(capture.get("num_steps")) is not int or capture["num_steps"] != profiling.get("num_steps"):
        raise ValueError("Magpie diagnostic capture step count disagrees with its server launch receipt")
    expected_ranks = benchmark["workload_spec"]["resolved_topology"]["gpu_count"]
    if type(expected_ranks) is not int or expected_ranks <= 0:
        raise ValueError("Magpie diagnostic configuration has no trusted GPU count")
    return _capture_rounds(capture, workspace, expected_ranks)
