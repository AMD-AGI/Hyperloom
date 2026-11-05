# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a conc sweep resolves before either arm runs, and the setup events it records about it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import SweepAnchor, SweepEnvironment
from hyperloom.inference_optimizer.session.session_paths import reports_dir
from ..state.shared_state import SharedState


@dataclass(frozen=True)
class SweepSetup:
    """Where a sweep runs and the materialized base config both arms launch from."""

    session_dir: Path
    task_id: str
    workspace: Path
    base_yaml_path: Path
    model_path: str
    gpu_type: str
    benchmark_script: str | None


def record_sweep_setup(
    recorder: Any, state: SharedState, setup: SweepSetup, *, extra_server_args: str, extra_envs: dict[str, str]
) -> None:
    """Record the workload, the anchor the optimized arm stands on, and where the sweep runs."""
    cb = state.current_best
    recorder.record_workload(
        session_id=getattr(state, "session_id", "") or setup.session_dir.name,
        isl=int(state.isl),
        osl=int(state.osl),
        tp=getattr(state, "tp", 0),
        benchmark_mode=getattr(state, "benchmark_mode", ""),
    )
    recorder.record_anchor(
        SweepAnchor(
            baseline_tput=float(state.baseline_tput),
            anchor_tput=cb.get("tput"),
            tp=getattr(state, "tp", 0),
            variant_id=cb.get("variant_name"),
            action=cb.get("action"),
            extra_server_args=extra_server_args,
            extra_envs=extra_envs,
        )
    )
    recorder.record_environment(
        SweepEnvironment(
            sweep_task_id=setup.task_id,
            workspace=setup.workspace.as_posix(),
            model_path=setup.model_path,
            gpu_type=setup.gpu_type,
            base_config_path=setup.base_yaml_path.as_posix(),
            report_json_path=(reports_dir(setup.session_dir) / "conc_sweep_summary.json").as_posix(),
            report_csv_path=(reports_dir(setup.session_dir) / "conc_sweep_raw.csv").as_posix(),
        )
    )
