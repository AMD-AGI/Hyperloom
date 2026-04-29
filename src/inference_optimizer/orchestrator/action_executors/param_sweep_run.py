"""``param_sweep_run`` executor — runs the CONC × ISL/OSL grid sweep.

Wraps ``run_sweep.sh`` which keeps a single server warm and sweeps a
matrix of (CONC, ISL, OSL) configs, writing one row per config into
``results.tsv``. We parse the TSV, find the best ``output_tput`` row
(with TP-normalised tok/s/GPU), and emit one ``update_state`` intent
that sets ``current_tput`` to the winning config + a ``send_message``
event with the full leaderboard so the LLM can adopt the best params.

User can override the grid via ``CONC_VALUES`` / ``ISL_OSL_CONFIGS``
env vars; otherwise the script's defaults apply
(``"4 16 64"`` × ``"1024:1024 8192:1024 1024:8192"``).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from ...paths import asset_script
from ._helpers import merged_env, send_message_intent, update_state_intent
from .base import (
    ActionExecutor,
    ExecutorContext,
    ExecutorResult,
    register_executor,
    run_subprocess,
)


log = logging.getLogger(__name__)


_REQUIRED_ENV = ("MODEL", "TP", "INFERENCEX_PATH")
_TSV_NAME = "results.tsv"


def _parse_tsv(path: Path) -> list[dict[str, str]]:
    """Return one dict per row keyed by header name."""
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _best_row(rows: list[dict[str, str]]) -> tuple[dict[str, str] | None, float]:
    """Pick the row with the highest numeric ``output_tput``. Returns
    ``(row, tput)`` or ``(None, 0.0)`` when no usable row is present."""
    best: dict[str, str] | None = None
    best_tput = -1.0
    for r in rows:
        if str(r.get("status", "")).strip() != "swept":
            continue
        try:
            t = float(r.get("output_tput", ""))
        except ValueError:
            continue
        if t > best_tput:
            best, best_tput = r, t
    return best, max(0.0, best_tput)


class ParamSweepRunExecutor(ActionExecutor):
    """Wraps ``run_sweep.sh`` (single server, multi-config grid)."""

    name = "param_sweep_run"
    timeout_s = 4 * 60 * 60  # 4 hours — sweeps can be long

    async def run(self, ctx: ExecutorContext) -> ExecutorResult:
        env_block = ctx.require_env(*_REQUIRED_ENV)

        results_dir = ctx.results_dir()

        env = merged_env(
            ctx.env, env_block,
            {
                "RESULT_DIR": str(results_dir),
                "PORT": ctx.env.get("PORT", "8888"),
                "FRAMEWORK": ctx.env.get("FRAMEWORK", "sglang"),
            },
        )

        script = asset_script("run_sweep.sh")
        log_path = results_dir / "run_sweep.log"

        rc = await run_subprocess(
            ["bash", str(script)],
            env=env, cwd=results_dir,
            timeout_s=self.timeout_s, log_path=log_path,
        )

        if rc != 0:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"run_sweep.sh exited rc={rc}; see {log_path}",
            )

        tsv = results_dir / _TSV_NAME
        rows = _parse_tsv(tsv)
        if not rows:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"no {_TSV_NAME} produced under {results_dir}",
            )

        best, best_tput = _best_row(rows)
        if best is None:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"no successful sweep rows in {tsv}",
            )

        try:
            tp = float(env_block["TP"])
        except ValueError:
            tp = 1.0
        best_per_gpu = best_tput / tp if tp > 0 else best_tput

        # Render a compact leaderboard for the LLM.
        leaderboard_lines = ["| conc | isl | osl | tput |", "|---:|---:|---:|---:|"]
        for r in sorted(
            (r for r in rows if r.get("status") == "swept"),
            key=lambda x: -float(x.get("output_tput") or 0),
        )[:10]:
            leaderboard_lines.append(
                f"| {r.get('conc','?')} | {r.get('isl','?')} | "
                f"{r.get('osl','?')} | {r.get('output_tput','?')} |"
            )

        intents = [
            update_state_intent(
                {
                    "current_tput": best_per_gpu,
                    "current_action": "param_sweep_run",
                },
                rationale=(
                    f"sweep best: conc={best.get('conc')} "
                    f"isl={best.get('isl')} osl={best.get('osl')} "
                    f"→ {best_per_gpu:.2f} tok/s/GPU"
                ),
            ),
            send_message_intent(
                topic="event",
                body_md=(
                    f"param sweep done; best={best_per_gpu:.2f} tok/s/GPU\n\n"
                    + "\n".join(leaderboard_lines)
                ),
                extras={
                    "kind": "sweep_done",
                    "tput_per_gpu": best_per_gpu,
                    "best_config": dict(best),
                    "result_path": str(tsv),
                },
            ),
        ]

        return ExecutorResult(
            status="succeeded", rc=rc,
            metrics={
                "tput_per_gpu": best_per_gpu,
                "tput_total": best_tput,
                "n_configs": len(rows),
                "best_conc": best.get("conc", ""),
                "best_isl": best.get("isl", ""),
                "best_osl": best.get("osl", ""),
            },
            artifacts=[str(tsv), str(log_path)],
            intents=intents,
            notes=(
                f"sweep best={best_per_gpu:.2f} tok/s/GPU "
                f"({len(rows)} configs)"
            ),
        )


register_executor(ParamSweepRunExecutor())


__all__ = ["ParamSweepRunExecutor"]
