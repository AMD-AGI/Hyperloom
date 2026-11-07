# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command a Magpie round runs, and the InferenceX checkout it runs against."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom.inference_optimizer import inferencex_checkout

from ._accuracy_gate import materialized_run_eval_disabled
from ._inferencex_patcher import (
    apply_round_seed,
    ensure_benchmark_lib_eval_start_patched,
    ensure_benchmark_serving_seed_patched,
    ensure_eval_probe_patched,
    ensure_eval_unbound_outputs_patched,
    eval_probe_targets_exist,
)
from ._magpie_patcher import ensure_lm_eval_install_once
from ._subprocess_kill import (
    EVAL_PROBE_UNPATCHABLE_RETURNCODE,
    INFERENCEX_CHECKOUT_INVALID_RETURNCODE,
    PreparationFailure,
)

log = logging.getLogger(__name__)


def resolve_benchmark_interpreter() -> str:
    """Return the Magpie-importable interpreter a Magpie round runs under."""
    from ._benchmark_interpreter import _resolve_magpie_python

    return _resolve_magpie_python()


def build_benchmark_command(
    *,
    python_exe: str,
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    """Return the Magpie ... --run-mode local argv, run through :mod:`.magpie_confined` so that Magpie's leftover-server
    sweep cannot kill another sandbox's server."""
    return [
        python_exe,
        str(Path(__file__).with_name("magpie_confined.py")),
        "-v",
        "benchmark",
        "--benchmark-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-mode",
        "local",
    ]


def _eval_probe_failure(config_path: Path, inferencex_path: str) -> PreparationFailure | None:
    # Asserted whether or not ``$INFERENCEX_PATH`` is set: unset falls back to the same env discovery the baseline
    # arm uses ($MAGPIE_PATH/InferenceX).
    probe_root = Path(inferencex_path) if inferencex_path else None
    # Best-effort: without it a warmup round pre-fills the measured round's prefix cache, which inflates the number
    # but never breaks the round.
    ensure_benchmark_serving_seed_patched(probe_root)
    # Best-effort, unlike the probe below: a missing guard only costs the reason a failed round reports.
    ensure_eval_unbound_outputs_patched(probe_root)
    if ensure_eval_probe_patched(probe_root) or materialized_run_eval_disabled(config_path):
        return None
    message = (
        "eval generation bounds + pathology probe are not installed "
        f"(utils/evals/patches/lm_eval_sitecustomize.py, inferencex={inferencex_path or '<unset>'}); this variant "
        "runs eval, so it would be scored against a baseline that truncated at a different point"
    )
    if eval_probe_targets_exist(probe_root):
        log.error("grid_runner: %s; failing this benchmark", message)
        return PreparationFailure(EVAL_PROBE_UNPATCHABLE_RETURNCODE, message)
    log.warning("grid_runner: %s", message)
    return None


def prepare_inferencex_checkout(config_path: Path, env: dict[str, str]) -> PreparationFailure | None:
    """Patch the InferenceX checkout a Magpie round runs against; native AgentX is owned by Magpie and left as it is."""
    if read_benchmark_point(config_path).agentx:
        return None
    # Pin Magpie's InferenceX resolution to ``$INFERENCEX_PATH`` (its highest-precedence rung) so it loads the patched
    # checkout, not a stale copy.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        invalid = inferencex_checkout.unusable_reason(inferencex_path)
        if invalid:
            log.error("grid_runner: %s; failing this benchmark", invalid)
            return PreparationFailure(INFERENCEX_CHECKOUT_INVALID_RETURNCODE, invalid)
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Explore and sweep never pass through the baseline's patch hook: re-assert here so a resumed session or a
        # re-cloned checkout still emits the eval-start marker and installs lm-eval once.
        ensure_benchmark_lib_eval_start_patched(Path(inferencex_path))
        ensure_lm_eval_install_once(Path(inferencex_path))
    return _eval_probe_failure(config_path, inferencex_path)


def apply_benchmark_round_seed(config_path: Path, env: dict[str, str], *, warmup: bool) -> None:
    """Reseed a discarded warmup round's client so it cannot pre-fill the measured round's prefix cache.

    Native AgentX replays its own workload through Magpie and is left exactly as it was.

    Args:
        config_path: The round's materialized Magpie YAML.
        env: The benchmark subprocess env, edited in place.
        warmup: Whether the round's result is discarded.
    """
    apply_round_seed(env, warmup=warmup and not read_benchmark_point(config_path).agentx)
