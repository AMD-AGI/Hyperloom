# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workload sweep that relaunches the GEAK-optimized server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from hyperloom.common.coerce import first_text
from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.common.jsonio import read_json
from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS, effective_mask_tokens, is_rocr_level
from hyperloom.orchestrator.kernel.geak_config import (
    _accepted_config_as_variant,
)
from hyperloom.orchestrator.actions.executors._gpu_pin import (
    coerce_tp,
    resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
)
from ._accuracy_gate import parse_eval_results
from ._gpu_power import (
    ServerLogPhaseDriver,
    build_gpu_power_recorder,
    read_measured_gpu_power,
    read_measured_gpu_power_by_gpu,
)
from ._subprocess_kill import _LogCursor
from ._launch_evidence import RequestedLaunch, build_launch_evidence, persist_launch_evidence
from ._workload_envs import GEAK_METRIC_OUTPUT

log = logging.getLogger(__name__)


def _write_benchmark_report(out_dir: Path, report: dict[str, Any]) -> None:
    """Write a session-breakdown-compatible ``benchmark_report.json``."""
    try:
        (out_dir / "benchmark_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        # Best-effort: a failed write must never break the sweep.
        log.warning("geak_sweep: could not write %s: %s", out_dir / "benchmark_report.json", exc)


def _parse_isl_osl(spec: str) -> tuple[int, int]:
    isl_s, _, osl_s = str(spec).partition(":")
    return int(isl_s or 1024), int(osl_s or 1024)


def _geak_replay_server_log(out_dir: Path) -> str | None:
    """Return the newest server log created inside this replay's output only."""
    try:
        existing = _replay_server_logs(out_dir)
        return max(existing, key=lambda path: Path(path).stat().st_mtime) if existing else None
    except OSError:
        # Replay-log discovery is best effort.
        logging.debug("Unable to enumerate GEAK replay server logs", exc_info=True)
        return None


def _replay_server_logs(out_dir: Path) -> list[str]:
    """Every server log this replay has written so far, oldest first: one per replica boot."""
    candidates = [out_dir / "server.log", *sorted(out_dir.glob("replica_*/attempt_*/server.log"))]
    return [str(path) for path in candidates if path.is_file()]


def _run_with_power_sampling(
    run: Callable[[], subprocess.CompletedProcess], out_dir: Path, env: Mapping[str, str], entry: dict[str, Any]
) -> subprocess.CompletedProcess:
    """Run one replay under the power recorder and record its measured power into ``entry``.

    Both power fields stay ``None`` when sampling is off, nothing was measured, or the replay raised.
    """
    entry.update(gpu_power_avg_w=None, gpu_power_by_gpu_w=None)
    proc = _run_sampled(run, out_dir, env)
    entry["gpu_power_avg_w"] = read_measured_gpu_power(out_dir)[1]
    entry["gpu_power_by_gpu_w"] = read_measured_gpu_power_by_gpu(out_dir)[1]
    return proc


def _run_sampled(
    run: Callable[[], subprocess.CompletedProcess], out_dir: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess:
    """Run one replay with the measured-phase power recorder a native round gets.

    The replay boots its own servers, so the watchdog's phase marks are not available; the driver reads the same
    ready / warmup / measured / eval markers from the logs the replay writes. Writes ``gpu_power.json`` into
    ``out_dir`` whenever sampling is on.
    """
    recorder = build_gpu_power_recorder(str(out_dir / "server.log"), dict(env))
    if recorder is None:
        return run()
    driver = ServerLogPhaseDriver(recorder, lambda: _replay_server_logs(out_dir), cursor=_LogCursor())
    driver.start()
    try:
        return run()
    finally:
        try:
            driver.stop()
        finally:
            recorder.close()


def _replay_logical_masks(pin: Mapping[str, Any], gpu_ids: str) -> dict[str, str]:
    var = first_text(pin.get("var"))
    tokens = effective_mask_tokens(pin.get("value"))
    if not is_rocr_level(var) or not tokens:
        raise ValueError("missing_replay_logical_gpu_pin")
    if any(not token.isdigit() or int(token) >= len(tokens) for token in effective_mask_tokens(gpu_ids)):
        raise ValueError("invalid_replay_logical_gpu_ids")
    return {var: ",".join(tokens)}


def _replay_serving_env(handoff: Mapping[str, Any], env_spec: Mapping[str, Any]) -> dict[str, str]:
    """Resolve serving identity from the frozen handoff or its materialized recipe."""
    recipe_path = first_text(handoff.get("launch_recipe"), env_spec.get("base_launch_recipe"))
    benchmark: dict[str, Any] = {}
    if recipe_path:
        parsed = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict) or not isinstance(parsed.get("benchmark"), dict):
            raise ValueError("invalid_replay_recipe")
        benchmark = parsed["benchmark"]
    recipe_envs = benchmark.get("envs") or {}
    if not isinstance(recipe_envs, dict):
        raise ValueError("invalid_replay_recipe_envs")
    legacy_env = {} if handoff or benchmark else os.environ
    tp = coerce_tp(handoff.get("tp"), recipe_envs.get("TP"), legacy_env.get("TP"))
    pin = handoff.get("gpu_pin")
    if not isinstance(pin, dict):
        pin = resolve_gpu_pin(recipe_envs=recipe_envs, environ=legacy_env)
    gpu_ids = ",".join(effective_mask_tokens(handoff.get("gpu_ids"))) or _resolve_handoff_gpu_ids(gpu_pin=pin, tp=tp)
    pin_space = _resolve_handoff_gpu_ids_space(gpu_pin=pin)
    space = str(handoff.get("gpu_ids_space") or pin_space)
    if space not in {"absolute", "logical"} or pin_space == "none":
        raise ValueError("invalid_replay_gpu_ids_space")
    if len(effective_mask_tokens(gpu_ids)) < tp:
        raise ValueError("replay_gpu_count_below_tp")
    masks = _replay_logical_masks(pin, gpu_ids) if space == "logical" else {}
    return {
        "MODEL": first_text(handoff.get("model_path"), benchmark.get("model"), legacy_env.get("MODEL_PATH")),
        "BACKEND": first_text(
            handoff.get("framework"), benchmark.get("framework"), legacy_env.get("FRAMEWORK"), "sglang"
        ),
        "TP": str(tp),
        "GPU": gpu_ids,
        **masks,
        "HIP_VISIBLE_DEVICES": gpu_ids,
        "CUDA_VISIBLE_DEVICES": gpu_ids,
    }


def _replay_point_failure(
    proc: subprocess.CompletedProcess, tput: object, basis: str, requested_basis: str
) -> dict[str, str] | None:
    """The fields of a replayed point that does not count, or ``None`` when it measured the requested basis."""
    measured = proc.returncode == 0 and isinstance(tput, (int, float)) and tput > 0
    if not measured:
        return {"status": "failed", "error": (proc.stderr or "")[-500:] or "no throughput"}
    if basis != requested_basis:
        error = f"measured {basis}, not the requested {requested_basis}"
        return {"status": "failed", "error": error, "metric_basis": basis}
    return None


@dataclass(frozen=True)
class _ReplayPlan:
    script: Path
    use_final_launch: bool
    flags: str
    accepted_env: dict[str, str]
    serving_env: dict[str, str]
    bench_client: str
    e2e_metric: str
    requested_basis: str
    overlay: str
    repeats: int
    timeout_sec: int
    protocol_env: dict[str, str]

    @property
    def replay_mode(self) -> str:
        return "final_launch_script" if self.use_final_launch else "bench_e2e_fallback"


def _requires_deployment(result: Mapping[str, Any]) -> bool:
    tuning = result.get("tuning_skillset")
    return (
        isinstance(tuning, dict)
        and tuning.get("gate") == "accepted"
        and bool(tuning.get("live_tree_files") or tuning.get("cache_invalidation"))
    )


def _executable_final_launcher(script: str) -> Path | None:
    path = Path(script) if script else None
    return path if path and path.is_file() and os.access(path, os.X_OK) else None


# Forward the validated measurement config + client trust onto every variant so the sweep measures on the same
# workload shape the KERNEL_AGENT phase accepted (else bench_e2e.sh falls back to its own defaults).
def _replay_protocol_env(result: Mapping[str, Any], flags: str) -> dict[str, str]:
    protocol = result.get("bench_protocol")
    if not isinstance(protocol, dict):
        regimes = result.get("validated_regimes") or []
        protocol = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
    fields = (
        ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        ("num_warmups", "NUM_WARMUPS"),
        ("seed", "SEED"),
        ("num_prompts", "NUM_PROMPTS"),
    )
    env = {dst: str(protocol[src]) for src, dst in fields if protocol.get(src) is not None}
    # Mirror the server's --trust-remote-code onto the bench client so its tokenizer load doesn't raise.
    if "trust-remote-code" in flags or "trust_remote_code" in flags:
        for key in ("BENCH_TRUST_REMOTE_CODE", "HF_HUB_TRUST_REMOTE_CODE", "MAGPIE_TRUST_REMOTE_CODE"):
            env.setdefault(key, "1")
    return env


def _sweep_point_env(plan: _ReplayPlan, out_dir: Path, *, conc: int, isl: int, osl: int) -> dict[str, str]:
    env = build_benchmark_env(
        {
            "OUT_DIR": str(out_dir),
            **plan.serving_env,
            "ISL": str(isl),
            "OSL": str(osl),
            "CONC": str(conc),
            "REPEATS": str(plan.repeats),
            "PROFILE": "0",
            "OVERLAY_PYTHONPATH": plan.overlay,
            "EXTRA_SERVER_ARGS": plan.flags,
            "EXTRA_ENV": shlex.join(f"{name}={value}" for name, value in plan.accepted_env.items()),
            "BENCH_CLIENT": plan.bench_client,
            "E2E_METRIC": plan.e2e_metric,
        }
    )
    for name in VISIBLE_DEVICE_VARS:
        env.pop(name, None)
    env.update(plan.serving_env)
    # setdefault: forwarded config/trust apply unless already pinned.
    for key, value in plan.protocol_env.items():
        env.setdefault(key, value)
    # Final launch scripts accept the output directory as their first positional argument.
    if plan.use_final_launch:
        env["REPLICAS"] = str(plan.repeats)
    return env


def _launch_evidence_fields(plan: _ReplayPlan, out_dir: Path) -> dict[str, Any]:
    actual_log = _geak_replay_server_log(out_dir)
    requested = RequestedLaunch(
        plan.script,
        plan.serving_env["BACKEND"],
        server_args=plan.flags,
        server_env=plan.accepted_env,
        model_path=plan.serving_env["MODEL"],
    )
    evidence = build_launch_evidence(requested, actual_server_log=actual_log, slot=out_dir)
    return {
        "server_log_path": actual_log or "",
        "launch_evidence": evidence,
        "launch_evidence_path": persist_launch_evidence(evidence, slot=out_dir),
    }


def _measured_number(value: object) -> float | int | None:
    return value if isinstance(value, (int, float)) else None


async def _replay_sweep_point(plan: _ReplayPlan, out_dir: Path, *, conc: int, isl: int, osl: int) -> dict[str, Any]:
    env = _sweep_point_env(plan, out_dir, conc=conc, isl=isl, osl=osl)
    cmd = ["bash", str(plan.script), str(out_dir)] if plan.use_final_launch else ["bash", str(plan.script)]

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=plan.timeout_sec,
            env=env,
            cwd=str(out_dir),
        )

    entry: dict[str, Any] = {
        "conc": conc,
        "isl": isl,
        "osl": osl,
        "variant_name": out_dir.name,
        "workspace": str(out_dir),
        "replay_script": str(plan.script),
        "replay_mode": plan.replay_mode,
    }
    ttft = tpot = e2el = tput = None
    basis = ""
    succeeded = False
    err: str | None = None
    try:
        proc = await asyncio.to_thread(_run_with_power_sampling, _run, out_dir, env, entry)
        summ = read_json(out_dir / "bench_summary.json", default={}, require_dict=True)
        # ``throughput_tok_s_median`` is the metric-neutral median of whatever basis GEAK
        # measured, and the only field populated on every axis: GEAK nulls the output-named
        # alias off the output axis precisely so nobody reads another axis under an "output"
        # name. On output the two are the same number, so synthetic sweeps stay
        # byte-identical. The output-named field stays as the fallback for summaries written
        # before the neutral one existed.
        tput = summ.get("throughput_tok_s_median")
        if tput is None:
            tput = summ.get("output_throughput_tok_s_median")
        # A summary that names no basis predates the field and was measured on output.
        basis = str(summ.get("metric_basis") or GEAK_METRIC_OUTPUT[1])
        ttft, tpot, e2el = summ.get("ttft_ms_median"), summ.get("tpot_ms_median"), summ.get("e2el_ms_median")
        failure = _replay_point_failure(proc, tput, basis, plan.requested_basis)
        if failure is None:
            succeeded = True
            evaluation = parse_eval_results(out_dir, framework=plan.serving_env["BACKEND"])
            entry.update(
                {
                    "status": "succeeded",
                    "measured_value": tput,
                    "metric_basis": basis,
                    # ``output_throughput`` is GRADED_OUTPUT, which the perf snapshot reads, so a number on
                    # any other axis published under it would be recorded as the session's output throughput.
                    **({"output_throughput": tput} if basis == GEAK_METRIC_OUTPUT[1] else {}),
                    "ttft_mean_ms": ttft,
                    "tpot_mean_ms": tpot,
                    "accuracy": evaluation.get("accuracy"),
                    "accuracy_source": evaluation.get("source_file"),
                }
            )
        else:
            err = failure["error"]
            entry.update(failure)
    except Exception as exc:  # isolation boundary: one GEAK replay sweep point
        log.warning("GEAK replay failed for sweep point %s", out_dir.name, exc_info=True)
        err = repr(exc)
        entry.update({"status": "failed", "error": err})

    entry.update(_launch_evidence_fields(plan, out_dir))
    # Emit a session-breakdown-compatible benchmark_report.json so the sweep collector parses this point like
    # the native sweep; bench_summary.json is kept as the raw artifact.
    report: dict[str, Any] = {
        "success": succeeded,
        "conc": conc,
        "isl": isl,
        "osl": osl,
        "output_throughput_tok_s": _measured_number(tput) if basis == GEAK_METRIC_OUTPUT[1] else None,
        "mean_ttft_ms": _measured_number(ttft),
        "mean_tpot_ms": _measured_number(tpot),
        "mean_e2el_ms": _measured_number(e2el),
        "source": "geak",
    }
    if err:
        report["error"] = err
    _write_benchmark_report(out_dir, report)
    return entry


def _sweep_outcome(entries: list[dict[str, Any]], plan: _ReplayPlan, output_root: Path) -> dict[str, Any]:
    # The replay runs one (conc, isl, osl) repeated, so the best succeeded point is the headline. Every succeeded
    # point was measured on the requested basis, and every axis GEAK measures is higher-is-better.
    succeeded = [e for e in entries if e["status"] == "succeeded"]
    measured = [e for e in succeeded if isinstance(e.get("measured_value"), (int, float))]
    return {
        "status": "succeeded" if succeeded else "failed",
        **({"error": str(entries[0].get("error") or "replay_failed")} if entries and not succeeded else {}),
        "grid_size": len(entries),
        "points": entries,
        "workspace": output_root.as_posix(),
        "source": "geak",
        "replay_mode": plan.replay_mode,
        "replay_script": str(plan.script),
        "promotion_measurement": max(measured, key=lambda e: e["measured_value"], default={}),
    }


@dataclass(frozen=True)
class GeakSweepSpec:
    """The CONC × (ISL, OSL) points a GEAK replay sweep measures, and how each point is measured."""

    conc_values: list[int]
    isl_osl_configs: list[str]
    variant_timeout_sec: int
    repeats: int = 3
    metric_axis: tuple[str, str] = GEAK_METRIC_OUTPUT


async def sweep_via_geak(
    *,
    result: dict[str, Any],
    spec: GeakSweepSpec,
    output_root: Path,
    handoff: Mapping[str, Any] | None = None,
    env_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a CONC × (ISL, OSL) sweep on the GEAK-optimized server.

    ``spec.metric_axis`` is the ``(E2E_METRIC, metric_basis)`` pair from ``geak_metric_axis``. Every point is measured
    on it, and a point whose summary records another basis fails: a GEAK that cannot measure the requested axis
    and falls back to another would otherwise have that number rank the points.
    """
    handoff = handoff or {}
    e2e_metric, requested_basis = spec.metric_axis
    bench_script = result.get("bench_script") or result.get("geak_bench_script")
    final_launch_script = first_text(result.get("final_launch_script"))
    final_launch = _executable_final_launcher(final_launch_script)
    if _requires_deployment(result) and final_launch is None:
        return {
            "status": "failed",
            "error_class": "missing_deployment_launcher",
            "error": "GEAK accepted tuning requires file deployment, but its final launch script is unavailable.",
        }
    replay_script = final_launch or Path(str(bench_script or ""))
    try:
        flags, accepted_env = _accepted_config_as_variant(result.get("accepted_config"))
    except ValueError as exc:
        return {"status": "failed", "error_class": "invalid_accepted_config", "error": str(exc)}
    if not replay_script.is_file():
        return {
            "status": "failed",
            "error_class": "missing_bench_script",
            "error": (
                f"GEAK final launch script is not executable ({final_launch_script}); "
                f"bench script not found: {bench_script}"
            ),
        }
    env_spec = handoff.get("baseline_env_spec") or env_spec or {}
    try:
        serving_env = _replay_serving_env(handoff, env_spec)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        return {"status": "failed", "error_class": "invalid_replay_identity", "error": str(exc)}

    plan = _ReplayPlan(
        script=replay_script,
        use_final_launch=final_launch is not None,
        flags=flags,
        accepted_env=accepted_env,
        serving_env=serving_env,
        bench_client=first_text(result.get("bench_client")) or "native",
        e2e_metric=e2e_metric,
        requested_basis=requested_basis,
        overlay=result.get("final_overlay") or "",
        repeats=spec.repeats,
        timeout_sec=spec.variant_timeout_sec,
        protocol_env=_replay_protocol_env(result, flags),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, (conc, shape) in enumerate((c, s) for c in spec.conc_values for s in spec.isl_osl_configs):
        isl, osl = _parse_isl_osl(shape)
        # Name matches the sweep collector's scanner regex (``variant_<idx>_conc<c>_isl<i>_osl<o>``) so it is
        # discovered.
        out_dir = output_root / f"variant_{index}_conc{conc}_isl{isl}_osl{osl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        entries.append(await _replay_sweep_point(plan, out_dir, conc=conc, isl=isl, osl=osl))

    return _sweep_outcome(entries, plan, output_root)
