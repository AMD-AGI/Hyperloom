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
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.common.jsonio import read_json
from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS, effective_mask_tokens, is_rocr_level
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _accepted_config_as_variant,
    _coerce_tp,
    _resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
)
from ._accuracy_gate import parse_eval_results
from ._launch_evidence import build_launch_evidence, persist_launch_evidence

log = logging.getLogger(__name__)


def _write_benchmark_report(
    out_dir: Path,
    *,
    conc: int,
    isl: int,
    osl: int,
    success: bool,
    output_throughput_tok_s: float | None,
    mean_ttft_ms: float | None,
    mean_tpot_ms: float | None,
    mean_e2el_ms: float | None,
    error: str | None = None,
) -> None:
    """Write a session-breakdown-compatible ``benchmark_report.json``."""
    report = {
        "success": bool(success),
        "conc": conc,
        "isl": isl,
        "osl": osl,
        "output_throughput_tok_s": output_throughput_tok_s,
        "mean_ttft_ms": mean_ttft_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "mean_e2el_ms": mean_e2el_ms,
        "source": "geak",
    }
    if error:
        report["error"] = error
    try:
        (out_dir / "benchmark_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        # Best-effort: a failed write must never break the sweep.
        log.warning("geak_sweep: could not write %s: %s", out_dir / "benchmark_report.json", exc)


def _serving_gpus(tp: int) -> str:
    return ",".join(str(i) for i in range(max(tp, 1)))


def _parse_isl_osl(spec: str) -> tuple[int, int]:
    isl_s, _, osl_s = str(spec).partition(":")
    return int(isl_s or 1024), int(osl_s or 1024)


def _geak_replay_server_log(out_dir: Path) -> str | None:
    """Return the newest server log created inside this replay's output only."""
    candidates = [out_dir / "server.log"]
    try:
        candidates.extend(out_dir.glob("replica_*/attempt_*/server.log"))
    except OSError:
        # Replay-log discovery is best effort; retain the direct log candidate.
        logging.debug("Unable to enumerate GEAK replica server logs", exc_info=True)
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    try:
        return str(max(existing, key=lambda path: path.stat().st_mtime))
    except OSError:
        return None


def _replay_serving_env(handoff: Mapping[str, Any], env_spec: Mapping[str, Any]) -> dict[str, str]:
    """Resolve serving identity from the frozen handoff or its materialized recipe."""
    recipe_path = str(handoff.get("launch_recipe") or env_spec.get("base_launch_recipe") or "")
    benchmark: dict[str, Any] = {}
    if recipe_path:
        parsed = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("benchmark"), dict):
            raise ValueError("invalid_replay_recipe")
        benchmark = parsed["benchmark"]
    recipe_envs = benchmark.get("envs") or {}
    if not isinstance(recipe_envs, dict):
        raise ValueError("invalid_replay_recipe_envs")
    frozen = bool(handoff or benchmark)
    legacy_env = {} if frozen else os.environ
    tp = _coerce_tp(handoff.get("tp"), recipe_envs.get("TP"), legacy_env.get("TP"))
    pin = handoff.get("gpu_pin")
    if not isinstance(pin, dict):
        pin = _resolve_gpu_pin(recipe_envs=recipe_envs, environ=legacy_env)
    gpu_ids = ",".join(effective_mask_tokens(handoff.get("gpu_ids"))) or _resolve_handoff_gpu_ids(gpu_pin=pin, tp=tp)
    space = str(handoff.get("gpu_ids_space") or _resolve_handoff_gpu_ids_space(gpu_pin=pin))
    if space not in {"absolute", "logical"} or _resolve_handoff_gpu_ids_space(gpu_pin=pin) == "none":
        raise ValueError("invalid_replay_gpu_ids_space")
    if len(effective_mask_tokens(gpu_ids)) < tp:
        raise ValueError("replay_gpu_count_below_tp")
    masks: dict[str, str] = {}
    if space == "logical":
        var = str(pin.get("var") or "")
        tokens = effective_mask_tokens(pin.get("value"))
        if not is_rocr_level(var) or not tokens:
            raise ValueError("missing_replay_logical_gpu_pin")
        if any(not token.isdigit() or int(token) >= len(tokens) for token in effective_mask_tokens(gpu_ids)):
            raise ValueError("invalid_replay_logical_gpu_ids")
        masks[var] = ",".join(tokens)
    masks["HIP_VISIBLE_DEVICES"] = gpu_ids
    masks["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return {
        "MODEL": str(handoff.get("model_path") or benchmark.get("model") or legacy_env.get("MODEL_PATH") or "").strip(),
        "BACKEND": str(
            handoff.get("framework") or benchmark.get("framework") or legacy_env.get("FRAMEWORK") or "sglang"
        ).strip(),
        "TP": str(tp),
        "GPU": gpu_ids,
        **masks,
    }


async def sweep_via_geak(
    *,
    result: dict[str, Any],
    conc_values: list[int],
    isl_osl_configs: list[str],
    output_root: Path,
    variant_timeout_sec: int,
    repeats: int = 3,
    pin_num_prompts: bool = False,
    handoff: Mapping[str, Any] | None = None,
    env_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a CONC × (ISL, OSL) sweep on the GEAK-optimized server."""
    handoff = handoff or {}
    bench_client = str(result.get("bench_client") or "native").strip() or "native"
    bench_script = result.get("bench_script") or result.get("geak_bench_script")
    final_launch_script = str(result.get("final_launch_script") or "").strip()
    final_launch_path = Path(final_launch_script) if final_launch_script else None
    use_final_launch = bool(final_launch_path and final_launch_path.is_file() and os.access(final_launch_path, os.X_OK))
    tuning = result.get("tuning_skillset")
    requires_deployment = (
        isinstance(tuning, dict)
        and tuning.get("gate") == "accepted"
        and bool(tuning.get("live_tree_files") or tuning.get("cache_invalidation"))
    )
    if requires_deployment and not use_final_launch:
        return {
            "status": "failed",
            "error_class": "missing_deployment_launcher",
            "error": "GEAK accepted tuning requires file deployment, but its final launch script is unavailable.",
        }
    replay_script = final_launch_path if use_final_launch else Path(str(bench_script or ""))
    overlay = result.get("final_overlay") or ""
    try:
        flags, accepted_env = _accepted_config_as_variant(result.get("accepted_config"))
    except ValueError as exc:
        return {"status": "failed", "error_class": "invalid_accepted_config", "error": str(exc)}
    env_str = shlex.join(f"{name}={value}" for name, value in accepted_env.items())

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
    model, backend = serving_env["MODEL"], serving_env["BACKEND"]

    # Forward the validated measurement config + client trust onto every variant so the sweep measures on the same
    # workload shape the KERNEL_AGENT phase accepted (else bench_e2e.sh falls back to its own defaults).
    _protocol = result.get("bench_protocol")
    if not isinstance(_protocol, dict):
        _regimes = result.get("validated_regimes") or []
        _protocol = _regimes[0] if _regimes and isinstance(_regimes[0], dict) else {}
    protocol_env: dict[str, str] = {}
    _protocol_map = [
        ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        ("num_warmups", "NUM_WARMUPS"),
        ("seed", "SEED"),
    ]
    # Single-point validated replay: also pin num_prompts (see docstring).
    if pin_num_prompts:
        _protocol_map.append(("num_prompts", "NUM_PROMPTS"))
    for _src, _dst in _protocol_map:
        _val = _protocol.get(_src)
        if _val is not None:
            protocol_env[_dst] = str(_val)
    # Mirror the server's --trust-remote-code onto the bench client so its tokenizer load doesn't raise.
    if "trust-remote-code" in flags or "trust_remote_code" in flags:
        for _tk in ("BENCH_TRUST_REMOTE_CODE", "HF_HUB_TRUST_REMOTE_CODE", "MAGPIE_TRUST_REMOTE_CODE"):
            protocol_env.setdefault(_tk, "1")

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    variant_idx = 0
    for conc in conc_values:
        for spec in isl_osl_configs:
            isl, osl = _parse_isl_osl(spec)
            # Name matches the sweep collector's scanner regex (``variant_<idx>_conc<c>_isl<i>_osl<o>``) so it is
            # discovered.
            variant_name = f"variant_{variant_idx}_conc{conc}_isl{isl}_osl{osl}"
            variant_idx += 1
            out_dir = output_root / variant_name
            out_dir.mkdir(parents=True, exist_ok=True)
            env = build_benchmark_env(
                {
                    "OUT_DIR": str(out_dir),
                    **serving_env,
                    "ISL": str(isl),
                    "OSL": str(osl),
                    "CONC": str(conc),
                    "REPEATS": str(repeats),
                    "PROFILE": "0",
                    "OVERLAY_PYTHONPATH": overlay,
                    "EXTRA_SERVER_ARGS": flags,
                    "EXTRA_ENV": env_str,
                    "BENCH_CLIENT": bench_client,
                }
            )
            for name in VISIBLE_DEVICE_VARS:
                env.pop(name, None)
            env.update(serving_env)
            # setdefault: forwarded config/trust apply unless already pinned.
            for _k, _v in protocol_env.items():
                env.setdefault(_k, _v)
            # Final launch scripts accept the output directory as their first positional argument.
            if use_final_launch:
                env["REPLICAS"] = str(repeats)
                cmd = ["bash", str(replay_script), str(out_dir)]
            else:
                cmd = ["bash", str(replay_script)]

            def _run() -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=variant_timeout_sec,
                    env=env,
                    cwd=str(out_dir),
                )

            entry: dict[str, Any] = {
                "conc": conc,
                "isl": isl,
                "osl": osl,
                "variant_name": variant_name,
                "workspace": str(out_dir),
                "replay_script": str(replay_script),
                "replay_mode": "final_launch_script" if use_final_launch else "bench_e2e_fallback",
            }
            ttft = tpot = e2el = None
            tput = None
            succeeded = False
            err: str | None = None
            try:
                proc = await asyncio.to_thread(_run)
                summ = read_json(out_dir / "bench_summary.json", default={}, require_dict=True)
                # ``throughput_tok_s_median`` is the metric-neutral median of
                # whatever basis GEAK measured, and the only field populated in
                # both modes: bench_e2e.sh nulls the output-named alias under
                # E2E_METRIC=total precisely so nobody reads total throughput
                # under an "output" name. In output mode the two are the same
                # number, so this keeps synthetic sweeps byte-identical while
                # letting an agentic one report at all. The output-named field
                # stays as the fallback for summaries written before it existed.
                tput = summ.get("throughput_tok_s_median")
                if tput is None:
                    tput = summ.get("output_throughput_tok_s_median")
                ttft = summ.get("ttft_ms_median")
                tpot = summ.get("tpot_ms_median")
                e2el = summ.get("e2el_ms_median")
                if proc.returncode == 0 and isinstance(tput, (int, float)) and tput > 0:
                    succeeded = True
                    evaluation = parse_eval_results(out_dir, framework=backend)
                    entry.update(
                        {
                            "status": "succeeded",
                            "output_throughput": tput,
                            "ttft_mean_ms": ttft,
                            "tpot_mean_ms": tpot,
                            "accuracy": evaluation.get("accuracy"),
                            "accuracy_source": evaluation.get("source_file"),
                        }
                    )
                else:
                    err = (proc.stderr or "")[-500:] or "no throughput"
                    entry.update({"status": "failed", "error": err})
            except Exception as exc:  # noqa: BLE001
                err = repr(exc)
                entry.update({"status": "failed", "error": err})

            actual_log = _geak_replay_server_log(out_dir)
            evidence = build_launch_evidence(
                config_path=replay_script,
                actual_server_log=actual_log,
                framework=backend,
                slot=out_dir,
                requested_server_args=flags,
                requested_server_env=accepted_env,
                model_path=model,
            )
            entry["server_log_path"] = actual_log or ""
            entry["launch_evidence"] = evidence
            entry["launch_evidence_path"] = persist_launch_evidence(evidence, slot=out_dir)

            # Emit a session-breakdown-compatible benchmark_report.json so the sweep collector parses this point like
            # the native sweep; bench_summary.json is kept as the raw artifact.
            _write_benchmark_report(
                out_dir,
                conc=conc,
                isl=isl,
                osl=osl,
                success=succeeded,
                output_throughput_tok_s=tput if isinstance(tput, (int, float)) else None,
                mean_ttft_ms=ttft if isinstance(ttft, (int, float)) else None,
                mean_tpot_ms=tpot if isinstance(tpot, (int, float)) else None,
                mean_e2el_ms=e2el if isinstance(e2el, (int, float)) else None,
                error=err,
            )
            entries.append(entry)

    # The replay runs one (conc, isl, osl) repeated, so the fastest succeeded point is the headline.
    succeeded = [e for e in entries if e["status"] == "succeeded"]
    measured = [e for e in succeeded if isinstance(e.get("output_throughput"), (int, float))]
    promotion_measurement = max(measured, key=lambda e: e["output_throughput"], default={})
    return {
        "status": "succeeded" if succeeded else "failed",
        **({"error": str(entries[0].get("error") or "replay_failed")} if entries and not succeeded else {}),
        "grid_size": len(entries),
        "points": entries,
        "workspace": output_root.as_posix(),
        "source": "geak",
        "replay_mode": "final_launch_script" if use_final_launch else "bench_e2e_fallback",
        "replay_script": str(replay_script),
        "promotion_measurement": promotion_measurement,
    }
