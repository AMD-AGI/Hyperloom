# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_GeakDispatchMixin``: GEAK runner handoff, launch, skip settlement and crash recovery for the KERNEL phase."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from . import geak_rebench as _geak_rebench, machine_state as _phase_state
from .kernel_round_budget import rebench_round_sec
from hyperloom_common.env import env_float, env_int
from hyperloom_common.jsonio import read_json
from hyperloom.common.benchmark_config import BenchmarkPoint, BenchmarkPointError, read_benchmark_point
from hyperloom.inference_optimizer.agentx.report import replayed_trace
from inference_suite import CLIENT_ROOT_ENV, CLIENT_SCRIPT, inferencex_root
from ..actions.executors._round_folder import read_round_launch
from ..actions.executors._suite.rounds import staged_tool_cache
from ..actions.executors._workload_envs import geak_grading_fields, geak_metric_axis
from ..bus.message_bus import Message
from ..kernel.geak_config import _geak_has_accepted_kernel
from ..kernel.lane_budget import INSUFFICIENT_BUDGET
from ..actions.executors._gpu_pin import (
    coerce_tp,
    resolve_gpu_pin,
    _resolve_handoff_gpu_identity,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
    _resolve_handoff_physical_gpus,
    _resolve_handoff_tp,
    _resolve_serving_fidelity,
)
from .kernel_geak_promotion import GeakRunnerFacts
from .kernel_gemm_tuning import _first_text

log = logging.getLogger(__name__)

_GEAK_RUNNER_MODULE = "hyperloom.orchestrator.kernel.geak_runner"
_NO_RECIPE = BenchmarkPoint.from_block({})
_TIMED_OUT = "timeout"

# How long the GEAK handoff waits for the servers it stopped to release their VRAM.
_GEAK_GPU_FREE_WAIT_SEC: float = 120.0

#: GEAK's names for the workload a handoff measures (``run_e2e.WORKLOAD_KIND_*``) and for its InferenceX client.
_AGENTX_WORKLOAD = "agentx_trace_replay"
_SYNTHETIC_WORKLOAD = "synthetic_isl_osl"
_INFERENCEX_CLIENT = "inferencex"


def _staged_inferencex() -> Path:
    return inferencex_root(staged_tool_cache())


def _geak_client_env(handoff: Mapping[str, Any]) -> dict[str, str]:
    """The client file GEAK's InferenceX adapter runs as ``python <file>``: the staged client is a package module."""
    if handoff.get("bench_client") != _INFERENCEX_CLIENT:
        return {}
    return {"INFERENCEX_BENCH_SERVING": str(CLIENT_SCRIPT), CLIENT_ROOT_ENV: str(handoff["inferencex_path"])}


def _with_framework_on_path(env_spec: dict[str, Any]) -> dict[str, Any]:
    """``env_spec`` with the measured round's framework interpreter first on ``PATH``, for GEAK's native launcher."""
    launch_json = str(env_spec.get("launch_json") or "")
    launch = read_round_launch(Path(launch_json).parent) if launch_json else None
    if launch is None or not Path(launch.framework.python).is_absolute():
        return env_spec
    path = os.pathsep.join(part for part in (str(Path(launch.framework.python).parent), os.environ.get("PATH")) if part)
    config = env_spec["config"]
    return {**env_spec, "config": {**config, "extra_envs": {**config["extra_envs"], "PATH": path}}}


def _observed_identity_digest(declared: str, flags: str, server_identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "declared_launch_identity": declared,
            "observed_server_launch_flags": flags,
            "observed_server_identity": server_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class EnvSpecUnserializable(Exception):
    """The accepted launch configuration could not be serialized into an env spec.

    Only this failure is an ``invalid_env_spec``. Everything else the GEAK handoff
    raises while it is built -- a native-AgentX proposal refusing an unverified
    launch, a malformed env var, a state field that is not a number -- is a different
    fault and propagates to the caller rather than being reported as one.
    """


@dataclass(frozen=True)
class GeakRunnerOutcome:
    """What one GEAK runner dispatch leaves for the phase to settle."""

    result: dict[str, Any] | None = None
    facts: GeakRunnerFacts = GeakRunnerFacts()
    recovered_from: str = ""


class _GeakDispatchMixin:
    @staticmethod
    def _resolve_bench_protocol(envs: Mapping[str, Any]) -> dict[str, Any]:
        """Extract Hyperloom's bench measurement protocol for the GEAK handoff.

        Reads an already-parsed ``benchmark.envs`` (falling back to the process
        env) and returns only the keys that resolve, so absent values leave GEAK
        on its standalone defaults. Never raises.
        """

        def _pick(key: str, cast: Callable[[str], Any]) -> Any:
            raw = envs.get(key)
            if raw is None or str(raw).strip() == "":
                raw = os.environ.get(key, "")
            raw = str(raw).strip()
            if not raw:
                return None
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return None

        protocol: dict[str, Any] = {}
        for proto_key, env_key, cast in (
            ("random_range_ratio", "RANDOM_RANGE_RATIO", float),
            ("num_prompts", "NUM_PROMPTS", int),
            ("num_warmups", "NUM_WARMUPS", int),
            ("seed", "SEED", int),
        ):
            val = _pick(env_key, cast)
            if val is not None:
                protocol[proto_key] = val
        return protocol

    @staticmethod
    def _recipe_point(recipe_path: str) -> BenchmarkPoint:
        """Read the materialized recipe once for every handoff resolver.

        A missing, unreadable or malformed recipe reads as the empty point, which each
        resolver already treats as "nothing to advertise". Never raises.
        """
        import yaml

        if recipe_path and Path(recipe_path).is_file():
            try:
                return read_benchmark_point(Path(recipe_path))
            except (OSError, BenchmarkPointError, yaml.YAMLError):
                log.warning("recipe: could not read %r", recipe_path, exc_info=True)
        return _NO_RECIPE

    def _geak_timeouts(self) -> tuple[int, int, bool]:
        """Resolve the GEAK e2e timeouts from the live run budget."""
        # Standalone fallback ONLY: the 12h (43200s) default applies when no run deadline is set (budget_known=False).
        env_default_timeout = env_int("GEAK_E2E_TIMEOUT_S", default=43200)
        deadline = self._coord.run_deadline
        if deadline is None:
            return env_default_timeout, env_default_timeout + 600, False
        remaining = deadline.remaining()
        grace = self.shared_state.closing_reserve_sec()
        margin = env_float("GEAK_BUDGET_MARGIN_S", default=300.0)
        # Reserve the closing window: kill the subprocess with at least ``grace`` left.
        kill_budget = remaining - grace
        # Also honour the KERNEL_AGENT phase's own wall-clock budget: cap by min(session, kernel_phase).
        phase_rem = _phase_state.phase_budget_remaining_seconds(self.shared_state)
        if phase_rem is not None:
            kill_budget = min(kill_budget, float(phase_rem))
        # A GEAK result reaches the headline only through this session's own re-measurement of it, which runs in this
        # phase after GEAK returns. Which results skip it is known only then, so every hand-off leaves room for it.
        rebench_sec = rebench_round_sec(self.shared_state)
        log.info("GEAK: holding back %.0fs of the KERNEL budget to re-measure GEAK's result", rebench_sec)
        kill_budget -= rebench_sec
        # The runner self-stops ``margin`` before the hard subprocess kill, which reserves the closing-grace window.
        kill_timeout = int(max(0.0, kill_budget))
        runner_timeout = int(max(0.0, kill_budget - margin))
        return runner_timeout, kill_timeout, True

    async def _free_gpus_for_geak(self, cards: list[int] | None) -> str | None:
        """Stop what this session left on GEAK's GPUs, then wait for them to drain; return why they did not, if so.

        The ``kernel_agent`` task holds every serving lane by now, but a lane only records that the round holding it
        ended, not that the server the round booted went with it.
        """
        from ..actions.executors._server_lifecycle import free_gpus_for_handoff

        gpus = await asyncio.to_thread(
            free_gpus_for_handoff, self.session_dir, cards, timeout_sec=_GEAK_GPU_FREE_WAIT_SEC
        )
        if gpus.reaped:
            log.warning(
                "geak handoff: stopped %d server(s) this session left running: pids=%s", len(gpus.reaped), gpus.reaped
            )
        if gpus.unverified:
            log.warning("geak handoff: cannot confirm GPUs %s are free (%s); delegating anyway", cards, gpus.unverified)
        if not gpus.busy:
            return None
        busy = ", ".join(f"GPU{card} {used:.0f}/{total:.0f} MiB" for card, used, total in gpus.busy)
        error = f"GPUs still hold VRAM after this session's leftover servers were stopped: {busy}"
        if gpus.holders:
            error += f"; serving processes still up: {'; '.join(gpus.holders)}"
        log.error("geak handoff: %s; not delegating", error)
        return error

    async def _wait_for_gpus_before_rebench(self, cards: list[int] | None) -> None:
        """Let the GPUs GEAK hands back drain before the rebench boots a server on them.

        A card still holding the VRAM of a process GEAK's run left behind fails the server's memory-balance check at
        boot, and the candidate is then dropped unvalidated. The rebench is attempted either way: this only waits.
        """
        from ..actions.executors._server_lifecycle import free_gpus_for_handoff

        gpus = await asyncio.to_thread(
            free_gpus_for_handoff, self.session_dir, cards, timeout_sec=_GEAK_GPU_FREE_WAIT_SEC
        )
        if gpus.reaped:
            log.warning(
                "geak rebench: stopped %d server(s) this session left running: pids=%s", len(gpus.reaped), gpus.reaped
            )
        if gpus.busy:
            busy = ", ".join(f"GPU{card} {used:.0f}/{total:.0f} MiB" for card, used, total in gpus.busy)
            holders = f"; serving processes still up: {'; '.join(gpus.holders)}" if gpus.holders else ""
            log.error("geak rebench: GPUs still hold VRAM after GEAK returned: %s%s; measuring anyway", busy, holders)
        elif gpus.unverified:
            log.warning("geak rebench: cannot confirm GPUs %s are free (%s)", cards, gpus.unverified)

    async def _launch_geak_runner(
        self, handoff_path: Path, *, timeouts: tuple[int, int, bool], from_phase: str, client_env: Mapping[str, str]
    ) -> subprocess.CompletedProcess:
        """Run geak_runner on ``handoff_path`` under :meth:`_geak_timeouts`; raise TimeoutExpired at its kill cap.

        geak_runner gives run_e2e its own SIGTERM grace to flush result.json before ``runner_timeout`` +
        ``GEAK_FLUSH_GRACE_S``, inside the kill cap. The launch marks the run, so whatever it left running is
        reaped when it returns, however it detached.
        """
        from hyperloom_common.proctree import REAP_MARK_ENV
        from ..actions.executors._subprocess_kill import RoundWatch, run_with_session_kill

        runner_timeout, kill_timeout, budget_known = timeouts
        cmd = [
            sys.executable,
            "-m",
            _GEAK_RUNNER_MODULE,
            str(handoff_path),
            str(handoff_path.parent),
            "--timeout-s",
            str(runner_timeout),
        ]
        log.info(
            "KERNEL entry: delegating to GEAK e2e (from=%s) runner_timeout=%ds kill_timeout=%ds budget_known=%s cmd=%s",
            from_phase or "<unknown>",
            runner_timeout,
            kill_timeout,
            budget_known,
            " ".join(cmd),
        )
        # GEAK must search and gate on the axis this session keeps candidates on: a kernel accepted on another
        # axis need not move that one. geak_metric_axis owns the mapping and its rationale.
        metric, _ = geak_metric_axis(
            benchmark_mode=str(self.shared_state.benchmark_mode or ""),
            grading=self.shared_state.grading or None,
        )
        runner_env = {**os.environ, **client_env}
        runner_env["E2E_METRIC"] = metric
        runner_env[REAP_MARK_ENV] = uuid.uuid4().hex
        # Only injection point needed for the whole GEAK chain: geak_runner and run_e2e both hand their full
        # environment to the child, so the tag reaches the Claude CLI that actually spends.
        from hyperloom_common.llm.llm_attribution import inject_env

        inject_env(runner_env, component="geak", operation="optimize_kernel")
        return await asyncio.to_thread(
            run_with_session_kill, cmd, env=runner_env, watch=RoundWatch(timeout=kill_timeout)
        )

    def _accepted_launch_spec(self) -> dict[str, Any]:
        try:
            return self._coord.writeback.build_env_spec()
        except (OSError, TypeError, ValueError) as exc:
            raise EnvSpecUnserializable(str(exc)) from exc

    @staticmethod
    def _reference_verification_status(evidence: Mapping[str, Any], *, identity_matches: bool, observed: bool) -> str:
        if identity_matches and observed:
            return "verified_observed"
        declared = (
            str(evidence.get("requested_server_args") or "").strip()
            or bool(evidence.get("requested_server_env"))
            or str(evidence.get("recipe_digest") or "").strip()
        )
        return "verified_declared_only" if identity_matches and declared else "unverified"

    def _same_config_reference(
        self, measurement: Mapping[str, Any], *, expected_identity: str, agentx: bool
    ) -> dict[str, Any]:
        measured_identity = _first_text(measurement, "declared_launch_identity")
        identity_matches = bool(expected_identity and expected_identity == measured_identity)
        evidence = measurement.get("launch_evidence")
        evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        observed_flags = str(
            measurement.get("resolved_server_launch_flags") or evidence.get("observed_server_launch_flags") or ""
        ).strip()
        server_identity = measurement.get("observed_server_identity") or evidence.get("observed_server_identity")
        server_identity = (
            {str(key): value for key, value in sorted(server_identity.items())}
            if isinstance(server_identity, Mapping)
            else {}
        )
        observed = bool(observed_flags or server_identity)
        status = (
            "unverified_workload"
            if agentx
            else self._reference_verification_status(evidence, identity_matches=identity_matches, observed=observed)
        )
        observed_identity = _first_text(measurement, "observed_launch_identity")
        if not observed_identity and identity_matches and observed:
            observed_identity = _observed_identity_digest(measured_identity, observed_flags, server_identity)
        verified = status == "verified_observed"
        return {
            # Zero means no verified same-config reference.
            "orchestrator_best_tput_same_config": float(measurement.get("tput") or 0.0) if verified else 0.0,
            "same_config_reference_status": "verified" if verified else "unverified",
            "same_config_reference_identity": measured_identity,
            "same_config_expected_identity": expected_identity,
            "same_config_reference_workspace": _first_text(measurement, "benchmark_workspace"),
            # Additive identity semantics.
            "same_config_reference_verification_status": status,
            "same_config_reference_declared_identity": measured_identity,
            "same_config_reference_observed_identity": observed_identity,
            # GEAK compares this map with its parsed ServerArgs.
            "same_config_observed_identity": server_identity,
            "observed_server_identity": server_identity,
            "measurement_evidence": evidence,
        }

    @staticmethod
    def _geak_device_fields(recipe_envs: Mapping[str, Any]) -> dict[str, Any]:
        gpu_pin = resolve_gpu_pin(recipe_envs=recipe_envs)
        tp = coerce_tp(recipe_envs.get("TP"), os.environ.get("TP"))
        gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=gpu_pin, tp=tp)
        space = _resolve_handoff_gpu_ids_space(gpu_pin=gpu_pin)
        if space == "none":
            # An empty visibility mask means the ids are placeholders, not devices.
            log.error(
                "geak handoff: %s is set but empty; the run has no visible GPUs. "
                "gpu_ids=%s is a placeholder (gpu_ids_space=none), not a device set.",
                gpu_pin.get("var"),
                gpu_ids,
            )
        return {
            "tp": _resolve_handoff_tp(gpu_ids=gpu_ids, tp=tp),
            # The serving/optimization device set, as HIP-level ids (what the
            # consumer exports as HIP_VISIBLE_DEVICES). Logical positions inside
            # an inherited ROCR mask, a HIP/CUDA mask as-is, else 0..tp-1.
            "gpu_ids": gpu_ids,
            # Which coordinate system ``gpu_ids`` is in: "logical" (positions
            # inside the ROCR mask the child inherits), "absolute" (whole-
            # machine device ids), or "none" (the mask is set but empty -- the
            # ids are placeholders and must not be launched on). Exporting them
            # as HIP_VISIBLE_DEVICES is correct in the first two; a consumer
            # that instead writes ROCR itself needs to know which it holds.
            "gpu_ids_space": space,
            # ABSOLUTE ids + the var they came from, so a consumer that writes
            # ROCR_VISIBLE_DEVICES itself re-applies the same pin instead of
            # resetting the child to card 0.
            **({"gpu_pin": gpu_pin} if gpu_pin else {}),
        }

    def _geak_serving_fidelity(self, spec_config: Mapping[str, Any], accepted_flags: str) -> dict[str, Any]:
        server_args = str(spec_config.get("server_launch_flags") or "")
        if not server_args:
            # Reuse baseline server settings so GEAK measures the same engine config.
            from hyperloom.inference_optimizer.roofline_ceiling import read_baseline_server_args

            inherited = read_baseline_server_args(self.shared_state) or ""
            from hyperloom.inference_optimizer.grid_server_args import compose_server_args

            server_args = compose_server_args(
                inherited_args=inherited,
                variant_extra_args=accepted_flags,
                remove_args=spec_config.get("remove_args"),
                args_mode=str(spec_config.get("args_mode") or "append"),
            )
        return _resolve_serving_fidelity(
            baseline_server_args=server_args,
            state_max_model_len=int(self.shared_state.max_model_len or 0),
        )

    def _geak_workload_spec(self, spec: Mapping[str, Any], *, agentx: bool) -> dict[str, Any]:
        # GEAK selects its client and its measurement protocol from the kind.
        if not agentx:
            return {"kind": _SYNTHETIC_WORKLOAD}
        state = self.shared_state
        session = {"concurrency": int(state.conc or 0), **geak_grading_fields(state.grading or None)}
        if spec:
            # Only an MLPerf AgentX recipe carries a spec; it names the graded workload and its client.
            return {**spec, **session}
        workspace = str(state.current_best_measurement.get("benchmark_workspace") or "")
        # GEAK keeps its own default for each field left out, so a trace it is not told about is its default corpus.
        replayed = replayed_trace(Path(workspace) if workspace else None)
        if replayed:
            log.info(
                "workload_spec: the accepted round replayed %s (%d entries)",
                replayed["corpus"],
                replayed["num_entries"],
            )
        else:
            log.warning("workload_spec: no aiperf export under %r names a corpus; GEAK replays its default", workspace)
        return {**replayed, **session, "kind": _AGENTX_WORKLOAD}

    def _geak_session_fields(self, *, agentx: bool) -> dict[str, Any]:
        state = self.shared_state
        # GEAK's E2E_METRIC and the workload spec's metric_basis are the same decision, so they
        # resolve through one helper: a handoff that named a different axis than the one KEEP is
        # decided on would have GEAK searching against a reference it was never measured against.
        e2e_metric, _ = geak_metric_axis(benchmark_mode=str(state.benchmark_mode or ""), grading=state.grading or None)
        return {
            # v2 adds baseline_env_spec; v3 adds actual GPU-pinning metadata.
            "schema_version": 3,
            "model_path": str(state.model_path or os.environ.get("MODEL_PATH", "")),
            "framework": str(state.framework or os.environ.get("FRAMEWORK", "") or "sglang"),
            **_resolve_handoff_gpu_identity(state.gpu_type or os.environ.get("GPU_TYPE", "")),
            "workload": {"isl": int(state.isl), "osl": int(state.osl), "conc": int(state.conc or 64)},
            # AgentX canonical throughput is not a reference for GEAK's proxy workload.
            "raw_baseline_tput": 0.0 if agentx else float(state.baseline_tput or 0.0),
            "exp_root": str(self.session_dir / "geak"),
            # Macro-cycle-scoped eval_dir so a same-cycle resume reuses the in-progress on-disk artifacts while a new
            # cycle gets a fresh dir.
            "eval_dir": str(self.session_dir / "geak" / f"e2e_cycle{int(state.macro_cycle or 0)}"),
            "e2e_metric": e2e_metric,
            # GEAK picks its AgentX replay client from the workload kind, and forces it back over any client named.
            **({} if agentx else {"bench_client": _INFERENCEX_CLIENT, "inferencex_path": str(_staged_inferencex())}),
        }

    def _build_geak_handoff(self, *, agentx: bool) -> dict[str, Any]:
        cb = self.shared_state.current_best
        env_spec = self._accepted_launch_spec()
        measurement = self.shared_state.current_best_measurement
        spec_config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
        accepted_flags = str(spec_config.get("extra_server_args", cb.get("extra_server_args")) or "")
        extra_envs = spec_config.get("extra_envs", cb.get("extra_envs")) or {}
        recipe = self._recipe_point(str(self.shared_state.baseline_config_path or ""))
        recipe_envs = recipe.envs
        handoff = {
            **self._geak_session_fields(agentx=agentx),
            **self._geak_device_fields(recipe_envs),
            "accepted_flags": accepted_flags,
            "accepted_env": shlex.join(f"{k}={v}" for k, v in dict(extra_envs).items()),
            **self._same_config_reference(
                measurement, expected_identity=str(env_spec.get("launch_identity") or ""), agentx=agentx
            ),
            "resolved_server_config": dict(measurement.get("resolved_server_config") or {}),
            **self._geak_serving_fidelity(spec_config, accepted_flags),
            # GEAK launches the server itself from the env spec, which carries the measured round's launch.json.
            "bench_launcher": "native",
            "workload_spec": self._geak_workload_spec(recipe.workload_spec, agentx=agentx),
            "baseline_env_spec": _with_framework_on_path(env_spec),
        }
        if protocol := self._resolve_bench_protocol(recipe_envs):
            handoff["bench_protocol"] = protocol
        if agentx:
            log.info("GEAK results remain proposal proxies; canonical AgentX validation remains in Hyperloom.")
            from hyperloom.inference_optimizer.agentx.geak_proxy import seed_native_geak_proxy

            seed_native_geak_proxy(handoff)
        return handoff

    def _geak_result_path(self) -> Path:
        return self.session_dir / "geak" / "result.json"

    def _geak_settled_replay(self, candidate: dict[str, Any], *, agentx: bool) -> bool:
        """Whether ``candidate`` is a result this session already settled."""
        return _geak_rebench.geak_candidate_is_adjudicated(
            self.shared_state.geak_result, candidate, harness_can_replay=not agentx
        )

    def _settle_geak_skip(self, result: dict[str, Any]) -> None:
        # A settled verdict is left standing: a later failure records itself without
        # retiring the candidate that already adjudicated.
        state = self.shared_state
        if not _geak_rebench.geak_verdict_is_terminal(state.geak_result):
            state.geak_result = result
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "error_class": result.get("error_class"),
                "error": (str(result.get("error") or "")[:500] or None),
            }
        )
        state.save(self.session_dir)

    def _finish_geak_skip(
        self, result: dict[str, Any], *, handoff: dict[str, Any], facts: GeakRunnerFacts = GeakRunnerFacts()
    ) -> None:
        self._record_geak_delegation_timeline(result, handoff=handoff, facts=facts)
        self._settle_geak_skip(result)

    def _promote_recovered_geak_result(
        self,
        result: dict[str, Any],
        *,
        handoff: dict[str, Any],
        recovered_from: str,
        facts: GeakRunnerFacts = GeakRunnerFacts(),
    ) -> bool:
        state = self.shared_state
        self._record_geak_delegation_timeline(result, handoff=handoff, recovered_from_disk=True, facts=facts)
        previous = state.geak_result or {}
        if previous.get("kernel_event_id") and _geak_rebench.geak_candidate_matches(previous, result):
            result["kernel_event_id"] = previous["kernel_event_id"]
        state.geak_result = result
        self._record_geak_measurement(result)
        # Rebench-first: record the recovered win as an UNVALIDATED candidate;
        # the caller enqueues the main-flow rebench that writes the headline.
        if not self._record_geak_candidate(result):
            state.save(self.session_dir)
            return False
        self._record_geak_kernel_journey(result)
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "recovered_from": recovered_from,
                **({"runner_timeout_s": facts.runner_timeout_sec} if facts.runner_timeout_sec is not None else {}),
            }
        )
        state.save(self.session_dir)
        return True

    def _refuse_geak_for_budget(self, handoff: dict[str, Any], *, runner_timeout: int, min_run: int) -> None:
        log.warning(
            "GEAK: only %ds budget remains (< min %ds); skipping e2e "
            "and winding down to SWEEP so the closing report runs in time.",
            runner_timeout,
            min_run,
        )
        self._finish_geak_skip(
            {
                "status": "skipped",
                "error_class": INSUFFICIENT_BUDGET,
                "error": (
                    f"only {runner_timeout}s of KERNEL budget remained "
                    f"(< min {min_run}s); skipped to protect the closing report window"
                ),
                "runner_timeout_s": runner_timeout,
            },
            handoff=handoff,
        )

    def _outer_timeout_record(
        self, flushed: dict[str, Any], *, runner_timeout: int, kill_timeout: int
    ) -> dict[str, Any]:
        """What an outer-deadline kill leaves in the session.

        A ``result.json`` GEAK managed to flush keeps its own status, phase, validation status, and accepted
        kernels. No file means Hyperloom killed the run before GEAK classified it.
        """
        if not flushed:
            return {
                "status": _TIMED_OUT,
                "error_class": "external_timeout",
                "phase_reached": "unknown",
                "error": (
                    f"GEAK e2e killed after {kill_timeout}s (budget-capped) "
                    "before it wrote result.json; closing window preserved"
                ),
                "runner_timeout_s": runner_timeout,
                "kill_timeout_s": kill_timeout,
            }
        error = str(flushed.get("error") or "").strip()
        record: dict[str, Any] = {
            "status": flushed.get("status") or _TIMED_OUT,
            "error_class": flushed.get("error_class") or flushed.get("status") or _TIMED_OUT,
            "error": error
            or f"GEAK e2e killed after {kill_timeout}s (budget-capped); retained the result GEAK flushed",
            "runner_timeout_s": runner_timeout,
            "kill_timeout_s": kill_timeout,
        }
        for key in ("phase_reached", "validation_status", "accepted_kernels", "stages_reached"):
            if key in flushed:
                record[key] = flushed[key]
        phase = record.get("phase_reached")
        if phase and "stages_reached" not in record:
            record["stages_reached"] = [phase]
        return record

    def _settle_geak_runner_exit(
        self, proc: subprocess.CompletedProcess, *, handoff: dict[str, Any], agentx: bool, facts: GeakRunnerFacts
    ) -> dict[str, Any] | None:
        result_path = self._geak_result_path()
        stderr_tail = (proc.stderr or "")[-2000:]
        result = read_json(result_path, {}, require_dict=True)
        if proc.returncode != 0:
            log.warning(
                "GEAK runner rc=%s: error_class=%s error=%s stderr_tail=%s",
                proc.returncode,
                result.get("error_class"),
                result.get("error"),
                stderr_tail,
            )
        if not result:
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": "no_result_json",
                    "error": f"runner rc={proc.returncode} produced no parseable result.json at {result_path}",
                    "stderr_tail": stderr_tail,
                    "returncode": proc.returncode,
                },
                handoff=handoff,
                facts=facts,
            )
            return None
        if agentx and self._geak_settled_replay(result, agentx=agentx):
            # The runner left the candidate this session already settled, so it shipped no product: recording it
            # would retire the verdict and re-enqueue the revalidation that produced it. The file stays for the
            # next run to overwrite.
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": "no_new_geak_product",
                    "error": f"runner rc={proc.returncode} left the already-adjudicated result.json at {result_path}",
                    "stderr_tail": stderr_tail,
                },
                handoff=handoff,
            )
            return None
        # Carry the actual exit code so the breakdown can audit a nonzero rc. Nothing else is added: the persisted
        # record stays the product GEAK wrote, so the next KERNEL entry recognizes its own settled candidate.
        result.setdefault("returncode", proc.returncode)
        return result

    async def _dispatch_geak_runner(
        self, handoff: dict[str, Any], *, from_phase: str, agentx: bool
    ) -> GeakRunnerOutcome:
        out_dir = self.session_dir / "geak"
        # Budget-aware timeouts: shrink to the remaining run deadline and always reserve the closing-grace window.
        runner_timeout, kill_timeout, budget_known = self._geak_timeouts()
        min_run = env_int("GEAK_MIN_RUN_S", default=600)
        if budget_known and runner_timeout < min_run:
            self._refuse_geak_for_budget(handoff, runner_timeout=runner_timeout, min_run=min_run)
            return GeakRunnerOutcome()
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()

        def _facts() -> GeakRunnerFacts:
            return GeakRunnerFacts(
                started_at=started_at,
                duration_sec=time.monotonic() - started,
                runner_timeout_sec=runner_timeout,
                kill_timeout_sec=kill_timeout,
            )

        try:
            proc = await self._launch_geak_runner(
                out_dir / "handoff.json",
                timeouts=(runner_timeout, kill_timeout, budget_known),
                from_phase=from_phase,
                client_env=_geak_client_env(handoff),
            )
        except subprocess.TimeoutExpired:
            log.warning(
                "GEAK runner exceeded kill_timeout=%ds; SIGTERM'd to let it flush, then reclaimed the closing window",
                kill_timeout,
            )
            # The graceful SIGTERM gives run_e2e a window to flush result.json; keep a real win instead of discarding
            # the phase as a timeout.
            facts = _facts()
            flushed = read_json(self._geak_result_path(), {}, require_dict=True)
            settled = bool(flushed) and agentx and self._geak_settled_replay(flushed, agentx=agentx)
            if flushed.get("status") == "ok" and not settled:
                log.info(
                    "GEAK flushed an OK result.json under SIGTERM grace; promoting the recovered win despite the cap."
                )
                return GeakRunnerOutcome(flushed, facts, "sigterm_flushed_result_json")
            if not settled:
                self._finish_geak_skip(
                    self._outer_timeout_record(flushed, runner_timeout=runner_timeout, kill_timeout=kill_timeout),
                    handoff=handoff,
                    facts=facts,
                )
                return GeakRunnerOutcome()
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": _TIMED_OUT,
                    "error": f"GEAK e2e killed after {kill_timeout}s (budget-capped); closing window preserved",
                    "runner_timeout_s": runner_timeout,
                    "kill_timeout_s": kill_timeout,
                },
                handoff=handoff,
                facts=facts,
            )
            return GeakRunnerOutcome()
        except (OSError, ValueError) as exc:
            log.exception("GEAK runner crashed")
            self._finish_geak_skip(
                {"status": "error", "error_class": "runner_crashed", "error": repr(exc)},
                handoff=handoff,
                facts=_facts(),
            )
            return GeakRunnerOutcome()
        facts = _facts()
        return GeakRunnerOutcome(
            self._settle_geak_runner_exit(proc, handoff=handoff, agentx=agentx, facts=facts), facts
        )

    def _open_geak_handoff(self, *, agentx: bool) -> dict[str, Any] | None:
        try:
            handoff = self._build_geak_handoff(agentx=agentx)
        except EnvSpecUnserializable as exc:
            log.exception("geak: cannot serialize the accepted launch configuration")
            recorder = self.timeline()
            if recorder is not None:
                recorder.finish_failed(stage="geak_handoff", error_class="invalid_env_spec", message=str(exc))
            self._settle_geak_skip({"status": "error", "error_class": "invalid_env_spec", "error": str(exc)})
            return None
        out_dir = self.session_dir / "geak"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")
        recorder = self.timeline()
        if recorder is not None:
            recorder.enter_stage("geak_delegation")
            recorder.record_geak_handoff(handoff)
        return handoff

    async def _promote_crashed_geak_result(
        self, handoff: dict[str, Any], *, agentx: bool, cards: list[int] | None
    ) -> bool:
        # A validated result.json written before a coordinator crash is promoted on resume, guarded by
        # ``_geak_win_already_recorded`` so a prior cycle's result.json does not short-circuit a fresh entry. A
        # result this session already settled is not recovered; new candidate evidence remains eligible.
        recovered = read_json(self._geak_result_path(), {}, require_dict=True)
        if (
            recovered.get("status") != "ok"
            or self._geak_win_already_recorded()
            or self._geak_settled_replay(recovered, agentx=agentx)
        ):
            return False
        log.info(
            "GEAK result.json exists but state has no recorded win (crash before handback); promoting recovered result."
        )
        if self._promote_recovered_geak_result(recovered, handoff=handoff, recovered_from="existing_result_json"):
            await self._revalidate_geak_candidate(reason="geak_e2e_win_recovered", handoff_cards=cards)
        return True

    def _refuse_phantom_baseline_gain(self, result: dict[str, Any]) -> None:
        # A GEAK run whose baseline ref failed to reproduce ``orchestrator_best_tput_same_config`` optimized
        # against a phantom baseline, so its gain is non-comparable -- never promote it.
        log.warning(
            "GEAK baseline_reproduction_failed: ref did not match "
            "orchestrator best (%s); refusing to promote a phantom-baseline gain",
            result.get("error"),
        )
        self._settle_geak_skip(
            {
                "status": "baseline_reproduction_failed",
                "error_class": "baseline_reproduction_failed",
                "error": (
                    str(result.get("error") or "")[:500] or "GEAK baseline ref != orchestrator best (env_spec mismatch)"
                ),
                "ref_tput": result.get("ref_tput"),
                "orchestrator_best_tput_same_config": result.get("orchestrator_best_tput_same_config"),
                **({"absent_backends": result["absent_backends"]} if result.get("absent_backends") else {}),
            }
        )

    async def _publish_geak_e2e_done(self, result: dict[str, Any], *, facts: GeakRunnerFacts) -> None:
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "runner_timeout_s": facts.runner_timeout_sec,
            }
        )
        self.shared_state.save(self.session_dir)
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "geak_e2e_done",
                    "status": str(result.get("status") or "unknown"),
                    "speedup": result.get("throughput_speedup"),
                    "result_path": str(self._geak_result_path()),
                },
            )
        )
        self.shared_state.save(self.session_dir)

    async def _run_geak_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to GEAK (one whole-pipeline e2e run)."""
        state = self.shared_state
        agentx = not _geak_rebench.geak_harness_replays_workload(state)
        handoff = self._open_geak_handoff(agentx=agentx)
        if handoff is None:
            return
        cards = _resolve_handoff_physical_gpus(
            gpu_ids=handoff["gpu_ids"], gpu_ids_space=handoff["gpu_ids_space"], gpu_pin=handoff.get("gpu_pin") or {}
        )
        if await self._promote_crashed_geak_result(handoff, agentx=agentx, cards=cards):
            return

        still_busy = await self._free_gpus_for_geak(cards)
        if still_busy:
            self._finish_geak_skip(
                {"status": "error", "error_class": "gpus_busy_at_handoff", "error": still_busy}, handoff=handoff
            )
            return

        outcome = await self._dispatch_geak_runner(handoff, from_phase=from_phase, agentx=agentx)
        result, facts = outcome.result, outcome.facts
        if result is None:
            return
        if outcome.recovered_from:
            if self._promote_recovered_geak_result(
                result, handoff=handoff, recovered_from=outcome.recovered_from, facts=facts
            ):
                await self._revalidate_geak_candidate(reason="geak_e2e_win_sigterm_recovered", handoff_cards=cards)
            return

        self._record_geak_delegation_timeline(result, handoff=handoff, facts=facts)
        state.geak_result = result
        self._record_geak_measurement(result)
        status = str(result.get("status") or "")
        if status == "baseline_reproduction_failed":
            self._refuse_phantom_baseline_gain(result)
            return

        # Rebench-first: record the win as an UNVALIDATED candidate only; the headline is written later from the
        # measured rebench. The same-harness rebench is the only path that writes it, and a no_gain headline over an
        # accepted, parity-checked kernel still earns the measurement -- without it the kernel is lost with no number.
        if not self._record_geak_candidate(result):
            state.save(self.session_dir)
            return
        self._record_geak_kernel_journey(result)
        if status == "ok":
            await self._revalidate_geak_candidate(reason="geak_e2e_win", handoff_cards=cards)
        elif _geak_has_accepted_kernel(result):
            await self._revalidate_geak_candidate(reason="geak_e2e_accepted_kernel", handoff_cards=cards)
        await self._publish_geak_e2e_done(result, facts=facts)
