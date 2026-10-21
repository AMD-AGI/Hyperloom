# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The payload the KERNEL phase hands a GEAK e2e run.

GEAK searches against a reference it must be able to reproduce, so everything
here answers one question: what did this session measure, under which launch, on
which devices, and how far can that be shown rather than merely asserted.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging as _logging
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, NamedTuple

from hyperloom.common.launch_log_evidence import recipe_server_launch

from ..actions.executors._gpu_pin import (
    _resolve_handoff_gpu_identity,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
    _resolve_handoff_tp,
    _resolve_serving_fidelity,
    coerce_tp,
    resolve_gpu_pin,
)
from ..actions.executors._recipe_script import resolve_launch_server_script

log = _logging.getLogger(__name__)

#: The reference launch was observed, not merely declared to have been asked for.
_VERIFIED_OBSERVED = "verified_observed"


class SameConfigReference(NamedTuple):
    """The accepted launch GEAK is asked to reproduce, and how far it was verified."""

    env_spec: dict[str, Any]
    config: Mapping[str, Any]
    measurement: Mapping[str, Any]
    accepted_flags: str
    accepted_env: str
    expected_identity: str
    declared_identity: str
    observed_identity: str
    observed_server_identity: dict[str, Any]
    launch_evidence: dict[str, Any]
    verification_status: str

    @property
    def verified(self) -> bool:
        """Whether the reference launch was observed rather than only declared."""
        return self.verification_status == _VERIFIED_OBSERVED

    @property
    def tput(self) -> float:
        """The throughput the verified reference measured; zero when there is no verified one."""
        return float(self.measurement.get("tput") or 0.0) if self.verified else 0.0


class DevicePin(NamedTuple):
    """The device set the handoff puts GEAK's servers on, in the space its ids are numbered in."""

    tp: int
    gpu_ids: str
    gpu_ids_space: str
    pin: dict[str, Any]


def geak_dir(session_dir: Path) -> Path:
    """The directory one GEAK delegation reads and writes everything under."""
    return session_dir / "geak"


def same_config_reference(state: Any, env_spec: dict[str, Any], *, writeback: Any, agentx: bool) -> SameConfigReference:
    """Read back the accepted launch this session measured its current best under."""
    best = state.current_best
    config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
    accepted_flags = str(config.get("extra_server_args", best.get("extra_server_args")) or "")
    extra_envs = config.get("extra_envs", best.get("extra_envs")) or {}
    measurement = _best_measurement(state, best)
    if agentx and not str(config.get("server_launch_flags") or "").strip():
        # Rebuild the launch spec from the recipe's own server log, where it has one.
        evidence = measurement.get("launch_evidence")
        server_log = str(evidence.get("actual_server_log_path") or "") if isinstance(evidence, Mapping) else ""
        recipe_launch = recipe_server_launch(server_log)
        if recipe_launch.flags or recipe_launch.env:
            env_spec = writeback.build_env_spec(
                measurement=measurement,
                server_launch_flags=recipe_launch.flags,
                server_launch_envs=recipe_launch.env,
            )
            config = env_spec["config"]
    identity = _launch_identity(env_spec, measurement)
    if agentx:
        # GEAK's proxy workload is not the canonical AgentX one, so no AgentX
        # measurement can stand as a same-config reference for it.
        identity["verification_status"] = "unverified_workload"
    return SameConfigReference(
        env_spec=env_spec,
        config=config,
        measurement=measurement,
        accepted_flags=accepted_flags,
        accepted_env=shlex.join(f"{k}={v}" for k, v in dict(extra_envs).items()),
        **identity,
    )


def _best_measurement(state: Any, best: Mapping[str, Any]) -> Mapping[str, Any]:
    """The measurement behind the current best, preferring the one state holds in its own right."""
    recorded = state.current_best_measurement or None
    if isinstance(recorded, Mapping) and recorded:
        return recorded
    inline = best.get("measurement")
    return inline if isinstance(inline, Mapping) else {}


def _launch_identity(env_spec: Mapping[str, Any], measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Pair what the launch declared about itself with what was observed of it."""
    expected = str(env_spec.get("launch_identity") or "")
    declared = str(measurement.get("declared_launch_identity") or measurement.get("launch_identity") or "")
    matches = bool(expected and expected == declared)
    evidence = measurement.get("launch_evidence")
    evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
    flags = str(
        measurement.get("resolved_server_launch_flags") or evidence.get("observed_server_launch_flags") or ""
    ).strip()
    server_identity = measurement.get("observed_server_identity") or evidence.get("observed_server_identity")
    server_identity = (
        {str(key): value for key, value in sorted(server_identity.items())}
        if isinstance(server_identity, Mapping)
        else {}
    )
    observed = str(measurement.get("observed_launch_identity") or "")
    if not observed and matches and (flags or server_identity):
        observed = _observed_identity_digest(declared, flags=flags, server_identity=server_identity)
    return {
        "expected_identity": expected,
        "declared_identity": declared,
        "observed_identity": observed,
        "observed_server_identity": server_identity,
        "launch_evidence": evidence,
        "verification_status": reference_verification_status(
            evidence, identity_matches=matches, observed_flags=flags, observed_server_identity=server_identity
        ),
    }


def _observed_identity_digest(declared: str, *, flags: str, server_identity: Mapping[str, Any]) -> str:
    """Mint an observed-launch identity for a measurement that carries the evidence but not the digest."""
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


def reference_verification_status(
    launch_evidence: Mapping[str, Any],
    *,
    identity_matches: bool,
    observed_flags: Any,
    observed_server_identity: Any,
) -> str:
    """Say how far the reference launch was verified against what was asked for."""
    if identity_matches and (observed_flags or observed_server_identity):
        return _VERIFIED_OBSERVED
    if identity_matches and (
        str(launch_evidence.get("requested_server_args") or "").strip()
        or bool(launch_evidence.get("requested_server_env"))
        or str(launch_evidence.get("recipe_digest") or "").strip()
    ):
        return "verified_declared_only"
    return "unverified"


def device_pin(recipe_envs: Mapping[str, Any]) -> DevicePin:
    """Resolve which cards the handoff puts GEAK's servers on, and in which coordinate space."""
    pin = resolve_gpu_pin(recipe_envs=recipe_envs)
    tp = coerce_tp(recipe_envs.get("TP"), os.environ.get("TP"))
    gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=pin, tp=tp)
    space = _resolve_handoff_gpu_ids_space(gpu_pin=pin)
    if space == "none":
        # An empty visibility mask means the ids are placeholders, not devices.
        log.error(
            "geak handoff: %s is set but empty; the run has no visible GPUs. "
            "gpu_ids=%s is a placeholder (gpu_ids_space=none), not a device set.",
            pin.get("var"),
            gpu_ids,
        )
    return DevicePin(tp=_resolve_handoff_tp(gpu_ids=gpu_ids, tp=tp), gpu_ids=gpu_ids, gpu_ids_space=space, pin=pin)


def build_handoff(
    state: Any, reference: SameConfigReference, *, session_dir: Path, agentx: bool, e2e_metric: str
) -> dict[str, Any]:
    """The handoff one GEAK e2e run reads its workload, reference and devices from.

    ``e2e_metric`` is the axis the phase also exports as ``E2E_METRIC`` to the
    runner: a handoff that named a different axis than the one the runner
    searches on would have GEAK measuring against a reference it was never
    graded against, so both come from the phase's single resolution of it.
    """
    recipe_path = str(state.baseline_config_path or "")
    recipe_envs = read_recipe_bench_envs(recipe_path)
    devices = device_pin(recipe_envs)
    handoff = _handoff_payload(state, reference, devices, session_dir=session_dir, agentx=agentx)
    handoff["e2e_metric"] = e2e_metric
    if devices.pin:
        # ABSOLUTE ids + the var they came from, so a consumer that writes
        # ROCR_VISIBLE_DEVICES itself re-applies the same pin instead of
        # resetting the child to card 0.
        handoff["gpu_pin"] = devices.pin
    protocol = bench_protocol(recipe_path, envs=recipe_envs)
    if protocol:
        handoff["bench_protocol"] = protocol
    # Parsed once for both resolvers below: they read the same recipe, so a
    # second read could only disagree with the first.
    bench = recipe_benchmark(recipe_path)
    # Only AgentX materializations carry ``benchmark.workload_spec``; every other
    # recipe omits it so GEAK keeps today's synthetic path unchanged.
    spec = bench.get("workload_spec")
    if isinstance(spec, dict) and spec:
        observed = observed_replay_shape(session_dir, float(state.baseline_tput or 0.0))
        if observed:
            # Give the kernel agents the shape they must actually optimize for;
            # absence leaves GEAK on the handoff's synthetic isl/osl.
            log.info(
                "workload_spec: observed replay shape isl=%d osl=%d over %d requests (%s)",
                observed["observed_isl"],
                observed["observed_osl"],
                observed["observed_requests"],
                observed["observed_source"],
            )
        handoff["workload_spec"] = {**spec, **observed}
    # Absent => GEAK keeps deriving the launcher from the recipe, which is
    # correct everywhere except AgentX (see launch_server_script).
    script = launch_server_script(bench)
    if script:
        handoff["launch_server_script"] = script
    handoff.update(
        _resolve_serving_fidelity(
            baseline_server_args=_current_best_server_args(state, reference),
            state_max_model_len=int(state.max_model_len or 0),
        )
    )
    # Full layered environment and its matching measurement identity.
    if reference.env_spec:
        handoff["baseline_env_spec"] = reference.env_spec
    if agentx:
        _fit_to_agentx(handoff, has_launch_script=bool(script))
    from hyperloom.inference_optimizer.agentx.geak_proxy import seed_native_geak_proxy

    seed_native_geak_proxy(handoff, benchmark=bench, measurement=reference.measurement or {})
    return handoff


def _handoff_payload(
    state: Any,
    reference: SameConfigReference,
    devices: DevicePin,
    *,
    session_dir: Path,
    agentx: bool,
) -> dict[str, Any]:
    """The fields every handoff carries, whatever the recipe adds to them."""
    measurement = reference.measurement
    return {
        # v2 adds baseline_env_spec; v3 adds actual GPU-pinning metadata.
        "schema_version": 3,
        "model_path": str(state.model_path or os.environ.get("MODEL_PATH", "")),
        "framework": str(state.framework or os.environ.get("FRAMEWORK", "") or "sglang"),
        **_resolve_handoff_gpu_identity(state.gpu_type or os.environ.get("GPU_TYPE", "")),
        "tp": devices.tp,
        "workload": {"isl": int(state.isl), "osl": int(state.osl), "conc": int(state.conc or 64)},
        "accepted_flags": reference.accepted_flags,
        "accepted_env": reference.accepted_env,
        "launch_recipe": str(state.baseline_config_path or ""),
        # AgentX canonical throughput is not a reference for GEAK's proxy workload.
        "raw_baseline_tput": 0.0 if agentx else float(state.baseline_tput or 0.0),
        # Zero means no verified same-config reference.
        "orchestrator_best_tput_same_config": reference.tput,
        "same_config_reference_status": "verified" if reference.verified else "unverified",
        "same_config_reference_identity": reference.declared_identity,
        "same_config_expected_identity": reference.expected_identity,
        "same_config_reference_workspace": str(measurement.get("benchmark_workspace") or ""),
        # Additive identity semantics.
        "same_config_reference_verification_status": reference.verification_status,
        "same_config_reference_declared_identity": reference.declared_identity,
        "same_config_reference_observed_identity": reference.observed_identity,
        # GEAK compares this map with its parsed ServerArgs.
        "same_config_observed_identity": reference.observed_server_identity,
        "observed_server_identity": reference.observed_server_identity,
        "measurement_evidence": reference.launch_evidence,
        "resolved_server_config": dict(measurement.get("resolved_server_config") or {}),
        "exp_root": str(geak_dir(session_dir)),
        # Macro-cycle-scoped eval_dir so a same-cycle resume reuses the in-progress on-disk artifacts while a new
        # cycle gets a fresh dir.
        "eval_dir": str(geak_dir(session_dir) / f"e2e_cycle{int(state.macro_cycle or 0)}"),
        # GEAK owns client selection; AgentX results are proposal proxies.
        "bench_client": "auto",
        "inferencex_path": str(os.environ.get("INFERENCEX_PATH", "")),
        # The serving/optimization device set, as HIP-level ids (what the
        # consumer exports as HIP_VISIBLE_DEVICES). Logical positions inside
        # an inherited ROCR mask, a HIP/CUDA mask as-is, else 0..tp-1.
        "gpu_ids": devices.gpu_ids,
        # Which coordinate system ``gpu_ids`` is in: "logical" (positions
        # inside the ROCR mask the child inherits), "absolute" (whole-
        # machine device ids), or "none" (the mask is set but empty -- the
        # ids are placeholders and must not be launched on). Exporting them
        # as HIP_VISIBLE_DEVICES is correct in the first two; a consumer
        # that instead writes ROCR itself needs to know which it holds.
        "gpu_ids_space": devices.gpu_ids_space,
    }


def _current_best_server_args(state: Any, reference: SameConfigReference) -> str:
    """The full server command line the current best was launched with."""
    args = str(reference.config.get("server_launch_flags") or "")
    if args:
        return args
    from hyperloom.inference_optimizer.grid_server_args import compose_server_args

    return compose_server_args(
        inherited_args=_baseline_server_args(state),
        variant_extra_args=reference.accepted_flags,
        remove_args=reference.config.get("remove_args"),
        args_mode=str(reference.config.get("args_mode") or "append"),
    )


def _baseline_server_args(state: Any) -> str:
    """The server settings the baseline ran with, so GEAK measures the same engine config."""
    try:
        from hyperloom.inference_optimizer.roofline_ceiling import read_baseline_server_args

        return read_baseline_server_args(state) or ""
    except Exception:  # noqa: BLE001 -- accessor is best-effort
        return ""


def _fit_to_agentx(handoff: dict[str, Any], *, has_launch_script: bool) -> None:
    """Fit the handoff to an AgentX session, whose canonical validation stays in Hyperloom."""
    log.info("GEAK results remain proposal proxies; canonical AgentX validation remains in Hyperloom.")
    # Force native only when the recipe builtin could not be named. The
    # AgentX client delegates its server phase to that builtin, so with it
    # in hand (set as ``launch_server_script``) GEAK derives the Magpie
    # launcher -- which writes a pid and recovers the non-flag presets --
    # instead of its adapter default; without it there is no Magpie server
    # phase to launch. Flags/env stay aligned either way: baseline_env_spec
    # still folds the recorded EXTRA_ATOM_ARGS.
    if not has_launch_script:
        handoff["bench_launcher"] = "native"


def write_handoff(handoff: Mapping[str, Any], *, session_dir: Path) -> Path:
    """Write the handoff file GEAK's runner reads, and return where it landed."""
    out_dir = geak_dir(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "handoff.json"
    path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    return path


def read_recipe_bench_envs(recipe_path: str) -> dict[str, Any]:
    """Read recipe ``benchmark.envs``; return ``{}`` if unavailable."""
    try:
        import yaml

        if recipe_path and Path(recipe_path).is_file():
            cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
            envs = ((cfg.get("benchmark") or {}).get("envs")) or {}
            return dict(envs) if isinstance(envs, dict) else {}
    except Exception:
        log.warning("geak handoff: could not read recipe %r", recipe_path, exc_info=True)
    return {}


def bench_protocol(recipe_path: str, *, envs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract Hyperloom's bench measurement protocol for the GEAK handoff.

    Reads the materialized baseline recipe's ``benchmark.envs`` (falling back
    to the process env) and returns only the keys that resolve, so absent
    values leave GEAK on its standalone defaults. Never raises.

    Args:
        recipe_path: Path to the baseline recipe YAML.
        envs: An already-parsed ``benchmark.envs`` for that path. Pass it
            when the caller needs the envs too, so the YAML is read once and
            both consumers see the same snapshot.
    """
    envs = read_recipe_bench_envs(recipe_path) if envs is None else envs

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


def recipe_benchmark(recipe_path: str) -> dict[str, Any]:
    """Parse a materialized recipe once and hand back its ``benchmark`` mapping."""
    try:
        import yaml

        if not recipe_path or not Path(recipe_path).is_file():
            return {}
        cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            return {}
        bench = cfg.get("benchmark") or {}
        return bench if isinstance(bench, dict) else {}
    except Exception:
        log.warning("recipe: could not read %r", recipe_path, exc_info=True)
        return {}


def launch_server_script(bench: Mapping[str, Any]) -> str:
    """Name the server-phase script GEAK should launch through Magpie."""
    try:
        from hyperloom.common.agentx_workload import is_agentx_client_script

        # Only the AgentX client misleads the inference; anything else in
        # this field is the launcher GEAK should keep deriving for itself.
        if not is_agentx_client_script(str(bench.get("benchmark_script") or "").strip()):
            return ""
        return resolve_launch_server_script(bench)
    except Exception:
        log.warning("launch_server_script: could not resolve from the recipe", exc_info=True)
        return ""


def observed_replay_shape(session_dir: Path, target_tput: float) -> dict[str, Any]:
    """Measure the trace replay's real sequence shape from OUR OWN baseline.

    The AgentX handoff's ``workload.isl/osl`` are the CLI's synthetic
    defaults (1024/1024) because a trace replay has no single ISL -- the
    corpus spans roughly 89k at p50 past 500k at p99. GEAK does not measure
    with them (the aiperf client ignores them outright), but the KERNEL
    agents still read them as the analytic serving call model when they
    synthesize GEMM/attention shapes. Left at 1024 they aim two orders of
    magnitude below the load, so the search tunes a regime nobody serves.

    Rather than hardcode corpus percentiles, derive the average shape from
    the baseline result THIS run already measured: the canonical result
    records ``total_input_tokens``/``total_output_tokens`` over ``completed``
    requests. That keeps the number sourced from a measurement instead of a
    constant that silently rots when the corpus is re-pinned.

    Returns ``{}`` whenever no usable result is on disk, so an unmeasurable
    run degrades to today's behavior rather than to a fabricated shape.
    """
    from hyperloom.inference_optimizer.session.session_paths import runs_root

    try:
        root = runs_root(session_dir)
        if not Path(root).is_dir():
            return {}
        shapes: list[tuple[float, int, int, int, str]] = []
        for path in glob.glob(str(Path(root) / "**" / "inferencex_result.json"), recursive=True):
            # GEAK's own results are not the orchestrator's baseline, and the
            # overlay probe is not a served measurement. Match path COMPONENTS
            # below the runs root, never the whole string: a substring test
            # excludes every result the moment an ancestor directory happens
            # to contain "geak" (a campaign rooted at .../hlgeak_24h_run/ hits
            # this), which silently degrades the shape to the 1024 default
            # this method exists to replace.
            try:
                parts = set(Path(path).relative_to(root).parts)
                raw = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if not isinstance(raw, dict) or parts & {"geak", "_baseline_source_overlay"}:
                continue
            done, tokens_in, tokens_out = (
                int(raw.get(key) or 0) for key in ("completed", "total_input_tokens", "total_output_tokens")
            )
            tput = float(raw.get("output_throughput") or 0.0)
            if min(done, tokens_in, tokens_out, tput) <= 0:
                continue
            # Prefer the result whose throughput IS the baseline we handed
            # over, so the shape describes the same measurement.
            error = abs(tput - target_tput) / target_tput if target_tput > 0 else 1.0
            shapes.append((error, round(tokens_in / done), round(tokens_out / done), done, path))
    except Exception:
        log.warning("workload_spec: could not derive observed replay shape", exc_info=True)
        return {}
    if not shapes:
        return {}
    error, isl, osl, done, source = min(shapes, key=lambda shape: shape[0])
    if isl <= 0 or osl <= 0:
        return {}
    return {
        "observed_isl": isl,
        "observed_osl": osl,
        "observed_requests": done,
        "observed_source": source,
        # 1.0 => no baseline throughput to match against, so this is the
        # best available replay result rather than a confirmed same-run one.
        "observed_tput_match_err": round(error, 6),
    }
