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
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable, NamedTuple

from hyperloom.common.launch_log_evidence import recipe_server_launch

from ..actions.executors._gpu_pin import (
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

#: Measurement protocol keys the handoff forwards, and the env var each is read from.
_BENCH_PROTOCOL_KEYS: tuple[tuple[str, str, Callable[[str], Any]], ...] = (
    ("random_range_ratio", "RANDOM_RANGE_RATIO", float),
    ("num_prompts", "NUM_PROMPTS", int),
    ("num_warmups", "NUM_WARMUPS", int),
    ("seed", "SEED", int),
)


class LaunchSpec(NamedTuple):
    """The env spec a GEAK run launches under, and the config it resolves to."""

    env_spec: dict[str, Any]
    config: Mapping[str, Any]


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
    env_spec, config = _recipe_launch_spec(
        env_spec, config, writeback=writeback, agentx=agentx, measurement=measurement
    )
    identity = _launch_identity(env_spec, measurement)
    return SameConfigReference(
        env_spec=env_spec,
        config=config,
        measurement=measurement,
        accepted_flags=accepted_flags,
        accepted_env=shlex.join(f"{k}={v}" for k, v in dict(extra_envs).items()),
        expected_identity=identity.expected_identity,
        declared_identity=identity.declared_identity,
        observed_identity=identity.observed_identity,
        observed_server_identity=identity.observed_server_identity,
        launch_evidence=identity.launch_evidence,
        # GEAK's proxy workload is not the canonical AgentX one, so no AgentX
        # measurement can stand as a same-config reference for it.
        verification_status="unverified_workload" if agentx else identity.verification_status,
    )


def _best_measurement(state: Any, best: Mapping[str, Any]) -> Mapping[str, Any]:
    """The measurement behind the current best, preferring the one state holds in its own right."""
    recorded = state.current_best_measurement or None
    if isinstance(recorded, Mapping) and recorded:
        return recorded
    inline = best.get("measurement")
    return inline if isinstance(inline, Mapping) else {}


def _recipe_launch_spec(
    env_spec: dict[str, Any], config: Mapping[str, Any], *, writeback: Any, agentx: bool, measurement: Mapping[str, Any]
) -> LaunchSpec:
    """Rebuild the launch spec from the recipe's own server log, where it has one."""
    if not agentx or str(config.get("server_launch_flags") or "").strip():
        return LaunchSpec(env_spec, config)
    evidence = measurement.get("launch_evidence")
    server_log = str(evidence.get("actual_server_log_path") or "") if isinstance(evidence, Mapping) else ""
    recipe_launch = recipe_server_launch(server_log)
    if recipe_launch.flags or recipe_launch.env:
        env_spec = writeback.build_env_spec(
            measurement=measurement,
            server_launch_flags=recipe_launch.flags,
            server_launch_envs=recipe_launch.env,
        )
        return LaunchSpec(env_spec, env_spec["config"])
    return LaunchSpec(env_spec, config)


class _LaunchIdentity(NamedTuple):
    """How the measured launch identifies itself against the one that was asked for."""

    expected_identity: str
    declared_identity: str
    observed_identity: str
    observed_server_identity: dict[str, Any]
    launch_evidence: dict[str, Any]
    verification_status: str


def _launch_identity(env_spec: Mapping[str, Any], measurement: Mapping[str, Any]) -> _LaunchIdentity:
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
    return _LaunchIdentity(
        expected_identity=expected,
        declared_identity=declared,
        observed_identity=observed,
        observed_server_identity=server_identity,
        launch_evidence=evidence,
        verification_status=reference_verification_status(
            evidence, identity_matches=matches, observed_flags=flags, observed_server_identity=server_identity
        ),
    )


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
    spec = advertised_workload_spec(bench, session_dir=session_dir, baseline_tput=float(state.baseline_tput or 0.0))
    if spec:
        handoff["workload_spec"] = spec
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
        "gpu_type": str(state.gpu_type or os.environ.get("GPU_TYPE", "")),
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
    """Extract Hyperloom's bench measurement protocol for the GEAK handoff."""
    envs = read_recipe_bench_envs(recipe_path) if envs is None else envs
    protocol: dict[str, Any] = {}
    for proto_key, env_key, cast in _BENCH_PROTOCOL_KEYS:
        value = _cast_env(envs, env_key, cast)
        if value is not None:
            protocol[proto_key] = value
    return protocol


def _cast_env(envs: Mapping[str, Any], key: str, cast: Callable[[str], Any]) -> Any:
    """Read ``key`` from the recipe's envs, then the process env, cast; ``None`` when neither resolves."""
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


def advertised_workload_spec(bench: Mapping[str, Any], *, session_dir: Path, baseline_tput: float) -> dict[str, Any]:
    """The self-describing workload the recipe carries, refined by the shape the replay actually had."""
    spec = bench.get("workload_spec")
    if not isinstance(spec, dict) or not spec:
        return {}
    observed = observed_replay_shape(session_dir, baseline_tput)
    if not observed:
        return dict(spec)
    # Give the kernel agents the shape they must actually optimize for;
    # absence leaves GEAK on the handoff's synthetic isl/osl.
    log.info(
        "workload_spec: observed replay shape isl=%d osl=%d over %d requests (%s)",
        observed["observed_isl"],
        observed["observed_osl"],
        observed["observed_requests"],
        observed["observed_source"],
    )
    return {**spec, **observed}


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


class _ReplayResult(NamedTuple):
    """One served replay result, as the observed-shape search reads it."""

    path: str
    completed: int
    input_tokens: int
    output_tokens: int
    tput: float


def observed_replay_shape(session_dir: Path, target_tput: float) -> dict[str, Any]:
    """Measure the trace replay's real sequence shape from OUR OWN baseline."""
    try:
        best = _closest_replay_result(session_dir, target_tput)
    except Exception:
        log.warning("workload_spec: could not derive observed replay shape", exc_info=True)
        return {}
    if best is None:
        return {}
    result, error = best
    isl = int(round(result.input_tokens / result.completed))
    osl = int(round(result.output_tokens / result.completed))
    if isl <= 0 or osl <= 0:
        return {}
    return {
        "observed_isl": isl,
        "observed_osl": osl,
        "observed_requests": result.completed,
        "observed_source": result.path,
        # 1.0 => no baseline throughput to match against, so this is the
        # best available replay result rather than a confirmed same-run one.
        "observed_tput_match_err": round(error, 6),
    }


def _closest_replay_result(session_dir: Path, target_tput: float) -> tuple[_ReplayResult, float] | None:
    """The served replay result whose throughput is nearest ``target_tput``, with that relative error."""
    best: tuple[_ReplayResult, float] | None = None
    for result in _served_replay_results(session_dir):
        # Prefer the result whose throughput IS the baseline we handed
        # over, so the shape describes the same measurement.
        error = abs(result.tput - target_tput) / target_tput if target_tput > 0 else 1.0
        if best is None or error < best[1]:
            best = (result, error)
    return best


def _served_replay_results(session_dir: Path) -> Iterator[_ReplayResult]:
    """Every orchestrator-served result under the session's runs root that carries a usable shape."""
    from hyperloom.inference_optimizer.session.session_paths import runs_root

    root = runs_root(session_dir)
    if not Path(root).is_dir():
        return
    for path in glob.glob(str(Path(root) / "**" / "inferencex_result.json"), recursive=True):
        if not _is_served_baseline(root, path):
            continue
        raw = _read_result(path)
        completed = int(raw.get("completed") or 0)
        tokens_in = int(raw.get("total_input_tokens") or 0)
        tokens_out = int(raw.get("total_output_tokens") or 0)
        tput = float(raw.get("output_throughput") or 0.0)
        if completed > 0 and tokens_in > 0 and tokens_out > 0 and tput > 0:
            yield _ReplayResult(path, completed, tokens_in, tokens_out, tput)


def _is_served_baseline(root: Any, path: str) -> bool:
    """Whether a result under ``root`` is one the orchestrator itself served."""
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        return False
    return "geak" not in parts and "_baseline_source_overlay" not in parts


def _read_result(path: str) -> Mapping[str, Any]:
    """A result file's mapping, or ``{}`` when it cannot be read as one."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}
