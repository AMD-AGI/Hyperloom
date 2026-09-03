# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared workload-env materialization (single source of truth).

This module is the single source of truth for rendering a Magpie YAML with the
user's actual process-env workload contract, so the baseline and grid-runner
paths render identical YAML:

* :func:`materialize_config_with_envs` — write a per-run YAML honoring process
  env (+ optional caller overrides).
* :func:`default_baseline_config` — pick the shipped YAML by ``$FRAMEWORK``.

Used by ``baseline.py`` (materializes once, surfaces the path) and the
``explore`` / ``sweep`` grid runs (fall back to materializing on their own).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env import env_int
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    BLOCKED_EXTERNAL_ENV_NAMES,
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
)
from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.orchestrator.framework.paths import ENV_FLYDSL_EXTRA_SOURCE_DIRS
from hyperloom.orchestrator.framework.paths import GENERIC_FRAMEWORK_ROOT_ENV
from hyperloom.orchestrator.framework.paths import flydsl_extra_source_dirs
from ._accuracy_gate import _RUN_EVAL_FALSE_VALUES
from ._grid_runner import (
    compact_json_server_args,
    dedup_vllm_server_args,
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    inject_sglang_moe_runner_backend,
    inject_sglang_watchdog_timeout,
    server_args_env_name,
)
from ._grid_server_args import remove_server_args
from ._grid_server_args import validate_server_args_shell_safe
from ._server_patcher import (
    ensure_sglang_patched_for_ck_blockscale,
    ensure_sglang_patched_for_tracelens,
    ensure_vllm_patched_for_tracelens,
)
from hyperloom.inference_optimizer.model_config_utils import (
    _fp8_is_per_channel_per_token,
    _load_model_config_dict,
    _model_is_gemma2,
    _sparse_kv_block_size,
)

log = logging.getLogger(__name__)

# gfx942 / CDNA3 dies (MI300X, MI308X, MI325X) that ship the aiter CK
# gemm_a8w8_bpreshuffle kernel. MI355X is gfx950 and excluded.
_GFX942_GPU_TYPES = frozenset({"mi300x", "mi308x", "mi325x"})


# Value is optional so a bare, value-less flag (an operator typo, or a flag
# left dangling at end-of-string) is stripped too rather than surviving into a
# retry that is supposed to launch without it.
_MOE_RUNNER_BACKEND_RE = re.compile(r"(?:^|\s)--moe-runner-backend(?:(?:[=\s]+)(?!--)\S+)?")

# Profile-phase capture defaults. Trace size scales with captured decode steps;
# an oversized capture serializes too slowly and kills the engine. 128 steps is
# serialization-safe on a large TP=8 MoE. Tunable via
# HYPERLOOM_PROFILE_MAX_STEPS_CAP.
_DEFAULT_PROFILE_MAX_STEPS = 128
# AgentX counterpart of the cap above. The 128 is calibrated in decode steps
# against the synthetic 1024/1024 shape; an agentic step carries a measured ISL
# p50 of 56k-96k tokens, so the same step count buffers orders of magnitude more
# profiler events in HOST RAM. Measured on DeepSeek-V4-Pro: eight vLLM workers at
# 113-127 GB each, Ray reported 1012/1024 GB and killed the capture three times.
# The AgentX client bounds the window by wall clock anyway (~20s of steady
# state), so the extra steps buy nothing. HYPERLOOM_PROFILE_MAX_ITERS overrides.
_AGENTX_PROFILE_MAX_ITERS = 8
# Default profile OSL ceiling when --profile-osl / PROFILE_OSL is unset: the
# profile reuses min(served OSL, this) so its trace stays light.
_PROFILE_DEFAULT_OSL = 1024
# SGLang graph-capture profiling flag shipped by the profile YAML, and the
# eager flag that makes it a no-op. Literals rather than an import from
# ``baseline``: that module imports this one, so the dependency only goes one
# way.
_SGLANG_PROFILE_CUDA_GRAPH_FLAG = "--enable-profile-cuda-graph"
_SGLANG_DISABLE_CUDA_GRAPH_FLAG = "--disable-cuda-graph"
# ``HYPERLOOM_PROFILE_DEGRADED_REASON`` value meaning the TraceLens runtime patch
# was attempted and did not apply. Distinct from "never attempted": patching can
# be disabled for an image that already ships the patch.
_TRACELENS_PATCH_UNAVAILABLE = "tracelens_runtime_patch_unavailable"
_AGENTX_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

# Quality-reference env names, in resolution order. Every scriptable workload
# needs this gate, so the contract is the framework-neutral ``HYPERLOOM_`` pair.
# The ``XDIT_`` pair predates ``--framework custom`` and is still both read and
# written because operator bench scripts live outside this repo and cannot be
# renamed in lockstep; drop it once those scripts have moved over.
_QUALITY_REF_ENVS = ("HYPERLOOM_QUALITY_REF", "XDIT_QUALITY_REF")
_QUALITY_REF_WRITE_ENVS = ("HYPERLOOM_QUALITY_REF_WRITE", "XDIT_QUALITY_REF_WRITE")


def _first_env(names: tuple[str, ...]) -> str:
    """Return the first non-empty stripped value among ``names`` in ``os.environ``.

    Args:
        names (tuple[str, ...]): Env var names in resolution order.

    Returns:
        str: The first non-empty value, or ``""`` when none is set.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def agentx_enabled(env: dict[str, str] | None = None) -> bool:
    """Return whether the AgentX benchmark wrapper is explicitly enabled."""
    raw = (env or os.environ).get("HYPERLOOM_AGENTX", "")
    return str(raw).strip().lower() in _AGENTX_TRUE_VALUES


def agentx_active(shared_state: Any = None) -> bool:
    """Whether AgentX is enabled, preferring persisted state over the ambient env var.

    ``benchmark_mode`` is stamped at seed precisely so it survives a restart,
    while ``HYPERLOOM_AGENTX`` only describes the shell that happens to be
    running -- an SDK caller, or a re-baseline/variant round driven from a
    subprocess that did not inherit it, would otherwise miss the AgentX-sized
    timeout or publish the agentic number under a synthetic tag. Either saying
    "agentx" is enough.

    Args:
        shared_state: Session state, when the caller has one.

    Returns:
        True when AgentX is enabled for this session, by either signal.
    """
    if agentx_enabled():
        return True
    return is_agentx_mode(getattr(shared_state, "benchmark_mode", ""))


def agentx_env_for_conc(conc: int | None = None) -> "Mapping[str, str]":
    """The environment the AgentX derivations read, carrying a rung's own CONC.

    Warmup is ``CANON_WARMUP_PER_LANE`` requests per lane across ``CONC`` lanes,
    so every bound derived from it is linear in the concurrency being measured,
    not in the one the session was launched at.

    Args:
        conc: The rung's concurrency, or ``None`` to read the session's.

    Returns:
        ``os.environ`` unchanged when no rung concurrency is given, else a copy
        with ``CONC`` replaced.
    """
    if not conc or conc <= 0:
        return os.environ
    return {**os.environ, "CONC": str(conc)}


def agentx_kb_write_blocked(shared_state: Any = None) -> bool:
    """Whether an agentic measurement must stay out of the cross-session KB.

    The recipe canonical id is a seven-tuple of model / hardware / framework /
    precision identity: no workload, no mode. Row workload tags are copied from
    ``SharedState.isl``/``osl``, which under AgentX are the inert 1024/1024
    placeholders the corpus overrides. So an agentic throughput would overwrite a
    synthetic ``best_throughput`` on a bare numeric comparison, and the row would
    be tagged as a 1024/1024 synthetic run -- which a later synthetic session's
    shape filter then matches positively. The store is machine-global and
    ``--reset-state`` does not clear it, so the damage outlives its session.

    One helper rather than a gate per sink: there are three writers (CLOSE
    finalize, the runtime amend, and the T0 anchor), they were not all found at
    once, and a fourth should have something obvious to call.

    Args:
        shared_state: Session state, when the caller has one.

    Returns:
        True when the caller must skip its Recipe KB write.
    """
    return agentx_active(shared_state)


def apply_agentx_switch(
    bench: dict[str, Any],
    model_path: str | None = None,
    *,
    conc: Any = None,
    active: bool | None = None,
) -> None:
    """Switch serving-framework benchmarks to the AgentX aiperf client.

    ``conc`` is the concurrency this round will run at; the inner benchmark cap
    and the client's warmup grace are both derived from it (see
    :func:`agentx_env_for_conc`).
    """
    if active is None:
        active = agentx_enabled()
    if not active:
        return
    from hyperloom.inference_optimizer import framework_registry

    framework = str(bench.get("framework") or "").strip().lower()
    if not framework or framework_registry.is_scriptable(framework):
        return
    envs = bench.setdefault("envs", {})
    bench["benchmark_script"] = "aiperf_client.sh"
    # The Magpie benchmark config's flat wall-clock cap (``benchmark.timeout_seconds``,
    # e.g. 7200s from baseline_vllm.yaml) is one deadline over server boot + warmup +
    # the measurement window + result export. AgentX runs at the model's native
    # context (``max_model_len`` lifted from the synthetic 6144 to e.g. 1M), so boot +
    # warmup alone can consume ~45 min before the window even opens; the flat cap then
    # SIGKILLs the benchmark before aiperf writes ``inferencex_result.json`` -- a 0-tput
    # baseline that fails the session. Raise the inner cap to the same AgentX budget the
    # outer subprocess timeout already uses (``agentx_baseline_timeout_sec``) so the two
    # layers stay consistent. AgentX-only: this function returned early above when AgentX
    # is off, so the default (synthetic) cap is untouched. The import is function-local
    # to avoid a circular dependency (``baseline`` imports this module at load time).
    #
    # max(), never assignment: this is the ONLY place in the AgentX path that
    # writes an existing cap, and a bare assignment LOWERS every config that
    # already declares more than the AgentX derivation. profile_sglang.yaml
    # declares 14400s ("Qwen-32B TP=1 profile with steady-state window can take
    # ~3 h") against a default derivation of 10800s, so an AgentX profile round
    # there was being cut from four hours to three -- the same mid-round kill this
    # module exists to prevent, introduced by the fix for it. A declared cap is a
    # measured statement about that config; the derivation is a floor under it,
    # not a replacement for it.
    from hyperloom.orchestrator.actions.executors.baseline import (
        agentx_baseline_timeout_sec,
        agentx_warmup_grace_sec,
    )

    _agentx_env = agentx_env_for_conc(conc)
    _derived = agentx_baseline_timeout_sec(_agentx_env)
    try:
        _declared = int(bench.get("timeout_seconds") or 0)
    except (TypeError, ValueError):
        _declared = 0
    if _declared > _derived:
        log.info(
            "AgentX: keeping the config's declared benchmark timeout %ds (> the AgentX "
            "derivation %ds). The derivation is a floor, never a ceiling -- lowering a "
            "cap the config measured for itself is how a round gets killed mid-window.",
            _declared,
            _derived,
        )
    bench["timeout_seconds"] = max(_declared, _derived)
    envs["RUN_EVAL"] = "false"
    envs["MODEL"] = str(model_path or bench.get("model") or os.environ.get("MODEL_PATH", "")).strip()
    envs["FRAMEWORK"] = framework
    # WEKA_LOADER_OVERRIDE is upstream's own per-recipe corpus pin, so it has no
    # AGENTX_ prefix and would not survive the loop below. aiperf_client.sh
    # documents it as a supported knob; without forwarding it only works when
    # the benchmark process happens to inherit the full parent environment,
    # which is exactly the kind of silent difference this path exists to remove.
    for key, value in os.environ.items():
        if key.startswith("AGENTX_") or key in ("AIPERF_BIN", "WEKA_LOADER_OVERRIDE"):
            envs[key] = value
    # ...but AGENTX_WARMUP_GRACE_PERIOD must not be forwarded raw. It is read by
    # TWO layers that have to agree: this process derives the subprocess cap from
    # it (scaled by CONC, because warmup is per-lane requests x CONC lanes), while
    # aiperf_client.sh hands it to aiperf as --warmup-grace-period, which is what
    # actually cuts the warmup off. The loop above copies the operator's raw
    # value, so the client was bounded at the UNSCALED number while the cap
    # budgeted the scaled one.
    #
    # Measured on a Kimi-K3 conc=32 round: cap 14400s of warmup vs client bound
    # 3600s. Warmup would have been cut at 106 of 354 requests -- not a crash, a
    # round that reports a prefix-reuse figure measured before the cache had
    # anything in it. Export the derived value so both layers see one number.
    #
    # AgentX-only by construction: this function returned early when AgentX is
    # off, and AGENTX_* has no meaning on the synthetic path.
    _grace = agentx_warmup_grace_sec(_agentx_env)
    _raw_grace = (os.environ.get("AGENTX_WARMUP_GRACE_PERIOD") or "").strip()
    envs["AGENTX_WARMUP_GRACE_PERIOD"] = str(_grace)
    if _raw_grace != str(_grace):
        log.info(
            "AgentX: exporting the CONC-scaled warmup grace %ds to the client "
            "(operator value %s). The client's --warmup-grace-period and this "
            "process's subprocess cap are derived from the same number, so a "
            "raw forward here would bound the warmup below what the cap pays for.",
            _grace,
            _raw_grace or "unset",
        )


def prepare_agentx_runtime(
    *,
    env: dict[str, str] | None = None,
    inferencex_path: str | None = None,
    config_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> str | None:
    """Deploy and preflight AgentX assets for baseline/profile runs."""
    runtime_env = env or os.environ
    if not agentx_enabled(runtime_env):
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError
        from hyperloom.inference_optimizer.agentx.runtime import maybe_prepare_agentx

        maybe_prepare_agentx(
            env=runtime_env,
            inferencex_path=str(inferencex_path or ""),
            config_path=Path(config_path) if config_path else "",
        )
    except AgentXPreflightError as exc:
        return f"AgentX preflight failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"AgentX runtime preparation failed: {type(exc).__name__}: {exc}"
    return None


def _scriptable_runner_type(bench: dict[str, Any], gpu_type: str | None) -> str:
    """Resolve the runner suffix a scriptable entrypoint is named after.

    Framework-independent: the suffix names the machine, not the workload.
    """
    return (
        str(gpu_type or "").strip().lower()
        or str(bench.get("runner_type") or "").strip().lower()
        or os.environ.get("GPU_TYPE", "").strip().lower()
        or os.environ.get("RUNNER_TYPE", "").strip().lower()
        or "mi355x"
    )


def _sync_repo_aliases(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    framework: str,
    prefer_dir: bool = False,
) -> None:
    """Keep ``<FRAMEWORK>_REPO_PATH`` and ``<FRAMEWORK>_DIR`` on one checkout.

    Both spellings reach the benchmark and a script may read either, so letting
    them drift points one half of a run at a different tree than the other.

    Args:
        bench: The ``benchmark`` section; the sync applies only to its framework.
        envs: The ``benchmark.envs`` mapping, updated in place.
        framework: Framework whose prefix the two aliases carry.
        prefer_dir: Resolve ``_DIR`` first, for the caller that saw the operator
            supply only that spelling.
    """
    if str(bench.get("framework") or "").strip().lower() != framework.strip().lower():
        return
    prefix = framework.strip().upper()
    order = (f"{prefix}_DIR", f"{prefix}_REPO_PATH") if prefer_dir else (f"{prefix}_REPO_PATH", f"{prefix}_DIR")
    repo_path = str(envs.get(order[0]) or envs.get(order[1]) or "").strip()
    if repo_path:
        envs[f"{prefix}_REPO_PATH"] = repo_path
        envs[f"{prefix}_DIR"] = repo_path


def _publish_scriptable_repo_root(framework: str, repo_path: str) -> None:
    """Publish a scriptable framework's checkout into the orchestrator's own env.

    A scriptable framework runs out of a repo checkout rather than a pip-installed
    package, so ``resolve_source_file_allowlist`` can only find it through
    ``<FRAMEWORK>_REPO_PATH`` / ``<FRAMEWORK>_DIR`` in ``os.environ``. Writing the
    resolved path into the materialized YAML reaches the *benchmark* subprocess
    but not the orchestrator, and it is the orchestrator that runs PolicyGate — so
    without this the source root is absent from the allowlist and every patch a
    specialist writes against the framework's own code is rejected, on a session
    that otherwise looks correctly configured.

    An operator-provided value always wins; this only fills the gap when the
    checkout was resolved (or is about to be cloned) by Hyperloom itself. The path
    is published even when it does not exist yet, because the benchmark wrapper
    clones on first use and the allowlist is recomputed per call.

    Args:
        framework: Framework name, used for the env prefix.
        repo_path: The resolved checkout path.
    """
    name = str(framework or "").strip().upper()
    path = str(repo_path or "").strip()
    if not name or not path:
        return
    for var in (f"{name}_REPO_PATH", f"{name}_DIR"):
        if not os.environ.get(var, "").strip():
            os.environ[var] = path
            log.info(
                "%s: published %s=%s so the framework source root reaches PolicyGate",
                framework,
                var,
                path,
            )


def _resolve_framework_repo_path(
    envs: dict[str, Any],
    *,
    framework: str,
    extra_aliases: tuple[str, ...] = (),
) -> str:
    """Resolve the framework checkout a session may patch, prefixed forms first.

    Resolution order is ``<FRAMEWORK>_REPO_PATH`` > ``<FRAMEWORK>_DIR`` > any
    per-framework alias > :data:`GENERIC_FRAMEWORK_ROOT_ENV`, with ``os.environ``
    ahead of the materialized ``envs`` at each rung. The prefixed names keep
    precedence because they are the more specific statement, so adding the generic
    form changes no existing behaviour; it exists because a session is
    single-framework by construction and an operator should not have to know the
    framework's name to point at its source.

    Args:
        envs: The materialized ``benchmark.envs`` mapping.
        framework: Framework name, used for the env prefix.
        extra_aliases: Additional per-framework spellings to accept, in order.

    Returns:
        The resolved path, or ``""`` when nothing was supplied.
    """
    prefix = str(framework or "").strip().upper()
    names = (f"{prefix}_REPO_PATH", f"{prefix}_DIR", *extra_aliases, GENERIC_FRAMEWORK_ROOT_ENV)
    for name in names:
        value = os.environ.get(name, "").strip() or str(envs.get(name) or "").strip()
        if value:
            return value
    return ""


def _custom_script_path(runner_type: str) -> str:
    """Locate the operator's entrypoint inside ``$HYPERLOOM_BYPASS_SCRIPTS_DIR``.

    Prefers the ``custom_{runner_type}.sh`` convention, then a lone ``.sh`` in
    the directory, which is what an operator who wrote one script for one
    machine actually has. Returns ``""`` when neither applies, leaving the
    resolution chain in ``bypass_scriptable`` to report the miss.
    """
    raw = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    if not raw:
        return ""
    scripts_dir = Path(raw)
    if not scripts_dir.is_dir():
        return ""
    preferred = scripts_dir / f"custom_{runner_type}.sh"
    if preferred.is_file():
        return str(preferred)
    candidates = sorted(p for p in scripts_dir.glob("*.sh") if p.is_file())
    return str(candidates[0]) if len(candidates) == 1 else ""


def _operator_extra_env() -> dict[str, str]:
    """Return the ``--extra-env`` pins the CLI serialized, or ``{}``.

    Args:
        None.

    Returns:
        The operator's ``NAME=VALUE`` pins; empty when unset or unparseable,
        because a malformed pin must not take the run down with it.
    """
    raw = os.environ.get("INFERENCE_OPTIMIZER_EXTRA_ENV", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        log.warning("custom: ignoring unparseable INFERENCE_OPTIMIZER_EXTRA_ENV")
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k).strip(): str(v) for k, v in parsed.items() if str(k).strip()}


def resolve_reference_base() -> tuple[str, dict[str, str]]:
    """Read the ``--reference-script`` server args / envs from SharedState.

    Returns ``("", {})`` when the run has no reference recipe.
    """
    from hyperloom.inference_optimizer.session.paths import session_dir

    from ...state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir())
    return state.reference_server_args.strip(), dict(state.reference_envs)


def _apply_custom_runtime_defaults(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    gpu_type: str | None,
    explicit_benchmark_script: bool,
) -> None:
    """Resolve an operator-supplied workload's checkout, entrypoint and pins.

    The other scriptable helpers know their framework's repo URL and shipped
    script; here both arrive at launch, so this only wires what was given and
    leaves anything missing to fail where it is diagnosable.

    ``--extra-env`` is the only channel an operator has for the knobs their own
    script reads, so those pins are written into ``benchmark.envs`` rather than
    left in the orchestrator's environment: the materialized config is what the
    measurement contract is read from, and a pin that never lands there is
    neither delivered to the benchmark nor protected from being overwritten by
    a variant.
    """
    if str(bench.get("framework") or "").strip().lower() != "custom":
        return

    runner_type = _scriptable_runner_type(bench, gpu_type)
    bench["runner_type"] = runner_type
    if not explicit_benchmark_script:
        script = _custom_script_path(runner_type)
        if script:
            bench["benchmark_script"] = script

    # A variant's own overrides are applied after this hook, so they still win
    # for keys the operator did not pin; setdefault only fills the gaps.
    for key, value in _operator_extra_env().items():
        envs.setdefault(key, value)

    repo_path = _resolve_framework_repo_path(envs, framework="custom")
    if not repo_path:
        return
    envs["CUSTOM_REPO_PATH"] = repo_path
    envs["CUSTOM_DIR"] = repo_path
    # Also under the framework-agnostic name, and in the config rather than only
    # in os.environ. An entrypoint Hyperloom did not write can only be expected
    # to know the generic spelling, and the benchmark inherits its environment
    # from whichever process launched the runner — a Ray worker carries the
    # environment the raylet booted with, hours or days earlier, so a stale
    # <FRAMEWORK>_DIR there outranks anything published later. Config envs are
    # applied on top of that inherited environment, so this is what actually
    # reaches the script.
    envs["FRAMEWORK_REPO_PATH"] = repo_path
    _publish_scriptable_repo_root("custom", repo_path)


def apply_scriptable_runtime_defaults(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    gpu_type: str | None,
    explicit_benchmark_script: bool,
) -> None:
    """Re-pin an operator's scriptable entrypoint and its runtime paths.

    Every config path that re-derives ``benchmark_script`` from ``gpu_type``
    must call this, otherwise the bare ``{framework}_{gpu_type}.sh`` it writes
    replaces the resolved absolute path. Each helper is a no-op for frameworks
    other than its own, so a new one is added here once.
    """
    for apply in (_apply_custom_runtime_defaults,):
        apply(
            bench,
            envs,
            gpu_type=gpu_type,
            explicit_benchmark_script=explicit_benchmark_script,
        )


def _remove_moe_runner_backend_arg(args: str) -> str:
    """Remove any existing SGLang MoE runner backend flag from an args string."""
    return " ".join(_MOE_RUNNER_BACKEND_RE.sub(" ", str(args or "")).split())


# Warn once per process when the accuracy gate is disabled.
_RUN_EVAL_DISABLED_WARN_EMITTED = False


def _model_requires_remote_code(model_path: str | None) -> bool:
    """Return whether benchmark server/client must trust custom HF code.

    Fires for any local config that advertises a custom AutoTokenizer (e.g.
    Kimi K2.6, ``model_type=kimi_k25``), so custom-code tokenizers do not need
    per-model special cases.
    """
    model = str(model_path or "").strip()
    if not model:
        return False
    data = _load_model_config_dict(model)
    basename = Path(model).name.lower()
    if data is None:
        # Fallback for mounted model dirs whose config is unreadable.
        return "kimi-k2" in basename or "kimi_k2" in basename
    model_type = str(data.get("model_type") or "").lower()
    archs = {str(a).lower() for a in data.get("architectures") or []}
    if model_type == "kimi_k25" or "kimik25forconditionalgeneration" in archs:
        return True
    auto_map = data.get("auto_map")
    return isinstance(auto_map, dict) and bool(auto_map.get("AutoTokenizer"))


def inject_vllm_expert_parallel(
    server_args: str | None,
    framework: Any,
    ep: Any,
) -> str:
    """Append vLLM expert-parallel flag when EP is enabled."""
    args = str(server_args or "").strip()
    if "vllm" not in str(framework or "").lower():
        return args
    try:
        ep_int = int(ep if ep not in (None, "") else 1)
    except (TypeError, ValueError):
        return args
    if ep_int <= 1:
        return args
    if re.search(r"(?:^|\s)--enable-expert-parallel(?:\s|$)", args):
        return args
    return f"{args} --enable-expert-parallel".strip()


class FrameworkScriptMismatchError(ValueError):
    """Raised when benchmark_script targets a different framework than the run.

    Subclasses ValueError so callers can catch it specifically and turn it
    into a structured action failure instead of an uncaught exception.
    """


def _visible_gpu_count() -> int:
    """Return how many GPUs are visible to this pod (0 = none / unknown).

    Prefers ``torch.cuda.device_count`` (no shell-out, keeps subprocess-mock
    tests happy), falls back to ``rocm-smi --showid``. Returns 0 on every
    failure so callers skip the clamp. Override via
    ``$INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT``.

    Returns:
        The number of visible GPUs, or 0 when none/unknown.
    """
    override = os.environ.get("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "").strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            log.warning(
                "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT=%r is not an int; ignoring override.",
                override,
            )
    try:
        import torch  # type: ignore[import-not-found]

        count = int(torch.cuda.device_count() or 0)
        if count > 0:
            return count
    except Exception:
        pass
    if shutil.which("rocm-smi"):
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, PermissionError, OSError):
            return 0
        if proc.returncode == 0:
            # ``rocm-smi --showid`` emits multiple ``GPU[N]`` lines per device;
            # dedup by index.
            indices: set[str] = set()
            for line in (proc.stdout or "").splitlines():
                stripped = line.strip()
                if stripped.startswith("GPU["):
                    idx, _, _ = stripped[4:].partition("]")
                    if idx:
                        indices.add(idx)
            return len(indices)
    return 0


def _tracelens_patch_enabled() -> bool:
    """Read the ``HYPERLOOM_ENABLE_PATCH`` kill switch (default on).

    Set ``HYPERLOOM_ENABLE_PATCH=0`` to disable runtime patching of vLLM /
    SGLang. Default on because the patches are backward-compatible.

    With the switch off, no *server flag* that only a patched build accepts is
    injected (vLLM ``--profiler-config.detailed_trace_annotation``, SGLang
    ``--enable-shape-discovery-for-cuda-graph-profile``), because an unpatched
    argparse rejects them. The SGLang ``PROFILE_EXTRA_BODY`` annotations
    (``shape_discovery`` / ``detailed_annotations``) are a different case and are
    **kept**: they ride the ``/start_profile`` API, which an unpatched server
    accepts, and the switch is also how a pre-patched image opts out of runtime
    patching while still supporting them. Only a patch that was *attempted and
    failed* clears them, which is why that gate reads
    ``HYPERLOOM_PROFILE_DEGRADED_REASON`` (set solely on the attempted-and-failed
    path) rather than ``tracelens_patch_ok``.

    Returns:
        True when runtime patching is enabled (default), else False.
    """
    return os.environ.get("HYPERLOOM_ENABLE_PATCH", "1").strip() != "0"


def _coerce_workload_int_env(env_key: str, raw: str) -> int:
    """Coerce workload env values, accepting ``CONC`` comma ladders."""
    text = str(raw or "").strip()
    if env_key == "CONC" and "," in text:
        values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
        if not values or any(v <= 0 for v in values):
            raise ValueError(f"{env_key}={raw!r} must contain positive integers")
        return values[0]
    value = int(text)
    if value <= 0:
        raise ValueError(f"{env_key}={raw!r} must be positive")
    return value


# ``$FRAMEWORK`` (lowercased) -> shipped Magpie YAML, relative to
# ``asset_root()``. Unknown / unset frameworks fall back to
# ``_DEFAULT_BASELINE_CONFIG`` (sglang) so existing sglang-default tests keep
# passing. Values are relative so ``asset_root()`` is still resolved at call
# time (honoring the ``$INFERENCE_OPTIMIZER_ASSET_ROOT`` override).
_BASELINE_CONFIG_BY_FRAMEWORK: dict[str, Path] = {
    "atom": Path("assets/configs/baseline_atom.yaml"),
    "vllm": Path("assets/configs/baseline_vllm.yaml"),
    "xdit": Path("assets/configs/baseline_xdit.yaml"),
    "custom": Path("assets/configs/baseline_custom.yaml"),
}
_DEFAULT_BASELINE_CONFIG = Path("assets/configs/baseline_sglang.yaml")


def default_baseline_config() -> Path:
    """Resolve the shipped Magpie YAML based on ``$FRAMEWORK`` env.

    Returns the sglang YAML when ``$FRAMEWORK`` is unset/unknown so existing
    sglang-default tests keep passing.

    Returns:
        Path: The shipped Magpie YAML config path for the resolved framework.
    """
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    rel = _BASELINE_CONFIG_BY_FRAMEWORK.get(fw, _DEFAULT_BASELINE_CONFIG)
    return asset_root() / rel


_PROFILER_FLAG_RE = re.compile(r"--profiler-config\.(\w+)[=\s]+(\S+)")
_TRUTHY_FLAG_VALUES = frozenset({"1", "on", "true", "yes"})


def _profiler_flag_value(server_args: str, name: str) -> str | None:
    """The value vLLM would resolve for ``--profiler-config.<name>``, or None.

    Returns the LAST occurrence, because repeated flags are how this layer overrides
    earlier ones and vLLM's argparse resolves them last-wins. Accepts both
    ``--flag value`` and ``--flag=value``. Regex rather than tokenization: this runs
    after JSON-valued flags have been compacted, and one unbalanced quote must not
    take the whole guard down with it.
    """
    value: str | None = None
    for match in _PROFILER_FLAG_RE.finditer(server_args or ""):
        if match.group(1) == name:
            value = match.group(2)
    return value


def _profiler_bound_holds(name: str, value: str | None, *, cap: int) -> bool:
    """Whether the value already present keeps this flag's guarantee.

    Name-only matching is not enough for the two flags that decide whether the
    capture is bounded at all: vLLM reads ``max_iterations`` of 0 (or absent) as "no
    limit", and a frontend profiler left on tracks no iterations and captures the
    entire ``start_profile``..``stop_profile`` range. A guard that accepted those
    would report success while the run stayed unbounded, which is worse than not
    guarding -- the warning would send the next investigation the wrong way.

    Every other flag only decides what the trace contains or where it lands, so its
    presence is the whole contract.
    """
    if value is None:
        return False
    if name == "max_iterations":
        try:
            iterations = int(value)
        except ValueError:
            return False
        return 0 < iterations <= cap
    if name == "ignore_frontend":
        return value.strip().lower() in _TRUTHY_FLAG_VALUES
    return True


def _finalize_framework_server_args(
    envs: dict[str, Any],
    bench: dict[str, Any],
    *,
    gpu_type: str | None,
    isl_val: int,
    osl_val: int,
    drop_moe_runner_backend: bool = False,
) -> None:
    """Apply the final framework server-arg guard pipeline in place
    (context-length/watchdog/attention/MoE/EP/dedup/compact/shell-safe); order is fixed.

    1. --context-length cap: sglang sizes max_total_tokens off the model's
       max_position_embeddings, so a huge native window balloons the aiter
       workspace past GPU memory. Cap to ISL+OSL+headroom, clamped to the
       native window AND to the run's MAX_MODEL_LEN.
    2. MI300X cold-compile guard: raise sglang's scheduler watchdog so the
       first-request aiter JIT compile survives (the 300s default fires
       SIGQUIT mid-warmup on a cold aiter cache).

    Steps 1-4b are sglang-scoped; steps 5 and 6 are vLLM/atom-scoped. The
    inline comments below carry the per-step rationale.

    ``drop_moe_runner_backend`` turns step 4 into a removal: the args are
    already merged from every source (task params, ``$INFERENCE_OPTIMIZER_
    SERVER_ARGS``, the reference recipe, the YAML base), so stripping here is
    what guarantees a retry launches without the backend that killed it."""
    framework_env = server_args_env_name(bench.get("framework"))
    resolved_server_args = str(envs.get(framework_env, "")).strip()
    resolved_server_args = inject_sglang_context_length(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        isl_val,
        osl_val,
        max_model_len=envs.get("MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN"),
    )
    resolved_server_args = inject_sglang_watchdog_timeout(
        resolved_server_args,
        bench.get("framework"),
    )
    # 3. Dual-chunk attention backend: Qwen 1M models need
    #    dual_chunk_flash_attn; inject it unless --attention-backend is pinned.
    resolved_server_args = inject_sglang_attention_backend(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        gpu_type=gpu_type or bench.get("runner_type"),
    )
    # 4. MoE runner backend: aiter's CK fused-MoE JIT build is broken in some
    #    images; inject the triton MoE runner unless --moe-runner-backend is
    #    pinned.
    if drop_moe_runner_backend:
        if framework_env == "EXTRA_SGLANG_ARGS":
            resolved_server_args = _remove_moe_runner_backend_arg(resolved_server_args)
    else:
        resolved_server_args = inject_sglang_moe_runner_backend(
            resolved_server_args,
            bench.get("framework"),
            bench.get("model"),
            gpu_type=gpu_type or bench.get("runner_type"),
        )
    resolved_server_args = inject_vllm_expert_parallel(
        resolved_server_args,
        bench.get("framework"),
        os.environ.get("EP", "").strip() or envs.get("EP"),
    )
    # 5. vLLM/atom argparse dedup: collapse repeated single-value flags to
    #    last-wins (vLLM crashes EngineCoreProc on a duplicate); no-op for
    #    sglang.
    resolved_server_args = dedup_vllm_server_args(
        resolved_server_args,
        bench.get("framework"),
    )
    # 6. JSON-valued flags (--speculative-config / --compilation-config /
    #    --hf-overrides ...): Magpie expands $EXTRA_VLLM_ARGS unquoted, so
    #    compact each JSON blob to be space-free so it survives as one shell
    #    word. No-op for sglang and for arg strings with no JSON.
    resolved_server_args = compact_json_server_args(
        resolved_server_args,
        bench.get("framework"),
    )
    resolved_server_args = validate_server_args_shell_safe(resolved_server_args)
    if resolved_server_args:
        envs[framework_env] = resolved_server_args


def materialize_config_with_envs(
    config_path: Path,
    output_dir: Path,
    *,
    extra_server_args: str = "",
    extra_envs: dict[str, Any] | None = None,
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None,
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None,
    args_mode: str = "append",
    model_path: str | None = None,
    gpu_type: str | None = None,
    inferencex_path: str | None = None,
    benchmark_script: str | None = None,
    out_name: str = "baseline_config.with_envs.yaml",
    establish_quality_ref: bool = False,
    drop_moe_runner_backend: bool = False,
    flydsl_source_dirs: bool = False,
    agentx_mode: bool | None = None,
) -> Path:
    """Render a per-run Magpie YAML with caller-provided overrides.

    Process env wins over YAML defaults: ``MODEL_PATH`` → ``benchmark.model``;
    ``GPU_TYPE`` → ``runner_type`` + pinned generic ``{framework}_{gpu_type}.sh``
    (so Magpie doesn't fall through to a native script hardcoding
    ``--result-dir /workspace/``); ``benchmark_script`` (pre-sanitized) re-pins
    after that; ``PRECISION`` → ``precision``; ``CONC/ISL/OSL/MAX_MODEL_LEN/TP/
    RANDOM_RANGE_RATIO`` → ``benchmark.envs``; ``ROCR_VISIBLE_DEVICES``
    reconciled against TP; ``RUN_EVAL`` defaulted; ``NUM_PROMPTS`` /
    ``NUM_WARMUPS`` computed adaptively. ``inferencex_path`` explicitly pins
    ``benchmark.inferencex_path`` for one task (falling back to
    ``$INFERENCEX_PATH`` for existing callers). ``extra_server_args`` routes
    into the framework env; ``--extra-env`` is copied into ``benchmark.envs``
    so Magpie forwards it to vLLM/sglang workers; ``extra_envs`` overrides any
    of the above.
    The session's ``--reference-script`` recipe is read from SharedState here
    rather than passed in, so it seeds a lowest-priority base (below the YAML
    base and ``extra_server_args``) for every caller.

    Args:
        config_path: Path to the source Magpie YAML to render.
        output_dir: Directory the materialized YAML is written into.
        extra_server_args: Extra framework server args merged into the env.
        extra_envs: Overrides applied last over any computed env values;
            shell/loader hijack names and credentials are dropped.
        remove_args: Inherited framework server args to remove before launch.
        unset_envs: Inherited env names to remove before applying
            ``extra_envs``; workload pins are refused.
        args_mode: ``"append"`` (default) or ``"replace"`` for
            ``extra_server_args``.
        model_path: Model path/id; overrides ``benchmark.model`` when set.
        gpu_type: GPU type; sets ``runner_type`` and pins the generic script.
        inferencex_path: Explicit InferenceX checkout to pin into the YAML.
        benchmark_script: Pre-sanitized benchmark script name to re-pin.
        out_name: File name for the materialized YAML.
        establish_quality_ref: When True (baseline only) the scriptable
            image-quality reference is ESTABLISHED (written) by this run;
            otherwise the run only COMPARES against it. See the quality-
            reference wiring block below.
        drop_moe_runner_backend: When True, skip the AMD MoE
            ``--moe-runner-backend`` injection and strip the flag from the
            merged args whatever source it came from (the one-shot fallback
            after a launch failure blamed on that backend).
        flydsl_source_dirs: When True, name the FlyDSL source roots in
            ``$FLYDSL_EXTRA_SOURCE_DIRS`` so a patched helper invalidates the JIT
            cache key. Off by default: only a run that applied such a patch needs it.
        agentx_mode: Explicit session-level AgentX decision. ``None`` preserves
            the legacy environment-based fallback.

    Returns:
        The materialized YAML path (stable file name across calls).

    Raises:
        FrameworkScriptMismatchError: If ``benchmark_script`` targets a
            different known framework than the run's framework.
    """
    server_args = (extra_server_args or "").strip()
    operator_server_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
    replace_args = str(args_mode or "append").strip().lower() == "replace"
    if operator_server_args and not replace_args:
        if server_args:
            from ._grid_runner import merge_server_args

            server_args = merge_server_args(operator_server_args, server_args)
        else:
            server_args = operator_server_args
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = cfg.setdefault("benchmark", {})
    if model_path:
        bench["model"] = str(model_path)
    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision
    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)
    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)
    envs = bench.setdefault("envs", {})
    apply_scriptable_runtime_defaults(
        bench,
        envs,
        gpu_type=gpu_type,
        explicit_benchmark_script=bool(benchmark_script),
    )
    apply_agentx_switch(bench, model_path, active=agentx_mode)
    # Fail fast on framework/script mismatch (e.g. vllm image + sglang script).
    # Only trip when the script carries a DIFFERENT known framework's prefix, so
    # custom/non-prefixed scripts are not falsely rejected.
    _script = str(bench.get("benchmark_script") or "").lower()
    _fw = str(bench.get("framework") or "").lower()
    from hyperloom.inference_optimizer import framework_registry

    _known_fw = framework_registry.names()
    if _script and _fw in _known_fw:
        _other = [k for k in _known_fw if k != _fw and _script.startswith(f"{k}_")]
        if _other:
            raise FrameworkScriptMismatchError(
                f"framework/script mismatch: framework={_fw!r} but "
                f"benchmark_script={_script!r} targets {_other[0]!r}; refusing "
                f"to boot server (would launch the wrong framework's entrypoint)"
            )
    effective_inferencex_path = str(inferencex_path or "").strip() or os.environ.get("INFERENCEX_PATH", "").strip()
    if effective_inferencex_path:
        # Persist the resolved InferenceX checkout so Magpie's runtime checkout
        # matches Hyperloom's patch target.
        bench["inferencex_path"] = effective_inferencex_path
    for env_key in (
        "CONC",
        "ISL",
        "OSL",
        "MAX_MODEL_LEN",
        "TP",
        "PORT",
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = _coerce_workload_int_env(env_key, val)
    # RANDOM_RANGE_RATIO is a float feeding the steady-state formulas below; do
    # NOT coerce to int.
    r_env = os.environ.get("RANDOM_RANGE_RATIO", "").strip()
    if r_env:
        envs["RANDOM_RANGE_RATIO"] = float(r_env)
    rocr_env = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if rocr_env:
        envs["ROCR_VISIBLE_DEVICES"] = rocr_env
    tp_from_env = os.environ.get("TP", "").strip()
    tp_from_yaml = envs.get("TP")
    rocr_yaml = str(envs.get("ROCR_VISIBLE_DEVICES") or "").strip()
    rocr_devices = [d.strip() for d in rocr_yaml.split(",") if d.strip()]
    if tp_from_env:
        resolved_tp = int(tp_from_env)
    elif rocr_yaml and not tp_from_yaml:
        # Derive TP from the user-pinned GPU list when the YAML doesn't set TP.
        resolved_tp = len(rocr_devices)
        envs["TP"] = resolved_tp
    else:
        resolved_tp = int(tp_from_yaml or 1)
    # Auto-clamp TP to the visible GPU count. Override via
    # $INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP=1.
    if os.environ.get("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "").strip() != "1":
        visible = _visible_gpu_count()
        if visible and resolved_tp > visible:
            log.warning(
                "TP=%d but only %d GPU(s) visible to this pod; clamping "
                "TP=%d so sglang/vllm can actually load weights. Export "
                "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP=1 to opt out (the "
                "subprocess will then fail at server launch).",
                resolved_tp,
                visible,
                visible,
            )
            resolved_tp = visible
    envs["TP"] = resolved_tp
    if not rocr_yaml or len(rocr_devices) < resolved_tp:
        derived = ",".join(str(i) for i in range(resolved_tp))
        if rocr_yaml and rocr_yaml != derived:
            log.warning(
                "ROCR_VISIBLE_DEVICES=%r has %d devices but TP=%d; "
                "expanding to %r so SGLang sees enough GPUs. Set "
                "ROCR_VISIBLE_DEVICES explicitly to override.",
                rocr_yaml,
                len(rocr_devices),
                resolved_tp,
                derived,
            )
        envs["ROCR_VISIBLE_DEVICES"] = derived

    # Last-resort fallbacks kept in sync with the CLI workload defaults
    # (parser.DEFAULT_ISL/OSL/CONC); normally the CLI has already projected the
    # resolved values into these envs before materialization.
    isl_val = int(envs.get("ISL") or 1024)
    osl_val = int(envs.get("OSL") or 1024)
    conc_val = int(envs.get("CONC") or 64)

    # Steady-state window for profiling configs (detected by YAML
    # ``benchmark.envs.PROFILE`` or ``profiler.torch_profiler.enabled``, not the process env).
    # The captured-step count is capped at
    # a serialization-safe budget; the profile OSL is resolved (and lowered if
    # needed) so the steady-state floor fits that cap:
    #   max_iters    = HYPERLOOM_PROFILE_MAX_STEPS_CAP (default 128)
    #   steady_floor = ceil(OSL * (1 + R) / (2 * CONC))   # must be <= max_iters
    #   delay_iters  = OSL * (R + 1) * 3 - max_iters / 2
    is_profile = str(envs.get("PROFILE", "")).strip() == "1" or (
        bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True
    )
    # Scriptable image frameworks (xDiT diffusion) have no LLM decode
    # steady-state window; the OSL/steady-floor math below is a serving concept
    # xDiT never consumes, so skip it.
    from hyperloom.inference_optimizer import framework_registry as _fw_reg

    _is_scriptable_profile = _fw_reg.is_scriptable(bench.get("framework"))
    profile_num_prompts: int | None = None
    # ``(sentinel, flag)`` pairs remembered so the re-assertion at the very end of
    # this function can restore exactly the profiler flags that some later step
    # dropped, without re-stating the ones that survived. See that block for why a
    # dropped ``max_iterations`` is an OOM and not a cosmetic problem.
    pending_vllm_profiler_flags: list[tuple[str, str]] = []
    # The serialization-safe capture cap computed below, so the re-assertion can reject
    # a max_iterations that is above it (or non-positive, which vLLM reads as no limit)
    # instead of accepting the flag on its name alone.
    pending_vllm_profiler_cap: int = 0
    if is_profile and not _is_scriptable_profile:
        try:
            r_val = float(envs.get("RANDOM_RANGE_RATIO", 1.0))
        except (TypeError, ValueError):
            r_val = 1.0
        safe_conc = max(conc_val, 1)
        # Cap captured decode steps at a serialization-safe default so the
        # torch-profiler trace can be written without starving the engine RPC.
        _cap_raw = os.environ.get("HYPERLOOM_PROFILE_MAX_STEPS_CAP", "").strip()
        cap_explicit = _cap_raw.isdigit() and int(_cap_raw) >= 1
        try:
            cap = int(_cap_raw or _DEFAULT_PROFILE_MAX_STEPS)
        except (TypeError, ValueError):
            cap = _DEFAULT_PROFILE_MAX_STEPS
        if cap < 1:
            cap = _DEFAULT_PROFILE_MAX_STEPS

        # Resolve the profile-scoped OSL. PROFILE_OSL (via --profile-osl) is
        # honored as-is; otherwise default to min(served OSL,
        # _PROFILE_DEFAULT_OSL). Scoped to is_profile.
        _profile_osl_raw = os.environ.get("PROFILE_OSL", "").strip()
        profile_osl_explicit = _profile_osl_raw.isdigit() and int(_profile_osl_raw) > 0
        if profile_osl_explicit:
            osl_val = int(_profile_osl_raw)
        else:
            osl_val = min(osl_val, _PROFILE_DEFAULT_OSL)
        safe_osl = max(osl_val, 1)

        # Steady-state floor: minimum captured decode steps for the splitter to
        # isolate a steady-state window (mirrors TraceLens
        # find_steady_state_window). The capture must be >= this or the splitter
        # reports trace_split_no_steady_state.
        steady_floor = math.ceil(safe_osl * (1.0 + r_val) / (2.0 * safe_conc))
        if steady_floor > cap:
            if profile_osl_explicit:
                # Honor the operator's explicit OSL; warn about the window.
                log.warning(
                    "PROFILE_OSL=%d needs %d captured steps to reach steady "
                    "state, above the profile cap of %d; the trace may lack a "
                    "steady-state window (trace_split_no_steady_state). Lower "
                    "--profile-osl or raise HYPERLOOM_PROFILE_MAX_STEPS_CAP.",
                    osl_val,
                    steady_floor,
                    cap,
                )
            else:
                # Auto path: lower the profile OSL so the floor fits the cap.
                fitted_osl = max(1, int(cap * 2 * safe_conc / (1.0 + r_val)))
                log.warning(
                    "profile OSL %d would need %d captured steps to reach "
                    "steady state (> cap %d); lowering profile OSL to %d so the "
                    "capture stays serializable. Baseline/optimize unaffected.",
                    osl_val,
                    steady_floor,
                    cap,
                    fitted_osl,
                )
                osl_val = fitted_osl
                safe_osl = max(osl_val, 1)
                steady_floor = math.ceil(safe_osl * (1.0 + r_val) / (2.0 * safe_conc))

        # Profile server runs at the resolved profile OSL, decoupled from --osl.
        envs["OSL"] = osl_val

        # Capture up to the cap (>= steady_floor in the auto path).
        max_iters = cap
        delay_iters = int(osl_val * (r_val + 1) * 3 - max_iters / 2)
        if delay_iters < 0:
            delay_iters = 0
        # The iteration-based delay assumes the client streams a predictable
        # number of decode steps before steady state. The AgentX client instead
        # brackets a WALL-CLOCK window with /start_profile and /stop_profile, so
        # an iteration delay computed from the placeholder OSL (6080 steps at the
        # 1024/1024 defaults) is never reached inside that window and the trace
        # comes back empty. Hand the delay to the client and keep only the
        # capture bound, which is what stops the worker accumulating events in
        # host RAM until the OOM killer arrives.
        if agentx_enabled():
            delay_iters = 0
            # ...and the bound itself has to come down, because the cap above is
            # sized in DECODE STEPS against the synthetic OSL. Under AgentX the
            # captured work per step is agentic: measured ISL p50 was 56k-96k
            # tokens, two orders of magnitude past the 1024/1024 shape the cap
            # was calibrated on. At the stock cap a DeepSeek-V4 profile round put
            # each of the eight vLLM workers at 113-127 GB of HOST RAM -- Ray
            # reported 1012/1024 GB and killed them mid-capture, three attempts
            # in a row, so the round produced no trace at all.
            #
            # A shorter capture is not a worse trace here: the client already
            # bounds the window by wall clock (~20s of steady state), so the
            # extra steps buy nothing and only inflate the in-memory event
            # buffer. HYPERLOOM_PROFILE_MAX_ITERS still overrides this below.
            if max_iters > _AGENTX_PROFILE_MAX_ITERS:
                if cap_explicit:
                    # The operator asked for this cap explicitly (e.g. to widen
                    # the steady-state window); silently overriding it with no
                    # trace of the original value would hide why a deliberate
                    # HYPERLOOM_PROFILE_MAX_STEPS_CAP setting had no effect.
                    log.warning(
                        "AgentX: explicit HYPERLOOM_PROFILE_MAX_STEPS_CAP=%d is "
                        "being overridden to %d. The cap is calibrated on the "
                        "synthetic ISL/OSL shape; an agentic step carries orders "
                        "of magnitude more, and the torch profiler buffers "
                        "events in host RAM until the OOM killer arrives.",
                        max_iters,
                        _AGENTX_PROFILE_MAX_ITERS,
                    )
                else:
                    log.info(
                        "AgentX: lowering captured profile steps %d -> %d. The cap is "
                        "calibrated on the synthetic ISL/OSL shape; an agentic step "
                        "carries orders of magnitude more, and the torch profiler "
                        "buffers events in host RAM until the OOM killer arrives.",
                        max_iters,
                        _AGENTX_PROFILE_MAX_ITERS,
                    )
                max_iters = _AGENTX_PROFILE_MAX_ITERS
                if max_iters < steady_floor:
                    log.warning(
                        "AgentX: capped profile steps %d is below the steady-state "
                        "floor of %d; the trace may lack a steady-state window "
                        "(trace_split_no_steady_state).",
                        max_iters,
                        steady_floor,
                    )
        # Operator hard-override of captured steps (e.g. a small eager FlyDSL
        # profile). Honored verbatim; warn when outside the safe band rather
        # than silently clamping.
        _ovr = os.environ.get("HYPERLOOM_PROFILE_MAX_ITERS", "").strip()
        if _ovr.isdigit() and int(_ovr) > 0:
            max_iters = int(_ovr)
            # Raising the capture bound must not revive the delay the AgentX
            # branch above zeroed; the two knobs are documented together.
            if agentx_enabled():
                _delay_ovr = os.environ.get("HYPERLOOM_PROFILE_DELAY_ITERS", "").strip()
                if _delay_ovr:
                    log.warning(
                        "ignoring HYPERLOOM_PROFILE_DELAY_ITERS=%s under AgentX: the "
                        "profiling window is wall-clock, so an iteration delay never elapses "
                        "inside it and the trace comes back empty",
                        _delay_ovr,
                    )
            else:
                delay_iters = max(0, env_int("HYPERLOOM_PROFILE_DELAY_ITERS", 8))
            # The AgentX clamp above is a HOST RAM bound, and this override
            # silently undoes it. Neither check below stands in for saying so:
            # ``cap`` defaults to _DEFAULT_PROFILE_MAX_STEPS, so the obvious
            # HYPERLOOM_PROFILE_MAX_ITERS=128 lands exactly on it, trips
            # neither branch, and restores the very bound that kept the
            # profiler from being OOM-killed -- without printing anything.
            if agentx_enabled() and max_iters > _AGENTX_PROFILE_MAX_ITERS:
                log.warning(
                    "HYPERLOOM_PROFILE_MAX_ITERS=%d overrides the AgentX capture "
                    "bound of %d. That bound is a host-RAM limit, not a "
                    "serialization one: an agentic step carries orders of "
                    "magnitude more events than the synthetic shape ``cap`` is "
                    "sized against, and at the stock cap a DeepSeek-V4 profile "
                    "round was OOM-killed mid-capture three times in a row. "
                    "Unset it to restore the bound.",
                    max_iters,
                    _AGENTX_PROFILE_MAX_ITERS,
                )
            if max_iters < steady_floor:
                log.warning(
                    "HYPERLOOM_PROFILE_MAX_ITERS=%d is below the steady-state "
                    "floor of %d; the trace may lack a steady-state window "
                    "(trace_split_no_steady_state).",
                    max_iters,
                    steady_floor,
                )
            elif max_iters > cap:
                log.warning(
                    "HYPERLOOM_PROFILE_MAX_ITERS=%d exceeds the serialization-"
                    "safe cap of %d; the trace may be too large to serialize "
                    "(EngineCore RPC timeout).",
                    max_iters,
                    cap,
                )
        # NUM_PROMPTS must let the engine reach ``delay_iters + max_iters``
        # decode steps before running out of prompts (N prompts ≈ N * OSL / CONC
        # iters; invert + 2x buffer). Hyperloom owns this under PROFILE.
        required_iters = delay_iters + max_iters
        iters_to_prompts = max(
            1,
            (required_iters * safe_conc + safe_osl - 1) // safe_osl,
        )
        profile_num_prompts = max(safe_conc, iters_to_prompts * 2)
        fw = str(bench.get("framework") or "").lower()
        # atom's profiler is HTTP-driven via atom_mi*x.sh, so this layer sets no
        # atom profiler envs and must NOT inject --profiler-config.* flags.
        is_atom = "atom" in fw
        # TraceLens profiler flags exist only in patched vLLM / SGLang builds;
        # try to patch, fall back to the safe set on failure. Default-on
        # (HYPERLOOM_ENABLE_PATCH=0 disables); skip for atom.
        tracelens_patch_ok = False
        patch_attempted = _tracelens_patch_enabled() and not is_atom
        if patch_attempted:
            if "vllm" in fw:
                tracelens_patch_ok = ensure_vllm_patched_for_tracelens()
            else:
                tracelens_patch_ok = ensure_sglang_patched_for_tracelens()
            if not tracelens_patch_ok:
                envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "unavailable"
                envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] = _TRACELENS_PATCH_UNAVAILABLE
                log.warning(
                    "TraceLens runtime patch unavailable for framework=%s; "
                    "profile will omit annotation-only flags and roofline "
                    "analysis may be degraded.",
                    fw or "<unset>",
                )
        if is_atom:
            # atom's profile window lives only in Magpie's atom_mi*x.sh
            # (ATOM_PROFILE_OSL / ATOM_PROFILE_NUM_PROMPTS); defer to Magpie.
            profile_num_prompts = None
        elif "vllm" in fw:
            existing_vllm_args = str(envs.get("EXTRA_VLLM_ARGS", ""))
            profiler_flags = [
                ("delay_iterations", f"--profiler-config.delay_iterations {delay_iters}"),
                ("max_iterations", f"--profiler-config.max_iterations {max_iters}"),
            ]
            if tracelens_patch_ok:
                profiler_flags.append(("capture_torch_profiler", "--profiler-config.capture_torch_profiler True"))
                profiler_flags.append(("detailed_trace_annotation", "--profiler-config.detailed_trace_annotation True"))
            # vLLM's AsyncLLM-side profiler tracks no iterations and captures the
            # whole start_profile..stop_profile range, so it has to stay off
            # wherever we bound the worker-side one. The YAML normally carries the
            # flag, but a replacing candidate wipes the YAML value too, so it
            # belongs in the set the re-assertion can restore.
            profiler_flags.append(("ignore_frontend", "--profiler-config.ignore_frontend True"))
            pending_vllm_profiler_flags = profiler_flags
            pending_vllm_profiler_cap = max_iters
            # Injected unconditionally so the computed cap WINS over anything the YAML
            # pins, via the repeated-flag last-wins that vLLM's argparse and this
            # layer both rely on. Filtering out a flag the YAML already carries would
            # hand the run e.g. a hand-written ``max_iterations 100000`` -- unbounded
            # in practice -- and quietly discard the serialization-safe budget
            # computed above; HYPERLOOM_PROFILE_MAX_ITERS is the override channel for
            # that, not the YAML. ``ignore_frontend`` is the exception: the shipped
            # profile YAML already sets it, and vLLM warns on duplicate keys.
            profiler_args = " ".join(
                flag
                for sentinel, flag in profiler_flags
                if sentinel != "ignore_frontend" or "ignore_frontend" not in existing_vllm_args
            )
            if "delay_iterations" not in existing_vllm_args:
                envs["EXTRA_VLLM_ARGS"] = f"{existing_vllm_args} {profiler_args}".strip()
        else:
            import json as _json

            try:
                _raw_body = _json.loads(str(envs.get("PROFILE_EXTRA_BODY", "{}")))
                extra_body = _raw_body if isinstance(_raw_body, dict) else {}
            except (ValueError, TypeError):
                extra_body = {}
            # Always override start_step/num_steps (the template has CONC=8
            # placeholders).
            extra_body["start_step"] = delay_iters
            extra_body["num_steps"] = max_iters
            # shape_discovery balloons an eager+with_stack trace; allow disabling
            # it via env for eager profiles.
            _shape_disc = os.environ.get(
                "HYPERLOOM_PROFILE_SHAPE_DISCOVERY",
                "1",
            ).strip().lower() not in {"0", "false", "no", "off"}
            # Gemma2 + shape-discovery crashes CUDA-graph capture, so disable
            # shape-discovery for Gemma2. Escape hatch
            # HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 only skips the Gemma2
            # gate; it does NOT override a global
            # HYPERLOOM_PROFILE_SHAPE_DISCOVERY=0.
            _force_shape_disc = os.environ.get(
                "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE",
                "0",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if _shape_disc and not _force_shape_disc:
                _model = str(bench.get("model") or "")
                if _model_is_gemma2(_model):
                    _shape_disc = False
                    log.info(
                        "Gemma2 roofline: disabling shape-discovery to avoid "
                        "CUDA-graph capture crash (hipErrorStreamCapture"
                        "Unsupported); CUDA graph + profiling kept. Set "
                        "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 to override.",
                    )
                    if _load_model_config_dict(_model) is None:
                        log.warning(
                            "Gemma2 detected via path heuristic (no readable "
                            "config.json at %r); shape-discovery skip may be "
                            "imprecise.",
                            _model,
                        )
            # Both capture options are annotation-only and need TraceLens
            # server-side support to land: without it the trace carries no
            # ``kernel_shape_profiler`` events (trace-health check 5), so asking
            # for them pays the capture cost for data nothing downstream reads.
            # Keyed on the degraded *reason* rather than ``tracelens_patch_ok``:
            # a patch that was never attempted (HYPERLOOM_ENABLE_PATCH=0) can
            # still be baked into the image, and must keep the annotations.
            _patch_degraded = envs.get("HYPERLOOM_PROFILE_DEGRADED_REASON") == _TRACELENS_PATCH_UNAVAILABLE
            if _patch_degraded:
                _shape_disc = False
            extra_body["shape_discovery"] = _shape_disc
            if _patch_degraded:
                extra_body["detailed_annotations"] = False
            else:
                extra_body.setdefault("detailed_annotations", True)
            # NOTE: this write happens before the per-task ``extra_envs`` merge, so
            # an ``extra_envs`` entry for PROFILE_EXTRA_BODY can still drop
            # start_step/num_steps the way ``args_mode="replace"`` used to drop
            # vLLM's --profiler-config bounds. The vLLM side is re-asserted at the
            # end of this function; SGLang is NOT, because deciding whether a
            # non-positive num_steps means "unbounded" or "no capture" needs a
            # SGLang-side answer this layer does not have. Every OOM observed so
            # far was vLLM.
            envs["PROFILE_EXTRA_BODY"] = _json.dumps(extra_body)
            if tracelens_patch_ok and _shape_disc:
                # TraceLens-patched SGLang exposes
                # --enable-shape-discovery-for-cuda-graph-profile; unpatched
                # SGLang errors on it.
                existing_sglang = str(envs.get("EXTRA_SGLANG_ARGS", ""))
                if "shape-discovery-for-cuda-graph-profile" not in existing_sglang:
                    envs["EXTRA_SGLANG_ARGS"] = (
                        f"{existing_sglang} --enable-shape-discovery-for-cuda-graph-profile"
                    ).strip()

    if not _is_scriptable_profile:
        # NUM_PROMPTS / NUM_WARMUPS are serving-request concepts; xDiT drives its
        # own iteration count, so leave them untouched.
        if profile_num_prompts is not None:
            # Profile mode: force-override NUM_PROMPTS to reach the steady-state
            # window.
            envs["NUM_PROMPTS"] = profile_num_prompts
        else:
            seq_cost = isl_val + osl_val
            if seq_cost <= 1024:
                factor = 10
            elif seq_cost <= 4096:
                factor = 5
            elif seq_cost <= 16384:
                factor = 3
            else:
                factor = 2
            if "NUM_PROMPTS" not in envs:
                envs["NUM_PROMPTS"] = max(conc_val * factor, conc_val)
        if "NUM_WARMUPS" not in envs:
            envs["NUM_WARMUPS"] = min(conc_val, 8)
    # ── reference-script base (lowest priority) ────────────────────────────
    # Seed the framework server-args env + envs from a reference recipe below
    # the YAML base and any per-task extra_server_args (reference flags leftmost,
    # so last-wins lets later merges override them).
    ref_args, reference_envs = resolve_reference_base()
    if ref_args:
        from ._grid_runner import merge_server_args

        _ref_fw_env = server_args_env_name(bench.get("framework"))
        _ref_existing = str(envs.get(_ref_fw_env, "")).strip()
        envs[_ref_fw_env] = merge_server_args(ref_args, _ref_existing) if _ref_existing else ref_args
    for _rk, _rv in reference_envs.items():
        envs.setdefault(str(_rk), str(_rv))  # never clobber YAML/CLI envs
    if server_args:
        # Merge into (not overwrite) the framework env so the profile path's
        # graph-capture flags aren't dropped.
        from ._grid_runner import merge_server_args

        framework_env = server_args_env_name(bench.get("framework"))
        existing = str(envs.get(framework_env, "")).strip()
        if replace_args:
            envs[framework_env] = server_args
        elif framework_env == "EXTRA_SGLANG_ARGS" and "--moe-runner-backend" in str(server_args):
            # For MoE backend exploration/tuning, the candidate value must
            # replace the baseline's injected default (usually triton) rather
            # than relying on duplicate last-wins flags.
            existing = _remove_moe_runner_backend_arg(existing)
        if not replace_args and existing:
            envs[framework_env] = merge_server_args(existing, server_args)
        elif not replace_args:
            envs[framework_env] = server_args
    # Magpie forwards only ``benchmark.envs``. ``--extra-env`` used to land
    # there only for ``custom``, so vLLM Ray workers never saw MTP pins.
    combined_extra: dict[str, Any] = dict(_operator_extra_env())
    if extra_envs:
        combined_extra.update(extra_envs)
    safe_extra_envs, dropped_extra_envs = filter_untrusted_env_mapping(
        combined_extra,
        allow_predicate=is_allowed_variant_env_key,
    )
    for _dk in dropped_extra_envs:
        log.warning("Dropping unsafe extra_envs key %s before benchmark materialization", _dk)
    for key, value in safe_extra_envs.items():
        envs[str(key)] = str(value)
    # ── aiter tuned-config lookup logging ────────────────────────────────────
    # aiter logs a line for every tuned-GEMM table lookup it MISSES
    # unconditionally, but the corresponding HIT line only when this is set. Two
    # things depend on having it on:
    #   * GEMM tuning takes its shape list from the misses -- shapes derived from
    #     config.json instead cover 0.4% of what the runtime actually asks for;
    #   * apply verification counts hits to decide whether a tuned artifact was
    #     ever read. Without hit lines, "0 hits" and "hit logging was off" are
    #     indistinguishable, and treating the latter as the former would REVERT
    #     every arm.
    # A scan of 60 production server logs found the flag set in none of them, so
    # this is not a hypothetical gap. setdefault keeps an operator override.
    envs.setdefault("AITER_LOG_TUNED_CONFIG", "1")
    _sync_repo_aliases(
        bench,
        envs,
        framework="custom",
        prefer_dir="CUSTOM_DIR" in safe_extra_envs and "CUSTOM_REPO_PATH" not in safe_extra_envs,
    )
    framework_env = server_args_env_name(bench.get("framework"))
    # Final dedup after reference/server_args merges: collapse repeated
    # vLLM/atom single-value flags to last-wins so recipe/variant values
    # override earlier ones (no-op for sglang).
    _final_args = str(envs.get(framework_env, "")).strip()
    if _final_args:
        envs[framework_env] = dedup_vllm_server_args(_final_args, bench.get("framework"))
    # An eager profile cannot capture a CUDA graph, so the profile YAML's
    # ``--enable-profile-cuda-graph`` is dead weight the server still sets up
    # for. The two flags reach the env from different layers -- the YAML
    # template vs the eager fallback folded into ``extra_server_args`` -- and
    # only meet here, after every merge above, so this is the first point that
    # can see the contradiction.
    if framework_env == "EXTRA_SGLANG_ARGS":
        _sg_tokens = str(envs.get(framework_env, "")).split()
        if _SGLANG_DISABLE_CUDA_GRAPH_FLAG in _sg_tokens and _SGLANG_PROFILE_CUDA_GRAPH_FLAG in _sg_tokens:
            envs[framework_env] = " ".join(t for t in _sg_tokens if t != _SGLANG_PROFILE_CUDA_GRAPH_FLAG)
            log.info(
                "Dropping %s: %s is set, so there is no graph capture to profile.",
                _SGLANG_PROFILE_CUDA_GRAPH_FLAG,
                _SGLANG_DISABLE_CUDA_GRAPH_FLAG,
            )
    # ── Quality-reference wiring (scriptable / server-less workloads) ──────
    # Magpie forwards only ``benchmark.envs`` to the wrapper subprocess, so
    # re-inject the quality reference here (the single scriptable choke point)
    # or the shipped empty YAML default silently skips the gate. What the
    # reference holds is the workload's business — an xDiT image, an operator's
    # own artifact — so the default filename is a convention, not a contract.
    # When the operator configures nothing, derive a stable reference path
    # under ``session_dir()`` (pinned via
    # ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``) so baseline + variants
    # resolve the same file:
    #   * BASELINE (establish_quality_ref=True): COMPARE off + WRITE a fresh
    #     reference (a stale prior-session file cannot mislead the baseline).
    #   * Every other variant: COMPARE only, write path empty (a degraded
    #     variant can never overwrite the baseline reference).
    #   * Profiling / roofline (is_profile): no gate and never write.
    if framework_registry.is_scriptable(bench.get("framework")):
        _qref = _first_env(_QUALITY_REF_ENVS)
        if not _qref:
            from hyperloom.inference_optimizer.session.paths import session_dir

            _qref = str(session_dir() / "storage" / "quality_ref" / "baseline.png")
        if is_profile:
            _ref_compare, _ref_write = "", ""
        elif establish_quality_ref:
            _ref_compare, _ref_write = "", (_first_env(_QUALITY_REF_WRITE_ENVS) or _qref)
        else:
            _ref_compare, _ref_write = _qref, ""
        for _name in _QUALITY_REF_ENVS:
            envs[_name] = _ref_compare
        for _name in _QUALITY_REF_WRITE_ENVS:
            envs[_name] = _ref_write
    # ── xDiT-only wiring ───────────────────────────────────────────────────
    # These three are read by the xDiT runner and by nothing else, so they are
    # keyed on the framework rather than on scriptability: an operator-supplied
    # ``custom`` workload must not have its baseline altered by settings it
    # never declared, least of all an attention backend.
    if str(bench.get("framework") or "").strip().lower() == "xdit":
        # The xDiT runner resolves models via MODEL_REGISTRY keys, not
        # filesystem paths. XDIT_MODEL_ARG selects the basename ("name",
        # registry-correct) vs the full path ("path", which fails lookup). Force
        # it onto benchmark.envs here so per-task overrides can't break model
        # resolution. Default "name".
        envs["XDIT_MODEL_ARG"] = os.environ.get("XDIT_MODEL_ARG", "").strip() or "name"
        # If set, the baked hyperloom_local_aliases map each registered name to
        # a local snapshot dir rooted at $XDIT_MODEL_ROOT/<slug>. Leave unset in
        # public/default deployments so the operator chooses the model cache
        # location explicitly.
        _xdit_model_root = os.environ.get("XDIT_MODEL_ROOT", "").strip()
        if _xdit_model_root:
            envs["XDIT_MODEL_ROOT"] = _xdit_model_root
        # For the baseline only, force the operator-pinned backend (default
        # 'aiter', the MI300X-verified path) so an invalid agent override cannot
        # poison the reference measurement. Explore/sweep variants keep their
        # freedom to try alternative backends.
        if establish_quality_ref:
            envs["XDIT_ATTENTION_BACKEND"] = os.environ.get("XDIT_ATTENTION_BACKEND", "").strip() or "aiter"
    # ── Per-model MI300X baseline work-arounds ─────────────────────────
    # A handful of flagship models SIGABRT during CUDA-graph capture on the
    # sglang ROCm image because their DEFAULT fused kernels are buggy on
    # gfx942. Inject the verified per-model work-around unless the caller
    # already pinned it (setdefault/merge, never overwrite). Matched on the
    # model basename.
    _model_basename = Path(str(model_path or os.environ.get("MODEL_PATH", ""))).name.lower()
    if "kimi-k2" in _model_basename:
        # Kimi K2.x at tp8 takes sglang's ROCm fused-decode-MLA path, whose RoPE
        # kernel aborts during CUDA-graph capture. Disabling the fused decode
        # pipeline keeps tp8 + the clean aiter MLA path.
        envs.setdefault("SGLANG_ROCM_FUSED_DECODE_MLA", "0")
    if "mimo-v2" in _model_basename:
        # MiMo-V2.x's DEFAULT aiter attention backend SIGABRTs during CUDA-graph
        # capture on gfx942. Pin the triton attention backend. Merge (never
        # overwrite) and skip when the caller already pinned an
        # --attention-backend.
        from ._grid_runner import merge_server_args

        _mimo_fw_env = server_args_env_name(bench.get("framework"))
        _mimo_existing = str(envs.get(_mimo_fw_env, "")).strip()
        _mimo_is_vllm = "vllm" in str(bench.get("framework") or "").lower()
        # sglang accepts lowercase `triton`; vLLM only knows TRITON_ATTN. Pick
        # the framework-correct spelling.
        _mimo_attn_backend = "TRITON_ATTN" if _mimo_is_vllm else "triton"
        if "attention-backend" not in _mimo_existing:
            envs[_mimo_fw_env] = (
                merge_server_args(_mimo_existing, f"--attention-backend {_mimo_attn_backend}")
                if _mimo_existing
                else f"--attention-backend {_mimo_attn_backend}"
            )
        # vLLM registers this checkpoint under MiMoV2FlashForCausalLM but the HF
        # config declares MiMoV2ForCausalLM, which the pod-local vLLM build
        # rejects at boot. Remap the arch via --hf-overrides. vLLM-only; JSON
        # kept space-free so it survives Magpie's unquoted splice. Merge (never
        # overwrite) and skip when --hf-overrides is already pinned.
        if "vllm" in str(bench.get("framework") or "").lower():
            _mimo_hf_existing = str(envs.get(_mimo_fw_env, "")).strip()
            if "hf-overrides" not in _mimo_hf_existing and "hf_overrides" not in _mimo_hf_existing:
                _mimo_arch_override = '--hf-overrides {"architectures":["MiMoV2FlashForCausalLM"]}'
                envs[_mimo_fw_env] = (
                    merge_server_args(_mimo_hf_existing, _mimo_arch_override)
                    if _mimo_hf_existing
                    else _mimo_arch_override
                )
    # Sparse-attention KV-cache block size (config-derived, model-agnostic).
    # Models like MiniMax-M3 (MSA) place the main K/V and the indexer side-cache
    # in one KV-cache group whose sparse backends only accept the model's
    # sparse_attention_config.sparse_block_size (e.g. 128). vLLM's default
    # --block-size 16 (and the value Magpie bakes in when EXTRA_VLLM_ARGS is
    # empty) has no common block size with it, so KV-cache init aborts with
    # "No common block size for 16" -- baseline, roofline, and every
    # explore variant crash at startup. Read the required size from the
    # model config and pin --block-size at this shared choke point so it rides
    # EXTRA_VLLM_ARGS on every path (the roofline path in particular seeds from
    # the current-best delta and would otherwise drop the baseline's block size
    # and fall back to the default). vLLM-only: --block-size is a vLLM flag
    # (sglang rejects it; its sparse page size is set differently). Merge (never
    # overwrite) and skip when a --block-size is already pinned so an explicit
    # operator/explore choice wins; the later dedup_vllm_server_args collapses
    # any duplicate last-wins anyway. Config unreadable (e.g. an uncached
    # hub-id) -> None -> no injection, prior behaviour preserved.
    if "vllm" in str(bench.get("framework") or "").lower():
        _sparse_bs = _sparse_kv_block_size(str(model_path or bench.get("model") or ""))
        if _sparse_bs:
            from ._grid_runner import merge_server_args

            _sp_fw_env = server_args_env_name(bench.get("framework"))
            _sp_existing = str(envs.get(_sp_fw_env, "")).strip()
            if "block-size" not in _sp_existing and "block_size" not in _sp_existing:
                envs[_sp_fw_env] = (
                    merge_server_args(_sp_existing, f"--block-size {_sparse_bs}")
                    if _sp_existing
                    else f"--block-size {_sparse_bs}"
                )
    # Single choke point every benchmark path funnels through: the final
    # server-arg guards, applied at the FINAL framework env so any
    # operator-pinned flag is honored and never doubled.
    _finalize_framework_server_args(
        envs,
        bench,
        gpu_type=gpu_type,
        isl_val=isl_val,
        osl_val=osl_val,
        drop_moe_runner_backend=drop_moe_runner_backend,
    )
    # ── Client trust-remote-code (model-agnostic) ─────────────────────────
    # The MI300X bench scripts always launch the SERVER with
    # --trust-remote-code, so a custom-tokenizer model's CLIENT must load the
    # same remote code or transformers raises mid-warmup. Mirror it onto every
    # client-trust env. setdefault never overrides an operator opt-out.
    for _trust_key in (
        "MAGPIE_TRUST_REMOTE_CODE",  # Magpie sglang remote-direct client
        "BENCH_TRUST_REMOTE_CODE",  # GEAK bench_e2e.sh inferencex client
        "HF_HUB_TRUST_REMOTE_CODE",  # transformers / HF hub tokenizer auto-load
    ):
        envs.setdefault(_trust_key, "1")
    if _model_requires_remote_code(model_path or bench.get("model")):
        _remote_code_existing = str(envs.get(framework_env, "")).strip()
        if "trust-remote-code" not in _remote_code_existing:
            from ._grid_runner import merge_server_args

            envs[framework_env] = (
                merge_server_args(_remote_code_existing, "--trust-remote-code")
                if _remote_code_existing
                else "--trust-remote-code"
            )
    # Server-side trust-remote-code for custom-code models (Qwen3.6 MoE): the
    # SERVER must also load remote code or it refuses the arch at boot. Scoped
    # to this family. Merge (never overwrite) and skip when --trust-remote-code
    # is already set.
    if "qwen3.6-35b-a3b" in _model_basename or "qwen3-6-35b-a3b" in _model_basename:
        _trust_existing = str(envs.get(framework_env, "")).strip()
        if "trust-remote-code" not in _trust_existing:
            from ._grid_runner import merge_server_args

            envs[framework_env] = (
                merge_server_args(_trust_existing, "--trust-remote-code") if _trust_existing else "--trust-remote-code"
            )
    # Accuracy eval (GSM8K) is ON by default; env / extra_envs may override.
    # Disabling it removes the per-variant accuracy gate — warn loudly, never
    # block.
    if "RUN_EVAL" not in envs:
        env_run_eval = os.environ.get("RUN_EVAL")
        envs["RUN_EVAL"] = env_run_eval if env_run_eval is not None else "true"
    # Persist eval task/limit into the YAML so they become part of the
    # fingerprint-able contract and bypass_runner reads them from a stable source.
    _eval_tasks_env = os.environ.get("MAGPIE_EVAL_TASKS", "").strip()
    if _eval_tasks_env and "MAGPIE_EVAL_TASKS" not in envs:
        envs["MAGPIE_EVAL_TASKS"] = _eval_tasks_env
    _eval_limit_env = os.environ.get("MAGPIE_EVAL_LIMIT", "").strip()
    if _eval_limit_env and "MAGPIE_EVAL_LIMIT" not in envs:
        envs["MAGPIE_EVAL_LIMIT"] = _eval_limit_env
    # PD (router-fronted): lm_eval's default token-id prompts are rejected by the
    # sglang_router's /v1/completions (HTTP 422, StringOrArray), collapsing the
    # accuracy eval. Force string prompts so the gate works. Explicit env /
    # extra_envs win; only disaggregated runs are touched (aggregated hits the
    # sglang server directly, which accepts token-id prompts).
    _eval_tok_env = os.environ.get("MAGPIE_EVAL_TOKENIZED_REQUESTS", "").strip()
    if not _eval_tok_env:
        try:
            from ._multi_node_env import resolve_kb_topology

            if str(resolve_kb_topology().get("pd_mode") or "").lower() == "disaggregated":
                _eval_tok_env = "false"
        except Exception as exc:  # noqa: BLE001 - best-effort; never block config materialization
            # Don't degrade silently: if this IS a PD run, failing to resolve the
            # topology leaves MAGPIE_EVAL_TOKENIZED_REQUESTS unset, lm_eval keeps
            # sending token-id prompts, and the sglang_router 422s every request
            # -> accuracy reads 0 with no other clue pointing back here.
            log.warning(
                "_workload_envs: could not resolve PD topology to gate the lm_eval prompt format (%s); "
                "if this is a disaggregated run, MAGPIE_EVAL_TOKENIZED_REQUESTS stays unset and the "
                "sglang_router may reject token-id prompts with HTTP 422 (accuracy eval would read 0)",
                exc,
            )
            _eval_tok_env = ""
    if _eval_tok_env and "MAGPIE_EVAL_TOKENIZED_REQUESTS" not in envs:
        envs["MAGPIE_EVAL_TOKENIZED_REQUESTS"] = _eval_tok_env
    if str(envs.get("RUN_EVAL", "")).strip().lower() in _RUN_EVAL_FALSE_VALUES:
        global _RUN_EVAL_DISABLED_WARN_EMITTED
        if not _RUN_EVAL_DISABLED_WARN_EMITTED:
            log.warning(
                "RUN_EVAL is disabled: no per-variant accuracy gate, so "
                "accuracy regressions will not be caught. Set RUN_EVAL=true "
                "to restore the gate. This warning fires once per process."
            )
            _RUN_EVAL_DISABLED_WARN_EMITTED = True
    # KernelForge fp8 block-scale CK backend switch: SGLANG_FP8_BLOCKSCALE_CK_MAX_M
    # only takes effect on a KernelForge-patched sglang fp8_utils.py. Ensure the
    # patch, scoped to sglang + the env present. Fail-soft (a failed patch leaves
    # the env a no-op). Honors the HYPERLOOM_ENABLE_PATCH kill switch.
    _fw = str(bench.get("framework") or "").lower()
    if _tracelens_patch_enabled() and "sglang" in _fw and "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" in envs:
        if not ensure_sglang_patched_for_ck_blockscale():
            log.warning(
                "CK fp8 block-scale patch could not be applied; "
                "SGLANG_FP8_BLOCKSCALE_CK_MAX_M will no-op on the unpatched "
                "sglang fp8_utils.py (serving run continues unaffected)."
            )
    # FlyDSL folds only same-directory helpers into its JIT cache key, so a patched
    # helper one directory over is served from a stale binary. Naming the roots
    # folds their sources into the key. Only the run that applied such a patch has
    # that hazard; setdefault so an operator-set value (YAML / extra_envs) wins.
    if flydsl_source_dirs:
        _flydsl_dirs = flydsl_extra_source_dirs()
        if _flydsl_dirs:
            envs.setdefault(ENV_FLYDSL_EXTRA_SOURCE_DIRS, _flydsl_dirs)

    # sglang FP8 per-channel/per-token CK fast path: a dense FP8 checkpoint
    # with per-channel weight + per-token (dynamic) activation falls into the
    # slow unfused _apply_fallback_scaled_mm in sglang's apply_fp8_linear
    # unless SGLANG_USE_AITER_FP8_PER_TOKEN=1 flips use_per_token_if_dynamic on
    # and routes the GEMM to aiter's CK gemm_a8w8_bpreshuffle. Inject it from
    # Hyperloom, strictly scoped to sglang + fp8 + gfx942 + that exact quant
    # scheme so per-tensor and block-scale FP8 are never touched. setdefault so
    # an operator-set value (YAML / extra_envs) always wins.
    from hyperloom.inference_optimizer.gpu_types import _resolve_amd_gpu_type

    _model_for_quant = str(model_path or os.environ.get("MODEL_PATH", ""))
    if (
        "sglang" in _fw
        and str(bench.get("precision") or "").strip().lower() == "fp8"
        and _resolve_amd_gpu_type(gpu_type or bench.get("runner_type")) in _GFX942_GPU_TYPES
        and _fp8_is_per_channel_per_token(_model_for_quant)
    ):
        envs.setdefault("SGLANG_USE_AITER_FP8_PER_TOKEN", "1")
    remove_list = to_str_list(remove_args)
    unset_list = to_str_list(unset_envs)
    if remove_list:
        envs[framework_env] = remove_server_args(envs.get(framework_env, ""), remove_list)
    for key in unset_list:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(key).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("Refusing to unset pinned benchmark env %s", key)
            continue
        envs.pop(str(key), None)
    for key in unset_list:
        if isinstance(extra_envs, dict) and key in extra_envs:
            envs[str(key)] = str(extra_envs[key])
    if pending_vllm_profiler_flags and framework_env == "EXTRA_VLLM_ARGS":
        # The profile path's iteration bounds have to be the LAST word on this env,
        # because three separate steps above can drop them: a candidate carrying
        # ``args_mode="replace"`` (which writeback sets as soon as a KEEP needs
        # ``remove_args``) overwrites the whole flag string, ``extra_envs`` can
        # override it outright, and ``remove_args`` itself strips flags by name.
        # vLLM reads a missing ``max_iterations`` as "profile until stop_profile",
        # and the worker then accumulates every profiler event in host RAM at
        # ~60 MiB/s until the cgroup OOM-killer takes it out mid-roofline. No
        # exploration result is worth that, so the bounds win over all three.
        from ._grid_runner import merge_server_args

        profile_args = str(envs.get(framework_env, "")).strip()
        restored = [
            flag
            for name, flag in pending_vllm_profiler_flags
            if not _profiler_bound_holds(
                name,
                _profiler_flag_value(profile_args, name),
                cap=pending_vllm_profiler_cap,
            )
        ]
        if restored:
            log.warning(
                "profile server args lost torch-profiler flags %r (replace_args=%s); "
                "restoring them so the profiler cannot run unbounded.",
                restored,
                bool(replace_args),
            )
            merged = " ".join(restored)
            if profile_args:
                merged = merge_server_args(profile_args, merged)
            # _finalize_framework_server_args already ran, so re-apply the sink-side
            # guard it ends with rather than shipping an unvalidated string.
            envs[framework_env] = validate_server_args_shell_safe(merged)
    # The rendered YAML is persisted, so credentials must not reach it.
    filtered_envs, dropped_credentials = filter_untrusted_env_mapping(
        envs,
        allow_predicate=lambda key: key not in BENCHMARK_SECRET_ENV_NAMES,
    )
    if dropped_credentials:
        log.warning(
            "Dropping control-plane credentials from benchmark envs: %s",
            ", ".join(sorted(dropped_credentials)),
        )
        envs.clear()
        envs.update(filtered_envs)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = output_dir / out_name
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return materialized


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
