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

:func:`~._server_argv.seal_server_argv` settles the server argument string at
the end of :func:`materialize_config_with_envs`; nothing may write it after.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from hyperloom.common.env import env_bool, env_flag, is_truthy
from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES
from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    INTVTY_OBJECTIVES,
    agentx_enabled as agentx_enabled,
    intvty_grading_enabled,
    agentx_active as _agentx_active,
    parse_intvty_noise_pct,
)
from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    BLOCKED_EXTERNAL_ENV_NAMES,
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
)
from hyperloom.common.workload_defaults import (
    DEFAULT_CONC,
    DEFAULT_ISL,
    DEFAULT_OSL,
)
from hyperloom.inference_optimizer.gpu_types import gpu_runner_type
from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.inference_optimizer.framework_paths import ENV_FLYDSL_EXTRA_SOURCE_DIRS, GENERIC_FRAMEWORK_ROOT_ENV
from hyperloom.inference_optimizer.framework_paths import flydsl_extra_source_dirs
from ._model_launch_workarounds import pin_model_launch_workarounds
from ._profile_capture import (
    _TRACELENS_PATCH_UNAVAILABLE,
    _AtomTracelensCaps,
    _ProfileConfig,
    _ProfileKnobs,
    _ProfileTarget,
    _fit_profile_capture_window,
    _profiler_bound_holds,
    _profiler_flag_value,
    _shape_profile_trace,
    profile_target,
)
from hyperloom.inference_optimizer.grid_server_args import (
    compact_json_server_args,
    dedup_vllm_server_args,
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    inject_sglang_watchdog_timeout,
    server_args_env_name,
)
from hyperloom.inference_optimizer.grid_server_args import merge_server_args
from hyperloom.inference_optimizer.grid_server_args import remove_server_args
from hyperloom.inference_optimizer.grid_server_args import validate_server_args_shell_safe
from hyperloom.inference_optimizer import framework_registry
from ._recipe_script import launcher_overwritten_envs
from ._server_argv import add_server_arg_unless_pinned, seal_server_argv
from ._server_patcher import (
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

# The aiter CK gemm_a8w8_bpreshuffle kernel ships for gfx942 / CDNA3 only; gfx950
# has no such kernel. The gate is the arch, so the board list is read off the
# identity table rather than typed out beside it.
_GFX942_GPU_TYPES = frozenset(board for board, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items() if arch == "gfx942")


# Value is optional so a bare, value-less flag (an operator typo, or a flag
# left dangling at end-of-string) is stripped too rather than surviving into a
# retry that is supposed to launch without it.
_MOE_RUNNER_BACKEND_RE = re.compile(r"(?:^|\s)--moe-runner-backend(?:(?:[=\s]+)(?!--)\S+)?")

# SGLang graph-capture profiling flag shipped by the profile YAML, and the
# eager flag that makes it a no-op. Literals rather than an import from
# ``baseline``: that module imports this one, so the dependency only goes one
# way.
_SGLANG_PROFILE_CUDA_GRAPH_FLAG = "--enable-profile-cuda-graph"
_SGLANG_DISABLE_CUDA_GRAPH_FLAG = "--disable-cuda-graph"


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
    return _agentx_active(benchmark_mode=getattr(shared_state, "benchmark_mode", ""))


def agentx_env_for_conc(conc: int | None = None) -> "Mapping[str, str]":
    """The environment the AgentX derivations read, carrying a rung's own CONC.

    The operator's ``--extra-env`` pins are already in the process environment,
    exported by ``_export_operator_launch_shape``, so no layer over it is needed
    here -- re-merging them would make this reader resolve a knob differently
    from every other one.

    Warmup is ``CANON_WARMUP_PER_LANE`` requests per lane across ``CONC`` lanes,
    so every bound derived from it is linear in the concurrency being measured,
    not in the one the session was launched at.

    Args:
        conc: The rung's concurrency, or ``None`` to read the session's.

    Returns:
        The process environment, with ``CONC`` replaced when a rung concurrency
        is given.
    """
    env = dict(os.environ)
    if conc and conc > 0:
        env["CONC"] = str(conc)
    return env


def _is_agentx_client_key(key: str, env: "Mapping[str, str]") -> bool:
    """True for a knob :func:`apply_agentx_switch` forwards to the AgentX client."""
    from hyperloom.common.agentx_workload import BACKEND_ENV, is_mlperf_backend

    if key.startswith("AGENTX_"):
        return True
    return is_mlperf_backend(env) and (key.startswith(("MLPERF_", "AGENTIC_")) or key == BACKEND_ENV)


# GEAK's own names for the axes Hyperloom hands it: ``E2E_METRIC`` selects one
# and ``bench_summary.json`` records the matching ``metric_basis``. Spelled in
# GEAK's vocabulary, not Hyperloom's curve-row names, so the handoff and GEAK's
# summary can be compared as strings on both sides. The interactivity token is
# the full field name: in InferenceX's vocabulary a bare ``intvty`` is the
# decode-only 1/ITL family, not this end-to-end one.
GEAK_METRIC_OUTPUT = ("output", "aggregate_output_tok_s")
GEAK_METRIC_INTVTY = (GRADED_INTVTY_P50, GRADED_INTVTY_P50)


def geak_metric_axis(
    *,
    benchmark_mode: str = "",
    grading: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """GEAK's ``(E2E_METRIC, metric_basis)`` pair for the axis this session keeps candidates on.

    GEAK accepts on the single axis it is handed, so that axis has to be the
    objective. An interactivity-graded session keeps a candidate only on a gain
    in the median ``e2e_norm_intvty_p50``, with the p90 tail and output
    throughput as noise-band guards; searching either guard selects wins the
    rebench reverts and passes over median wins. Total token throughput is in
    neither: on a trace replay under a ~97% prefix cache it is ~99% input tokens
    that were never computed, so a kernel win cannot surface on it. Every other
    session is graded on output.

    A GEAK build that cannot measure ``e2e_norm_intvty_p50`` refuses it, or, predating
    that check, falls back to output and records that in ``metric_basis``;
    ``sweep_via_geak`` rejects a replay recorded on any basis but the one
    requested.

    Args:
        benchmark_mode: The session's persisted mode, when the caller holds one.
            Passing it matters for a round driven from a subprocess that did not
            inherit ``HYPERLOOM_AGENTX``.
        grading: The session's ``SharedState.grading``. Its ``objective`` was
            resolved at seed, where the run could still see its own
            configuration, so it wins outright over the mode-based derivation
            below -- which reads this process's environment.

    Returns:
        The ``E2E_METRIC`` value and the ``metric_basis`` name that goes with it.
    """
    objective = str((grading or {}).get("objective") or "").strip()
    if objective:
        on_intvty = objective in INTVTY_OBJECTIVES
    else:
        on_intvty = intvty_grading_enabled(benchmark_mode=benchmark_mode)
    if on_intvty:
        return GEAK_METRIC_INTVTY
    return GEAK_METRIC_OUTPUT


def geak_acceptance(metric_basis: str, noise_pct: float) -> dict[str, Any] | None:
    """Hyperloom's KEEP rule on the axis it hands GEAK, so GEAK's verdict applies the same bar.

    The rebench of a GEAK candidate keeps an interactivity-graded one only on a median gain of
    at least ``AGENTX_KEEP_P50_THRESHOLD_PCT`` with the p90 tail and output throughput each
    inside the noise band. Without the rule GEAK reports any gain past its own noise band as a
    win, and the rebench reverts each one short of the threshold. Guards are named by GEAK
    basis, like the objective; GEAK records the p90 tail under the curve-row name.

    Args:
        metric_basis: The ``metric_basis`` handed to GEAK (see :func:`geak_metric_axis`).
        noise_pct: The session's noise band, the one its own verdict holds both guards to.

    Returns:
        The rule, or ``None`` for any other axis, which leaves GEAK's verdict as it was.
    """
    if metric_basis != GEAK_METRIC_INTVTY[1]:
        return None
    return {
        "objective": metric_basis,
        "min_gain_pct": AGENTX_KEEP_P50_THRESHOLD_PCT,
        "guard_max_drop_pct": {GRADED_INTVTY: noise_pct, GEAK_METRIC_OUTPUT[1]: noise_pct},
    }


def cli_workload_defaults() -> tuple[int, int, int]:
    """The CLI's ``ISL``/``OSL``/``CONC`` defaults, as one source for fallbacks.

    Every last-resort fallback in this module reads them from here, so a
    materialized recipe and the workload spec published beside it cannot
    disagree about what "unset" means. The CLI resolves its own flags from the
    same ``hyperloom.common`` constants, so the two cannot drift apart.
    """
    return DEFAULT_ISL, DEFAULT_OSL, DEFAULT_CONC


def build_agentx_workload_spec(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    model_path: str | None = None,
    env: Mapping[str, str] | None = None,
    grading: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the AgentX workload for downstream consumers.

    Written into the materialized recipe and forwarded in the GEAK handoff so
    GEAK can select the AgentX client and refuse to treat the CLI's synthetic
    ``isl``/``osl`` placeholders as the served load. Omitted entirely on non-AgentX
    runs, so the fixed-ISL/OSL path stays byte-identical.

    Call only from :func:`apply_agentx_switch`, and only after it has written its
    derived values into ``envs`` -- the spec must publish the numbers the client
    will actually run with, not the operator's raw inputs.

    Args:
        bench: The benchmark mapping being materialized.
        envs: That mapping's ``envs``, already carrying the switch's overrides.
        model_path: The model this round serves, when the caller knows it.
        env: The resolved process environment, carrying this round's ``CONC``
            (see :func:`agentx_env_for_conc`). Defaults to ``os.environ``.
    """
    from hyperloom.common.agentx_workload import (
        MLPERF_CORPUS,
        MLPERF_PORT,
        is_mlperf_backend,
        mlperf_flow,
        mlperf_trajectories,
    )

    proc_env: Mapping[str, str] = os.environ if env is None else env

    def client_knob(key: str, default: Any) -> str:
        """A knob the client reads from ``envs``, so ``envs`` wins.

        :func:`apply_agentx_switch` forwards the process env into ``envs`` and
        then overwrites the entries it derives, so ``envs`` holds the value the
        benchmark subprocess is actually handed.
        """
        return str(envs.get(key) or proc_env.get(key) or default)

    def served_knob(key: str, default: Any) -> str:
        """A serving knob the process env owns, so the process env wins.

        ``CONC``/``ISL``/``OSL`` carry no ``AGENTX_`` prefix, so the forwarding
        loop skips them and ``materialize_config_with_envs`` projects them into
        ``envs`` only after this runs. Reading ``envs`` first publishes the base
        YAML's stale value -- which is how a spec pinned at the parser default 64
        shipped to GEAK while the recipe served the operator's 8.
        """
        return str(proc_env.get(key) or envs.get(key) or default)

    default_isl, default_osl, default_conc = cli_workload_defaults()
    conc = int(served_knob("CONC", default_conc))
    metric_basis = geak_metric_axis(benchmark_mode="agentx", grading=grading)[1]
    intvty_p90_veto_pct = (
        float(grading["noise_pct"])
        if isinstance(grading, Mapping) and isinstance(grading.get("noise_pct"), (int, float))
        else parse_intvty_noise_pct()
    )
    acceptance = geak_acceptance(metric_basis, intvty_p90_veto_pct)
    keep_rule = {"acceptance": acceptance} if acceptance else {}
    isl_osl_placeholder = {
        "isl": int(served_knob("ISL", default_isl)),
        "osl": int(served_knob("OSL", default_osl)),
        "note": "CLI defaults only; agentic replay ignores fixed ISL/OSL",
    }
    if is_mlperf_backend(proc_env) or is_mlperf_backend(envs):
        return {
            "kind": "agentx_mlperf_agentic",
            "client": "mlperf",
            "scenario": "mlperf-agentic-v6",
            "corpus": MLPERF_CORPUS,
            "canonical_corpus": MLPERF_CORPUS,
            "num_entries": mlperf_trajectories(envs),
            "duration_s": 0,
            "geak_loop_duration_s": 0,
            "concurrency": conc,
            "metric_basis": metric_basis,
            "intvty_p90_veto_pct": intvty_p90_veto_pct,
            **keep_rule,
            "metric_window_s": 0.0,
            "flow": mlperf_flow(envs),
            "port": int(envs.get("PORT") or MLPERF_PORT),
            "failed_request_threshold": float(client_knob("AGENTX_FAILED_REQUEST_THRESHOLD", 0.10)),
            "isl_osl_placeholder": isl_osl_placeholder,
        }

    raise ValueError("Only the MLPerf client uses a Hyperloom workload spec")


def pin_mlperf_round_concurrency(envs: dict[str, Any]) -> None:
    """Make ``AGENTIC_CONCURRENCY`` follow this round's ``CONC``.

    The client used to prefer ``AGENTIC_CONCURRENCY``, and the baseline YAML
    carries it. A conc-sweep rung that only changes ``CONC`` would then be
    recorded at one concurrency and measured at another.
    """
    from hyperloom.common.agentx_workload import is_mlperf_backend

    if not is_mlperf_backend(envs):
        return
    conc = envs.get("CONC")
    if conc not in (None, ""):
        envs["AGENTIC_CONCURRENCY"] = str(conc)


def apply_agentx_switch(
    bench: dict[str, Any],
    model_path: str | None = None,
    *,
    conc: Any = None,
    active: bool | None = None,
    grading: Mapping[str, Any] | None = None,
) -> None:
    """Switch serving-framework benchmarks to the independent MLPerf client.

    ``conc`` is the concurrency this round will run at; the inner benchmark cap,
    the client's warmup grace and the published ``workload_spec.concurrency`` are
    all derived from it (see :func:`agentx_env_for_conc`).

    ``grading`` is the session's own ``SharedState.grading``, resolved once at
    seed. It settles the axis and band the published ``workload_spec`` hands to
    GEAK; callers that cannot reach the live state leave it ``None`` and the
    spec derives both from the environment as before.
    """
    from hyperloom.common.agentx_workload import is_mlperf_backend

    if active is None:
        active = agentx_enabled()
    if not active or not is_mlperf_backend():
        return
    from hyperloom.inference_optimizer import framework_registry

    framework = str(bench.get("framework") or "").strip().lower()
    if not framework or framework_registry.is_scriptable(framework):
        return
    envs = bench.setdefault("envs", {})
    from hyperloom.common.agentx_workload import (
        BACKEND_ENV,
        MLPERF_PORT,
        MLPERF_SERVED_MODEL,
        agentx_client_script,
        is_mlperf_backend,
        mlperf_flow,
        mlperf_trajectories,
    )

    _agentx_env = agentx_env_for_conc(conc)
    bench["benchmark_script"] = agentx_client_script(_agentx_env)
    envs["RUN_EVAL"] = "false"
    envs["MODEL"] = str(model_path or bench.get("model") or os.environ.get("MODEL_PATH", "")).strip()
    envs["FRAMEWORK"] = framework
    for key, value in _agentx_env.items():
        if _is_agentx_client_key(key, _agentx_env):
            envs[key] = value
    if is_mlperf_backend(_agentx_env):
        envs[BACKEND_ENV] = "mlperf"
        # The harness dials a fixed port and model name. Pinning both here is the
        # one place they are enforced, so a recipe PORT cannot leave the server
        # and the client on different sockets.
        envs["PORT"] = str(MLPERF_PORT)
        envs["MLPERF_AGENTIC_MODEL"] = MLPERF_SERVED_MODEL
        # Settled once here; the client and the published spec read these back.
        envs["MLPERF_AGENTIC_FLOW"] = mlperf_flow(envs)
        envs["AGENTIC_NUM_TRAJECTORIES"] = str(mlperf_trajectories(envs))
        envs.setdefault("MLPERF_ENDPOINTS_DIR", os.environ.get("MLPERF_ENDPOINTS_DIR") or "/opt/mlperf-endpoints")
        envs.setdefault("MLPERF_AGENTIC_HARDWARE", os.environ.get("MLPERF_AGENTIC_HARDWARE") or "mi355x")
        # The round's CONC is what the measurement is recorded under. A baseline
        # YAML that already carries AGENTIC_CONCURRENCY must not freeze it.
        envs["CONC"] = str(conc or envs.get("CONC") or os.environ.get("CONC") or 16)
        pin_mlperf_round_concurrency(envs)
        from ._server_argv import add_server_arg_unless_pinned

        add_server_arg_unless_pinned(
            envs,
            framework,
            f"--served-model-name {MLPERF_SERVED_MODEL}",
            pinned_by=("served-model-name", "served_model_name"),
        )
    bench["workload_spec"] = build_agentx_workload_spec(
        bench,
        envs,
        model_path=model_path,
        env=_agentx_env,
        grading=grading,
    )


def prepare_agentx_runtime(
    *,
    env: dict[str, str] | None = None,
    inferencex_path: str | None = None,
    config_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    active: bool | None = None,
) -> str | None:
    """Deploy and preflight AgentX assets for baseline/profile runs."""
    from hyperloom.common.agentx_workload import is_mlperf_backend

    runtime_env = env or os.environ
    if not is_mlperf_backend(runtime_env):
        return None
    if active is None:
        active = agentx_enabled(runtime_env)
        if not active and config_path:
            try:
                materialized = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
                benchmark = materialized.get("benchmark") if isinstance(materialized, dict) else {}
                from hyperloom.common.agentx_workload import is_agentx_client_script

                active = isinstance(benchmark, dict) and is_agentx_client_script(
                    str(benchmark.get("benchmark_script") or "")
                )
            except (OSError, ValueError, TypeError):
                active = False
    if not active:
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
    package, so ``resolve_kernel_search_roots`` can only find it through
    ``<FRAMEWORK>_REPO_PATH`` / ``<FRAMEWORK>_DIR`` in ``os.environ``. Writing the
    resolved path into the materialized YAML reaches the *benchmark* subprocess
    but not the orchestrator — so without this the framework's own code is absent
    from the roots the specialist is pointed at and from the trees a patch can be
    grounded against, on a session that otherwise looks correctly configured.

    An operator-provided value always wins; this only fills the gap when the
    checkout was resolved (or is about to be cloned) by Hyperloom itself. The path
    is published even when it does not exist yet, because the benchmark wrapper
    clones on first use and the roots are recomputed per call.

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


#: Source-root envs Hyperloom writes into a vLLM launch; vLLM reports them as unknown.
VLLM_SOURCE_ROOT_ENVS: tuple[str, ...] = ("FRAMEWORK_REPO_PATH", "VLLM_REPO_PATH", "VLLM_DIR")


def _apply_vllm_source_runtime(bench: dict[str, Any], envs: dict[str, Any]) -> None:
    """Route a prepared image checkout into the vLLM server launch."""
    if str(bench.get("framework") or "").strip().lower() != "vllm":
        return
    if os.environ.get("HYPERLOOM_VLLM_IMAGE_SOURCE", "").strip() != "1":
        return
    repo_path = _resolve_framework_repo_path(envs, framework="vllm")
    if not repo_path:
        return
    for name in VLLM_SOURCE_ROOT_ENVS:
        envs[name] = repo_path
        os.environ[name] = repo_path
    existing = str(envs.get("PYTHONPATH") or os.environ.get("PYTHONPATH") or "")
    entries = [repo_path, *(part for part in existing.split(os.pathsep) if part)]
    envs["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))


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
    """Return the ``--extra-env`` pins the CLI serialized, or ``{}``."""
    from hyperloom.common.env_safety import operator_extra_env

    return {key: value for key, value in operator_extra_env().items() if key.strip().upper() != "RUN_EVAL"}


def resolve_reference_launch() -> tuple[str, dict[str, str], dict[str, Any]]:
    """Read the accepted reference recipe and its dedicated launch controls."""
    from hyperloom.inference_optimizer.session.paths import session_dir

    from ...state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir())
    return state.reference_server_args.strip(), dict(state.reference_envs), dict(state.reference_launch_controls)


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


#: Tokenizer modes the benchmark client actually implements a loader for.
#:
#: A tokenizer mode is a loader backend, not a model type: naming one the client
#: does not implement makes it reject the flag, which fails exactly the way the
#: unnamed tokenizer did. So "transformers cannot map this model_type" is
#: necessary but not sufficient -- the mode has to be one the client knows.
#: ``deepseek_v4`` is routed by InferenceX's ``benchmark_serving.py`` to vLLM's
#: own loader; other custom-code checkpoints (``kimi_k25``, say) are served by
#: the trust-remote-code path above and must NOT be named here.
_CLIENT_TOKENIZER_MODES: frozenset[str] = frozenset({"deepseek_v4"})


def _client_tokenizer_mode(model_path: str | None) -> str:
    """Return the tokenizer mode the benchmark client must be told, or ``""``.

    The generic bench client resolves a checkpoint through HF ``AutoConfig``. A
    model whose ``model_type`` this transformers build does not know dies there
    with ``KeyError: '<model_type>'`` before issuing a request, so no throughput
    result is written and the round is graded a boot failure with the server up
    and serving. InferenceX's client already routes ``--tokenizer-mode`` to
    vLLM's own loader, which does know it; this names the mode to pass.

    Model-agnostic on purpose: the question asked is "can HF resolve this
    model_type", not "is this DeepSeek-V4". A model HF understands returns ``""``
    and the client argv is unchanged.
    """
    model = str(model_path or "").strip()
    if not model:
        return ""
    data = _load_model_config_dict(model)
    if data is None:
        return ""
    model_type = str(data.get("model_type") or "").strip().lower()
    if model_type not in _CLIENT_TOKENIZER_MODES:
        return ""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError:
        # No transformers, no HF resolution to reason about; naming a mode here
        # would be a guess.
        return ""
    if model_type in CONFIG_MAPPING:
        return ""
    return model_type


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
    except (ImportError, RuntimeError):
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


def default_baseline_config() -> Path:
    """Resolve the shipped Magpie YAML for ``$FRAMEWORK``, or for the default framework when it is unset.

    Resolved at call time so ``$INFERENCE_OPTIMIZER_ASSET_ROOT`` is honoured.

    Returns:
        Path: The shipped Magpie YAML config path for the resolved framework.
    """
    from hyperloom.common.benchmark_config import benchmark_config_path

    source = benchmark_config_path()
    if source is not None:
        return source
    fw = os.environ.get("FRAMEWORK") or framework_registry.DEFAULT_FRAMEWORK
    return asset_root() / "assets" / "configs" / framework_registry.shipped_config_name("baseline", fw)


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
    (context-length/watchdog/attention/MoE/dedup/compact/shell-safe); order is fixed.

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
    # 4. MoE runner backend: left to sglang, whose ``--moe-runner-backend auto``
    #    (the default) follows SGLANG_USE_AITER on current sglang/ROCm images.
    #    A baseline retry can still ask to strip an inherited/pinned flag that
    #    crashed the server.
    if drop_moe_runner_backend and framework_env == "EXTRA_SGLANG_ARGS":
        resolved_server_args = _remove_moe_runner_backend_arg(resolved_server_args)
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


def prepare_grid_benchmark_config(config_path: Path, output_dir: Path, **overrides: Any) -> Path:
    """Keep the verified native baseline intact until the grid composes its candidate."""
    from .magpie_agentx import native_config

    if native_config(config_path):
        return config_path
    return materialize_config_with_envs(config_path, output_dir, **overrides)


def materialize_benchmark_config(config_path: Path, output_dir: Path, **overrides: Any) -> Path:
    """Resolve the workload owner before applying server candidate changes."""
    import yaml

    from hyperloom.common.benchmark_config import is_magpie_agentx
    from hyperloom.common.coerce import to_str_list
    from .magpie_agentx import LaunchChanges, apply_launch_changes, materialize_agentx, recipe_option_names

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not is_magpie_agentx(config["benchmark"]):
        return materialize_config_with_envs(config_path, output_dir, **overrides)
    replacing = overrides.get("args_mode") == "replace"
    inherited = "" if replacing else os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "")
    recipe_options = recipe_option_names(config_path)
    apply_launch_changes(config["benchmark"], LaunchChanges(args=inherited), recipe_options=recipe_options)
    unset_envs = to_str_list(overrides.get("unset_envs"))
    server_env, dropped = filter_untrusted_env_mapping(
        {
            key: value
            for key, value in _operator_extra_env().items()
            if key not in {"MODEL_PATH", "FRAMEWORK", "PRECISION", "TP", "EP", "CONC"}
            and key not in unset_envs
            and not key.startswith(("HYPERLOOM_", "INFERENCE_OPTIMIZER_"))
        },
        allow_predicate=is_allowed_variant_env_key,
    )
    for key in dropped:
        log.warning("Dropping extra_envs key %s before AgentX materialization", key)
    server_env.update(overrides.get("extra_envs") or {})
    if is_truthy(server_env.pop("RUN_EVAL", False)):
        raise ValueError("AgentX uses Magpie's replay validity; RUN_EVAL cannot enable GSM8K for this workload")
    changes = LaunchChanges(
        overrides.get("extra_server_args", ""),
        server_env,
        to_str_list(overrides.get("remove_args")),
        unset_envs,
        replacing,
    )
    apply_launch_changes(config["benchmark"], changes, recipe_options=recipe_options)
    return materialize_agentx(
        config,
        output_dir / overrides.get("out_name", "baseline_config.with_envs.yaml"),
        model_path=overrides.get("model_path"),
        inferencex_path=overrides.get("inferencex_path"),
    )


class _Workload(NamedTuple):
    """The benchmark being materialised, and where its model and session live."""

    bench: dict[str, Any]
    model_path: str


class _WorkloadShape(NamedTuple):
    """The request shape this run serves, and which kind of run it is."""

    isl_val: int
    osl_val: int
    conc_val: int
    is_profile: bool
    #: Scriptable image frameworks (xDiT diffusion) have no LLM decode
    #: steady-state window, so none of the serving-request math applies to them.
    is_scriptable: bool


def _apply_runner_type(bench: dict[str, Any], gpu_type: str) -> None:
    """Name the benchmark script the resolved GPU type calls for."""
    if gpu_type:
        bench["runner_type"] = gpu_runner_type(gpu_type)
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{bench['runner_type']}.sh"
        else:
            bench.pop("benchmark_script", None)


def _resolve_tensor_parallel(
    envs: dict[str, str],
    *,
    tp_from_env: Any,
    tp_from_yaml: Any,
    rocr_yaml: Any,
    rocr_devices: Any,
) -> int:
    """Settle how many GPUs this run is told to use, and say so in the envs."""
    resolved_tp = 0
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
    return resolved_tp


class _TracelensPatch(NamedTuple):
    """How a profile obtained its TraceLens instrumentation, if at all."""

    sglang_sitecustomize: bool
    applied: bool


def _patch_sglang_for_graph_capture() -> None:
    """Apply the SGLang graph-capture patch a sitecustomize profile still needs."""
    if ensure_sglang_patched_for_tracelens():
        return
    log.warning(
        "SGLang graph-capture patch (sglang_gc_patch) not applied; profiling "
        "continues on the unpatched server, where per-batch-size CUDA-graph "
        "capture can IndexError on multi-variant models (DSA dense+sparse). "
        "It needs TRACELENS_ROOT with an exact "
        "examples/custom_workflows/inference_analysis/sglang_gc_patch/sglang_<X_Y_Z> dir."
    )


def _record_tracelens_patch_status(envs: dict[str, str], framework: str) -> bool:
    """Patch the framework for TraceLens and say in the envs whether it took."""
    applied = ensure_vllm_patched_for_tracelens() if "vllm" in framework else ensure_sglang_patched_for_tracelens()
    envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "ok" if applied else "unavailable"
    if applied:
        return True
    envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] = _TRACELENS_PATCH_UNAVAILABLE
    log.warning(
        "TraceLens runtime patch unavailable for framework=%s; "
        "profile will omit annotation-only flags and roofline "
        "analysis may be degraded.",
        framework or "<unset>",
    )
    return False


def _settle_tracelens_patch(envs: dict[str, str], framework: str) -> _TracelensPatch:
    """Settle how this profile is instrumented, and record that in the envs."""
    from ._server_patcher import resolve_sglang_shape_mode

    sitecustomize = "sglang" in framework and resolve_sglang_shape_mode() == "sitecustomize"
    attempted = _tracelens_patch_enabled() and "atom" not in framework and not sitecustomize
    envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "not_attempted"
    if attempted:
        return _TracelensPatch(
            sglang_sitecustomize=sitecustomize,
            applied=_record_tracelens_patch_status(envs, framework),
        )
    if sitecustomize and _tracelens_patch_enabled():
        _patch_sglang_for_graph_capture()
    return _TracelensPatch(sglang_sitecustomize=sitecustomize, applied=False)


def _shape_discovery_enabled(bench: dict[str, Any]) -> bool:
    """Whether this profile may ask the server to record kernel shapes."""
    forced = env_bool("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE")
    if not env_flag("HYPERLOOM_PROFILE_SHAPE_DISCOVERY", default=True):
        return False
    if forced:
        return True
    model = str(bench.get("model") or "")
    if not _model_is_gemma2(model):
        return True
    log.info(
        "Gemma2 roofline: disabling shape-discovery to avoid "
        "CUDA-graph capture crash (hipErrorStreamCapture"
        "Unsupported); CUDA graph + profiling kept. Set "
        "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 to override.",
    )
    if _load_model_config_dict(model) is None:
        log.warning(
            "Gemma2 detected via path heuristic (no readable "
            "config.json at %r); shape-discovery skip may be "
            "imprecise.",
            model,
        )
    return False


def _apply_profile_config(envs: dict[str, str], bench: dict[str, Any], shape: _WorkloadShape) -> _ProfileConfig:
    """Shape a non-scriptable profile run, and report what it capped."""
    config = _ProfileConfig(osl_val=shape.osl_val, conc_val=shape.conc_val)
    if not shape.is_profile or shape.is_scriptable:
        return config
    window = _fit_profile_capture_window(envs, osl_val=shape.osl_val, conc_val=shape.conc_val)
    # The profile server runs at the resolved profile OSL, decoupled from --osl.
    envs["OSL"] = window.osl_val
    framework = str(bench.get("framework") or "").lower()
    target = profile_target(framework)
    patch = _settle_tracelens_patch(envs, framework)
    return _shape_profile_trace(
        envs,
        config._replace(osl_val=window.osl_val, profile_num_prompts=window.num_prompts),
        _ProfileKnobs(
            target=target,
            delay_iters=window.delay_iters,
            max_iters=window.max_iters,
            sglang_sitecustomize=patch.sglang_sitecustomize,
            tracelens_patch_ok=patch.applied,
            shape_discovery=target is _ProfileTarget.REQUEST_BODY and _shape_discovery_enabled(bench),
        ),
    )


def _drop_redundant_graph_capture_flag(envs: dict[str, str], framework_env: str) -> None:
    """Drop SGLang's graph-capture profiling flag from a run that is eager anyway."""
    if framework_env != "EXTRA_SGLANG_ARGS":
        return
    tokens = str(envs.get(framework_env, "")).split()
    if _SGLANG_DISABLE_CUDA_GRAPH_FLAG not in tokens or _SGLANG_PROFILE_CUDA_GRAPH_FLAG not in tokens:
        return
    envs[framework_env] = " ".join(t for t in tokens if t != _SGLANG_PROFILE_CUDA_GRAPH_FLAG)
    log.info(
        "Dropping %s: %s is set, so there is no graph capture to profile.",
        _SGLANG_PROFILE_CUDA_GRAPH_FLAG,
        _SGLANG_DISABLE_CUDA_GRAPH_FLAG,
    )


def _publish_quality_reference(
    envs: dict[str, str],
    bench: dict[str, Any],
    *,
    is_profile: bool,
    establish_quality_ref: bool,
) -> None:
    """Name the image a scriptable workload is graded against, and who may write it."""
    if not framework_registry.is_scriptable(bench.get("framework")):
        return
    reference = _first_env(_QUALITY_REF_ENVS)
    if not reference:
        from hyperloom.inference_optimizer.session.paths import session_dir

        reference = str(session_dir() / "storage" / "quality_ref" / "baseline.png")
    if is_profile:
        compared, written = "", ""
    elif establish_quality_ref:
        compared, written = "", (_first_env(_QUALITY_REF_WRITE_ENVS) or reference)
    else:
        compared, written = reference, ""
    for name in _QUALITY_REF_ENVS:
        envs[name] = compared
    for name in _QUALITY_REF_WRITE_ENVS:
        envs[name] = written


def _publish_xdit_model_resolution(
    envs: dict[str, str],
    bench: dict[str, Any],
    *,
    establish_quality_ref: bool,
) -> None:
    """Pin how the xDiT runner resolves its model, and which backend serves the baseline."""
    if str(bench.get("framework") or "").strip().lower() != "xdit":
        return
    # The runner resolves models through MODEL_REGISTRY keys: "name" is
    # registry-correct, "path" fails lookup. Forcing it onto benchmark.envs keeps
    # a per-task override from breaking model resolution.
    envs["XDIT_MODEL_ARG"] = os.environ.get("XDIT_MODEL_ARG", "").strip() or "name"
    # The baked hyperloom_local_aliases map each registered name to a local
    # snapshot dir rooted here. Left unset in public deployments so the operator
    # chooses the model cache location explicitly.
    model_root = os.environ.get("XDIT_MODEL_ROOT", "").strip()
    if model_root:
        envs["XDIT_MODEL_ROOT"] = model_root
    # The baseline forces the operator-pinned backend (default 'aiter', the
    # MI300X-verified path) so an invalid agent override cannot poison the
    # reference measurement. Explore/sweep variants keep their freedom.
    if establish_quality_ref:
        envs["XDIT_ATTENTION_BACKEND"] = os.environ.get("XDIT_ATTENTION_BACKEND", "").strip() or "aiter"


def _pin_sparse_kv_block_size(envs: dict[str, str], workload: _Workload) -> None:
    """Pin the KV-cache block size a sparse-attention model can actually allocate."""
    if "vllm" not in str(workload.bench.get("framework") or "").lower():
        return
    block_size = _sparse_kv_block_size(str(workload.model_path or workload.bench.get("model") or ""))
    if not block_size:
        return
    add_server_arg_unless_pinned(
        envs,
        workload.bench.get("framework"),
        f"--block-size {block_size}",
        pinned_by=("block-size", "block_size"),
    )


def _pin_client_model_access(envs: dict[str, str], workload: _Workload) -> None:
    """Tell the benchmark client how to load this model."""
    for trust_key in (
        "MAGPIE_TRUST_REMOTE_CODE",  # Magpie sglang remote-direct client
        "BENCH_TRUST_REMOTE_CODE",  # GEAK bench_e2e.sh inferencex client
        "HF_HUB_TRUST_REMOTE_CODE",  # transformers / HF hub tokenizer auto-load
    ):
        envs.setdefault(trust_key, "1")
    tokenizer_mode = _client_tokenizer_mode(workload.model_path or workload.bench.get("model"))
    if tokenizer_mode:
        envs.setdefault("HYPERLOOM_CLIENT_TOKENIZER_MODE", tokenizer_mode)


def _pin_fp8_per_token_fast_path(envs: dict[str, str], workload: _Workload) -> None:
    """Route a per-channel, per-token FP8 checkpoint to aiter's fused GEMM."""
    from hyperloom.inference_optimizer.gpu_types import _resolve_amd_gpu_type

    if (
        "sglang" in str(workload.bench.get("framework") or "").lower()
        and str(workload.bench.get("precision") or "").strip().lower() == "fp8"
        and _resolve_amd_gpu_type(workload.bench.get("runner_type")) in _GFX942_GPU_TYPES
        and _fp8_is_per_channel_per_token(str(workload.model_path or os.environ.get("MODEL_PATH", "")))
    ):
        envs.setdefault("SGLANG_USE_AITER_FP8_PER_TOKEN", "1")


def _apply_framework_envs(
    envs: dict[str, str],
    workload: _Workload,
    *,
    framework_env: str,
    is_profile: bool,
    establish_quality_ref: bool,
) -> str:
    """Apply the env this framework and this model need, and name the model."""
    _drop_redundant_graph_capture_flag(envs, framework_env)
    _publish_quality_reference(
        envs,
        workload.bench,
        is_profile=is_profile,
        establish_quality_ref=establish_quality_ref,
    )
    _publish_xdit_model_resolution(envs, workload.bench, establish_quality_ref=establish_quality_ref)
    model_basename = Path(str(workload.model_path or os.environ.get("MODEL_PATH", ""))).name.lower()
    pin_model_launch_workarounds(envs, workload.bench, model_basename)
    _pin_sparse_kv_block_size(envs, workload)
    return model_basename


def _tokenized_requests_env() -> str:
    """Whether the eval harness should send token ids rather than text."""
    tokenized = os.environ.get("MAGPIE_EVAL_TOKENIZED_REQUESTS", "").strip()
    if not tokenized:
        try:
            from ._multi_node_env import resolve_kb_topology

            if str(resolve_kb_topology().get("pd_mode") or "").lower() == "disaggregated":
                tokenized = "false"
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
            tokenized = ""
    return tokenized


def _apply_eval_envs(
    envs: dict[str, str],
    workload: _Workload,
    *,
    model_basename: str,
    flydsl_source_dirs: Any,
) -> None:
    """Settle what the run evaluates, and where the harness reads it from."""
    if _model_requires_remote_code(workload.model_path or workload.bench.get("model")):
        add_server_arg_unless_pinned(
            envs,
            workload.bench.get("framework"),
            "--trust-remote-code",
            pinned_by=("trust-remote-code",),
        )
    # Server-side trust-remote-code for custom-code models (Qwen3.6 MoE): the
    # SERVER must also load remote code or it refuses the arch at boot. Scoped
    # to this family.
    if "qwen3.6-35b-a3b" in model_basename or "qwen3-6-35b-a3b" in model_basename:
        add_server_arg_unless_pinned(
            envs,
            workload.bench.get("framework"),
            "--trust-remote-code",
            pinned_by=("trust-remote-code",),
        )
    # Accuracy eval (GSM8K) is ON unless the round passes ``extra_envs["RUN_EVAL"]`` or the workload is AgentX.
    if "RUN_EVAL" not in envs:
        envs["RUN_EVAL"] = "true"
    # Persist eval task/limit into the YAML so they become part of the
    # fingerprint-able contract and bypass_runner reads them from a stable source.
    _eval_tasks_env = os.environ.get("MAGPIE_EVAL_TASKS", "").strip()
    if _eval_tasks_env and "MAGPIE_EVAL_TASKS" not in envs:
        envs["MAGPIE_EVAL_TASKS"] = _eval_tasks_env
    _eval_limit_env = os.environ.get("MAGPIE_EVAL_LIMIT", "").strip()
    if _eval_limit_env and "MAGPIE_EVAL_LIMIT" not in envs:
        envs["MAGPIE_EVAL_LIMIT"] = _eval_limit_env
    _eval_tok_env = _tokenized_requests_env()
    if _eval_tok_env and "MAGPIE_EVAL_TOKENIZED_REQUESTS" not in envs:
        envs["MAGPIE_EVAL_TOKENIZED_REQUESTS"] = _eval_tok_env
    # FlyDSL folds only same-directory helpers into its JIT cache key, so a patched
    # helper one directory over is served from a stale binary. Naming the roots
    # folds their sources into the key. Only the run that applied such a patch has
    # that hazard; setdefault so an operator-set value (YAML / extra_envs) wins.
    if flydsl_source_dirs:
        _flydsl_dirs = flydsl_extra_source_dirs()
        if _flydsl_dirs:
            envs.setdefault(ENV_FLYDSL_EXTRA_SOURCE_DIRS, _flydsl_dirs)


class _LaunchIdentity(NamedTuple):
    """What this run launches: the model, the GPU, the entrypoint, the checkout and the client.

    ``agentx_mode`` picks the client: the AgentX switch replaces the benchmark
    script, so it settles the entrypoint alongside ``benchmark_script``.
    ``grading`` is the session's own, and settles the axis and band the published
    AgentX workload spec hands to GEAK.
    """

    model_path: str | None
    gpu_type: str | None
    benchmark_script: str | None
    inferencex_path: str | None
    agentx_mode: bool | None
    grading: Mapping[str, Any] | None


class _ArgOverrides(NamedTuple):
    """What the caller asked to add to, and take out of, the inherited launch."""

    extra_server_args: str
    args_mode: str
    extra_envs: dict[str, Any] | None
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None

    @property
    def replacing(self) -> bool:
        """Whether the caller's args replace the inherited string instead of merging."""
        return str(self.args_mode or "append").strip().lower() == "replace"


class _ReferenceLaunch(NamedTuple):
    """The reference recipe's envs and launch controls, as the later merges read them."""

    envs: dict[str, str]
    controls: dict[str, Any]


class _ComposedArgs(NamedTuple):
    """Where the composed launch ended up, and what the caller asked of it."""

    framework_env: str
    reference: _ReferenceLaunch
    extra_envs: dict[str, str]
    overrides: _ArgOverrides


def _read_benchmark_config(config_path: Path) -> dict[str, Any]:
    """Read the source Magpie YAML, with the benchmark section it renders present."""
    with config_path.open(encoding="utf-8") as f:
        document = yaml.safe_load(f) or {}
    document.setdefault("benchmark", {})
    return document


def _write_benchmark_config(document: dict[str, Any], materialized: Path) -> Path:
    """Write the rendered YAML where the launch will open it."""
    materialized.parent.mkdir(parents=True, exist_ok=True)
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(document, f, sort_keys=False)
    return materialized


def _refuse_framework_script_mismatch(bench: dict[str, Any]) -> None:
    """Refuse to boot an entrypoint that belongs to another framework."""
    script = str(bench.get("benchmark_script") or "").lower()
    framework = str(bench.get("framework") or "").lower()
    known = framework_registry.names()
    if not script or framework not in known:
        return
    other = [name for name in known if name != framework and script.startswith(f"{name}_")]
    if other:
        raise FrameworkScriptMismatchError(
            f"framework/script mismatch: framework={framework!r} but "
            f"benchmark_script={script!r} targets {other[0]!r}; refusing "
            f"to boot server (would launch the wrong framework's entrypoint)"
        )


def _pin_inferencex_checkout(bench: dict[str, Any], inferencex_path: str | None) -> None:
    """Persist the InferenceX checkout Magpie must use, so it matches the patch target."""
    checkout = str(inferencex_path or "").strip() or os.environ.get("INFERENCEX_PATH", "").strip()
    if checkout:
        bench["inferencex_path"] = checkout


def _adopt_recipe_envs(bench: dict[str, Any], agentx_mode: bool | None) -> dict[str, Any]:
    """Adopt the recipe's env block, dropping any accuracy-eval opt-out it carries."""
    keep_run_eval = agentx_enabled() if agentx_mode is None else agentx_mode
    envs = {key: value for key, value in (bench.get("envs") or {}).items() if keep_run_eval or key != "RUN_EVAL"}
    bench["envs"] = envs
    return envs


def _settle_benchmark_identity(bench: dict[str, Any], identity: _LaunchIdentity) -> None:
    """Settle what this run launches: its model, GPU, entrypoint and checkouts."""
    if identity.model_path:
        bench["model"] = str(identity.model_path)
    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision
    _apply_runner_type(bench, identity.gpu_type)
    if identity.benchmark_script:
        bench["benchmark_script"] = str(identity.benchmark_script)
    envs = _adopt_recipe_envs(bench, identity.agentx_mode)
    runtime_path = os.environ.get("PATH", "").strip()
    if "PATH" in envs and runtime_path:
        envs["PATH"] = runtime_path
    apply_scriptable_runtime_defaults(
        bench,
        envs,
        gpu_type=identity.gpu_type,
        explicit_benchmark_script=bool(identity.benchmark_script),
    )
    apply_agentx_switch(bench, identity.model_path, active=identity.agentx_mode, grading=identity.grading)
    _refuse_framework_script_mismatch(bench)
    _pin_inferencex_checkout(bench, identity.inferencex_path)


def _pin_workload_env_values(envs: dict[str, Any]) -> None:
    """Project the operator's workload pins out of the process env into the benchmark."""
    for env_key in ("CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "EP", "PORT"):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = _coerce_workload_int_env(env_key, val)
    # RANDOM_RANGE_RATIO is a float feeding the steady-state formulas; do NOT
    # coerce it to int.
    r_env = os.environ.get("RANDOM_RANGE_RATIO", "").strip()
    if r_env:
        envs["RANDOM_RANGE_RATIO"] = float(r_env)
    rocr_env = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if rocr_env:
        envs["ROCR_VISIBLE_DEVICES"] = rocr_env


def _pin_visible_devices(envs: dict[str, Any]) -> None:
    """Reconcile the visible-device list with the parallelism the run was given."""
    rocr_yaml = str(envs.get("ROCR_VISIBLE_DEVICES") or "").strip()
    rocr_devices = [d.strip() for d in rocr_yaml.split(",") if d.strip()]
    resolved_tp = _resolve_tensor_parallel(
        envs,
        tp_from_env=os.environ.get("TP", "").strip(),
        tp_from_yaml=envs.get("TP"),
        rocr_yaml=rocr_yaml,
        rocr_devices=rocr_devices,
    )
    envs["TP"] = resolved_tp
    if rocr_yaml and len(rocr_devices) >= resolved_tp:
        return
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


def _is_profile_run(envs: Mapping[str, Any], bench: Mapping[str, Any]) -> bool:
    """Whether this materialization renders a profiling run."""
    if str(envs.get("PROFILE", "")).strip() == "1":
        return True
    return bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True


def _resolve_workload_shape(envs: dict[str, Any], bench: dict[str, Any]) -> _WorkloadShape:
    """Settle the request shape this run serves and the devices it may use."""
    _pin_workload_env_values(envs)
    _pin_visible_devices(envs)
    default_isl, default_osl, default_conc = cli_workload_defaults()
    return _WorkloadShape(
        isl_val=int(envs.get("ISL") or default_isl),
        osl_val=int(envs.get("OSL") or default_osl),
        conc_val=int(envs.get("CONC") or default_conc),
        is_profile=_is_profile_run(envs, bench),
        is_scriptable=framework_registry.is_scriptable(bench.get("framework")),
    )


def _prompt_batch_factor(isl_val: int, osl_val: int) -> int:
    """Requests per concurrent slot a benchmark of this sequence cost should run."""
    seq_cost = isl_val + osl_val
    if seq_cost <= 1024:
        return 10
    if seq_cost <= 4096:
        return 5
    if seq_cost <= 16384:
        return 3
    return 2


def _pin_request_counts(envs: dict[str, Any], shape: _WorkloadShape, profile: _ProfileConfig) -> None:
    """Size the request stream: a profile to reach its window, a benchmark to the load."""
    if shape.is_scriptable:
        return
    if profile.profile_num_prompts is not None:
        envs["NUM_PROMPTS"] = profile.profile_num_prompts
    elif "NUM_PROMPTS" not in envs:
        envs["NUM_PROMPTS"] = max(shape.conc_val * _prompt_batch_factor(shape.isl_val, shape.osl_val), shape.conc_val)
    envs.setdefault("NUM_WARMUPS", min(shape.conc_val, 8))


def _seed_reference_server_args(
    envs: dict[str, Any],
    bench: dict[str, Any],
    ref_args: str,
    controls: Mapping[str, Any],
) -> None:
    """Seed the framework argument env from the reference recipe."""
    from hyperloom.inference_optimizer.grid_server_args import compose_server_args

    reference_env = server_args_env_name(bench.get("framework"))
    replacing = controls.get("args_mode") == "replace"
    envs[reference_env] = compose_server_args(
        base_extra_args=ref_args,
        variant_extra_args="" if replacing else str(envs.get(reference_env, "")),
        remove_args=controls.get("remove_args"),
        args_mode="replace",
    )


def _apply_reference_launch(envs: dict[str, Any], bench: dict[str, Any]) -> _ReferenceLaunch:
    """Seed this launch from the session's accepted reference recipe."""
    ref_args, reference_envs, controls = resolve_reference_launch()
    for name in controls.get("unset_envs", []):
        envs.pop(name, None)
    if ref_args or controls.get("remove_args") or controls.get("args_mode") == "replace":
        _seed_reference_server_args(envs, bench, ref_args, controls)
    for key, value in reference_envs.items():
        envs.setdefault(str(key), str(value))  # never clobber YAML/CLI envs
    if controls.get("overlay_pythonpath"):
        from hyperloom.common.overlay import validate_overlay_pythonpath

        prepend_launch_path(envs, "PYTHONPATH", validate_overlay_pythonpath(controls["overlay_pythonpath"]))
    return _ReferenceLaunch(envs=reference_envs, controls=controls)


def _merge_caller_server_args(envs: dict[str, Any], bench: dict[str, Any], overrides: _ArgOverrides) -> None:
    """Merge the caller's and the operator's server args into the framework env."""
    server_args = str(overrides.extra_server_args or "").strip()
    operator_server_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
    if operator_server_args and not overrides.replacing:
        server_args = merge_server_args(operator_server_args, server_args)
    if not server_args:
        return
    framework_env = server_args_env_name(bench.get("framework"))
    if overrides.replacing:
        envs[framework_env] = server_args
        return
    existing = str(envs.get(framework_env, "")).strip()
    if framework_env == "EXTRA_SGLANG_ARGS" and "--moe-runner-backend" in server_args:
        # For MoE backend exploration/tuning the candidate value must replace the
        # baseline's injected default (usually triton) rather than relying on
        # duplicate last-wins flags.
        existing = _remove_moe_runner_backend_arg(existing)
    envs[framework_env] = merge_server_args(existing, server_args)


def _merge_caller_envs(
    envs: dict[str, Any],
    bench: dict[str, Any],
    overrides: _ArgOverrides,
    recipe_overwritten: frozenset[str],
) -> dict[str, str]:
    """Copy the operator's and the caller's extra envs into the benchmark envs."""
    from hyperloom.common.agentx_workload import is_agentx_client_script

    combined: dict[str, Any] = dict(_operator_extra_env())
    if is_agentx_client_script(str(bench.get("benchmark_script") or "")):
        # The AgentX switch already settled these from the same pins; the raw
        # pin would undo its CONC scaling of the warmup grace.
        agentx_env = agentx_env_for_conc()
        combined = {k: v for k, v in combined.items() if not _is_agentx_client_key(k, agentx_env)}
    if overrides.extra_envs:
        combined.update(overrides.extra_envs)
    safe_envs, dropped = filter_untrusted_env_mapping(
        combined,
        # A name the recipe re-exports unconditionally cannot be overridden
        # here; carrying it into the YAML publishes a value the run never used.
        allow_predicate=lambda key: is_allowed_variant_env_key(key) and key not in recipe_overwritten,
    )
    for name in dropped:
        log.warning("Dropping extra_envs key %s before benchmark materialization", name)
    for key, value in safe_envs.items():
        envs[str(key)] = str(value)
    return safe_envs


def _compose_server_args(envs: dict[str, Any], bench: dict[str, Any], overrides: _ArgOverrides) -> _ComposedArgs:
    """Compose the launch from every source of arguments and envs that feeds it."""
    reference = _apply_reference_launch(envs, bench)
    recipe_overwritten = launcher_overwritten_envs(bench)
    _merge_caller_server_args(envs, bench, overrides)
    extra_envs = _merge_caller_envs(envs, bench, overrides, recipe_overwritten)
    # aiter logs a line for every tuned-GEMM table lookup it MISSES
    # unconditionally, but the corresponding HIT line only when this is set. Two
    # things depend on having it on: GEMM tuning takes its shape list from the
    # misses (shapes derived from config.json instead cover 0.4% of what the
    # runtime actually asks for), and apply verification counts hits to decide
    # whether a tuned artifact was ever read -- without hit lines, "0 hits" and
    # "hit logging was off" are indistinguishable, and treating the latter as the
    # former would REVERT every arm. A scan of 60 production server logs found the
    # flag set in none of them, so this is not a hypothetical gap.
    envs.setdefault("AITER_LOG_TUNED_CONFIG", "1")
    _sync_repo_aliases(
        bench,
        envs,
        framework="custom",
        prefer_dir="CUSTOM_DIR" in extra_envs and "CUSTOM_REPO_PATH" not in extra_envs,
    )
    framework_env = server_args_env_name(bench.get("framework"))
    # Final dedup after the reference and caller merges: collapse repeated
    # vLLM/atom single-value flags to last-wins so recipe and variant values
    # override earlier ones (no-op for sglang).
    merged_args = str(envs.get(framework_env, "")).strip()
    if merged_args:
        envs[framework_env] = dedup_vllm_server_args(merged_args, bench.get("framework"))
    return _ComposedArgs(
        framework_env=framework_env,
        reference=reference,
        extra_envs=extra_envs,
        overrides=overrides,
    )


def _apply_caller_removals(envs: dict[str, Any], composed: _ComposedArgs) -> None:
    """Take back out of the launch what the caller asked this run not to carry."""
    for key in composed.reference.controls.get("unset_envs", []):
        if key not in composed.reference.envs and key not in composed.extra_envs:
            envs.pop(key, None)
    framework_env = composed.framework_env
    remove_list = to_str_list(composed.overrides.remove_args)
    if remove_list:
        envs[framework_env] = remove_server_args(envs.get(framework_env, ""), remove_list)
    for key in to_str_list(composed.overrides.unset_envs):
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(key).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("Refusing to unset pinned benchmark env %s", key)
            continue
        envs.pop(str(key), None)


def _restore_vllm_profiler_bounds(envs: dict[str, Any], composed: _ComposedArgs, profile: _ProfileConfig) -> None:
    """Put the profile's iteration bounds back after a later merge dropped them."""
    framework_env = composed.framework_env
    profile_args = str(envs.get(framework_env, "")).strip()
    restored = [
        flag
        for name, flag in profile.pending_vllm_profiler_flags
        if not _profiler_bound_holds(
            name,
            _profiler_flag_value(profile_args, name),
            cap=profile.pending_vllm_profiler_cap,
        )
    ]
    if not restored:
        return
    log.warning(
        "profile server args lost torch-profiler flags %r (replace_args=%s); "
        "restoring them so the profiler cannot run unbounded.",
        restored,
        bool(composed.overrides.replacing),
    )
    # The seal applies the sink-side guard to whatever is left here.
    envs[framework_env] = merge_server_args(profile_args, " ".join(restored))


def _apply_atom_tracelens_knobs(envs: dict[str, Any], caps: _AtomTracelensCaps) -> None:
    """Turn on the TraceLens knobs the installed ATOM actually supports."""
    if caps.detailed_annotation:
        envs["ATOM_ENABLE_DETAILED_ANNOTATION"] = "1"
    if caps.profiler_more:
        envs["ATOM_PROFILER_MORE"] = "1"
    if not caps.mark_trace:
        return
    extra = str(envs.get("EXTRA_ATOM_ARGS", "")).strip()
    if "--mark-trace" not in extra:
        envs["EXTRA_ATOM_ARGS"] = f"{extra} --mark-trace".strip()


def _restate_profile_capture(envs: dict[str, Any], composed: _ComposedArgs, profile: _ProfileConfig) -> None:
    """Give the profile's capture knobs the last word over every later merge."""
    if profile.pending_vllm_profiler_flags and composed.framework_env == "EXTRA_VLLM_ARGS":
        _restore_vllm_profiler_bounds(envs, composed, profile)
    if profile.atom_caps is not None:
        _apply_atom_tracelens_knobs(envs, profile.atom_caps)


def _drop_benchmark_credentials(envs: dict[str, Any]) -> None:
    """Keep control-plane credentials out of the rendered YAML, which is persisted."""
    filtered_envs, dropped = filter_untrusted_env_mapping(
        envs,
        allow_predicate=lambda key: key not in BENCHMARK_SECRET_ENV_NAMES,
    )
    if not dropped:
        return
    log.warning(
        "Dropping control-plane credentials from benchmark envs: %s",
        ", ".join(sorted(dropped)),
    )
    envs.clear()
    envs.update(filtered_envs)


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
    grading: Mapping[str, Any] | None = None,
) -> Path:
    """Render a per-run Magpie YAML with caller-provided overrides.

    The process env wins over the YAML defaults, and the body is the order the
    phases run in: what the run launches, the request shape it serves, the
    profile shaping, the server arguments composed from every source that feeds
    them, the per-framework and per-model envs, and the caller's own removals --
    after which the argument string is sealed and nothing writes it again.

    Args:
        config_path: Path to the source Magpie YAML to render.
        output_dir: Directory the materialized YAML is written into.
        extra_server_args: Extra framework server args merged into the env.
        extra_envs: Overrides applied last over any computed env values;
            shell/loader hijack names and credentials are dropped.
        remove_args: Inherited framework server args to remove before launch.
        unset_envs: Env names to remove after applying ``extra_envs``;
            workload pins are refused.
        args_mode: ``"append"`` (default) or ``"replace"`` for
            ``extra_server_args``.
        model_path: Model path/id; overrides ``benchmark.model`` when set.
        gpu_type: Product or runner type; resolves ``runner_type`` and pins the generic script.
        inferencex_path: Explicit InferenceX checkout to pin into the YAML.
        benchmark_script: Pre-sanitized benchmark script name to re-pin.
        out_name: File name for the materialized YAML.
        establish_quality_ref: When True (baseline only) the scriptable
            image-quality reference is ESTABLISHED (written) by this run;
            otherwise the run only COMPARES against it.
        drop_moe_runner_backend: When True, skip the AMD MoE
            ``--moe-runner-backend`` injection and strip the flag from the
            merged args whatever source it came from (the one-shot fallback
            after a launch failure blamed on that backend).
        flydsl_source_dirs: When True, name the FlyDSL source roots in
            ``$FLYDSL_EXTRA_SOURCE_DIRS`` so a patched helper invalidates the JIT
            cache key. Off by default: only a run that applied such a patch needs it.
        agentx_mode: Explicit session-level AgentX decision. ``None`` preserves
            the legacy environment-based fallback.
        grading: The session's ``SharedState.grading``, which settles the axis
            and noise band the AgentX ``workload_spec`` publishes to GEAK.
            ``None`` preserves the environment-derived fallback.

    Returns:
        The materialized YAML path (stable file name across calls).

    Raises:
        FrameworkScriptMismatchError: If ``benchmark_script`` targets a
            different known framework than the run's framework.
    """
    document = _read_benchmark_config(config_path)
    bench = document["benchmark"]
    identity = _LaunchIdentity(
        model_path=model_path,
        gpu_type=gpu_type,
        benchmark_script=benchmark_script,
        inferencex_path=inferencex_path,
        agentx_mode=agentx_mode,
        grading=grading,
    )
    _settle_benchmark_identity(bench, identity)
    envs = bench["envs"]
    shape = _resolve_workload_shape(envs, bench)
    profile = _apply_profile_config(envs, bench, shape)
    _pin_request_counts(envs, shape, profile)
    composed = _compose_server_args(
        envs,
        bench,
        _ArgOverrides(
            extra_server_args=extra_server_args,
            args_mode=args_mode,
            extra_envs=extra_envs,
            remove_args=remove_args,
            unset_envs=unset_envs,
        ),
    )
    workload = _Workload(bench=bench, model_path=model_path)
    model_basename = _apply_framework_envs(
        envs,
        workload,
        framework_env=composed.framework_env,
        is_profile=shape.is_profile,
        establish_quality_ref=establish_quality_ref,
    )
    _finalize_framework_server_args(
        envs,
        bench,
        gpu_type=gpu_type,
        isl_val=shape.isl_val,
        osl_val=profile.osl_val,
        drop_moe_runner_backend=drop_moe_runner_backend,
    )
    _pin_client_model_access(envs, workload)
    _apply_eval_envs(envs, workload, model_basename=model_basename, flydsl_source_dirs=flydsl_source_dirs)
    _pin_fp8_per_token_fast_path(envs, workload)
    _apply_caller_removals(envs, composed)
    _restate_profile_capture(envs, composed, profile)
    _apply_vllm_source_runtime(bench, envs)
    _drop_benchmark_credentials(envs)
    seal_server_argv(envs, bench.get("framework"))
    return _write_benchmark_config(document, output_dir / out_name)


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
