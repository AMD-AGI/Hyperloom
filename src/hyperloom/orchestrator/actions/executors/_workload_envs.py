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
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import yaml

from hyperloom.common.agentx_workload import pin_mlperf_round_concurrency
from hyperloom.common.coerce import to_str_list
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from hyperloom.common.gpu_identity import AMD_GPU_BOARDS
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
from hyperloom.inference_optimizer.grid_server_args import (
    WorkloadTokenWindow,
    compact_json_server_args,
    dedup_vllm_server_args,
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    inject_sglang_watchdog_timeout,
)
from hyperloom.inference_optimizer.grid_server_args import merge_server_args
from hyperloom.inference_optimizer.grid_server_args import remove_moe_runner_backend_arg
from hyperloom.inference_optimizer.grid_server_args import remove_server_args
from hyperloom.inference_optimizer.grid_server_args import validate_server_args_shell_safe
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from ._recipe_script import launcher_overwritten_envs
from ._server_argv import add_server_arg_unless_pinned, seal_server_argv
from ._profile_setup import _AtomTracelensCaps, _profiler_bound_holds, _profiler_flag_value, apply_atom_profile_caps
from ._profile_setup import fit_capture_to_speculative
from hyperloom.inference_optimizer.model_config_utils import (
    _fp8_is_per_channel_per_token,
    load_model_config_dict,
    _sparse_kv_block_size,
)

log = logging.getLogger(__name__)

_XDIT_EXTRA_ARGS_ENV = framework_registry.extra_args_env("xdit")

# The aiter CK gemm_a8w8_bpreshuffle kernel ships for gfx942 / CDNA3 only; gfx950
# has no such kernel. The gate is the arch, so the board list is read off the
# board table rather than typed out beside it.
_GFX942_GPU_TYPES = frozenset(product for product, board in AMD_GPU_BOARDS.items() if board.gfx == "gfx942")


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
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as exc:
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


def _model_requires_remote_code(model_path: str | None) -> bool:
    """Return whether benchmark server/client must trust custom HF code.

    Fires for any local config that advertises a custom AutoTokenizer (e.g.
    Kimi K2.6, ``model_type=kimi_k25``), so custom-code tokenizers do not need
    per-model special cases.
    """
    model = str(model_path or "").strip()
    if not model:
        return False
    data = load_model_config_dict(model)
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
    data = load_model_config_dict(model)
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
    bench: dict[str, Any], window: _RequestWindow, *, drop_moe_runner_backend: bool = False
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
    envs = bench["envs"]
    framework_env = server_args_env_name(bench.get("framework"))
    resolved_server_args = str(envs.get(framework_env, "")).strip()
    resolved_server_args = inject_sglang_context_length(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        WorkloadTokenWindow(window.isl, window.osl, envs.get("MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN")),
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
    )
    # 4. MoE runner backend: left to sglang, whose ``--moe-runner-backend auto``
    #    (the default) follows SGLANG_USE_AITER on current sglang/ROCm images.
    #    A baseline retry can still ask to strip an inherited/pinned flag that
    #    crashed the server.
    if drop_moe_runner_backend and framework_env == "EXTRA_SGLANG_ARGS":
        resolved_server_args = remove_moe_runner_backend_arg(resolved_server_args)
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
    #    word. No-op for arg strings with no JSON.
    resolved_server_args = compact_json_server_args(resolved_server_args)
    resolved_server_args = validate_server_args_shell_safe(resolved_server_args)
    if resolved_server_args:
        envs[framework_env] = resolved_server_args


class _RequestWindow(NamedTuple):
    isl: int
    osl: int
    profiling: bool
    serves_requests: bool
    #: ``(sentinel, flag)`` pairs the re-assertion restores when a later step drops one;
    #: a dropped ``max_iterations`` is an OOM, not a cosmetic problem.
    vllm_profiler_flags: list[tuple[str, str]]
    #: The serialization-safe cap, above which (or at zero, which vLLM reads as no
    #: limit) a surviving ``max_iterations`` is rejected rather than trusted.
    vllm_profiler_cap: int
    atom_caps: _AtomTracelensCaps | None
    xdit_profiler_args: tuple[str, ...] = ()


def _pin_benchmark_target(bench: dict[str, Any], target: BenchmarkTarget) -> None:
    if target.model_path:
        bench["model"] = str(target.model_path)
    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision
    if target.gpu_type:
        bench["runner_type"] = gpu_runner_type(target.gpu_type)
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{bench['runner_type']}.sh"
        else:
            bench.pop("benchmark_script", None)
    if target.benchmark_script:
        bench["benchmark_script"] = str(target.benchmark_script)
    effective_inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if effective_inferencex_path:
        # Persist the resolved InferenceX checkout so Magpie's runtime checkout
        # matches Hyperloom's patch target.
        bench["inferencex_path"] = effective_inferencex_path


def _refuse_framework_script_mismatch(bench: dict[str, Any]) -> None:
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


def _pin_workload_shape_envs(envs: dict[str, Any]) -> None:
    for env_key in ("CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "EP", "PORT"):
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
    tp_from_yaml = envs.get("TP")
    rocr_yaml = str(envs.get("ROCR_VISIBLE_DEVICES") or "").strip()
    rocr_devices = [d.strip() for d in rocr_yaml.split(",") if d.strip()]
    if rocr_yaml and not tp_from_yaml:
        # Derive TP from the user-pinned GPU list when the YAML doesn't set TP.
        resolved_tp = len(rocr_devices)
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


def _apply_xdit_profile(envs: dict[str, Any]) -> tuple[str, ...]:
    """Set xDiT profiler flags and write patch status; return the flags for restore."""
    from ._server_patcher import ensure_xdit_patched_for_tracelens
    from ._profile_setup import _tracelens_patch_enabled, write_tracelens_patch_status

    xdit_patch_attempted = _tracelens_patch_enabled()
    xdit_patched = xdit_patch_attempted and ensure_xdit_patched_for_tracelens()
    needed = ["--profile_wait 1"]
    if xdit_patched:
        needed += ["--profile_capture_phase", "--profile_with_stack"]
    existing = str(envs.get(_XDIT_EXTRA_ARGS_ENV, "")).strip()
    missing = [f for f in needed if f not in existing]
    if missing:
        envs[_XDIT_EXTRA_ARGS_ENV] = f"{existing} {' '.join(missing)}".strip()
    write_tracelens_patch_status(envs, attempted=xdit_patch_attempted, patched=xdit_patched, fw="xdit")
    return tuple(needed)


def _shape_request_window(bench: dict[str, Any]) -> _RequestWindow:
    from hyperloom.inference_optimizer import framework_registry

    envs = bench["envs"]
    # Last-resort fallbacks, resolved from the CLI workload defaults so the
    # recipe and the workload spec published beside it cannot disagree about what
    # "unset" means; normally the CLI has already projected the resolved values
    # into these envs before materialization.
    _default_isl, _default_osl, _default_conc = cli_workload_defaults()
    isl_val = int(envs.get("ISL") or _default_isl)
    osl_val = int(envs.get("OSL") or _default_osl)
    conc_val = int(envs.get("CONC") or _default_conc)
    # Steady-state window for profiling configs (detected by YAML
    # ``benchmark.envs.PROFILE`` or ``profiler.torch_profiler.enabled``, not the process env).
    # The captured-step count is capped at
    # a serialization-safe budget; the profile OSL is resolved (and lowered if
    # needed) so the steady-state floor fits that cap:
    #   max_iters    = HYPERLOOM_PROFILE_MAX_STEPS_CAP (default 128)
    #   steady_floor = ceil(OSL * (1 + R) / (2 * CONC)), must be <= max_iters
    #   delay_iters  = OSL * (R + 1) * 3 - max_iters / 2 iterations
    is_profile = str(envs.get("PROFILE", "")).strip() == "1" or (
        bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True
    )
    # Scriptable image frameworks (xDiT diffusion) have no LLM decode steady-state
    # window, so the OSL/steady-floor math below is a serving concept they never consume.
    serves_requests = not framework_registry.is_scriptable(bench.get("framework"))
    _is_xdit_profile = is_profile and framework_registry.extra_args_env(bench.get("framework")) == _XDIT_EXTRA_ARGS_ENV
    xdit_saved_args = _apply_xdit_profile(envs) if _is_xdit_profile else ()
    profile_num_prompts: int | None = None
    atom_caps: _AtomTracelensCaps | None = None
    vllm_flags: list[tuple[str, str]] = []
    vllm_cap = 0
    if is_profile and serves_requests:
        from ._profile_setup import prepare_profile

        outcome = prepare_profile(bench, envs)
        osl_val = int(envs["OSL"])
        profile_num_prompts = outcome.num_prompts
        atom_caps = outcome.atom_caps
        vllm_flags = outcome.vllm_flags
        vllm_cap = outcome.vllm_cap
    if serves_requests:
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
    return _RequestWindow(
        isl_val, osl_val, is_profile, serves_requests, vllm_flags, vllm_cap, atom_caps, xdit_saved_args
    )


def _seed_reference_launch(bench: dict[str, Any], framework_env: str) -> tuple[dict[str, Any], dict[str, Any]]:
    envs = bench["envs"]
    # Seed the framework server-args env + envs from a reference recipe below
    # the YAML base and any per-task extra_server_args (reference flags leftmost,
    # so last-wins lets later merges override them).
    ref_args, reference_envs, reference_controls = resolve_reference_launch()
    for name in reference_controls.get("unset_envs", []):
        envs.pop(name, None)
    if ref_args or reference_controls.get("remove_args") or reference_controls.get("args_mode") == "replace":
        from hyperloom.inference_optimizer.grid_server_args import compose_server_args

        envs[framework_env] = compose_server_args(
            base_extra_args=ref_args,
            variant_extra_args=""
            if reference_controls.get("args_mode") == "replace"
            else str(envs.get(framework_env, "")),
            remove_args=reference_controls.get("remove_args"),
            args_mode="replace",
        )
    for _rk, _rv in reference_envs.items():
        envs.setdefault(str(_rk), str(_rv))  # never clobber YAML/CLI envs
    if reference_controls.get("overlay_pythonpath"):
        from hyperloom.common.overlay import validate_overlay_pythonpath

        prepend_launch_path(envs, "PYTHONPATH", validate_overlay_pythonpath(reference_controls["overlay_pythonpath"]))
    return reference_envs, reference_controls


def _merge_caller_overrides(
    bench: dict[str, Any], framework_env: str, server_args: str, replace_args: bool, extra_envs: dict[str, Any] | None
) -> dict[str, Any]:
    envs = bench["envs"]
    recipe_overwritten = launcher_overwritten_envs(bench)
    if server_args:
        # Merge into (not overwrite) the framework env so the profile path's
        # graph-capture flags aren't dropped.
        existing = str(envs.get(framework_env, "")).strip()
        if replace_args:
            envs[framework_env] = server_args
        else:
            if framework_env == "EXTRA_SGLANG_ARGS" and "--moe-runner-backend" in str(server_args):
                # For MoE backend exploration/tuning, the candidate value must
                # replace the baseline's injected default (usually triton) rather
                # than relying on duplicate last-wins flags.
                existing = remove_moe_runner_backend_arg(existing)
            envs[framework_env] = merge_server_args(existing, server_args)
    # Magpie forwards only ``benchmark.envs``, so ``--extra-env`` lands there for
    # every framework: vLLM Ray workers see MTP pins only this way.
    combined_extra: dict[str, Any] = dict(_operator_extra_env())
    from hyperloom.common.agentx_workload import is_agentx_client_script

    if is_agentx_client_script(str(bench.get("benchmark_script") or "")):
        # The AgentX switch already settled these from the same pins; the raw
        # pin would undo its CONC scaling of the warmup grace.
        agentx_env = agentx_env_for_conc()
        combined_extra = {k: v for k, v in combined_extra.items() if not _is_agentx_client_key(k, agentx_env)}
    if extra_envs:
        combined_extra.update(extra_envs)
    safe_extra_envs, dropped_extra_envs = filter_untrusted_env_mapping(
        combined_extra,
        # A name the recipe re-exports unconditionally cannot be overridden
        # here; carrying it into the YAML publishes a value the run never used.
        allow_predicate=lambda key: is_allowed_variant_env_key(key) and key not in recipe_overwritten,
    )
    for _dk in dropped_extra_envs:
        log.warning("Dropping extra_envs key %s before benchmark materialization", _dk)
    for key, value in safe_extra_envs.items():
        envs[str(key)] = str(value)
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
    prefer_dir = "CUSTOM_DIR" in safe_extra_envs and "CUSTOM_REPO_PATH" not in safe_extra_envs
    _sync_repo_aliases(bench, envs, framework="custom", prefer_dir=prefer_dir)
    return safe_extra_envs


def _settle_merged_server_args(bench: dict[str, Any], framework_env: str) -> None:
    envs = bench["envs"]
    # Final dedup after reference/server_args merges: collapse repeated
    # vLLM/atom single-value flags to last-wins so recipe/variant values
    # override earlier ones (no-op for sglang).
    if _final_args := str(envs.get(framework_env, "")).strip():
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


def _publish_scriptable_wiring(bench: dict[str, Any], window: _RequestWindow, establish_quality_ref: bool) -> None:
    envs = bench["envs"]
    # Magpie forwards only ``benchmark.envs`` to the wrapper subprocess, so
    # re-inject the quality reference here (the single scriptable choke point)
    # or the shipped empty YAML default silently skips the gate. What the
    # reference holds is the workload's business — an xDiT image, an operator's
    # own artifact — so the default filename is a convention, not a contract.
    # When the operator configures nothing, derive a stable reference path
    # under ``session_dir()`` (pinned via
    # ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``) so baseline + variants
    # resolve the same file.
    if not window.serves_requests:
        _qref = _first_env(_QUALITY_REF_ENVS)
        if not _qref:
            from hyperloom.inference_optimizer.session.paths import session_dir

            _qref = str(session_dir() / "storage" / "quality_ref" / "baseline.png")
        if window.profiling:
            _ref_compare, _ref_write = "", ""
        elif establish_quality_ref:
            _ref_compare, _ref_write = "", (_first_env(_QUALITY_REF_WRITE_ENVS) or _qref)
        else:
            _ref_compare, _ref_write = _qref, ""
        for _name in _QUALITY_REF_ENVS:
            envs[_name] = _ref_compare
        for _name in _QUALITY_REF_WRITE_ENVS:
            envs[_name] = _ref_write
    # These three are read by the xDiT runner and by nothing else, so they are
    # keyed on the framework rather than on scriptability: an operator-supplied
    # ``custom`` workload must not have its baseline altered by settings it
    # never declared, least of all an attention backend.
    if framework_registry.extra_args_env(bench.get("framework")) == _XDIT_EXTRA_ARGS_ENV:
        _xdit_model_name = os.environ.get("XDIT_MODEL_NAME", "").strip()
        if _xdit_model_name:
            envs["XDIT_MODEL_NAME"] = _xdit_model_name
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


def _pin_model_launch_workarounds(bench: dict[str, Any], model_path: str | None) -> str:
    envs = bench["envs"]
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
        # capture on gfx942. Pin the triton attention backend. sglang accepts
        # lowercase `triton`; vLLM only knows TRITON_ATTN.
        _mimo_is_vllm = "vllm" in str(bench.get("framework") or "").lower()
        _mimo_attn_backend = "TRITON_ATTN" if _mimo_is_vllm else "triton"
        add_server_arg_unless_pinned(
            envs,
            bench.get("framework"),
            f"--attention-backend {_mimo_attn_backend}",
            pinned_by=("attention-backend",),
        )
        # vLLM registers this checkpoint under MiMoV2FlashForCausalLM but the HF
        # config declares MiMoV2ForCausalLM, which the pod-local vLLM build
        # rejects at boot. Remap the arch via --hf-overrides. vLLM-only; JSON
        # kept space-free so it survives Magpie's unquoted splice.
        if _mimo_is_vllm:
            add_server_arg_unless_pinned(
                envs,
                bench.get("framework"),
                '--hf-overrides {"architectures":["MiMoV2FlashForCausalLM"]}',
                pinned_by=("hf-overrides", "hf_overrides"),
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
        if _sparse_bs := _sparse_kv_block_size(str(model_path or bench.get("model") or "")):
            add_server_arg_unless_pinned(
                envs,
                bench.get("framework"),
                f"--block-size {_sparse_bs}",
                pinned_by=("block-size", "block_size"),
            )
    return _model_basename


def _pin_client_model_access(bench: dict[str, Any], model_path: str | None, model_basename: str) -> None:
    envs = bench["envs"]
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
    # Trusting remote code is not enough when transformers cannot map the
    # model_type at all: the client has to be told which loader to use.
    _model = model_path or bench.get("model")
    if _client_tok_mode := _client_tokenizer_mode(_model):
        envs.setdefault("HYPERLOOM_CLIENT_TOKENIZER_MODE", _client_tok_mode)
    # Qwen3.6 MoE declares an arch the SERVER must also load remote code for, or it
    # refuses the arch at boot.
    _custom_code = "qwen3.6-35b-a3b" in model_basename or "qwen3-6-35b-a3b" in model_basename
    if _custom_code or _model_requires_remote_code(_model):
        add_server_arg_unless_pinned(
            envs,
            bench.get("framework"),
            "--trust-remote-code",
            pinned_by=("trust-remote-code",),
        )


def _pin_accuracy_eval_envs(envs: dict[str, Any]) -> None:
    # Accuracy eval (GSM8K) is ON unless the round passes ``extra_envs["RUN_EVAL"]`` or the workload is AgentX.
    if "RUN_EVAL" not in envs:
        envs["RUN_EVAL"] = "true"
    # Persist eval task/limit into the YAML so they become part of the
    # fingerprint-able contract and bypass_runner reads them from a stable source.
    for _eval_key in ("MAGPIE_EVAL_TASKS", "MAGPIE_EVAL_LIMIT"):
        _eval_value = os.environ.get(_eval_key, "").strip()
        if _eval_value and _eval_key not in envs:
            envs[_eval_key] = _eval_value
    # PD (router-fronted): lm_eval's default token-id prompts are rejected by the
    # sglang_router's /v1/completions (HTTP 422, StringOrArray), collapsing the
    # accuracy eval. Force string prompts so the gate works. Explicit env /
    # extra_envs win; only disaggregated runs are touched (aggregated hits the
    # sglang server directly, which accepts token-id prompts).
    _eval_tok_env = os.environ.get("MAGPIE_EVAL_TOKENIZED_REQUESTS", "").strip()
    if not _eval_tok_env:
        from ._multi_node_env import resolve_kb_topology

        if str(resolve_kb_topology().pd_mode or "").lower() == "disaggregated":
            _eval_tok_env = "false"
    if _eval_tok_env and "MAGPIE_EVAL_TOKENIZED_REQUESTS" not in envs:
        envs["MAGPIE_EVAL_TOKENIZED_REQUESTS"] = _eval_tok_env


def _pin_runtime_build_envs(
    bench: dict[str, Any], model_path: str | None, gpu_type: str | None, flydsl_source_dirs: bool
) -> None:
    envs = bench["envs"]
    # FlyDSL folds only same-directory helpers into its JIT cache key, so a patched
    # helper one directory over is served from a stale binary. Naming the roots
    # folds their sources into the key. Only the run that applied such a patch has
    # that hazard; setdefault so an operator-set value (YAML / extra_envs) wins.
    if flydsl_source_dirs and (_flydsl_dirs := flydsl_extra_source_dirs()):
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

    if (
        "sglang" in str(bench.get("framework") or "").lower()
        and str(bench.get("precision") or "").strip().lower() == "fp8"
        and _resolve_amd_gpu_type(gpu_type or bench.get("runner_type")) in _GFX942_GPU_TYPES
        and _fp8_is_per_channel_per_token(str(model_path or os.environ.get("MODEL_PATH", "")))
    ):
        envs.setdefault("SGLANG_USE_AITER_FP8_PER_TOKEN", "1")


def _apply_caller_removals(
    envs: dict[str, Any], framework_env: str, remove_args: Any, unset_envs: Any, stale: list[str]
) -> None:
    remove_list = to_str_list(remove_args)
    unset_list = to_str_list(unset_envs)
    for key in stale:
        envs.pop(key, None)
    if remove_list:
        envs[framework_env] = remove_server_args(envs.get(framework_env, ""), remove_list)
    for key in unset_list:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(key).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("Refusing to unset pinned benchmark env %s", key)
            continue
        envs.pop(str(key), None)


def _restate_profile_capture(
    envs: dict[str, Any], framework_env: str, window: _RequestWindow, replace_args: bool
) -> None:
    if window.vllm_profiler_flags and framework_env == "EXTRA_VLLM_ARGS":
        # The profile path's iteration bounds have to be the LAST word on this env,
        # because three separate steps above can drop them: a candidate carrying
        # ``args_mode="replace"`` (which writeback sets as soon as a KEEP needs
        # ``remove_args``) overwrites the whole flag string, ``extra_envs`` can
        # override it outright, and ``remove_args`` itself strips flags by name.
        # vLLM reads a missing ``max_iterations`` as "profile until stop_profile",
        # and the worker then accumulates every profiler event in host RAM at
        # ~60 MiB/s until the cgroup OOM-killer takes it out mid-roofline. No
        # exploration result is worth that, so the bounds win over all three.
        profile_args = str(envs.get(framework_env, "")).strip()
        restored = [
            flag
            for name, flag in window.vllm_profiler_flags
            if not _profiler_bound_holds(
                name,
                _profiler_flag_value(profile_args, name),
                cap=window.vllm_profiler_cap,
            )
        ]
        if restored:
            log.warning(
                "profile server args lost torch-profiler flags %r (replace_args=%s); "
                "restoring them so the profiler cannot run unbounded.",
                restored,
                bool(replace_args),
            )
            # The seal applies the sink-side guard to whatever is left here.
            envs[framework_env] = merge_server_args(profile_args, " ".join(restored))
    if window.xdit_profiler_args and framework_env == _XDIT_EXTRA_ARGS_ENV:
        current = str(envs.get(framework_env, "")).strip()
        missing = [f for f in window.xdit_profiler_args if f not in current]
        if missing:
            log.warning(
                "profile server args lost xDiT profiler flags %r (replace_args=%s); restoring.",
                missing,
                bool(replace_args),
            )
            envs[framework_env] = merge_server_args(current, " ".join(missing))
    # After the restore, so a vLLM delay it re-added is divided like any other.
    if window.profiling and window.serves_requests:
        fit_capture_to_speculative(envs, framework_env)
    # After extra_envs / replace_args / remove_args so TraceLens knobs survive the
    # same last-wins path as vLLM profiler bounds, and before the seal.
    apply_atom_profile_caps(envs, window.atom_caps)


def _drop_benchmark_credentials(envs: dict[str, Any]) -> None:
    # The rendered YAML is persisted, so credentials must not reach it.
    filtered_envs, dropped_credentials = filter_untrusted_env_mapping(
        envs,
        allow_predicate=lambda key: key not in BENCHMARK_SECRET_ENV_NAMES,
    )
    if dropped_credentials:
        log.warning(
            "Dropping control-plane credentials from benchmark envs: %s", ", ".join(sorted(dropped_credentials))
        )
        envs.clear()
        envs.update(filtered_envs)


@dataclass(frozen=True)
class LaunchLayer:
    """One layer of server-arg and env changes composed over a config's launch."""

    extra_server_args: str = ""
    extra_envs: dict[str, Any] | None = None
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None
    args_mode: str = "append"


@dataclass(frozen=True)
class BenchmarkTarget:
    """The model, GPU type and benchmark script a rendered config is pinned to."""

    model_path: str | None = None
    gpu_type: str | None = None
    benchmark_script: str | None = None


@dataclass(frozen=True)
class MaterializeOptions:
    """Per-run choices for a materialized config beyond its launch layer and target."""

    out_name: str = "baseline_config.with_envs.yaml"
    establish_quality_ref: bool = False
    drop_moe_runner_backend: bool = False
    flydsl_source_dirs: bool = False
    agentx_mode: bool | None = None
    grading: Mapping[str, Any] | None = None


def prepare_grid_benchmark_config(
    config_path: Path,
    output_dir: Path,
    *,
    launch: LaunchLayer = LaunchLayer(),
    target: BenchmarkTarget = BenchmarkTarget(),
    options: MaterializeOptions = MaterializeOptions(),
) -> Path:
    """Keep the verified native baseline intact until the grid composes its candidate."""
    from .magpie_agentx import native_config

    if native_config(config_path):
        return config_path
    return materialize_config_with_envs(config_path, output_dir, launch=launch, target=target, options=options)


def materialize_benchmark_config(
    config_path: Path,
    output_dir: Path,
    *,
    launch: LaunchLayer = LaunchLayer(),
    target: BenchmarkTarget = BenchmarkTarget(),
    options: MaterializeOptions = MaterializeOptions(),
) -> Path:
    """Resolve the workload owner before applying server candidate changes."""
    import yaml

    from hyperloom.common.benchmark_config import is_magpie_agentx
    from hyperloom.common.coerce import to_str_list
    from .magpie_agentx import LaunchChanges, apply_launch_changes, materialize_agentx, recipe_option_names

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not is_magpie_agentx(config["benchmark"]):
        return materialize_config_with_envs(config_path, output_dir, launch=launch, target=target, options=options)
    replacing = launch.args_mode == "replace"
    inherited = "" if replacing else os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "")
    recipe_options = recipe_option_names(config_path)
    apply_launch_changes(config["benchmark"], LaunchChanges(args=inherited), recipe_options=recipe_options)
    from hyperloom.common.env import is_truthy

    unset_envs = to_str_list(launch.unset_envs)
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
    server_env.update(launch.extra_envs or {})
    if is_truthy(server_env.pop("RUN_EVAL", False)):
        raise ValueError("AgentX uses Magpie's replay validity; RUN_EVAL cannot enable GSM8K for this workload")
    changes = LaunchChanges(
        launch.extra_server_args,
        server_env,
        to_str_list(launch.remove_args),
        unset_envs,
        replacing,
    )
    apply_launch_changes(config["benchmark"], changes, recipe_options=recipe_options)
    return materialize_agentx(
        config, output_dir / options.out_name, model_path=target.model_path, recipe_options=recipe_options
    )


def materialize_config_with_envs(
    config_path: Path,
    output_dir: Path,
    *,
    launch: LaunchLayer = LaunchLayer(),
    target: BenchmarkTarget = BenchmarkTarget(),
    options: MaterializeOptions = MaterializeOptions(),
) -> Path:
    """Render a per-run Magpie YAML with caller-provided overrides.

    The process env wins over the YAML defaults, and the body is the order the phases run
    in: what the run launches, the request shape it serves, the server arguments composed
    from every source, the per-framework and per-model envs, and the caller's own removals.

    Raises:
        FrameworkScriptMismatchError: If ``target.benchmark_script`` targets a
            different known framework than the run's framework.
    """
    server_args = (launch.extra_server_args or "").strip()
    operator_server_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
    replace_args = str(launch.args_mode or "append").strip().lower() == "replace"
    if operator_server_args and not replace_args:
        server_args = merge_server_args(operator_server_args, server_args)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = cfg.setdefault("benchmark", {})
    _pin_benchmark_target(bench, target)
    agentx_run = agentx_enabled() if options.agentx_mode is None else options.agentx_mode
    envs = bench["envs"] = {k: v for k, v in (bench.get("envs") or {}).items() if agentx_run or k != "RUN_EVAL"}
    runtime_path = os.environ.get("PATH", "").strip()
    if "PATH" in envs and runtime_path:
        envs["PATH"] = runtime_path
    apply_scriptable_runtime_defaults(
        bench, envs, gpu_type=target.gpu_type, explicit_benchmark_script=bool(target.benchmark_script)
    )
    apply_agentx_switch(bench, target.model_path, active=agentx_run, grading=options.grading)
    _refuse_framework_script_mismatch(bench)
    _pin_workload_shape_envs(envs)
    framework_env = server_args_env_name(bench.get("framework"))
    window = _shape_request_window(bench)
    reference_envs, reference_controls = _seed_reference_launch(bench, framework_env)
    safe_extra_envs = _merge_caller_overrides(bench, framework_env, server_args, replace_args, launch.extra_envs)
    _settle_merged_server_args(bench, framework_env)
    _publish_scriptable_wiring(bench, window, options.establish_quality_ref)
    model_basename = _pin_model_launch_workarounds(bench, target.model_path)
    _finalize_framework_server_args(bench, window, drop_moe_runner_backend=options.drop_moe_runner_backend)
    _pin_client_model_access(bench, target.model_path, model_basename)
    _pin_accuracy_eval_envs(envs)
    _pin_runtime_build_envs(bench, target.model_path, target.gpu_type, options.flydsl_source_dirs)
    unset_by_reference = reference_controls.get("unset_envs", [])
    stale = [k for k in unset_by_reference if k not in reference_envs and k not in safe_extra_envs]
    _apply_caller_removals(envs, framework_env, launch.remove_args, launch.unset_envs, stale)
    _restate_profile_capture(envs, framework_env, window, replace_args)
    _apply_vllm_source_runtime(bench, envs)
    _drop_benchmark_credentials(envs)
    seal_server_argv(envs, bench.get("framework"))
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = output_dir / options.out_name
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return materialized


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
