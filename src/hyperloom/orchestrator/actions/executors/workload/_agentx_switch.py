# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX MLPerf client switch and the workload spec and GEAK axis it publishes."""

from __future__ import annotations

import os
from typing import Any, Mapping

from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.agentx_workload import is_agentx_client_key, pin_mlperf_round_concurrency
from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    INTVTY_OBJECTIVES,
    agentx_enabled,
    intvty_grading_enabled,
    parse_intvty_noise_pct,
)
from hyperloom.common.workload_defaults import DEFAULT_CONC, DEFAULT_ISL, DEFAULT_OSL

from ..serving._server_argv import add_server_arg_unless_pinned


def agentx_env_for_conc(conc: int | None = None) -> "Mapping[str, str]":
    """The environment the AgentX derivations read, carrying a rung's own CONC.

    The operator's ``--extra-env`` pins are already in the process environment,
    exported by ``_export_operator_launch_shape``, so no layer over it is needed
    here -- re-merging them would make this reader resolve a knob differently
    from every other one.

    The AgentX warmup scales with ``CONC``, so every bound derived from it is
    linear in the concurrency being measured, not in the one the session was
    launched at.

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


# GEAK's own names for the axes Hyperloom hands it: ``E2E_METRIC`` selects one
# and ``bench_summary.json`` records the matching ``metric_basis``. Spelled in
# GEAK's vocabulary, not Hyperloom's curve-row names, so the handoff and GEAK's
# summary can be compared as strings on both sides. The interactivity token is
# the full field name: in InferenceX's vocabulary a bare ``intvty`` is the
# decode-only 1/ITL family, not this end-to-end one.
GEAK_METRIC_OUTPUT = ("output", "aggregate_output_tok_s")
GEAK_METRIC_INTVTY = (GRADED_INTVTY_P50, GRADED_INTVTY_P50)
# The guards GEAK's bench summary records beside the P50 objective, by Hyperloom metric. GEAK fails a guard it did not
# record on every candidate, and it records no total token throughput guard yet, so that guard is left to the rebench.
_GEAK_GUARD_BASES = {GRADED_INTVTY: GRADED_INTVTY}


def geak_metric_axis(
    *,
    benchmark_mode: str = "",
    grading: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """GEAK's ``(E2E_METRIC, metric_basis)`` pair for the axis this session keeps candidates on.

    GEAK accepts on the single axis it is handed, so that axis has to be the
    objective. An interactivity-graded session keeps a candidate only on a gain
    in the median ``e2e_norm_intvty_p50``, with the p90 tail and total token
    throughput as noise-band guards (``hyperloom.common.agentx_objective``);
    searching a guard selects wins the rebench reverts and passes over median
    wins. Every other session is graded on output.

    A GEAK build that cannot measure ``e2e_norm_intvty_p50`` refuses it, or, predating
    that check, falls back to output and records that in ``metric_basis``.

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

    The rebench keeps an interactivity-graded GEAK candidate only on the P50 objective's bar with its guards inside
    the noise band; without the rule GEAK reports any gain past its own band as a win the rebench then reverts.
    Guards are named by GEAK basis, like the objective, and only those GEAK records are handed over.

    Args:
        metric_basis: The ``metric_basis`` handed to GEAK (see :func:`geak_metric_axis`).
        noise_pct: The session's noise band, the one its own verdict holds the guards to.

    Returns:
        The rule, or ``None`` for any other axis, which leaves GEAK's verdict as it was.
    """
    if metric_basis != GEAK_METRIC_INTVTY[1]:
        return None
    objective = AgentXObjective.P50_INTERACTIVITY
    guards = {_GEAK_GUARD_BASES[g.metric]: noise_pct for g in objective.guards if g.metric in _GEAK_GUARD_BASES}
    return {"objective": metric_basis, "min_gain_pct": objective.bar_pct, "guard_max_drop_pct": guards}


def geak_grading_fields(grading: Mapping[str, Any] | None) -> dict[str, Any]:
    """The axis, noise band and KEEP rule an AgentX ``workload_spec`` hands GEAK for ``SharedState.grading``."""
    metric_basis = geak_metric_axis(benchmark_mode="agentx", grading=grading)[1]
    noise = grading.get("noise_pct") if isinstance(grading, Mapping) else None
    noise_pct = float(noise) if isinstance(noise, (int, float)) else parse_intvty_noise_pct()
    acceptance = geak_acceptance(metric_basis, noise_pct)
    keep_rule = {"acceptance": acceptance} if acceptance else {}
    return {"metric_basis": metric_basis, "intvty_p90_veto_pct": noise_pct, **keep_rule}


def build_agentx_workload_spec(
    envs: dict[str, Any],
    *,
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
        envs: The benchmark mapping's ``envs``, already carrying the switch's overrides.
        env: The resolved process environment, carrying this round's ``CONC``
            (see :func:`agentx_env_for_conc`). Defaults to ``os.environ``.
    """
    from hyperloom.common.agentx_workload import (
        MLPERF_CORPUS,
        MLPERF_PORT,
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

    conc = int(served_knob("CONC", DEFAULT_CONC))
    isl_osl_placeholder = {
        "isl": int(served_knob("ISL", DEFAULT_ISL)),
        "osl": int(served_knob("OSL", DEFAULT_OSL)),
        "note": "CLI defaults only; agentic replay ignores fixed ISL/OSL",
    }
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
        **geak_grading_fields(grading),
        "metric_window_s": 0.0,
        "flow": mlperf_flow(envs),
        "port": int(envs.get("PORT") or MLPERF_PORT),
        "failed_request_threshold": float(client_knob("AGENTX_FAILED_REQUEST_THRESHOLD", 0.10)),
        "isl_osl_placeholder": isl_osl_placeholder,
    }


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
        mlperf_flow,
        mlperf_trajectories,
    )

    _agentx_env = agentx_env_for_conc(conc)
    bench["benchmark_script"] = agentx_client_script(_agentx_env)
    envs["RUN_EVAL"] = "false"
    envs["MODEL"] = str(model_path or bench.get("model") or os.environ.get("MODEL_PATH", "")).strip()
    envs["FRAMEWORK"] = framework
    for key, value in _agentx_env.items():
        if is_agentx_client_key(key, _agentx_env):
            envs[key] = value
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
    add_server_arg_unless_pinned(
        envs,
        framework,
        f"--served-model-name {MLPERF_SERVED_MODEL}",
        pinned_by=("--served-model-name", "--served_model_name"),
    )
    bench["workload_spec"] = build_agentx_workload_spec(envs, env=_agentx_env, grading=grading)
