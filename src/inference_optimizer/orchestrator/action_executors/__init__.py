"""Python ↔ shell bridge for actions.

Each :class:`ActionExecutor` wraps one row of the action catalogue
(``actions/_meta/<name>.yaml``) and turns a queued ``delegate`` task
into real subprocess calls against the GPU-side scripts shipped under
``.cursor/skills/inference-optimizer/scripts/`` (resolved via
:func:`paths.skill_scripts_dir`).

Why a separate layer?
=====================

* The :class:`SubAgentRunner` (LLM path) only knows how to compose a
  prompt + parse intents — it cannot launch sglang or run GSM8K.
* The shipped scripts (``run_baseline.sh`` / ``eval_accuracy.sh`` / ...)
  were written for the standalone ``inference-optimization`` skill and
  expect a fully BYOI-bootstrapped GPU sandbox. They are 100% bash and
  rely on ``set -u`` to refuse to run without env vars.
* Executors are the seam where Python decides *whether* to invoke the
  real path (when the env says GPU + InferenceX are present) or fall
  back to the LLM-driven path (when the user is doing a smoke run).

Lookup
------

``EXECUTOR_REGISTRY`` is the global ``name -> ActionExecutor`` map. The
:class:`SubAgentRunner` checks it before composing the LLM prompt; if
the action has a registered executor, that executor wins.

External callers should use :func:`get_executor` rather than touching
the dict directly so the lookup is normalisation-safe (``kernel-opt``
and ``kernel_opt`` both resolve to the same executor).
"""
from __future__ import annotations

from .base import (
    EXECUTOR_REGISTRY,
    ActionExecutor,
    ExecutorContext,
    ExecutorEnvError,
    ExecutorResult,
    get_executor,
    register_executor,
)


# Executors are imported eagerly so they self-register on package import.
# Each module is independent — failures in one don't tear down the others.
def _import_default_executors() -> list[str]:
    failed: list[str] = []
    for mod_name in ("baseline", "bench_runner", "profile",
                     "param_sweep_run", "kernel_opt"):
        try:
            __import__(f"{__name__}.{mod_name}")
        except Exception:  # noqa: BLE001 — defer real failure to use site
            failed.append(mod_name)
    return failed


_failed = _import_default_executors()


__all__ = [
    "ActionExecutor",
    "EXECUTOR_REGISTRY",
    "ExecutorContext",
    "ExecutorEnvError",
    "ExecutorResult",
    "get_executor",
    "register_executor",
]
