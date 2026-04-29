"""Python ↔ shell bridge for actions.

Each :class:`ActionExecutor` wraps one row of the action catalogue
(``actions/_meta/<name>.yaml``) and turns a queued ``delegate`` task
into real subprocess calls against the GPU-side scripts shipped under
``src/inference_optimizer/scripts/`` (resolved via
:func:`paths.asset_scripts_dir`).

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

Discovery (Phase 3 — plug-and-play)
====================================

Executor modules are loaded by **scanning this package directory** at
import time rather than via a hard-coded list. Any ``*.py`` file in
this directory (other than ``__init__``, ``base``, ``_helpers``) that
calls :func:`register_executor` (e.g. via a module-level
``register_executor(MyExec())`` line) is auto-registered.

This means:

* Adding a new action executor is "drop a new ``actions/_meta/foo.yaml``
  + ``actions/foo.md`` + ``orchestrator/action_executors/foo.py`` that
  calls ``register_executor(...)``" — no edit to this file required.
* External integrators can also point ``INFERENCE_OPTIMIZER_EXTRA_EXECUTORS``
  at a colon-separated list of additional Python module names; each is
  imported here, giving them the same self-register hook.

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

import importlib
import logging
import os
from pathlib import Path

from .base import (
    EXECUTOR_REGISTRY,
    ActionExecutor,
    ExecutorContext,
    ExecutorEnvError,
    ExecutorResult,
    get_executor,
    register_executor,
)


_log = logging.getLogger(__name__)


# Files that are part of the framework, NOT executor implementations.
_SKIP_FILENAMES: frozenset[str] = frozenset(
    {"__init__.py", "base.py", "_helpers.py"}
)

# When set, the discovery loop does NOT short-circuit even if a module
# raised. Mainly for debugging — production runs swallow + log so a
# single broken executor can't take down the orchestrator at import.
_STRICT_ENV = "INFERENCE_OPTIMIZER_EXECUTOR_STRICT"
_EXTRA_ENV = "INFERENCE_OPTIMIZER_EXTRA_EXECUTORS"


def _discover_default_executors() -> tuple[list[str], list[tuple[str, str]]]:
    """Import every executor module in this package directory.

    Returns ``(loaded, failed)`` where ``failed`` is a list of
    ``(module_name, error_repr)`` tuples for the modules that raised
    during import.
    """
    here = Path(__file__).resolve().parent
    loaded: list[str] = []
    failed: list[tuple[str, str]] = []
    strict = os.environ.get(_STRICT_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    for entry in sorted(here.iterdir()):
        if entry.suffix != ".py":
            continue
        if entry.name in _SKIP_FILENAMES:
            continue
        if entry.name.startswith("_"):
            continue  # _helpers.py and any other underscore-prefixed module
        mod_name = f"{__name__}.{entry.stem}"
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            failed.append((mod_name, repr(exc)))
            _log.warning("action_executors: failed to import %s: %r", mod_name, exc)
            if strict:
                raise
            continue
        loaded.append(mod_name)
    # Optional external-module list (colon-separated, e.g.
    # "myco.executors.fancy:other.thing").
    extras = os.environ.get(_EXTRA_ENV, "").strip()
    if extras:
        for mod_name in [s.strip() for s in extras.split(":") if s.strip()]:
            try:
                importlib.import_module(mod_name)
            except Exception as exc:  # noqa: BLE001
                failed.append((mod_name, repr(exc)))
                _log.warning("action_executors: failed to import extra %s: %r",
                             mod_name, exc)
                if strict:
                    raise
                continue
            loaded.append(mod_name)
    if loaded:
        _log.info("action_executors: discovered %d modules: %s",
                  len(loaded), [m.rsplit('.', 1)[-1] for m in loaded])
    if failed:
        _log.info("action_executors: %d failures during discovery", len(failed))
    return loaded, failed


_loaded, _failed = _discover_default_executors()


def discovered_executor_modules() -> list[str]:
    """Return the list of module names loaded by auto-discovery.

    Useful for tests + diagnostics; does NOT trigger re-discovery.
    """
    return list(_loaded)


def discovery_failures() -> list[tuple[str, str]]:
    """Return any discovery failures recorded during package import."""
    return list(_failed)


__all__ = [
    "ActionExecutor",
    "EXECUTOR_REGISTRY",
    "ExecutorContext",
    "ExecutorEnvError",
    "ExecutorResult",
    "discovered_executor_modules",
    "discovery_failures",
    "get_executor",
    "register_executor",
]
