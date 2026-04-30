"""Tests for the action_executors auto-discovery loop (Phase 3)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import action_executors as ae_pkg
from inference_optimizer.orchestrator.action_executors import (
    EXECUTOR_REGISTRY,
    discovered_executor_modules,
    discovery_failures,
    get_executor,
)


def test_bundled_executors_all_discovered():
    """Plan A: KernelOptExecutor was removed (kernel agent owns kernel-opt).
    The four remaining bundled executors must self-register."""
    discovered = discovered_executor_modules()
    short = {m.rsplit(".", 1)[-1] for m in discovered}
    expected = {"baseline", "bench_runner", "profile", "param_sweep_run"}
    assert expected.issubset(short), (
        f"missing expected executors: discovered={short}"
    )
    assert "kernel_opt" not in short, (
        f"kernel_opt should be removed from executor discovery (Plan A): "
        f"discovered={short}"
    )


def test_no_discovery_failures_in_bundled_set():
    """A failure here means an executor module raised at import time —
    that breaks the orchestrator's ability to dispatch the action."""
    fails = discovery_failures()
    # We allow zero failures from the package itself; INFERENCE_OPTIMIZER_EXTRA_EXECUTORS
    # could have added more (none in CI).
    in_pkg = [(m, e) for m, e in fails
              if m.startswith("inference_optimizer.orchestrator.action_executors.")]
    assert in_pkg == []


def test_registry_keys_are_normalised():
    # ``bench-runner`` must resolve identically to ``bench_runner``.
    # (Plan A: kernel-opt removed; param_sweep_run is a representative
    # multi-word action whose dash form should normalise.)
    a = get_executor("param-sweep-run")
    b = get_executor("param_sweep_run")
    assert a is not None
    assert a is b


def test_underscore_prefixed_modules_skipped():
    """``_helpers.py`` is in the package but must NOT show up as an
    executor module (it has no ``register_executor`` call and would
    pollute the registry if discovery was naive)."""
    discovered = discovered_executor_modules()
    short = {m.rsplit(".", 1)[-1] for m in discovered}
    assert "_helpers" not in short
    assert "base" not in short


def test_extra_executor_env_var_is_honored(tmp_path: Path, monkeypatch):
    """Operators can drop a custom executor module path on
    INFERENCE_OPTIMIZER_EXTRA_EXECUTORS and it gets imported on
    discovery."""
    # Create a tmp package with a module that registers a fake executor.
    pkg_dir = tmp_path / "extras_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "fake_extra.py").write_text(
        "from inference_optimizer.orchestrator.action_executors import (\n"
        "    ActionExecutor, ExecutorContext, ExecutorResult,\n"
        "    register_executor,\n"
        ")\n"
        "\n"
        "class FakeExtra(ActionExecutor):\n"
        "    name = 'fake_extra_action'\n"
        "    async def run(self, ctx: ExecutorContext) -> ExecutorResult:\n"
        "        return ExecutorResult(status='succeeded')\n"
        "\n"
        "register_executor(FakeExtra())\n",
        encoding="utf-8",
    )
    import sys
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_EXECUTORS",
                       "extras_pkg.fake_extra")

    # Re-trigger discovery by re-running the discovery function. The
    # registry is module-global so a second call adds the new exec
    # without nuking the existing ones.
    loaded, failed = ae_pkg._discover_default_executors()
    assert any(m.endswith("fake_extra") for m in loaded), (
        f"extra module not picked up: loaded={loaded} failed={failed}"
    )
    assert get_executor("fake_extra_action") is not None


def test_strict_mode_propagates_import_errors(monkeypatch, tmp_path: Path):
    """When INFERENCE_OPTIMIZER_EXECUTOR_STRICT=1, a broken extra module
    should raise instead of being swallowed.

    We use the EXTRAS env var to inject a broken module so we don't
    have to mutate the in-tree package files.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXECUTOR_STRICT", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_EXECUTORS",
                       "definitely.not.a.real.module.path.qq")
    with pytest.raises(Exception):
        ae_pkg._discover_default_executors()


def test_registry_lookup_returns_singleton_per_action():
    """Multiple ``get_executor`` calls for the same action return the
    same instance (executors are stateless singletons by convention)."""
    a = get_executor("baseline")
    b = get_executor("baseline")
    assert a is b
    assert a is not None
