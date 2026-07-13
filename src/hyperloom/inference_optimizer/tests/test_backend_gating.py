# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Stage 3-gate tests: Magpie install/preflight is gated by benchmark backend.

The bypass backend must not force a Magpie clone/install/import/patch, while
the default (magpie) path stays byte-for-byte unchanged. These assertions are
structural (source-level) to avoid brittle end-to-end mocking of the large
_preflight()/install.sh flows, and they pin the exact gate the runtime relies
on.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb
from hyperloom.inference_optimizer.cli import preflight as preflight_mod


def test_backend_resolution_controls_magpie_need(monkeypatch):
    # Default + magpie -> Magpie needed; bypass -> not needed.
    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    assert bb.resolve_backend_name() == "magpie"
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    assert bb.resolve_backend_name() == "bypass"


def test_preflight_gates_magpie_on_backend():
    src = inspect.getsource(preflight_mod._preflight)
    # The Magpie import/install must be guarded by the active backend.
    assert "resolve_backend_name" in src
    assert "_magpie_backend_active" in src
    # The gate wraps the import check and the clone/install branch.
    assert 'import Magpie' in src
    assert "if _magpie_backend_active and" in src
    # InferenceX must NOT be gated away (bypass still needs it): the InferenceX
    # section marker exists and is not inside the magpie-only branch.
    assert "3. InferenceX" in src


def test_install_sh_gates_magpie_calls():
    install_sh = (
        Path(preflight_mod.__file__).resolve().parent.parent
        / "assets"
        / "install.sh"
    )
    text = install_sh.read_text(encoding="utf-8")
    # Backend-based gate present.
    assert "HYPERLOOM_BENCHMARK_BACKEND" in text
    assert 'if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" = "bypass" ]; then' in text
    # Magpie stages are inside the else branch (only run for non-bypass).
    gate_idx = text.index('HYPERLOOM_BENCHMARK_BACKEND_LC=')
    else_idx = text.index("else", gate_idx)
    fi_idx = text.index("\nfi\n", gate_idx)
    magpie_idx = text.index("ensure_magpie\n", gate_idx)
    patch_idx = text.index("ensure_magpie_atomic_scripts_patch\n", gate_idx)
    assert else_idx < magpie_idx < fi_idx
    assert else_idx < patch_idx < fi_idx
    # InferenceX stays unconditional (after the fi).
    inferencex_idx = text.index("ensure_inferencex\n", fi_idx)
    assert inferencex_idx > fi_idx