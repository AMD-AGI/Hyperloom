"""``_fill_integrate_defaults_from_state`` + integrate_handler defaulting.

Pre-existing ``_resolve_integrate_payload`` fills ``patch_path`` and
``source_file`` from SharedState when Orchestration sends only
``kernel_id``. The sibling helper added here fills the three other
Magpie re-baseline inputs that Orchestration omits just as often:

* ``base_tput``         from ``state.baseline_tput``
* ``config_path``       from ``state.baseline_config_path``
* ``extra_server_args`` from ``state.current_best["extra_server_args"]``

These tests pin (a) per-field defaulting, (b) explicit-payload wins,
and (c) the wiring through ``integrate_handler`` so the pre-existing
hard ``base_tput > 0`` check no longer panics when the value is
already on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _seed_state(
    session_dir: Path,
    *,
    baseline_tput: float = 0.0,
    baseline_config_path: str = "",
    current_best_args: str = "",
) -> SharedState:
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = baseline_tput
    state.baseline_config_path = baseline_config_path
    if current_best_args:
        state.current_best = {
            "action": "kernel_opt",
            "tput": 900.0,
            "extra_server_args": current_best_args,
        }
    state.save(session_dir)
    return state


class TestFillIntegrateDefaultsFromState:
    def test_all_three_defaults_fired(self, session_dir):
        _seed_state(
            session_dir,
            baseline_tput=800.0,
            baseline_config_path="/tmp/base.yaml",
            current_best_args="--page-size 16",
        )

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"}, session_dir=session_dir,
        )

        assert out["base_tput"] == 800.0
        assert out["config_path"] == "/tmp/base.yaml"
        assert out["extra_server_args"] == "--page-size 16"
        assert out["kernel_id"] == "k_abc"

    def test_payload_base_tput_wins(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "base_tput": 999.0},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 999.0

    def test_payload_config_path_wins(self, session_dir):
        _seed_state(
            session_dir,
            baseline_tput=800.0,
            baseline_config_path="/tmp/state.yaml",
        )

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "config_path": "/tmp/explicit.yaml"},
            session_dir=session_dir,
        )

        assert out["config_path"] == "/tmp/explicit.yaml"

    def test_payload_extra_args_wins(self, session_dir):
        _seed_state(session_dir, current_best_args="--from-state")

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "extra_server_args": "--from-payload"},
            session_dir=session_dir,
        )

        assert out["extra_server_args"] == "--from-payload"

    def test_empty_state_no_op(self, session_dir):
        _seed_state(session_dir)  # all defaults zero/empty

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"}, session_dir=session_dir,
        )

        assert "base_tput" not in out or out["base_tput"] in (0.0, 0)
        assert not out.get("config_path")
        assert not out.get("extra_server_args")

    def test_returns_shallow_copy_not_mutating_input(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)

        payload = {"kernel_id": "k_abc"}
        out = krh._fill_integrate_defaults_from_state(
            payload, session_dir=session_dir,
        )

        assert "base_tput" not in payload
        assert out["base_tput"] == 800.0

    def test_zero_base_tput_in_payload_triggers_fallback(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "base_tput": 0.0},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 800.0

    def test_zero_state_does_not_overwrite_explicit_payload(self, session_dir):
        _seed_state(session_dir, baseline_tput=0.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "base_tput": 750.0},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 750.0


class TestIntegrateHandlerHonoursStateDefault:
    @pytest.mark.asyncio
    async def test_missing_base_tput_in_payload_still_runs_when_state_has_one(
        self, session_dir, monkeypatch,
    ):
        """Pre-existing hard-check must not fire when state has a baseline.

        We don't run the full re-baseline pipeline — just confirm that
        integrate_handler advances PAST the ``base_tput <= 0`` early-out
        and reports a different failure mode (e.g. missing patch_path).
        That's enough to lock the defaulting wiring.
        """
        _seed_state(session_dir, baseline_tput=800.0)

        result = await krh.integrate_handler(
            {"kernel_id": "k_no_artifact"}, session_dir=session_dir,
        )

        assert result["status"] == "failed"
        assert result.get("error") != (
            "integrate_handler requires base_tput > 0 to compute KEEP/REVERT"
        )

    @pytest.mark.asyncio
    async def test_no_base_tput_anywhere_still_fails_with_clear_error(
        self, session_dir,
    ):
        result = await krh.integrate_handler(
            {"kernel_id": "k_orphan"}, session_dir=session_dir,
        )

        assert result["status"] == "failed"
        assert "base_tput" in result["error"]
