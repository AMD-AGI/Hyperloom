# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the scriptable/diffusion (xDiT) latency-domain surfacing:

* ``framework_registry.primary_metric_name`` (which field is the result).
* the close-out's final recipe carrying e2el into ``outcome.final``.
* ``SharedState._backfill_scriptable_latency`` deriving e2el from tput.
"""

from __future__ import annotations

from hyperloom.inference_optimizer import framework_registry as fr
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_outcome


class TestPrimaryMetricName:
    def test_scriptable_uses_e2el(self):
        assert fr.primary_metric_name("xdit") == "e2el_mean_ms"

    def test_serving_uses_throughput(self):
        assert fr.primary_metric_name("sglang") == "throughput_tok_s_per_gpu"
        assert fr.primary_metric_name(None) == "throughput_tok_s_per_gpu"


class TestFinalRecipeCarriesLatency:
    """The close-out's recipe is what ``outcome.final`` reports the latency from.

    A scriptable session's result *is* its latency, and a run that never
    validated its whole stack has no validation row to read it off -- so the
    recipe the close-out settled is the only author-time record of it.
    """

    def _final(self, current_best: dict) -> dict:
        """``outcome.final`` for a session whose close settled ``current_best``."""
        outcome = collect_v6_outcome(
            session={"stop_reason": "target_reached"},
            close={
                "final_recipe": {
                    "throughput": current_best.get("tput"),
                    "ttft_mean_ms": current_best.get("ttft_mean_ms"),
                    "e2el_mean_ms": current_best.get("e2el_mean_ms"),
                    "action_path": [str(current_best.get("action") or "")],
                    "extra_server_args": "",
                    "extra_envs": {},
                }
            },
            state={},
            timeline=[],
        )
        return outcome["final"]

    def test_scriptable_final_surfaces_the_derived_e2el(self):
        """``save`` derives the latency; the close-out records what it wrote.

        ``_backfill_scriptable_latency`` runs before ``state.json`` is written,
        so ``current_best`` already carries ``e2el_mean_ms`` at the close.
        """
        from hyperloom.orchestrator.state.shared_state import SharedState

        st = SharedState(session_id="s", model_name="m", model_path="/m")
        st.framework = "xdit"
        st.current_best = {"action": "explore", "tput": 1.098901}
        st._backfill_scriptable_latency()

        # 1000 / 1.098901 ~= 910.0
        assert self._final(st.current_best)["e2el_mean_ms"] == round(1000.0 / 1.098901, 4)

    def test_scriptable_final_prefers_measured_e2el(self):
        final = self._final({"action": "explore", "tput": 1.098901, "e2el_mean_ms": 980.0})
        assert final["e2el_mean_ms"] == 980.0

    def test_serving_final_has_no_derived_e2el(self):
        final = self._final({"action": "grid", "tput": 123.4})
        assert final["throughput_tok_s_per_gpu"] == 123.4
        assert final["e2el_mean_ms"] is None


class TestBackfillScriptableLatency:
    def _state(self, framework: str, cb: dict):
        from hyperloom.orchestrator.state.shared_state import SharedState

        st = SharedState(session_id="s", model_name="m", model_path="/m")
        st.framework = framework
        st.current_best = cb
        return st

    def test_scriptable_backfills_e2el_from_tput(self):
        st = self._state("xdit", {"action": "explore", "tput": 1.098901})
        st._backfill_scriptable_latency()
        assert st.current_best["e2el_mean_ms"] == round(1000.0 / 1.098901, 4)

    def test_measured_e2el_not_overwritten(self):
        st = self._state("xdit", {"action": "explore", "tput": 1.098901, "e2el_mean_ms": 980.0})
        st._backfill_scriptable_latency()
        assert st.current_best["e2el_mean_ms"] == 980.0

    def test_serving_is_noop(self):
        st = self._state("sglang", {"action": "grid", "tput": 123.4})
        st._backfill_scriptable_latency()
        assert "e2el_mean_ms" not in st.current_best

    def test_non_positive_tput_is_noop(self):
        st = self._state("xdit", {"action": "explore", "tput": 0.0})
        st._backfill_scriptable_latency()
        assert st.current_best.get("e2el_mean_ms") is None
