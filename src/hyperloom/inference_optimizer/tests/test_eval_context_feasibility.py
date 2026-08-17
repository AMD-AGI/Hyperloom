# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Hyperloom project
"""The accuracy gate must not spend a KEEP attempt on an eval that cannot run.

Observed on 195 sessions: the parameter search picks ``--max-model-len 2048``
for throughput, the gsm8k harness asks for a 2048-token completion on top of a
~1k-token five-shot prompt, and every request comes back HTTP 400. No verdict is
ever produced, so ``accuracy_pass`` stays ``None``. Because a positive baseline
accuracy (measured earlier, under the larger context the run started with) is
read as proof that eval works here, the missing verdict blocks the KEEP and the
round is recorded as a fair attempt. Three of those and the kernel is discarded
for a reason that has nothing to do with the kernel.
"""

import pytest

from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag
from hyperloom.orchestrator.state.shared_state import SharedState

# The generation budget the harness requests for gsm8k.
GSM8K_GEN_TOKENS = 2048


class TestServedContextHostsEval:
    """Can the serving configuration physically answer an eval request?"""

    def test_context_equal_to_the_generation_budget_cannot_host_a_prompt(self):
        """2048 of context and 2048 requested output leaves nothing for the prompt."""
        fits, reason = ag.served_context_hosts_eval(
            served_max_model_len=2048,
            eval_max_tokens=GSM8K_GEN_TOKENS,
        )
        assert fits is False
        assert "2048" in reason

    def test_the_real_session_configuration_is_rejected(self):
        """The exact shape seen in session e268b0be: env asks 6144, the server
        args override it to 2048, and the override is what the server honours."""
        served = ag.resolve_served_context(
            server_args=("--kv-cache-dtype fp8 --max-num-batched-tokens 32768 --max-model-len 2048 --async-scheduling"),
            env_max_model_len=6144,
        )
        assert served == 2048
        fits, _ = ag.served_context_hosts_eval(
            served_max_model_len=served,
            eval_max_tokens=GSM8K_GEN_TOKENS,
        )
        assert fits is False

    def test_the_context_the_env_asked_for_does_host_the_eval(self):
        fits, reason = ag.served_context_hosts_eval(
            served_max_model_len=6144,
            eval_max_tokens=GSM8K_GEN_TOKENS,
        )
        assert fits is True
        assert reason == ""

    def test_equals_form_of_the_flag_is_understood(self):
        assert (
            ag.resolve_served_context(
                server_args="--max-model-len=2048",
                env_max_model_len=6144,
            )
            == 2048
        )

    def test_env_is_used_when_the_server_args_are_silent(self):
        assert (
            ag.resolve_served_context(
                server_args="--kv-cache-dtype fp8",
                env_max_model_len=6144,
            )
            == 6144
        )

    def test_an_unknown_context_is_not_treated_as_infeasible(self):
        """Nothing is known, so nothing is claimed: never block on a guess."""
        fits, _ = ag.served_context_hosts_eval(
            served_max_model_len=0,
            eval_max_tokens=GSM8K_GEN_TOKENS,
        )
        assert fits is True

    @pytest.mark.parametrize("budget", [0, -1])
    def test_an_unbounded_generation_budget_is_not_treated_as_infeasible(self, budget):
        fits, _ = ag.served_context_hosts_eval(
            served_max_model_len=2048,
            eval_max_tokens=budget,
        )
        assert fits is True


class TestInfeasibleEvalIsAFault:
    """An eval that could not run is an environment fault, not a gate verdict."""

    def test_the_error_class_routes_to_the_fault_budget(self):
        """Faults get their own retry budget and never burn the REVERT quota."""
        assert SharedState._is_integrate_fault({"status": "ok", "error_class": ag.EVAL_KIND_CONTEXT_TOO_SMALL}) is True

    def test_a_genuine_regression_is_still_a_verdict_not_a_fault(self):
        assert SharedState._is_integrate_fault({"status": "ok", "error_class": "accuracy_regression"}) is False


class TestGradeMarksTheRoundInfeasible:
    """``_grade_integrate_accuracy`` must separate "eval broke" from "eval
    cannot run here"."""

    @staticmethod
    def _grade(monkeypatch, tmp_path, server_args):
        from hyperloom.orchestrator.kernel import request_handlers as rh

        # No score anywhere: the state this bug is about.
        monkeypatch.setattr(rh, "_maybe_revert_kernel_patch", lambda *_a, **_k: {})
        monkeypatch.setenv("MAX_MODEL_LEN", "6144")
        monkeypatch.delenv("HYPERLOOM_EVAL_MAX_TOKENS", raising=False)
        return rh._grade_integrate_accuracy(
            {"accuracy": None},
            session_dir=tmp_path,
            workspace=tmp_path,
            server_args=server_args,
        )

    def test_a_context_that_cannot_host_the_eval_is_flagged(self, monkeypatch, tmp_path):
        out = self._grade(monkeypatch, tmp_path, "--max-model-len 2048")
        assert out["infeasible"] is True
        assert out["accuracy_pass"] is None
        assert "2048" in out["reason"]

    def test_a_sufficient_context_is_not_flagged(self, monkeypatch, tmp_path):
        out = self._grade(monkeypatch, tmp_path, "--max-model-len 16384")
        assert out["infeasible"] is False

    def test_the_env_context_is_used_when_no_flag_is_present(self, monkeypatch, tmp_path):
        """MAX_MODEL_LEN 6144 against a 4096 budget leaves 2048 for the prompt."""
        out = self._grade(monkeypatch, tmp_path, "--kv-cache-dtype fp8")
        assert out["infeasible"] is False
