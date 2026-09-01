# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The door to the generated tier, and what it takes to get through it.

Five checkpoints in order -- gate, generate, sandbox, contract, referee -- each
ruling out a different kind of wrong and each ending the attempt without
touching the tuning run that hosts it. The referee is last and decisive:
everything before it can be satisfied by a script that reports what it was asked
to report, and only the referee establishes that a kernel actually got faster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kernelforge.gemm_tune.tier3 import gate, ledger, sandbox
from kernelforge.gemm_tune.tier3.coverage import CoverageGap
from kernelforge.gemm_tune.tier3.runner import attempt_generated_tuner


def _gap(table="odd.csv", misses=100, kind="no_tuner", tuner=None):
    return CoverageGap(
        table=table,
        tuner=tuner,
        env_var="AITER_CONFIG_ODD",
        key_schema=["M", "N", "K"],
        miss_count=misses,
        distinct_keys=misses,
        reason="no tuner is registered",
        kind=kind,
    )


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    for var in (gate.ALLOW_ENV, gate.DISABLE_ENV, gate.MIN_MISSES_ENV):
        monkeypatch.delenv(var, raising=False)


class TestGate:
    def test_open_by_default_for_a_table_nothing_owns(self):
        # The other conditions already restrict this to gaps where no tuner
        # exists, so the time a generated one spends is not taken from a tuner
        # that would have covered the table -- there is none.
        d = gate.should_generate([_gap()])
        assert d.allowed and d.gap.table == "odd.csv"

    def test_the_kill_switch_closes_it_without_naming_tables(self, monkeypatch):
        monkeypatch.setenv(gate.DISABLE_ENV, "1")
        d = gate.should_generate([_gap()])
        assert not d.allowed and gate.DISABLE_ENV in d.reasons[0]

    def test_a_list_narrows_rather_than_enables(self, monkeypatch):
        monkeypatch.setenv(gate.ALLOW_ENV, "odd.csv")
        assert gate.should_generate([_gap()]).allowed
        monkeypatch.setenv(gate.ALLOW_ENV, "something_else.csv")
        d = gate.should_generate([_gap()])
        assert not d.allowed and "does not list it" in d.reasons[0]

    def test_a_wildcard_is_the_same_as_no_list(self, monkeypatch):
        monkeypatch.setenv(gate.ALLOW_ENV, "*")
        assert gate.should_generate([_gap()]).allowed

    def test_a_tuner_that_exists_never_opens_the_gate(self, monkeypatch):
        # Whatever the whitelist says. Generating a second tuner for a table
        # that already has one papers over whatever stopped the first.
        monkeypatch.setenv(gate.ALLOW_ENV, "*")
        for kind in ("not_selected", "skipped"):
            d = gate.should_generate([_gap(kind=kind, tuner="a8w8")])
            assert not d.allowed
            assert kind in d.reasons[0]

    def test_too_little_demand_stays_closed(self, monkeypatch):
        monkeypatch.setenv(gate.ALLOW_ENV, "*")
        monkeypatch.setenv(gate.MIN_MISSES_ENV, "50")
        d = gate.should_generate([_gap(misses=12)])
        assert not d.allowed and "below the floor" in d.reasons[0]

    def test_a_gap_with_no_key_schema_cannot_be_written_against(self, monkeypatch):
        monkeypatch.setenv(gate.ALLOW_ENV, "*")
        g = _gap()
        g.key_schema = []
        d = gate.should_generate([g])
        assert not d.allowed and "no key schema" in d.reasons[0]

    def test_the_most_demanded_eligible_gap_wins(self, monkeypatch):
        monkeypatch.setenv(gate.ALLOW_ENV, "*")
        d = gate.should_generate([_gap("small.csv", 30), _gap("big.csv", 900)])
        assert d.gap.table == "big.csv"


class TestSandbox:
    def _script(self, tmp_path, body):
        p = tmp_path / "tuner.py"
        p.write_text(body, encoding="utf-8")
        return p

    def test_expected_files_decide_the_outcome_not_the_exit_code(self, tmp_path):
        # The aiter tuners in this same pipeline exit 1 on complete success, so
        # a script's return code cannot be the signal here either.
        out = tmp_path / "out.csv"
        script = self._script(
            tmp_path,
            f"""
import sys
open({str(out)!r}, "w").write("done\\n")
sys.exit(1)
""",
        )
        r = sandbox.run_generated_tuner(script, tmp_path, expect=[out], timeout_s=60)
        assert r.ok and r.returncode == 1

    def test_a_script_that_writes_nothing_fails(self, tmp_path):
        out = tmp_path / "out.csv"
        script = self._script(tmp_path, "print('I did nothing')\n")
        r = sandbox.run_generated_tuner(script, tmp_path, expect=[out], timeout_s=60)
        assert not r.ok and out.name not in "".join(r.produced)

    def test_stale_output_from_a_previous_attempt_is_cleared(self, tmp_path):
        out = tmp_path / "out.csv"
        out.write_text("last week's rows\n", encoding="utf-8")
        script = self._script(tmp_path, "print('nothing new')\n")
        r = sandbox.run_generated_tuner(script, tmp_path, expect=[out], timeout_s=60)
        assert not r.ok, "a leftover file would otherwise pass as this run's output"

    def test_a_timeout_keeps_what_was_already_written(self, tmp_path):
        out = tmp_path / "out.csv"
        script = self._script(
            tmp_path,
            f"""
import time
open({str(out)!r}, "w").write("partial\\n")
time.sleep(30)
""",
        )
        r = sandbox.run_generated_tuner(script, tmp_path, expect=[out], timeout_s=3)
        assert r.timed_out and r.ok, "partial output is the contract check's to judge"

    def test_a_crash_is_a_result_not_an_exception(self, tmp_path):
        script = self._script(tmp_path, "raise SystemExit(139)\n")
        r = sandbox.run_generated_tuner(script, tmp_path, expect=[tmp_path / "out.csv"], timeout_s=60)
        assert not r.ok and r.returncode == 139

    def test_the_child_is_confined_to_one_device(self, tmp_path):
        out = tmp_path / "env.txt"
        script = self._script(
            tmp_path,
            f"""
import os
open({str(out)!r}, "w").write(os.environ.get("HIP_VISIBLE_DEVICES", "?"))
""",
        )
        sandbox.run_generated_tuner(script, tmp_path, expect=[out], gpu_id="3", timeout_s=60)
        assert out.read_text(encoding="utf-8") == "3"


class TestLedger:
    def test_an_edited_script_starts_over(self, tmp_path):
        p = tmp_path / "t.py"
        p.write_text("a", encoding="utf-8")
        first = ledger.script_digest(p)
        p.write_text("b", encoding="utf-8")
        assert ledger.script_digest(p) != first

    def test_trust_needs_successes_across_models(self, tmp_path):
        path = tmp_path / "ledger.json"
        d = "abc123"
        for i in range(3):
            r = ledger.record_outcome(
                path,
                digest=d,
                table="t.csv",
                model="model-a",
                improved=True,
                speedup=1.2,
            )
        # Three successes, one model: not enough.
        assert r.successes == 3 and not r.eligible_for_trust
        r = ledger.record_outcome(
            path,
            digest=d,
            table="t.csv",
            model="model-b",
            improved=True,
            speedup=1.1,
        )
        assert r.eligible_for_trust

    def test_one_measured_regression_disqualifies_it(self, tmp_path):
        path = tmp_path / "ledger.json"
        d = "abc123"
        for model in ("a", "b", "c"):
            ledger.record_outcome(
                path,
                digest=d,
                table="t.csv",
                model=model,
                improved=True,
                speedup=1.2,
            )
        r = ledger.record_outcome(
            path,
            digest=d,
            table="t.csv",
            model="d",
            improved=False,
            speedup=0.8,
        )
        assert r.regressions == 1 and not r.eligible_for_trust

    def test_finding_nothing_is_not_a_regression(self, tmp_path):
        # A tuner that searched honestly and found no win has not misbehaved.
        path = tmp_path / "ledger.json"
        r = ledger.record_outcome(
            path,
            digest="d",
            table="t.csv",
            model="a",
            improved=False,
            speedup=None,
        )
        assert r.regressions == 0

    def test_eligibility_is_not_trust(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ledger.TRUST_ENV, raising=False)
        path = tmp_path / "ledger.json"
        for model in ("a", "b", "c"):
            r = ledger.record_outcome(
                path,
                digest="d1",
                table="t.csv",
                model=model,
                improved=True,
                speedup=1.3,
            )
        assert r.eligible_for_trust
        assert not ledger.is_trusted("d1"), "only an operator grants trust"
        monkeypatch.setenv(ledger.TRUST_ENV, "d1")
        assert ledger.is_trusted("d1")

    def test_the_ledger_survives_a_corrupt_file(self, tmp_path):
        path = tmp_path / "ledger.json"
        path.write_text("{not json", encoding="utf-8")
        r = ledger.record_outcome(
            path,
            digest="d",
            table="t.csv",
            model="a",
            improved=True,
            speedup=1.1,
        )
        assert r.successes == 1
        assert json.loads(path.read_text(encoding="utf-8"))["d"]["successes"] == 1


class TestPlanPreviewsWhatRunDoes:
    """A preview that answers a different question than the thing it previews
    is worse than no preview: it is consulted precisely when someone is
    unsure, and it was showing TunableOp skipped for inputs under which the
    real run selects it."""

    def _src(self, name):
        import inspect

        from kernelforge.gemm_tune import cli

        return inspect.getsource(getattr(cli, name).callback)

    def test_plan_derives_demand_from_the_serving_log_like_run(self):
        for name in ("run", "plan"):
            assert "_demand_from_serving_log(" in self._src(name), name

    def test_plan_counts_demand_as_a_shape_source_like_run(self):
        for name in ("run", "plan"):
            src = self._src(name)
            idx = src.index("has_shapes_json=")
            assert "demand_json" in src[idx : idx + 90], name

    def test_plan_accepts_an_explicit_demand_file(self):
        from kernelforge.gemm_tune import cli

        assert any("--demand" in (p.opts or []) for p in cli.plan.params), (
            "run takes --demand; plan must too or they diverge again"
        )


class TestTheCliActuallyReachesTier3:
    """The whole tier was unreachable from production and nothing said so.

    Every stage had tests and they all passed, because they called the stages
    directly. Nothing asserted that the CLI ever calls any of them, so the
    tier sat fully built and entirely disconnected.
    """

    def test_the_cli_has_a_call_site(self):
        import inspect

        from kernelforge.gemm_tune import cli

        source = inspect.getsource(cli)
        assert "_attempt_tier3(" in source
        # Defined and called, not merely defined.
        assert source.count("_attempt_tier3(") >= 2

    def test_it_runs_after_the_selected_tuners_not_beside_them(self):
        # This ordering is the guarantee that a generated tuner cannot take
        # time from one that was going to produce something.
        import inspect

        from kernelforge.gemm_tune import cli

        # click wraps the command, so reach the function it decorated.
        source = inspect.getsource(cli.run.callback)
        assert source.index("tuner_instance.execute()") < source.index("_attempt_tier3(")

    def test_a_table_with_no_adapter_is_refused_rather_than_approximated(self):
        from kernelforge.gemm_tune.tier3.dispatch import adapters_for

        assert adapters_for("a4w4_blockscale_tuned_gemm.csv") is None
        assert adapters_for("bf16_tuned_gemm.csv") is not None


class TestTheProviderCallMatchesTheProviderAPI:
    """The authoring call is only exercised with a real provider installed.

    Nothing else here reaches it, so a wrong argument list sits undetected
    until the one run that tries to generate -- and that run is exactly the
    one nobody is watching. These pin the call against the real signatures.
    """

    def test_the_runtime_call_binds_against_the_real_signature(self):
        import inspect

        registry = pytest.importorskip("kernelforge.agent_backends.registry")

        # What generate.call_agent passes. ``provider`` is positional and
        # required; calling it with keywords only raises TypeError.
        inspect.signature(registry.resolve_agent_runtime).bind(
            "claude",
            model="",
            timeout_sec=1800,
        )

    def test_provider_selection_yields_something_with_a_name(self):
        import inspect

        registry = pytest.importorskip("kernelforge.agent_backends.registry")

        inspect.signature(registry.select_default_agent_provider).bind("")
        assert "name" in inspect.get_annotations(registry.AgentProvider, eval_str=False) or hasattr(
            registry.AgentProvider, "name"
        )


class TestTheWholeChain:
    """A generated tuner only counts once our own clock agrees."""

    def _writes(self, tmp_path, rows, candidates):
        out = tmp_path / "out.csv"
        cj = tmp_path / "candidates.json"

        def _fake_generate(mandate, work_dir, **kw):
            from kernelforge.gemm_tune.tier3.generate import GeneratedTuner

            script = work_dir / "tuner.py"
            script.write_text("# generated\n", encoding="utf-8")
            Path(mandate.output_csv).write_text(rows, encoding="utf-8")
            Path(mandate.candidates_json).write_text(json.dumps(candidates), encoding="utf-8")
            return GeneratedTuner(True, script, "", "fake", "s1")

        return out, cj, _fake_generate

    @pytest.fixture
    def open_gate(self, monkeypatch):
        monkeypatch.setenv(gate.ALLOW_ENV, "*")
        monkeypatch.delenv(ledger.TRUST_ENV, raising=False)

    _HDR = "M,N,K,backend,config,default_us,tuned_us,improved"
    _ROWS = _HDR + "\n16,1536,7168,x,c=1,10.0,5.0,True\n"
    _CANDS = {"16x1536x7168": [{"backend": "x", "config": "c=1"}]}

    def _run(self, tmp_path, monkeypatch, *, rows, cands, dispatch_cost=1e-6, correct=True, with_dispatch=True):
        _, _, fake_gen = self._writes(tmp_path, rows, cands)
        monkeypatch.setattr("kernelforge.gemm_tune.tier3.runner.generate_tuner", fake_gen)
        monkeypatch.setattr(
            "kernelforge.gemm_tune.tier3.runner.run_generated_tuner",
            lambda script, wd, **kw: sandbox.SandboxResult(
                True, 0, 1.0, produced=[str(p) for p in kw.get("expect", [])]
            ),
        )
        now = {"t": 0.0}
        monkeypatch.setattr("kernelforge.gemm_tune.tier3.referee.time.perf_counter", lambda: now["t"])

        def _call(cost):
            def _fn():
                now["t"] += cost

            return _fn

        kwargs = {}
        if with_dispatch:
            kwargs = {
                "make_baseline": lambda shape: _call(2e-6),
                "make_dispatch": lambda shape: lambda c: _call(dispatch_cost),
                "make_correctness": lambda shape: lambda call: correct,
            }
        return attempt_generated_tuner(
            [_gap()],
            lambda g: [{"M": 16, "N": 1536, "K": 7168}],
            tmp_path,
            model_name="qwen3-8b",
            **kwargs,
        )

    def test_a_genuinely_faster_candidate_passes(self, tmp_path, monkeypatch, open_gate):
        out = self._run(tmp_path, monkeypatch, rows=self._ROWS, cands=self._CANDS)
        assert out.stage == "referee" and out.ok
        assert out.improved_shapes == 1
        # Recorded, but still a candidate until a person says otherwise.
        assert not out.operator_signed

    def test_the_scripts_own_claim_does_not_survive_re_timing(self, tmp_path, monkeypatch, open_gate):
        # Its CSV says 2x. Our clock says it is slower.
        out = self._run(tmp_path, monkeypatch, rows=self._ROWS, cands=self._CANDS, dispatch_cost=8e-6)
        assert not out.ok and "no shape improved" in out.reason

    def test_output_that_contradicts_itself_is_rejected_before_the_gpu(self, tmp_path, monkeypatch, open_gate):
        rows = self._HDR + "\n16,1536,7168,x,c=1,5.0,10.0,True\n"
        out = self._run(tmp_path, monkeypatch, rows=rows, cands=self._CANDS)
        assert out.stage == "contract" and not out.ok
        assert "contradicts" in out.reason

    def test_an_incorrect_candidate_never_becomes_a_win(self, tmp_path, monkeypatch, open_gate):
        out = self._run(tmp_path, monkeypatch, rows=self._ROWS, cands=self._CANDS, correct=False)
        assert not out.ok
        assert out.judgements[0].rejected_incorrect == 1

    def test_without_a_dispatch_nothing_is_emitted(self, tmp_path, monkeypatch, open_gate):
        # An unverified generated tuner is exactly what this tier must not emit.
        out = self._run(tmp_path, monkeypatch, rows=self._ROWS, cands=self._CANDS, with_dispatch=False)
        assert out.stage == "referee" and not out.ok
        assert "cannot be re-timed" in out.reason

    def test_the_kill_switch_stops_it_before_anything_happens(self, tmp_path, monkeypatch):
        monkeypatch.setenv(gate.DISABLE_ENV, "1")
        out = attempt_generated_tuner(
            [_gap()],
            lambda g: [],
            tmp_path,
        )
        assert not out.attempted and out.stage == "gate"

    def test_the_outcome_is_written_where_a_human_can_read_it(self, tmp_path, monkeypatch, open_gate):
        self._run(tmp_path, monkeypatch, rows=self._ROWS, cands=self._CANDS)
        written = list((tmp_path / "tier3").rglob("outcome.json"))
        assert written
        payload = json.loads(written[0].read_text(encoding="utf-8"))
        assert payload["ok"] is True and payload["ledger"]["successes"] == 1
        assert (tmp_path / "tier3" / "ledger.json").is_file()


def test_the_generator_degrades_when_no_agent_provider_is_installed(tmp_path, monkeypatch):
    """The standalone wheel must tune without the LLM stack present."""
    from kernelforge.gemm_tune.tier3.generate import generate_tuner
    from kernelforge.gemm_tune.tier3.mandate import build_mandate

    monkeypatch.setitem(sys.modules, "kernelforge.agent_backends.registry", None)
    m = build_mandate(_gap(), [{"M": 16, "N": 1536, "K": 7168}])
    result = generate_tuner(m, tmp_path)
    assert not result.ok
    assert "no agent provider" in result.reason or "unusable" in result.reason
