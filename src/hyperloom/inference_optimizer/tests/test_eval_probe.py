# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the lm-eval generation-pathology probe and the per-request bounds.

Both ship as one source string in ``_inferencex_patcher`` (it is appended to
lm-eval's ``lm_eval_sitecustomize.py``), so no linter or import ever type-checks
it. These tests exec it against stub lm-eval modules — hermetic, so they pin the
contract whether or not lm-eval is installed.

The two answer different failures and must not be conflated. The probe handles a
model that never terminates at all, and pays for it by voiding the whole eval
(~0 score). The bounds handle a healthy model whose hardest samples do not
converge: those are truncated one at a time so the rest of the measurement
survives — which is why the probe's ratio threshold cannot be lowered to cover
them.

What the probe must guarantee:

* a model whose answers terminate is never short-circuited;
* once a decisive share of responses hit the ``max_tokens`` cap, the remaining
  generate requests are answered with an empty string so lm-eval still writes a
  ``results*.json`` scoring ~0 instead of running for hours;
* loglikelihood requests are never short-circuited (they have no EOS to emit);
* a malformed response degrades to "eval runs as before", never a crash.

What the bounds must guarantee:

* every generate request carries a ceiling, with no cooperation from the caller
  — the accuracy gate is differential, so a ceiling only one arm has is worse
  than none;
* an existing lower ceiling is never raised;
* loglikelihood payloads are left alone;
* caller-supplied stop strings outrank upstream's, which sends at most four;
* the run reports how often the ceiling was hit, so it can be falsified.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    EVAL_KIND_GENERATION_PATHOLOGY,
    EVAL_PROBE_FILENAME,
    eval_probe_summary,
    read_eval_probe,
)
from hyperloom.orchestrator.actions.executors._inferencex_patcher import _EVAL_PROBE_PY


class _StubTemplateAPI:
    """Stands in for ``lm_eval.models.api_models.TemplateAPI``."""

    _concurrent = 4

    def __init__(self) -> None:
        self.inner_calls = 0
        self.cached: list[tuple[str, Any, str]] = []
        self.cache_hook = types.SimpleNamespace(
            add_partial=lambda method, key, res: self.cached.append((method, key, res))
        )

    async def amodel_call(self, session, sem, messages, **kwargs):
        self.inner_calls += 1
        return ["real answer"] * len(messages)


class _StubLocalChatCompletion:
    """Stands in for ``lm_eval.models.openai_completions.LocalChatCompletion``."""

    model = "stub-model"
    _max_gen_toks = 256

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        """Reproduce the pinned upstream payload builder.

        Faithful in the two respects the bounds shim depends on: ``stop`` is
        truncated to four entries (so ordering decides which terminators
        survive), and the loglikelihood branch carries ``max_tokens=1``.
        """
        gen_kwargs = dict(gen_kwargs or {})
        if not generate:
            return {"model": self.model, "prompt": messages, "max_tokens": 1, "logprobs": 1, "seed": seed}
        max_tokens = gen_kwargs.pop("max_tokens", self._max_gen_toks)
        stop = list(gen_kwargs.pop("until", None) or [])
        if eos and eos not in stop:
            stop.append(eos)
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "stop": stop[:4],
            "seed": seed,
        }

    @staticmethod
    def parse_generations(outputs, **kwargs):
        return ["upstream"]


def _install_stub_lm_eval(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Put stub ``lm_eval`` modules on ``sys.modules`` for the probe to patch.

    The injected block patches by class attribute assignment, so each install
    gets throwaway subclasses. Handing out the shared classes instead makes the
    wrappers accumulate across tests: ``monkeypatch`` would record the previous
    test's wrapper as the value to "restore", so the leak outlives every attempt
    to undo it.

    Returns:
        The stub ``api_models`` and ``openai_completions`` modules.
    """
    pkg = types.ModuleType("lm_eval")
    models = types.ModuleType("lm_eval.models")
    api_models = types.ModuleType("lm_eval.models.api_models")
    openai_completions = types.ModuleType("lm_eval.models.openai_completions")
    api_models.TemplateAPI = type("TemplateAPI", (_StubTemplateAPI,), {})
    openai_completions.LocalChatCompletion = type("LocalChatCompletion", (_StubLocalChatCompletion,), {})
    pkg.models = models
    models.api_models = api_models
    models.openai_completions = openai_completions
    for name, mod in (
        ("lm_eval", pkg),
        ("lm_eval.models", models),
        ("lm_eval.models.api_models", api_models),
        ("lm_eval.models.openai_completions", openai_completions),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return api_models, openai_completions


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str | None):
    """Install the injected block with an explicit env; ``None`` unsets a variable.

    Returns a namespace whose ``flush()`` runs only the exit hooks this install
    registered. Interpreter-wide ``atexit`` is left alone on purpose: the hooks
    resolve ``RESULT_DIR`` when they fire, so a real registration would have
    every past test's hook write into the current test's directory.
    """
    env.setdefault("RESULT_DIR", str(tmp_path))
    monkeypatch.delenv("HYPERLOOM_EVAL_PROBE", raising=False)
    for key, val in env.items():
        if val is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, val)
    api_models, openai_completions = _install_stub_lm_eval(monkeypatch)
    exit_hooks: list[Any] = []
    monkeypatch.setattr(atexit, "register", exit_hooks.append)
    exec(compile(_EVAL_PROBE_PY, "<probe>", "exec"), {"__name__": "sitecustomize"})
    return types.SimpleNamespace(
        api_models=api_models,
        openai_completions=openai_completions,
        cls=openai_completions.LocalChatCompletion,
        result_dir=tmp_path,
        flush=lambda: [hook() for hook in exit_hooks],
    )


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install the probe over stub lm-eval modules with a low trip threshold."""
    return _install(
        monkeypatch,
        tmp_path,
        HYPERLOOM_EVAL_PROBE_MIN_SAMPLES="8",
        HYPERLOOM_EVAL_PROBE_LENGTH_RATIO="0.75",
    )


def _response(finish_reason: str, completion_tokens: int = 16384) -> dict[str, Any]:
    """One OpenAI-shaped chat-completion response."""
    return {
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"content": "x"}}],
        "usage": {"completion_tokens": completion_tokens},
    }


def _feed(probe, finish_reason: str, count: int) -> None:
    """Push ``count`` responses through the observation hook."""
    for _ in range(count):
        probe.openai_completions.LocalChatCompletion.parse_generations(outputs=_response(finish_reason))


async def _call(probe, obj: _StubTemplateAPI, *, generate: bool = True, cache_keys=None):
    return await probe.api_models.TemplateAPI.amodel_call(
        obj, None, None, ["msg"], generate=generate, cache_keys=cache_keys
    )


def test_probe_installs_over_upstream_patches(probe):
    """The probe wraps rather than replaces, so InferenceX's own
    parse_generations fix (appended just above it) stays in effect."""
    out = probe.openai_completions.LocalChatCompletion.parse_generations(outputs=_response("stop"))
    assert out == ["upstream"]


def test_below_min_samples_does_not_short_circuit(probe):
    """One sample short of the minimum is not yet evidence; the eval runs on."""
    obj = _StubTemplateAPI()
    _feed(probe, "length", 7)
    assert asyncio.run(_call(probe, obj)) == ["real answer"]
    assert obj.inner_calls == 1


def test_trips_and_short_circuits_once_decisive(probe):
    obj = _StubTemplateAPI()
    _feed(probe, "length", 8)
    assert asyncio.run(_call(probe, obj)) == [""]
    assert obj.inner_calls == 0, "a tripped probe must not reach the server at all"


def test_terminating_model_is_never_short_circuited(probe):
    """The whole point: a model that emits EOS must be graded normally even
    though some answers legitimately hit the cap."""
    obj = _StubTemplateAPI()
    _feed(probe, "stop", 8)
    _feed(probe, "length", 1)
    assert asyncio.run(_call(probe, obj)) == ["real answer"]
    assert not (probe.result_dir / EVAL_PROBE_FILENAME).exists()


def test_loglikelihood_requests_are_never_short_circuited(probe):
    """Loglikelihood scoring emits no tokens, so the pathology cannot apply."""
    obj = _StubTemplateAPI()
    _feed(probe, "length", 8)
    assert asyncio.run(_call(probe, obj, generate=False)) == ["real answer"]
    assert obj.inner_calls == 1


def test_short_circuit_still_populates_the_harness_cache(probe):
    """lm-eval reconciles answers against cache_keys; skipping the hook would
    desync the run it is supposed to let finish cleanly."""
    obj = _StubTemplateAPI()
    _feed(probe, "length", 8)
    asyncio.run(_call(probe, obj, cache_keys=[("ctx", "kwargs")]))
    assert obj.cached == [("generate_until", ("ctx", "kwargs"), "")]


def test_gate_survives_a_fresh_event_loop(probe):
    """lm-eval calls asyncio.run() once per batch. A semaphore binds to the
    first loop that awaits it, so a non-loop-keyed gate would raise here."""
    obj = _StubTemplateAPI()
    asyncio.run(_call(probe, obj))
    _feed(probe, "length", 8)
    assert asyncio.run(_call(probe, obj)) == [""]


def test_sidecar_records_the_evidence(probe):
    _feed(probe, "length", 8)
    sidecar = probe.result_dir / EVAL_PROBE_FILENAME
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["reason"] == "model_not_terminating"
    assert record["observed_samples"] == 8
    assert record["finish_reason_length"] == 8
    assert record["cap_hits"] == 8
    assert record["max_completion_tokens_seen"] == 16384
    # parse_eval_results globs results*.json for the score; a probe sidecar
    # matching that name would be read as an lm-eval result file.
    assert not sidecar.name.startswith("results")


def test_sidecar_is_written_once(probe):
    """Every subsequent response would otherwise rewrite it with a diluted
    ratio, since short-circuited requests never report a finish_reason."""
    _feed(probe, "length", 8)
    first = (probe.result_dir / EVAL_PROBE_FILENAME).read_text(encoding="utf-8")
    _feed(probe, "length", 20)
    assert (probe.result_dir / EVAL_PROBE_FILENAME).read_text(encoding="utf-8") == first


def test_probe_can_be_disabled(monkeypatch, tmp_path):
    """Only the probe's own hooks come off. The bounds shim shares this block and
    installs regardless, so ``parse_generations`` stays wrapped — it just has to
    keep chaining through to upstream's."""
    monkeypatch.setenv("RESULT_DIR", str(tmp_path))
    monkeypatch.setenv("HYPERLOOM_EVAL_PROBE", "0")
    api_models, openai_completions = _install_stub_lm_eval(monkeypatch)
    pristine = api_models.TemplateAPI.amodel_call

    exec(compile(_EVAL_PROBE_PY, "<probe>", "exec"), {"__name__": "sitecustomize"})

    assert api_models.TemplateAPI.amodel_call is pristine
    assert openai_completions.LocalChatCompletion.parse_generations(outputs={}) == ["upstream"]


def test_probe_surfaces_import_failure_when_lm_eval_absent(monkeypatch):
    """A missing lm-eval must raise, not be swallowed: CPython's
    ``site.execsitecustomize()`` prints it to stderr and keeps the interpreter
    alive, so the failure is visible in the benchmark log."""
    import pytest

    for name in ("lm_eval", "lm_eval.models", "lm_eval.models.api_models"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError):
        exec(compile(_EVAL_PROBE_PY, "<probe>", "exec"), {"__name__": "sitecustomize"})


def test_probe_survives_malformed_responses(probe):
    """A server that answers with something unexpected must not take the eval
    down with it."""
    obj = _StubTemplateAPI()
    for junk in (None, [], {"choices": "not-a-list"}, {"choices": [None]}, {"usage": "nope"}):
        probe.openai_completions.LocalChatCompletion.parse_generations(outputs=junk)
    assert asyncio.run(_call(probe, obj)) == ["real answer"]


def test_read_eval_probe_finds_a_nested_sidecar(probe, tmp_path):
    """The baseline double-run evaluates in the warmup round, whose RESULT_DIR
    nests under the task workspace."""
    nested = tmp_path / "warmup_round"
    nested.mkdir()
    (nested / EVAL_PROBE_FILENAME).write_text(json.dumps({"reason": "model_not_terminating"}), encoding="utf-8")

    record = read_eval_probe(tmp_path)

    assert record is not None
    assert record["kind"] == EVAL_KIND_GENERATION_PATHOLOGY
    assert record["source_file"].endswith(EVAL_PROBE_FILENAME)


def test_long_answers_below_the_ceiling_do_not_trip(probe):
    """``finish_reason=length`` alone is not the pathology. lm-eval sizes
    max_tokens per request from the remaining context, so a truncated-but-
    terminating model produces capped responses at several different lengths;
    only the ones piled on the ceiling are evidence of a runaway loop."""
    obj = _StubTemplateAPI()
    for tokens in (1024,) * 4 + (2048,) * 4:
        probe.openai_completions.LocalChatCompletion.parse_generations(outputs=_response("length", tokens))
    assert asyncio.run(_call(probe, obj)) == ["real answer"]
    assert not (probe.result_dir / EVAL_PROBE_FILENAME).exists()


def test_length_ratio_zero_falls_back_to_the_default(monkeypatch, tmp_path):
    """0 is exactly what an operator reaches for to disable the probe. Taken
    literally it makes the ratio test vacuously true and guillotines every eval,
    so an out-of-range value must fall back to the default, not be clamped."""
    p = _install(
        monkeypatch,
        tmp_path,
        HYPERLOOM_EVAL_PROBE_MIN_SAMPLES="8",
        HYPERLOOM_EVAL_PROBE_LENGTH_RATIO="0",
    )
    obj = _StubTemplateAPI()
    _feed(p, "stop", 8)
    assert asyncio.run(_call(p, obj)) == ["real answer"]
    assert not (tmp_path / EVAL_PROBE_FILENAME).exists()


def test_min_samples_below_the_floor_falls_back_to_the_default(monkeypatch, tmp_path):
    """A one-sample window would end the eval on the first capped response."""
    p = _install(
        monkeypatch,
        tmp_path,
        HYPERLOOM_EVAL_PROBE_MIN_SAMPLES="1",
        HYPERLOOM_EVAL_PROBE_LENGTH_RATIO="0.75",
    )
    obj = _StubTemplateAPI()
    _feed(p, "length", 8)
    assert asyncio.run(_call(p, obj)) == ["real answer"]


def test_no_result_dir_keeps_the_sidecar_out_of_the_cwd(monkeypatch, tmp_path):
    """Without ``$RESULT_DIR`` the cwd is InferenceX's checkout, and writing
    there is the artifact escape the _EVAL_DEST_* patch exists to prevent. The
    record still reaches stderr, and the eval is still cut short."""
    monkeypatch.chdir(tmp_path)
    p = _install(monkeypatch, tmp_path, RESULT_DIR=None, HYPERLOOM_EVAL_PROBE_MIN_SAMPLES="8")
    obj = _StubTemplateAPI()
    _feed(p, "length", 8)
    assert asyncio.run(_call(p, obj)) == [""]
    assert list(tmp_path.iterdir()) == []


def test_install_drops_a_stale_sidecar(monkeypatch, tmp_path):
    """The eval-failure retry reuses ``$RESULT_DIR``, so a sidecar left by the
    previous attempt would be read as this run's verdict."""
    stale = tmp_path / EVAL_PROBE_FILENAME
    stale.write_text(json.dumps({"reason": "model_not_terminating"}), encoding="utf-8")

    _install(monkeypatch, tmp_path, HYPERLOOM_EVAL_PROBE_MIN_SAMPLES="8")

    assert not stale.exists()


def test_read_eval_probe_prefers_the_newest_sidecar(tmp_path):
    """``integrate_patch`` searches the grid slot, where sibling variants each
    own a sidecar, and attempt dirs are hash-named — so path order says nothing
    about which eval ran last."""
    older = tmp_path / "zzz_first" / EVAL_PROBE_FILENAME
    newer = tmp_path / "aaa_second" / EVAL_PROBE_FILENAME
    for path, hits in ((older, 11), (newer, 22)):
        path.parent.mkdir()
        path.write_text(json.dumps({"cap_hits": hits}), encoding="utf-8")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    record = read_eval_probe(tmp_path)

    assert record is not None
    assert record["cap_hits"] == 22


def test_read_eval_probe_is_none_without_a_sidecar(tmp_path):
    """No sidecar is the ordinary case: the model terminated its answers."""
    assert read_eval_probe(tmp_path) is None


def test_read_eval_probe_tolerates_corrupt_json(tmp_path):
    (tmp_path / EVAL_PROBE_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_eval_probe(tmp_path) is None


def test_eval_probe_summary_is_empty_without_a_probe():
    assert eval_probe_summary(None) == ""


def test_eval_probe_summary_names_the_kind_and_the_evidence():
    summary = eval_probe_summary({"observed_samples": 16, "cap_hits": 16, "max_completion_tokens_seen": 16384})
    assert EVAL_KIND_GENERATION_PATHOLOGY in summary
    assert "16/16" in summary
    assert "16384" in summary


def test_probe_record_reaches_session_breakdown(tmp_path):
    """End of the traceability chain: the writeback audit stores the record in
    the attempt's ``extras``, and the collector must carry it into
    ``session_breakdown.json``. Without this, a baseline accuracy of 0 gives a
    reader no way to tell a broken generation loop from wrong answers."""
    from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_baseline

    probe_record = {
        "kind": EVAL_KIND_GENERATION_PATHOLOGY,
        "reason": "model_not_terminating",
        "observed_samples": 16,
        "finish_reason_length": 16,
    }
    state = {
        "baseline_tput": 1234.0,
        "baseline_accuracy": 0.0,
        "baseline_attempts": [
            {
                "ts": "2026-08-03T00:00:00+00:00",
                "task_id": "t1",
                "status": "succeeded",
                "decision": "promoted",
                "key_metric": 1234.0,
                "error_class": None,
                "extras": {"eval_probe": probe_record},
            }
        ],
    }

    section = collect_baseline(tmp_path, state, [])

    assert section["attempts_history"][0]["extras"]["eval_probe"] == probe_record


def test_breakdown_attempt_extras_default_to_empty(tmp_path):
    """Attempts recorded before this field existed must still render."""
    from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_baseline

    state = {"baseline_attempts": [{"ts": "2026-08-03T00:00:00+00:00", "task_id": "t1", "status": "failed"}]}

    section = collect_baseline(tmp_path, state, [])

    assert section["attempts_history"][0]["extras"] == {}


# ---------------------------------------------------------------------------
# Per-request bounds
# ---------------------------------------------------------------------------

BOUNDS_FILENAME = "hyperloom_eval_bounds.json"


@pytest.fixture
def bounds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install the block with the probe disabled, isolating the bounds shim."""
    return _install(
        monkeypatch,
        tmp_path,
        HYPERLOOM_EVAL_PROBE="0",
        HYPERLOOM_EVAL_MAX_TOKENS=None,
        HYPERLOOM_EVAL_STOP_STRINGS=None,
    )


def _payload(env, generate=True, **kwargs):
    """Build one payload through the patched class, as lm-eval would."""
    return env.cls()._create_payload([{"role": "user", "content": "hi"}], generate=generate, **kwargs)


def test_bounds_clamp_max_tokens_with_no_env_set(bounds):
    """The ceiling cannot depend on the caller remembering to set it: InferenceX
    passes ``max_tokens=min(16384, ctx-4096)`` and both gate arms must share
    whatever bound applies."""
    assert _payload(bounds, gen_kwargs={"max_tokens": 16384})["max_tokens"] == 4096


def test_bounds_env_overrides_the_default(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, HYPERLOOM_EVAL_PROBE="0", HYPERLOOM_EVAL_MAX_TOKENS="512")

    assert _payload(env, gen_kwargs={"max_tokens": 16384})["max_tokens"] == 512


def test_bounds_never_raise_an_existing_lower_ceiling(bounds):
    """Clamping is one-directional. A task that asked for less knows something we
    do not, and granting it more would change what is being measured."""
    assert _payload(bounds, gen_kwargs={"max_tokens": 64})["max_tokens"] == 64


def test_bounds_zero_disables_the_clamp(monkeypatch, tmp_path):
    """An operator reproducing an upstream number needs a way out."""
    env = _install(monkeypatch, tmp_path, HYPERLOOM_EVAL_PROBE="0", HYPERLOOM_EVAL_MAX_TOKENS="0")

    assert _payload(env, gen_kwargs={"max_tokens": 16384})["max_tokens"] == 16384


@pytest.mark.parametrize("raw", ["", "   ", "lots", "4096.5", "-1"])
def test_bounds_fall_back_to_the_default_when_the_env_is_unusable(monkeypatch, tmp_path, raw):
    """A typo must not silently mean "unbounded" — that is the failure this
    exists to prevent. Only an explicit 0 disables it."""
    env = _install(monkeypatch, tmp_path, HYPERLOOM_EVAL_PROBE="0", HYPERLOOM_EVAL_MAX_TOKENS=raw)

    assert _payload(env, gen_kwargs={"max_tokens": 16384})["max_tokens"] == 4096


def test_bounds_leave_loglikelihood_payloads_untouched(bounds):
    """Scoring requests share this seam and generate nothing."""
    payload = _payload(bounds, generate=False)

    assert payload["max_tokens"] == 1
    assert "stop" not in payload


def test_bounds_prepend_stop_strings_ahead_of_upstreams(monkeypatch, tmp_path):
    """Upstream keeps only the first four, so the caller's terminators — the ones
    derived from the model actually under test — have to go first."""
    env = _install(
        monkeypatch,
        tmp_path,
        HYPERLOOM_EVAL_PROBE="0",
        HYPERLOOM_EVAL_STOP_STRINGS="<|im_end|>\x1f<|endoftext|>",
    )

    payload = _payload(env, gen_kwargs={"until": ["Q:", "\n\n", "Question:"]}, eos="</s>")

    assert payload["stop"] == ["<|im_end|>", "<|endoftext|>", "Q:", "\n\n"]


def test_bounds_do_not_duplicate_a_stop_string_upstream_already_sent(monkeypatch, tmp_path):
    """Duplicates would burn one of the four slots for nothing."""
    env = _install(monkeypatch, tmp_path, HYPERLOOM_EVAL_PROBE="0", HYPERLOOM_EVAL_STOP_STRINGS="Q:")

    assert _payload(env, gen_kwargs={"until": ["Q:", "\n\n"]})["stop"] == ["Q:", "\n\n"]


def test_bounds_summary_reports_how_often_the_ceiling_was_hit(bounds):
    """Truncation is only defensible while it is rare, so the run has to say how
    rare it was: too low a ceiling depresses both arms' scores and nothing else
    would show it."""
    cls = bounds.cls
    cls.parse_generations(outputs=[{"choices": [{"finish_reason": "length"}, {"finish_reason": "stop"}]}])
    cls.parse_generations(outputs=[{"choices": [{"finish_reason": "stop"}]}])
    _payload(bounds, gen_kwargs={"max_tokens": 16384})
    bounds.flush()

    record = json.loads((bounds.result_dir / BOUNDS_FILENAME).read_text(encoding="utf-8"))
    assert record["generations"] == 3
    assert record["truncated"] == 1
    assert record["max_tokens"] == 4096


def test_bounds_summary_is_silent_when_nothing_was_generated(bounds):
    """A run that never reached lm-eval must not leave a misleading sidecar."""
    bounds.flush()

    assert not (bounds.result_dir / BOUNDS_FILENAME).exists()


def test_bounds_counting_survives_a_response_it_cannot_read(bounds):
    """Measuring the eval must never break it."""
    for junk in (None, [], {"choices": "not-a-list"}, {"choices": [None]}):
        assert bounds.cls.parse_generations(outputs=junk) == ["upstream"]


def test_bounds_and_probe_coexist(probe):
    """Both wrap ``parse_generations``; the chain has to reach upstream's, and
    the probe still has to trip."""
    assert probe.cls.parse_generations(outputs=[{"choices": [{"finish_reason": "stop"}]}]) == ["upstream"]
    assert _payload(probe, gen_kwargs={"max_tokens": 16384})["max_tokens"] == 4096
