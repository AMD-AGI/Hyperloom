# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A variant that ablates a flag and measures nothing: did the flag come off, or was the idea wrong?"""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors.explore import _ablation_realized

_RADIX = "--disable-radix-cache"


def _observed(flags: str) -> dict[str, str]:
    return {"observed_server_launch_flags": flags}


def _resolved(**settings: object) -> dict[str, object]:
    """Launch evidence as SGLang leaves it: no argv line, but every setting the engine resolved."""
    return {"framework": "sglang", "observed_server_launch_flags": "", "observed_server_config": dict(settings)}


def test_a_flag_absent_from_the_launch_confirms_the_ablation_took():
    realized = _ablation_realized([_RADIX], _observed("--tp 8 --enable-metrics --context-length 1048576"))

    assert realized == {"requested": [_RADIX], "still_present": [], "took_effect": True}


def test_a_flag_still_on_the_launch_says_the_ablation_was_blocked():
    realized = _ablation_realized([_RADIX], _observed(f"--tp 8 {_RADIX} --enable-metrics"))

    assert realized["still_present"] == [_RADIX]
    assert realized["took_effect"] is False


def test_a_flag_the_launcher_spells_with_a_value_is_matched_on_its_name():
    realized = _ablation_realized(["--mem-fraction-static"], _observed("--tp 8 --mem-fraction-static=0.8"))

    assert realized["took_effect"] is False


def test_the_engines_own_settings_answer_when_no_argv_was_printed():
    """SGLang prints no argv line, so without this every SGLang ablation went unjudged."""
    realized = _ablation_realized([_RADIX], _resolved(disable_radix_cache=True, tp_size=8))

    assert realized == {"requested": [_RADIX], "still_present": [_RADIX], "took_effect": False}

    assert _ablation_realized([_RADIX], _resolved(disable_radix_cache=False))["took_effect"] is True


def test_a_valued_flag_is_unknown_without_an_argv_record():
    """Dropped, it takes an engine default no resolved-settings record names, whatever it reads now."""
    assert _ablation_realized(["--mem-fraction-static=0.9"], _resolved(mem_fraction_static=0.8)) is None
    assert _ablation_realized(["--tensor-parallel-size"], _resolved(tp_size=8)) is None
    assert _ablation_realized(["--speculative-num-steps"], _resolved(speculative_num_steps=0)) is None


def test_one_unanswerable_flag_leaves_the_whole_verdict_unknown():
    """Reporting took_effect over the flags that happened to be judgeable would overstate it."""
    evidence = _resolved(disable_radix_cache=False)

    assert _ablation_realized([_RADIX, "--mem-fraction-static=0.9"], evidence) is None


def test_a_setting_the_engine_never_reported_is_unknown():
    assert _ablation_realized([_RADIX], _resolved(tp_size=8)) is None


def test_an_unobserved_launch_is_unknown_rather_than_confirmed():
    """Returning a took_effect=True here would assert an ablation nobody watched."""
    assert _ablation_realized([_RADIX], {}) is None
    assert _ablation_realized([_RADIX], _observed("")) is None
    assert _ablation_realized([_RADIX], None) is None


def test_a_variant_that_ablates_nothing_reports_nothing():
    assert _ablation_realized([], _observed("--tp 8")) is None
    assert _ablation_realized(None, _observed("--tp 8")) is None
    assert _ablation_realized(["  "], _observed("--tp 8")) is None
