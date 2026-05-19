"""Roofline-v2 N4: layered `_format_discovered_flags` Z-scheme tests.

Pins the contract N5 (orchestration prompt guidance section) builds
on top of: the main Orchestration LLM sees every real flag name in
the prompt with a per-flag tested-status tag, so it can pick
"untested + bottleneck-matched" flags directly without reverse-
engineering ``params_search.tested`` fingerprints.

Cf. design/roofline-v2.md §8.6 for the rendering spec.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


def _state_with_flags(**discovered_flags_kwargs) -> SharedState:
    """SharedState with `discovered_flags` seeded for tests."""
    s = SharedState()
    s.discovered_flags = discovered_flags_kwargs
    return s


# ---------------------------------------------------------------------------
# Empty / placeholder paths
# ---------------------------------------------------------------------------
def test_empty_discovered_flags_returns_placeholder():
    s = SharedState()
    assert s.discovered_flags == {}
    out = s._format_discovered_flags()
    assert out == "(none — first backends/params round will populate)"


def test_framework_present_but_no_flag_lists_returns_none():
    """Framework key exists but backend_flags + param_flags both empty.
    With nothing to render, we surface a stable "(none)" — distinct
    from the empty-discovered-flags placeholder (different meanings).
    """
    s = _state_with_flags(sglang={"backend_flags": [], "param_flags": []})
    assert s._format_discovered_flags() == "(none)"


def test_non_dict_entry_skipped_silently():
    s = _state_with_flags(sglang="garbage")
    assert s._format_discovered_flags() == "(none)"


def test_non_list_flag_value_skipped_silently():
    s = _state_with_flags(sglang={
        "backend_flags": "not-a-list",
        "param_flags": [],
    })
    assert s._format_discovered_flags() == "(none)"


# ---------------------------------------------------------------------------
# Happy path — single framework / single action group
# ---------------------------------------------------------------------------
def test_renders_single_framework_backends_only():
    s = _state_with_flags(sglang={
        "backend_flags": [
            "--enable-two-batch-overlap",
            "--enable-aiter-allreduce-fusion",
        ],
        "param_flags": [],
    })
    out = s._format_discovered_flags()
    assert out.startswith("\n")  # leading newline so prompt summary indents nicely
    assert "sglang.backends (2 flags):" in out
    assert "--enable-aiter-allreduce-fusion" in out
    assert "--enable-two-batch-overlap" in out
    # sorted alphabetically
    a_idx = out.find("--enable-aiter-allreduce-fusion")
    b_idx = out.find("--enable-two-batch-overlap")
    assert a_idx < b_idx
    # all-untested when no tested ledger
    assert "[untested]" in out
    # no params group rendered (empty list)
    assert "sglang.params" not in out


def test_renders_single_framework_both_groups():
    s = _state_with_flags(sglang={
        "backend_flags": ["--enable-two-batch-overlap"],
        "param_flags": ["--cuda-graph-max-bs", "--enable-torch-compile"],
    })
    out = s._format_discovered_flags()
    assert "sglang.backends (1 flags):" in out
    assert "sglang.params (2 flags):" in out
    assert "--enable-two-batch-overlap" in out
    assert "--cuda-graph-max-bs" in out
    # backends group comes before params (per Z scheme: action-kind
    # ordering is fixed for prompt cache stability)
    b_idx = out.find("sglang.backends")
    p_idx = out.find("sglang.params")
    assert b_idx < p_idx


def test_renders_multiple_frameworks_sorted():
    s = _state_with_flags(
        sglang={"backend_flags": ["--sg-x"], "param_flags": []},
        vllm={"backend_flags": ["--vll-y"], "param_flags": []},
    )
    out = s._format_discovered_flags()
    # framework sort: sglang < vllm
    sg_idx = out.find("sglang.backends")
    vl_idx = out.find("vllm.backends")
    assert sg_idx < vl_idx


# ---------------------------------------------------------------------------
# Tested status tags — cross-ref with params_search / backends_search
# ---------------------------------------------------------------------------
def test_untested_tag_when_search_ledger_empty():
    s = _state_with_flags(sglang={
        "backend_flags": ["--enable-two-batch-overlap"],
        "param_flags": [],
    })
    # No params_search / backends_search seeded
    out = s._format_discovered_flags()
    assert "[untested]" in out


def test_tested_single_variant_surfaces_gain():
    s = _state_with_flags(sglang={
        "backend_flags": ["--enable-two-batch-overlap"],
        "param_flags": [],
    })
    s.backends_search = {
        "tested": {
            "fp1": {
                "name": "tbo",
                "extra_sglang_args": "--enable-two-batch-overlap",
                "gain_pct": 1.23,
            },
        },
    }
    out = s._format_discovered_flags()
    assert "[tested: +1.23%]" in out


def test_tested_multiple_variants_surfaces_best():
    s = _state_with_flags(sglang={
        "param_flags": ["--cuda-graph-max-bs"],
        "backend_flags": [],
    })
    s.params_search = {
        "tested": {
            "fp1": {"extra_sglang_args": "--cuda-graph-max-bs 64", "gain_pct": 0.3},
            "fp2": {"extra_sglang_args": "--cuda-graph-max-bs 128", "gain_pct": 0.85},
            "fp3": {"extra_sglang_args": "--cuda-graph-max-bs 256", "gain_pct": -0.2},
            "fp4": {"extra_sglang_args": "--mem-fraction-static 0.9"},  # other flag
        },
    }
    out = s._format_discovered_flags()
    # 3 of the 4 tested variants reference --cuda-graph-max-bs
    assert "[tested 3 vars, best +0.85%]" in out


def test_tested_with_negative_gain_renders_with_sign():
    s = _state_with_flags(sglang={
        "backend_flags": ["--bad-flag"],
        "param_flags": [],
    })
    s.backends_search = {
        "tested": {
            "fp1": {"extra_sglang_args": "--bad-flag", "gain_pct": -2.5},
        },
    }
    out = s._format_discovered_flags()
    assert "[tested: -2.50%]" in out


def test_tested_without_gain_pct_treated_as_untested():
    """A fingerprint with no `gain_pct` value (the variant ran but
    didn't produce a measurable gain) should not pretend to be a
    tested-with-result entry."""
    s = _state_with_flags(sglang={
        "backend_flags": ["--foo"],
        "param_flags": [],
    })
    s.backends_search = {
        "tested": {
            "fp1": {"extra_sglang_args": "--foo"},  # no gain_pct
        },
    }
    out = s._format_discovered_flags()
    assert "[untested]" in out


def test_tested_lookup_uses_action_kind_specific_search():
    """A backend flag should only consult backends_search, not
    params_search; otherwise a flag could show "tested" because it
    happens to appear in a params variant."""
    s = _state_with_flags(sglang={
        "backend_flags": ["--moe-a2a-backend"],
        "param_flags": [],
    })
    s.params_search = {
        "tested": {
            "fp1": {"extra_sglang_args": "--moe-a2a-backend deepep", "gain_pct": 1.0},
        },
    }
    s.backends_search = {}  # empty for the backends group
    out = s._format_discovered_flags()
    # The flag is rendered in the backends group; backends_search is
    # empty → should be untested
    assert "[untested]" in out
    assert "[tested" not in out


def test_malformed_tested_dict_handled():
    """Non-dict values inside `tested` are silently skipped (defence
    against state.json corruption)."""
    s = _state_with_flags(sglang={
        "backend_flags": ["--foo"],
        "param_flags": [],
    })
    s.backends_search = {
        "tested": {
            "fp1": "not-a-dict",
            "fp2": {"extra_sglang_args": "--foo", "gain_pct": 0.5},
        },
    }
    out = s._format_discovered_flags()
    assert "[tested: +0.50%]" in out


# ---------------------------------------------------------------------------
# Integration with to_prompt_summary
# ---------------------------------------------------------------------------
def test_to_prompt_summary_includes_layered_flags():
    s = _state_with_flags(sglang={
        "backend_flags": ["--enable-two-batch-overlap"],
        "param_flags": ["--cuda-graph-max-bs"],
    })
    out = s.to_prompt_summary()
    assert "discovered_flags=" in out
    # The layered rendering pushes the actual flag names into the
    # prompt; before N4 the LLM only saw the count summary.
    assert "sglang.backends (1 flags):" in out
    assert "--enable-two-batch-overlap" in out
    assert "sglang.params (1 flags):" in out
    assert "--cuda-graph-max-bs" in out
    assert "[untested]" in out


def test_to_prompt_summary_with_empty_flags_still_renders():
    """Pre-N4 behaviour (placeholder when discovered_flags empty) must
    survive."""
    s = SharedState()
    out = s.to_prompt_summary()
    assert "discovered_flags=(none — first backends/params round will populate)" in out


# ---------------------------------------------------------------------------
# Tagging helper unit tests
# ---------------------------------------------------------------------------
def test_tag_helper_returns_untested_for_unknown_action_kind():
    s = SharedState()
    s.params_search = {"tested": {"x": {"extra_sglang_args": "--foo", "gain_pct": 1}}}
    # action_kind not in {params, backends} → must short-circuit untested
    assert s._tested_tag_for_flag("--foo", "garbage") == "[untested]"


def test_tag_helper_handles_missing_search_attribute():
    s = SharedState()  # neither params_search nor backends_search populated
    assert s._tested_tag_for_flag("--foo", "params") == "[untested]"
    assert s._tested_tag_for_flag("--foo", "backends") == "[untested]"
