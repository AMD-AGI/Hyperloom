"""Verify SharedState renderers expose numeric ``gain_pct`` values.

Pre-Phase the explore ledgers rendered as "name + counts" only, which left
Orchestration unable to rank retry candidates by observed impact. The
:meth:`SharedState._format_backend_winners_history` and
:meth:`SharedState._format_search_state` renderers now multi-line the
ledger and surface ``gain_pct``, ``tput``, and resolved flags/envs per
variant. These tests pin that contract:

* accepted entries that lack ``gain_pct`` on disk pull it from
  ``tested[fingerprint]`` at render time (params_search shape);
* rejected entries (which always carry ``gain_pct``) render directly;
* ``backend_winners_history`` shows per-winner gain plus the round-level
  best gain;
* legacy ``key=value`` prefixes survive so existing prefix-style
  assertions across the suite keep passing.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


def _make_state() -> SharedState:
    """Construct a SharedState with one accepted/one rejected entry per
    ``*_search`` ledger plus one populated ``backend_winners_history``
    round. Mirrors the shape executors persist (see params.py /
    backends.py / record_backends_accepted)."""
    s = SharedState(session_id="gain-render-test")
    s.params_search = {
        "schema_version": 2,
        # Accepted entry uses _variant_to_dict() shape — no gain_pct key.
        "accepted": [
            {
                "name": "mem_fraction_0_85",
                "fingerprint": "fp_a",
                "extra_server_args": "--mem-fraction-static 0.85",
                "extra_envs": {},
                "note": "memory",
            },
        ],
        "rejected": [
            {
                "name": "cuda_graph_max_bs_8",
                "fingerprint": "fp_b",
                "extra_server_args": "--cuda-graph-max-bs 8",
                "extra_envs": {},
                "note": "cuda_graph",
                "gain_pct": -1.05,
                "tput": 1750.0,
                "reason": "not_keep",
            },
        ],
        "tested": {
            # tested[fp] carries gain — renderer must pick it up for the
            # accepted row that lacks the field on disk.
            "fp_a": {
                "name": "mem_fraction_0_85",
                "gain_pct": 0.27,
                "tput": 1773.4,
            },
            "fp_b": {
                "name": "cuda_graph_max_bs_8",
                "gain_pct": -1.05,
                "tput": 1750.0,
            },
        },
        "name_index": {},
        "cursor": 2,
        "last_round": {},
    }
    s.backends_search = {
        "schema_version": 1,
        # backends accepted entries carry gain_pct directly (stamped by
        # record_backends_accepted) — exercise that path too.
        "accepted": [
            {
                "name": "qr_int4",
                "fingerprint": "fp_c",
                "extra_server_args": "",
                "extra_envs": {"VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4"},
                "note": "tier5_comm",
                "gain_pct": 3.33,
                "tput": 1768.67,
            },
        ],
        "rejected": [
            {
                "name": "attn_aiter",
                "fingerprint": "fp_d",
                "extra_server_args": "--attention-backend aiter",
                "extra_envs": {},
                "note": "tier1_attention",
                "gain_pct": -2.10,
                "tput": 1607.0,
            },
        ],
        "tested": {"fp_c": {}, "fp_d": {}},
        "name_index": {},
        "cursor": 2,
    }
    s.push_backend_winners_round(
        action="backends",
        base_tput=1641.16,
        base_extra_args="",
        winners=[
            {
                "name": "qr_int4",
                "output_throughput": 1768.67,
                "gain_pct": 3.33,
                "extra_server_args": "",
                "extra_envs": {
                    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
                },
                "note": "tier5_comm",
            },
        ],
        best={
            "name": "qr_int4",
            "output_throughput": 1768.67,
            "gain_pct": 3.33,
            "extra_server_args": "",
            "extra_envs": {
                "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
            },
            "note": "tier5_comm",
        },
    )
    return s


def test_search_state_renders_gain_pct_for_accepted_via_tested_lookup():
    """params_search.accepted lacks gain_pct on disk; renderer pulls it
    from tested[fingerprint] so the LLM still sees the +0.27%."""
    s = _make_state()
    out = s._format_params_search()
    assert "+0.27%" in out
    assert "mem_fraction_0_85" in out
    assert "--mem-fraction-static 0.85" in out


def test_search_state_renders_gain_pct_for_rejected_directly():
    """*_search.rejected entries carry gain_pct natively; render them
    with the matching sign so 'avoid' vs 'tweak value' is one read."""
    s = _make_state()
    out_p = s._format_params_search()
    assert "-1.05%" in out_p
    assert "cuda_graph_max_bs_8" in out_p
    out_b = s._format_backends_search()
    assert "-2.10%" in out_b
    assert "+3.33%" in out_b
    assert "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4" in out_b


def test_backend_winners_history_renders_gain_pct_per_winner():
    """backend_winners_history surfaces both round-level best gain and
    per-winner gain so synergy combos can be composed by impact."""
    s = _make_state()
    out = s._format_backend_winners_history()
    assert "best=qr_int4 +3.33%" in out
    assert "+3.33%" in out
    assert "(tput=1768.7)" in out


def test_search_state_handles_empty_buckets():
    """Empty ledgers must not crash and must not emit body sections."""
    s = SharedState(session_id="empty")
    s.params_search = {
        "schema_version": 2,
        "accepted": [],
        "rejected": [],
        "tested": {},
        "name_index": {},
        "cursor": 0,
        "last_round": {},
    }
    out = s._format_params_search()
    assert "cursor=0" in out
    assert "accepted=0" in out
    assert "accepted:" not in out
    assert "rejected (last 5):" not in out


def test_to_prompt_summary_keeps_legacy_prefixes_and_adds_values():
    """Existing tests across the suite assert prefixes like
    ``params_search=`` / ``backends_search=`` / ``backend_winners_history=``;
    the multi-line body must still be preceded by those prefixes."""
    s = _make_state()
    summary = s.to_prompt_summary()
    assert "params_search=" in summary
    assert "backends_search=" in summary
    assert "backend_winners_history=" in summary
    # And the new numeric signal is now visible end-to-end.
    assert "+3.33%" in summary
    assert "-1.05%" in summary
    assert "+0.27%" in summary  # accepted backfilled from tested[fp]


def test_format_variant_line_handles_missing_gain_and_envs():
    """Variants without measurement render ``no_meas`` instead of crashing;
    flag-less env-only variants render ``(no-flag)`` so the row is still
    a single, parseable line."""
    line = SharedState._format_variant_line({
        "name": "mystery", "extra_server_args": "", "extra_envs": {},
    })
    assert "mystery" in line
    assert "no_meas" in line
    assert "(no-flag)" in line


def test_backend_winners_history_caps_at_five_rounds():
    """Older rounds collapse to ``[+N earlier rounds elided]`` so the
    prompt doesn't grow unbounded across long sessions."""
    s = SharedState(session_id="long")
    for i in range(7):
        s.push_backend_winners_round(
            action="backends",
            base_tput=1000.0 + i,
            base_extra_args="",
            winners=[],
            best=None,
        )
    out = s._format_backend_winners_history()
    assert "[+2 earlier rounds elided]" in out
    # Most recent round is still present.
    assert "backends-007" in out
    # Oldest two rounds should have been elided from the body.
    assert "backends-001" not in out
    assert "backends-002" not in out
