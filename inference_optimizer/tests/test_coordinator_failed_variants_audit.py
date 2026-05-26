"""Regression tests for ``_summarize_failed_variants`` + PR-3 timeout.

Pins偏差 3 PR-2 + PR-3 contract:

PR-2 (audit-trail per-variant failure surfacing)
  ``_summarize_failed_variants`` projects the ``status=='failed'`` rows of
  a ``run_grid`` ``all_results`` dump into a compact list suitable for
  ``record_action_attempt(extras={"failed_variants": ...})``. The LLM
  critic prompt assembled from SharedState then sees which variants
  silently aborted and won't re-propose them on the next round.

PR-3 (faster silent-abort detection)
  ``DEFAULT_HEALTH_TIMEOUT_S`` tightened from 1800s (30 min) to 900s
  (15 min) after multi-node sessions burned GPU-hours on a single
  variant whose sglang launcher never reached /health 200. Override
  path via ``HYPERLOOM_MN_HEALTH_WAIT_S`` is preserved for slow
  workloads that genuinely need more headroom.
"""

from __future__ import annotations

from inference_optimizer.orchestrator import coordinator
from inference_optimizer.orchestrator.action_executors import (
    _multi_node_server_lifecycle,
)


# ---------------------------------------------------------------------------
# PR-2 - _summarize_failed_variants
# ---------------------------------------------------------------------------
def test_summarize_failed_variants_returns_empty_when_input_not_list():
    """Defensive guard: garbage / None input shouldn't crash audit write."""
    assert coordinator._summarize_failed_variants(None) == []
    assert coordinator._summarize_failed_variants("not a list") == []
    assert coordinator._summarize_failed_variants(42) == []
    assert coordinator._summarize_failed_variants({}) == []


def test_summarize_failed_variants_returns_empty_when_no_failures():
    """All variants succeeded -> empty list (caller still sees the field
    but it carries no signal, which is fine for prompt rendering)."""
    rows = [
        {"name": "v1", "status": "succeeded", "output_throughput": 640.0},
        {"name": "v2", "status": "succeeded", "output_throughput": 643.9},
    ]
    assert coordinator._summarize_failed_variants(rows) == []


def test_summarize_failed_variants_projects_expected_keys():
    """Failed rows are projected to a stable 4-key shape."""
    rows = [
        {
            "name": "max_num_seqs_128",
            "status": "failed",
            "error_class": "mn_server_restart_failed",
            "error": (
                "server /health did not return 200 within 1800s "
                "(url=http://10.245.131.67:8888/health, "
                "last_err=ConnectError: All connection attempts failed)"
            ),
            "extra_sglang_args": "--max-num-seqs 128",
        },
        {
            "name": "max_num_seqs_512",
            "status": "succeeded",
            "output_throughput": 510.0,
        },
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert len(out) == 1
    assert out[0] == {
        "name": "max_num_seqs_128",
        "error_class": "mn_server_restart_failed",
        "error_excerpt": (
            "server /health did not return 200 within 1800s "
            "(url=http://10.245.131.67:8888/health, "
            "last_err=ConnectError: All connection attempts failed)"
        ),
        "extra_sglang_args": "--max-num-seqs 128",
    }


def test_summarize_failed_variants_truncates_error_excerpt_at_400_chars():
    """Per-entry error blob is capped so a runaway grid can't inflate
    attempts_history past prompt-budget limits."""
    huge_err = "x" * 5000
    rows = [
        {
            "name": "v",
            "status": "failed",
            "error_class": "ec",
            "error": huge_err,
            "extra_sglang_args": "",
        },
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert out[0]["error_excerpt"] is not None
    assert len(out[0]["error_excerpt"]) == 400


def test_summarize_failed_variants_caps_max_entries():
    """A runaway 100-variant grid should not bloat audit extras -
    the helper truncates at ``max_entries`` (default 10)."""
    rows = [
        {
            "name": f"v{i}",
            "status": "failed",
            "error_class": "ec",
            "error": "boom",
            "extra_sglang_args": f"--arg {i}",
        }
        for i in range(50)
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert len(out) == 10
    # FIFO ordering: first 10 of the input list are kept
    assert [e["name"] for e in out] == [f"v{i}" for i in range(10)]


def test_summarize_failed_variants_skips_non_dict_rows():
    """Tolerate malformed entries mixed into the list (defensive)."""
    rows = [
        None,
        "garbage",
        {"name": "real", "status": "failed", "error_class": "ec", "error": "msg"},
        42,
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert len(out) == 1
    assert out[0]["name"] == "real"


def test_summarize_failed_variants_handles_missing_optional_fields():
    """Rows missing ``error_class`` / ``error`` / ``extra_sglang_args``
    still get a normalized entry - None for optionals, empty str for
    name/extra_args defaults - so prompt rendering doesn't crash."""
    rows = [{"name": "v", "status": "failed"}]
    out = coordinator._summarize_failed_variants(rows)
    assert out == [{
        "name": "v",
        "error_class": None,
        "error_excerpt": None,
        "extra_sglang_args": "",
    }]


# ---------------------------------------------------------------------------
# PR-3 - DEFAULT_HEALTH_TIMEOUT_S regression
# ---------------------------------------------------------------------------
def test_default_health_timeout_is_900s_not_1800s():
    """Pin the tightening: 30 min ceiling was 3x normal cold-start
    headroom; 15 min is ~2x and still matches the env-override knob
    ``HYPERLOOM_MN_HEALTH_WAIT_S``."""
    assert _multi_node_server_lifecycle.DEFAULT_HEALTH_TIMEOUT_S == 900
