# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parsing a serving log has to be bounded by something other than uptime.

aiter prints a line for every tuned-config miss unconditionally, and hit logging
is now on for every serving run, so a long production run's server.log is large.
Deriving demand from it is on the tuning path, and the parser walks it line by
line while holding one entry per distinct key in memory.

Truncation is reported rather than silent: a demand list that stopped early is
still the runtime's own shapes, and far better than config-derived ones, but a
reader has to be able to tell it is a prefix.
"""

from __future__ import annotations

from kernelforge.gemm_tune import evidence as ev

_MISS = (
    "[aiter] shape is M:{m}, N:4096, K:4096 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
    "not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv"
)


def _log(n_keys: int, repeats: int = 1) -> str:
    return "\n".join(_MISS.format(m=m) for _ in range(repeats) for m in range(1, n_keys + 1))


class TestUnbounded:
    def test_a_normal_log_reports_no_truncation(self):
        report = ev.parse_log(_log(20))
        assert "truncated" not in report
        assert report["demands"][0]["distinct_keys"] == 20


class TestLineBound:
    def test_reading_stops_at_the_line_limit(self, monkeypatch):
        monkeypatch.setenv(ev._MAX_LINES_ENV, "10")
        report = ev.parse_log(_log(50))
        assert report["truncated"]["lines"] == 10
        assert report["apply_verdict"]["miss"] == 10

    def test_the_shapes_read_before_the_limit_are_still_usable(self, monkeypatch):
        monkeypatch.setenv(ev._MAX_LINES_ENV, "5")
        entry = ev.demand_for_tuner(ev.parse_log(_log(50)), "sglang_dense_bf16")
        # bucket=False: this is about which lines were read before the bound,
        # so it wants the raw M values rather than a padded cover of them.
        shapes = ev.demand_shapes(entry, bucket=False)
        assert [s["M"] for s in shapes] == [1, 2, 3, 4, 5]


class TestKeyBound:
    def test_new_keys_stop_being_listed_at_the_limit(self, monkeypatch):
        monkeypatch.setenv(ev._MAX_KEYS_ENV, "8")
        report = ev.parse_log(_log(40))
        entry = report["demands"][0]
        assert entry["distinct_keys"] == 8
        assert report["truncated"]["tables"]["bf16_tuned_gemm.csv"] == 8

    def test_the_miss_count_still_counts_everything(self, monkeypatch):
        # The count is what the apply verdict reads; capping the *list* must not
        # silently shrink the number of lookups the runtime made.
        monkeypatch.setenv(ev._MAX_KEYS_ENV, "8")
        report = ev.parse_log(_log(40))
        assert report["apply_verdict"]["miss"] == 40
        assert report["demands"][0]["miss_count"] == 40

    def test_repeats_of_a_known_key_are_still_counted_past_the_limit(self, monkeypatch):
        # Request counts are the only ordering demand_shapes has, so a key
        # already in the list must keep accruing even once the set is full.
        monkeypatch.setenv(ev._MAX_KEYS_ENV, "3")
        report = ev.parse_log(_log(10, repeats=4))
        entry = report["demands"][0]
        assert entry["distinct_keys"] == 3
        assert all(k["requests"] == 4 for k in entry["keys"])


class TestOverrides:
    def test_limits_are_raisable_for_an_offline_audit(self, monkeypatch):
        monkeypatch.setenv(ev._MAX_KEYS_ENV, "100000")
        monkeypatch.setenv(ev._MAX_LINES_ENV, "100000")
        assert "truncated" not in ev.parse_log(_log(50))

    def test_garbage_and_non_positive_values_fall_back_to_the_default(self, monkeypatch):
        for bad in ("", "0", "-5", "lots"):
            monkeypatch.setenv(ev._MAX_KEYS_ENV, bad)
            assert ev._env_int(ev._MAX_KEYS_ENV, 7) == 7


class TestKeySchemaMatchesInstalledAiter:
    """Pinned to headers read off two independent MI355X aiter installs.

    A documented claim that blockscale carried a scaling-granularity column,
    and bpreshuffle a preshuffle marker, went unchallenged for a while because
    the only thing contradicting it was another document. Neither column
    exists. Getting this wrong would under-key the generated untuned CSV, and
    rows tuned under the wrong key are rows the runtime never finds.
    """

    # table -> the untuned CSV header, which *is* the tuner's input key.
    MEASURED = {
        "a8w8_blockscale_tuned_gemm.csv": ("M", "N", "K"),
        "a8w8_blockscale_bpreshuffle_tuned_gemm.csv": ("M", "N", "K"),
        "a4w4_blockscale_tuned_gemm.csv": ("M", "N", "K"),
        "a8w8_tuned_gemm.csv": ("M", "N", "K", "q_dtype_w"),
        "a8w8_bpreshuffle_tuned_gemm.csv": ("M", "N", "K", "q_dtype_w"),
    }

    def test_each_schema_matches_what_aiter_ships(self):
        for table, expected in self.MEASURED.items():
            assert ev.TABLE_KEY_SCHEMA[table] == expected, table

    def test_blockscale_carries_no_granularity_column(self):
        for table in (
            "a8w8_blockscale_tuned_gemm.csv",
            "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
            "a4w4_blockscale_tuned_gemm.csv",
        ):
            assert not [c for c in ev.TABLE_KEY_SCHEMA[table] if c not in ("M", "N", "K")], (
                f"{table} gained a key column aiter does not have"
            )

    def test_q_dtype_w_is_a_key_the_log_cannot_supply(self):
        # Both facts matter together: it belongs in the key, and evidence can
        # never fill it, so it has to come from the hardware downstream.
        assert "q_dtype_w" in ev.TABLE_KEY_SCHEMA["a8w8_tuned_gemm.csv"]
        assert "q_dtype_w" in ev.UNLOGGABLE_KEY_FIELDS
