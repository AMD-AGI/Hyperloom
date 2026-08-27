# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for tuned-artifact apply verification.

The load-bearing distinction: aiter logs a miss unconditionally but a hit only
when AITER_LOG_TUNED_CONFIG=1. "No hit lines" therefore does not mean "zero
hits", and a check that conflates them would revert every arm that ran without
the flag -- which, in a scan of 60 production logs, was all of them.

``kernelforge.gemm_tune`` is not a Hyperloom dependency, so importorskip on it left
this whole module -- and therefore the KEEP gate's entire decision surface --
without automated coverage in CI. The verdict logic is exercised against a
stand-in parser instead, and the real parser is used as well wherever forge
happens to be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from hyperloom.orchestrator.measurement import apply_verification as av
from hyperloom.orchestrator.measurement.apply_verification import verify_applied

_MISS = (
    "[aiter] shape is M:{m}, N:4096, K:4096 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
    "not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv"
)
# Transcribed from a real MI355X run: the hit line names the table it resolved
# in, which is the only place the path appears once AITER_CONFIG_* is set.
_HIT = (
    "[aiter] shape is M:{m}, N:4096, K:4096 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False "
    "found padded_M: {m}, N:4096, K:4096 is tuned on cu_num = 256 in "
    "/tmp/aiter_configs/bf16_tuned_gemm.csv, libtype is asm, kernel name is knl"
)
_MERGE = "[aiter] merge tuned file under model_configs/ and configs/ /srv/cfg/bf16_tuned_gemm.csv:/srv/cfg/other.csv"
_BF16 = ["bf16_tuned_gemm.csv"]


def _log(tmp_path, lines):
    p = tmp_path / "server.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture(params=["stub_parser", "real_forge"])
def parser(request, monkeypatch):
    """Run every case against a stand-in parser, and against forge when present.

    The stand-in is deliberately minimal -- it reproduces only the three facts
    the verdict depends on (hit/miss counts, merged tables, consulted tables) --
    so the decision logic stays under test on a machine that has no forge.
    """
    if request.param == "real_forge":
        # Skip on the submodule production actually imports, not the top-level
        # package. A box can have kernelforge.gemm_tune installed without
        # ``evidence`` in it, and then the top-level check passes, the parser
        # comes back None, every verdict is "unknown", and eleven cases fail on
        # a developer machine for a reason that has nothing to do with them.
        pytest.importorskip("kernelforge.gemm_tune.evidence", reason="real parser unavailable")
        return None

    import re

    def _fake_parse_log_file(path):
        text = __import__("pathlib").Path(path).read_text(encoding="utf-8", errors="replace")
        hits = misses = 0
        merged: list[str] = []
        consulted: set[str] = set()
        for line in text.splitlines():
            if "merge tuned file" in line:
                merged.extend(p for p in line.split()[-1].split(":") if p)
            elif "not found tuned config in" in line:
                misses += 1
                consulted.add(line.split("not found tuned config in")[1].split(",")[0].strip())
            elif "found padded_M" in line:
                hits += 1
                m = re.search(r"is tuned on cu_num\s*=\s*\d+\s+in\s+([^,]+)", line)
                if m:
                    consulted.add(m.group(1).strip())
        return {
            "apply_verdict": {"hit": hits, "miss": misses},
            "merged_tables": sorted(set(merged)),
            "consulted_tables": sorted(consulted),
        }

    fake = types.ModuleType("kernelforge.gemm_tune")
    fake_ev = types.ModuleType("kernelforge.gemm_tune.evidence")
    fake_ev.parse_log_file = _fake_parse_log_file  # type: ignore[attr-defined]
    fake.evidence = fake_ev  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kernelforge.gemm_tune", fake)
    monkeypatch.setitem(sys.modules, "kernelforge.gemm_tune.evidence", fake_ev)
    return None


class TestServed:
    def test_hits_mean_served(self, tmp_path, parser):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16), _MISS.format(m=15)])
        v = verify_applied(p, ["/work/bf16_tuned_gemm.csv"], runtime_table_names=_BF16)
        assert v.verdict == "served"
        assert v.hits == 1 and v.misses == 1
        assert not v.blocks_keep and v.conclusive


class TestArtifactArrival:
    def test_an_override_run_prints_no_merge_line_and_still_counts_as_arrived(self, tmp_path, parser):
        # Setting AITER_CONFIG_* makes aiter skip the merge step: no merge line
        # at all, and the lookups name our own file. Reading that as "not
        # merged" would revert every candidate, which is what the merge-list
        # comparison used to do.
        ours = "/work/run/merged_tuned_dense_bf16.csv"
        p = _log(
            tmp_path,
            [
                _MISS.format(m=15).replace("/tmp/aiter_configs/bf16_tuned_gemm.csv", ours),
            ],
        )
        v = verify_applied(p, [ours], runtime_table_names=_BF16)
        assert v.verdict != "not_merged"

    def test_the_runtime_table_name_is_accepted_too(self, tmp_path, parser):
        # The deployed file is named after the candidate; the server resolves it
        # under the canonical table name. Both are the same artifact.
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        v = verify_applied(p, ["/work/run/merged_tuned_dense_bf16.csv"], runtime_table_names=_BF16)
        assert v.verdict == "served"

    def test_a_genuinely_absent_artifact_still_blocks(self, tmp_path, parser):
        # Nothing the runtime touched resembles what we deployed.
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15)])
        v = verify_applied(
            p,
            ["/work/a8w8_blockscale_tuned_gemm.csv"],
            runtime_table_names=["a8w8_blockscale_tuned_gemm.csv"],
        )
        assert v.verdict == "not_merged" and v.blocks_keep
        assert v.unmerged_artifacts == ["/work/a8w8_blockscale_tuned_gemm.csv"]

    def test_match_is_by_basename(self, tmp_path, parser):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        v = verify_applied(p, ["/some/other/dir/bf16_tuned_gemm.csv"], runtime_table_names=_BF16)
        assert v.verdict == "served"


class TestHitLoggingTrap:
    def test_misses_without_hit_logging_is_not_a_failure(self, tmp_path, parser):
        # No hit lines because the flag was off -- must NOT block.
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15), _MISS.format(m=17)])
        v = verify_applied(
            p,
            ["/work/bf16_tuned_gemm.csv"],
            runtime_table_names=_BF16,
            hit_logging=False,
        )
        assert v.verdict == "inconclusive_no_hit_logging"
        assert not v.blocks_keep
        assert not v.conclusive
        assert "AITER_LOG_TUNED_CONFIG" in v.detail

    def test_unknown_hit_logging_stays_inconclusive(self, tmp_path, parser):
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15)])
        v = verify_applied(p, ["/work/bf16_tuned_gemm.csv"], runtime_table_names=_BF16)
        assert v.verdict == "inconclusive_no_hit_logging" and not v.blocks_keep

    def test_zero_hits_with_logging_on_is_a_real_failure(self, tmp_path, parser):
        # This is the verdict the gate exists for, and it was unreachable: the
        # parser answers "inconclusive" for hits==0 whatever the flag, so the
        # branch was dead. Now that every serving run sets the flag, "0 hits and
        # N misses" is a genuine zero and has to block.
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15), _MISS.format(m=17)])
        v = verify_applied(
            p,
            ["/work/bf16_tuned_gemm.csv"],
            runtime_table_names=_BF16,
            hit_logging=True,
        )
        assert v.verdict == "zero_hit"
        assert v.blocks_keep and v.conclusive
        assert v.misses == 2

    def test_every_blocking_verdict_is_reachable(self, tmp_path, parser):
        """A verdict listed as blocking that nothing can return is not a gate."""
        reached = set()
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15)])
        reached.add(
            verify_applied(
                p,
                ["/work/a8w8_tuned_gemm.csv"],
                runtime_table_names=["a8w8_tuned_gemm.csv"],
            ).verdict
        )
        reached.add(
            verify_applied(
                p,
                ["/work/bf16_tuned_gemm.csv"],
                runtime_table_names=_BF16,
                hit_logging=True,
            ).verdict
        )
        assert av.BLOCKING_VERDICTS <= reached


class TestDegraded:
    def test_missing_log(self, tmp_path, parser):
        v = verify_applied(tmp_path / "nope.log", ["/work/x.csv"])
        assert v.verdict == "unknown" and not v.blocks_keep

    def test_no_lookups_at_all(self, tmp_path, parser):
        p = _log(tmp_path, ["server started", "ready"])
        v = verify_applied(p, [])
        assert v.verdict == "no_lookups" and not v.blocks_keep

    def test_no_artifacts_supplied_skips_the_arrival_check(self, tmp_path, parser):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        assert verify_applied(p, None).verdict == "served"

    def test_to_dict_is_serialisable(self, tmp_path, parser):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        d = verify_applied(p, ["/work/bf16_tuned_gemm.csv"], runtime_table_names=_BF16).to_dict()
        assert d["verdict"] == "served" and d["blocks_keep"] is False


class TestTheEnvToTableMapDoesNotDrift:
    """The same mapping exists here and in KernelForge, and cannot be shared.

    A name that drifts makes the apply check compare our deployed file against
    the wrong runtime table, conclude the artifact never arrived, and revert a
    candidate that was fine. The two same-repo copies are now one constant;
    this covers the copy that lives in the other repository.
    """

    def test_every_env_var_maps_to_the_same_table_as_kernelforge(self):
        forge_utils = pytest.importorskip("kernelforge.gemm_tune.utils", reason="KernelForge not installed here")
        from hyperloom.orchestrator.phases.kernel import _AITER_ENV_TO_TABLE

        forge_env_vars = set(getattr(forge_utils, "TUNER_ENV_VARS", {}).values())
        aiter_only = {v for v in forge_env_vars if v.startswith("AITER_CONFIG")}

        missing = aiter_only - set(_AITER_ENV_TO_TABLE)
        assert not missing, (
            f"KernelForge writes {sorted(missing)} but the apply check has no "
            "table for them, so artifacts under those names read as never "
            "having arrived"
        )

    def test_the_fp4_key_is_the_one_aiter_actually_reads(self):
        # AITER_CONFIG_GEMM_A4W4, not the "_BLOCKSCALE" variant. The suffixed
        # name was a dead key that silently dropped every tuned fp4 GEMM.
        from hyperloom.orchestrator.phases.kernel import _AITER_ENV_TO_TABLE

        assert "AITER_CONFIG_GEMM_A4W4" in _AITER_ENV_TO_TABLE
        assert "AITER_CONFIG_GEMM_A4W4_BLOCKSCALE" not in _AITER_ENV_TO_TABLE

    def test_the_merge_step_and_the_apply_check_read_one_map(self):
        # They were separate copies until one was almost edited alone.
        import inspect

        from hyperloom.orchestrator.phases import kernel

        src = inspect.getsource(kernel.KernelPhase._merge_gemm_candidate_with_runtime)
        assert "_AITER_ENV_TO_TABLE" in src
        assert "a8w8_blockscale_tuned_gemm.csv" not in src
