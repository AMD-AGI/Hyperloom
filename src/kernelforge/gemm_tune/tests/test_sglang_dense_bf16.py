# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the sglang dense BF16 tuner.

Three production failures are covered here, all observed on MI355X (gfx950)
against aiter at /sgl-workspace/aiter:

1. forge pointed at ``gradlib/gradlib/gemm_tuner.py`` and passed ``--libtype``,
   which that script does not accept -- every call died with
   ``unrecognized arguments: --libtype hipblaslt`` and produced nothing.
2. Moving to ``csrc/gemm_a16w16/`` fixed the argument error but still tuned 0
   shapes, because ``hipblaslt`` is additionally gated on ``--with-hipblaslt``.
   With the flag, 11 of 11 real shapes tuned; without it, 0 of 2.
3. Success was judged by exit code. The tuner returns 1 even when every shape
   tuned, and its shim rewrites that same 1 into a 0, so both directions
   misjudge. Row count is the only reliable signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.report import build_report
from kernelforge.gemm_tune.script_probe import ScriptSurface
from kernelforge.gemm_tune.tuners import sglang_dense_bf16 as sd
from kernelforge.gemm_tune.tuners.base import TuneContext

# Real header written by csrc/gemm_a16w16/gemm_a16w16_tune.py for both the
# tuned (-o) and the full-candidate profile (-o2) CSV.
_HDR = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
)

_NK = [(4096, 4096)]
_M = [1, 512]


def _row(m, n, k, libtype, us, tflops=750.0):
    return (
        f"gfx950,256,{m},{n},{k},False,torch.bfloat16,torch.bfloat16,False,False,"
        f"{libtype},438410,0,{us},knl,0.0,{tflops},3000.0"
    )


def _csv(rows):
    return "\n".join([_HDR, *rows]) + "\n"


def _aiter_root(tmp_path, *, direct=True, shim=False, gradlib=False) -> Path:
    root = tmp_path / "aiter"
    targets = []
    if direct:
        targets.append(root / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py")
    if shim:
        targets.append(root / "csrc" / "gemm_a16w16" / "gemm_tuner.py")
    if gradlib:
        targets.append(root / "gradlib" / "gradlib" / "gemm_tuner.py")
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("# stub", encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ctx(tmp_path, **overrides) -> TuneContext:
    base = dict(
        profile=ModelProfile(
            model_path="/fake",
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
        framework="sglang",
        precision="bf16",
        quant_type="none",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[1, 512],
        mp=1,
        output_dir=tmp_path,
        iters=20,
        warmup=5,
        min_improvement_pct=1.0,
        timeout_s=3600,
    )
    base.update(overrides)
    return TuneContext(**base)


def _permissive_surface(script):
    """What probe_script returns when --help could not be read: veto nothing."""
    return ScriptSurface(str(script), frozenset(), False)


def _prep(tmp_path, monkeypatch, *, tuned_rows, profile_rows=(), rc=1, root=None, stderr="", surface=None):
    """Wire the tuner so run() executes without aiter, and capture its argv.

    ``tuned_rows=None`` means the tuner wrote no CSV at all, as happens when the
    invocation is rejected outright.
    """
    monkeypatch.setattr(sd, "probe_script", surface or _permissive_surface)
    monkeypatch.setattr(sd, "resolve_aiter_root", lambda: root or _aiter_root(tmp_path))
    monkeypatch.setattr(sd, "_compute_nk_shapes", lambda **kw: list(_NK))
    monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(_M))
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        if tuned_rows is not None:
            Path(cmd[cmd.index("-o") + 1]).write_text(_csv(tuned_rows), encoding="utf-8")
            Path(cmd[cmd.index("-o2") + 1]).write_text(_csv(profile_rows), encoding="utf-8")
        return rc, "", stderr

    monkeypatch.setattr(sd, "run_subprocess", _fake_run)
    return captured


def _run(tmp_path, **ctx_kwargs):
    return sd.SglangDenseBf16Tuner(_ctx(tmp_path, **ctx_kwargs)).run()


# ── script resolution: the gradlib path is gone for good ─────────────────────


class TestScriptResolution:
    def test_prefers_direct_tuner_over_shim(self, tmp_path):
        root = _aiter_root(tmp_path, direct=True, shim=True)
        assert sd._resolve_tuner_script(root).name == "gemm_a16w16_tune.py"

    def test_falls_back_to_shim(self, tmp_path):
        root = _aiter_root(tmp_path, direct=False, shim=True)
        assert sd._resolve_tuner_script(root).name == "gemm_tuner.py"

    def test_gradlib_is_not_a_fallback(self, tmp_path):
        # gradlib cannot parse the CSV schema this tuner writes (it reads the
        # `dtype` column value "torch.bfloat16" as an --indtype key), so falling
        # back to it would guarantee a KeyError rather than a tuned artifact.
        root = _aiter_root(tmp_path, direct=False, shim=False, gradlib=True)
        assert sd._resolve_tuner_script(root) is None

    def test_validate_names_the_legacy_layout(self, tmp_path, monkeypatch):
        root = _aiter_root(tmp_path, direct=False, shim=False, gradlib=True)
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: root)
        err = sd.SglangDenseBf16Tuner(_ctx(tmp_path)).validate()
        assert err and "gradlib" in err

    def test_validate_passes_with_new_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        assert sd.SglangDenseBf16Tuner(_ctx(tmp_path)).validate() is None


class TestValidateAsksWhetherRunCanDeriveShapes:
    """``validate`` refuses only when ``run`` would derive nothing.

    Keying the refusal on ``intermediate_size`` was wrong in both directions. A
    MoE-only config still yields the attention projections, because
    ``compute_dense_nk_shapes`` skips only the FFN pair -- refusing it threw away
    GEMMs that were derivable and correctly keyed. Meanwhile a config that
    yields nothing at all was waved through whenever an input ``run`` never
    reads happened to be supplied. Putting the question to the derivation itself
    is the only judgement that matches what ``run`` does, and it needs no
    special case for sparse MLA.
    """

    def _no_ffn_profile(self):
        # MoE-only checkout: FFN width lives in moe_intermediate_size. Not the
        # sparse-MLA shape either, so no exemption applies -- only the plain
        # attention projections are derivable.
        return ModelProfile(
            model_path="/fake",
            hidden_size=4096,
            intermediate_size=0,
            num_attention_heads=32,
            num_key_value_heads=8,
        )

    def _barren_profile(self):
        """Nothing to derive from: no hidden size means no attention shapes."""
        return ModelProfile(model_path="/fake", hidden_size=0, intermediate_size=0)

    def test_moe_only_config_passes_on_its_attention_shapes(self, tmp_path, monkeypatch):
        """Regression: refusing this discarded the QKV and O GEMMs it can derive."""
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        ctx = _ctx(tmp_path, profile=self._no_ffn_profile())

        assert sd.SglangDenseBf16Tuner(ctx).validate() is None

    def test_config_yielding_no_shapes_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        ctx = _ctx(tmp_path, profile=self._barren_profile())

        err = sd.SglangDenseBf16Tuner(ctx).validate()

        assert err and "shape" in err.lower()

    def test_an_input_run_never_reads_cannot_waive_the_check(self, tmp_path, monkeypatch):
        """``untuned_csv`` is not a shape source here, so it cannot rescue a
        config that derives nothing -- crediting it is what silently dropped the
        caller's shapes in the first place."""
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        csv = tmp_path / "untuned.csv"
        csv.write_text("M,N,K\n64,4096,4096\n", encoding="utf-8")
        ctx = _ctx(tmp_path, profile=self._barren_profile(), untuned_csv=csv)

        err = sd.SglangDenseBf16Tuner(ctx).validate()

        assert err and "shape" in err.lower()

    def test_demand_waives_it(self, tmp_path, monkeypatch):
        """Demand is the one external source ``run`` does read."""
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        demand = tmp_path / "demand.json"
        demand.write_text("{}", encoding="utf-8")
        ctx = _ctx(tmp_path, profile=self._barren_profile(), demand_json=demand)

        assert sd.SglangDenseBf16Tuner(ctx).validate() is None

    def test_sparse_mla_needs_no_exemption(self, tmp_path, monkeypatch):
        """DeepSeek-V4 sparse MLA: ``q_lora_rank`` without ``kv_lora_rank``.
        It passes because its shapes derive, not because it is named."""
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        profile = ModelProfile(
            model_path="/fake",
            hidden_size=4096,
            intermediate_size=0,
            num_attention_heads=64,
            num_key_value_heads=64,
            head_dim=512,
            q_lora_rank=1024,
        )
        ctx = _ctx(tmp_path, profile=profile)

        assert sd.SglangDenseBf16Tuner(ctx).validate() is None


class TestValidateCannotEscapeExecute:
    """``execute`` must convert any validate failure into a TuneResult.

    ``validate`` now asks the shape derivation, which puts ``raw_config`` values
    through ``int()``; a raise there would leave the CLI with no sentinel JSON.
    """

    def _tuner(self, tmp_path, monkeypatch, raw_config: dict):
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        profile = ModelProfile(
            model_path="/fake",
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            raw_config=raw_config,
        )
        return sd.SglangDenseBf16Tuner(_ctx(tmp_path, profile=profile))

    @pytest.mark.parametrize(
        "bad_heads",
        ["auto", [32], {"n": 32}, "32.0"],
        ids=["string", "list", "dict", "float-string"],
    )
    def test_a_malformed_config_yields_a_result_not_a_traceback(self, tmp_path, monkeypatch, bad_heads):
        tuner = self._tuner(tmp_path, monkeypatch, {"num_attention_heads": bad_heads})

        result = tuner.execute()

        assert result.status == "failed"
        assert result.error, "the failure has to name itself"

    def test_a_clean_config_still_runs(self, tmp_path, monkeypatch):
        """The guard must not swallow the ordinary path."""
        tuner = self._tuner(tmp_path, monkeypatch, {"num_attention_heads": 32, "num_key_value_heads": 8})

        assert tuner.validate() is None


class TestRunNamesTheInputsItIgnores:
    """Dropping a caller's shapes without a word is the failure this line exists
    to remove. ``run`` reads demand or the config; anything else that arrives
    has to be reported as unused rather than silently discarded."""

    def test_untuned_csv_is_reported_as_ignored(self, tmp_path, monkeypatch, caplog):
        csv = tmp_path / "untuned.csv"
        csv.write_text("M,N,K\n64,4096,4096\n", encoding="utf-8")
        _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])

        with caplog.at_level("WARNING"):
            _run(tmp_path, untuned_csv=csv)

        assert any("untuned_csv" in r.message for r in caplog.records), caplog.text

    def test_nothing_is_said_when_no_such_input_arrives(self, tmp_path, monkeypatch, caplog):
        _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])

        with caplog.at_level("WARNING"):
            _run(tmp_path)

        assert not any("ignor" in r.message.lower() for r in caplog.records), caplog.text


# ── the one flag that decides whether anything gets tuned at all ─────────────


class TestWithHipblasltFlag:
    def test_fast_mode_enables_hipblaslt(self, tmp_path, monkeypatch):
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        _run(tmp_path)
        cmd = cap["cmd"]
        assert "--with-hipblaslt" in cmd
        assert cmd[cmd.index("--libtype") + 1] == "hipblaslt,torch"

    def test_fast_mode_asks_for_torch_so_the_run_has_a_baseline(self, tmp_path, monkeypatch):
        # torch is not a serious contender against hipblaslt; it is the kernel
        # aiter falls back to when a shape is untuned, so its profile row is the
        # only baseline _parse_profile_defaults can read. Dropping it made every
        # fast run report improved_shapes=0 for want of a comparison.
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        _run(tmp_path)
        libtypes = cap["cmd"][cap["cmd"].index("--libtype") + 1].split(",")
        assert "torch" in libtypes and "hipblaslt" in libtypes

    def test_thorough_mode_also_enables_hipblaslt(self, tmp_path, monkeypatch):
        # --libtype all is gated on --with-hipblaslt too: the `all` variants
        # measured on MI355X left the large-M shapes untuned without it.
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "flydsl", 9.5)])
        _run(tmp_path, thorough=True)
        cmd = cap["cmd"]
        assert "--with-hipblaslt" in cmd
        assert cmd[cmd.index("--libtype") + 1] == "all"

    def test_invokes_the_gemm_a16w16_script(self, tmp_path, monkeypatch):
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        _run(tmp_path)
        assert cap["cmd"][1].endswith("gemm_a16w16_tune.py")
        assert "gradlib" not in cap["cmd"][1]


# ── --timeout is a whole-batch budget under --shape_grouped ──────────────────


class TestBatchTimeout:
    def test_scales_with_shape_count(self, tmp_path):
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=100_000))
        assert tuner._batch_timeout_s(1) == sd._PER_SHAPE_BUDGET_S
        assert tuner._batch_timeout_s(10) == 10 * sd._PER_SHAPE_BUDGET_S

    def test_capped_by_the_outer_timeout(self, tmp_path):
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=1_000))
        assert tuner._batch_timeout_s(100) == 1_000 - sd._TIMEOUT_RESERVE_S

    def test_never_exceeds_the_outer_kill_timeout(self, tmp_path):
        # The floor used to be raised back to per_shape, handing aiter a
        # deadline past the point the outer watchdog kills it -- so it never
        # reached its own timeout and never flushed what it had.
        for timeout_s in (1, 30, 60, 120, 180, 240):
            for thorough in (False, True):
                tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=timeout_s, thorough=thorough))
                assert 1 <= tuner._batch_timeout_s(8) <= timeout_s, (timeout_s, thorough)


class TestStaleArtifactsAreCleared:
    """Row count only means "this run" if last run's rows are gone.

    Judging by output instead of exit code is the point of this tuner, and a
    tuned CSV left in the work dir by an earlier attempt would be read as this
    run's output -- turning an invocation that wrote nothing into a full,
    successful-looking result.
    """

    def test_previous_output_is_removed_before_launching(self, tmp_path, monkeypatch):
        work = tmp_path / "tuners" / "sglang_dense_bf16"
        work.mkdir(parents=True)
        stale = work / "tuned_dense_bf16.csv"
        stale.write_text(_csv([_row(1, 4096, 4096, "hipblaslt", 1.0)]), encoding="utf-8")
        (work / "profile_dense_bf16.csv").write_text(_csv([]), encoding="utf-8")

        seen: dict = {}

        def _writes_nothing(cmd, **kwargs):
            # Whatever the previous run left must already be gone by now.
            seen["tuned_exists"] = Path(cmd[cmd.index("-o") + 1]).exists()
            seen["profile_exists"] = Path(cmd[cmd.index("-o2") + 1]).exists()
            return 1, "", ""

        monkeypatch.setattr(sd, "probe_script", _permissive_surface)
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: _aiter_root(tmp_path))
        monkeypatch.setattr(sd, "_compute_nk_shapes", lambda **kw: list(_NK))
        monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(_M))
        monkeypatch.setattr(sd, "run_subprocess", _writes_nothing)

        result = _run(tmp_path)

        assert seen == {"tuned_exists": False, "profile_exists": False}
        # And the run that wrote nothing is reported as such, not as the stale row.
        assert result.total_shapes == 0
        assert result.status != "ok"

    def test_thorough_gets_a_bigger_per_shape_budget(self, tmp_path):
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=100_000, thorough=True))
        assert tuner._batch_timeout_s(2) == 2 * sd._PER_SHAPE_BUDGET_THOROUGH_S

    def test_timeout_is_actually_passed(self, tmp_path, monkeypatch):
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        _run(tmp_path, timeout_s=100_000)
        cmd = cap["cmd"]
        # 2 shapes (1 NK pair x 2 M values) at the fast per-shape budget.
        assert cmd[cmd.index("--timeout") + 1] == str(2 * sd._PER_SHAPE_BUDGET_S)


class TestShapeBudgetIsModeAware:
    """A thorough shape costs ~5.5x a fast one, so it cannot be counted the same.

    Measured per-backend on an 8-GPU MI355X box over four shapes: 169s for
    hipblaslt+asm+triton+skinny+opus+torch together, 1458s for flydsl alone.
    Sizing a `--libtype all` run with the fast figure claims 5.5x the shapes the
    batch can finish, and `--shape_grouped` then spends the whole allowance on
    the first few while the rest are written as nothing -- which the report
    cannot tell apart from a tuner that found no improvement.
    """

    def test_fast_and_thorough_use_their_own_cost(self, tmp_path):
        fast = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=3_600))
        thorough = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=3_600, thorough=True))

        assert fast._shape_budget() == (3_600 - sd._TIMEOUT_RESERVE_S) // sd._PER_SHAPE_COST_S
        assert thorough._shape_budget() == ((3_600 - sd._TIMEOUT_RESERVE_S) // sd._PER_SHAPE_COST_THOROUGH_S)
        # The whole point: an hour buys far fewer shapes when every backend is
        # searched, and claiming otherwise is what produced empty results.
        assert thorough._shape_budget() < fast._shape_budget()

    def test_thorough_budget_is_finishable(self, tmp_path):
        """The claimed shapes must fit inside the aiter timeout they are given."""
        for timeout_s in (1_800, 3_600, 7_200):
            tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=timeout_s, thorough=True))
            n = tuner._shape_budget()
            assert n * sd._PER_SHAPE_COST_THOROUGH_S <= timeout_s

    def test_explicit_override_still_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(sd._MAX_SHAPES_ENV, "7")
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=3_600, thorough=True))
        assert tuner._shape_budget() == 7

    def test_at_least_one_shape_even_on_a_tiny_budget(self, tmp_path):
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=10, thorough=True))
        assert tuner._shape_budget() == 1

    def test_default_timeout_fits_all_82_measured_buckets(self, tmp_path):
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=10_800))
        assert tuner._shape_budget() >= 82

    def test_demand_log_compares_buckets_to_buckets(self, tmp_path, monkeypatch, caplog):
        entry = {
            "distinct_keys": 3,
            "miss_count": 15,
            "keys": [
                {"M": 300, "N": 4096, "K": 4096, "requests": 7},
                {"M": 400, "N": 4096, "K": 4096, "requests": 3},
                {"M": 64, "N": 4096, "K": 4096, "requests": 5},
            ],
        }
        monkeypatch.setattr(sd, "load_demand", lambda _path: {"demands": [entry]})
        monkeypatch.setattr(sd, "demand_for_tuner", lambda _report, _name: entry)
        monkeypatch.setenv(sd._MAX_SHAPES_ENV, "1")
        tuner = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=10_800, demand_json=tmp_path / "demand.json"))

        with caplog.at_level("INFO"):
            shapes = tuner._demand_shapes()

        assert [shape["M"] for shape in shapes] == [512]
        assert "1 of 2 padded-M buckets selected" in caplog.text
        assert "covering 2 of 3 distinct raw keys" in caplog.text
        assert "from 10800s timeout" in caplog.text


class TestDerivedShapesRespectTheBudget:
    """The derived cross product used to ignore the budget the demand list honours.

    A 1800s thorough run generated 4 NK pairs x 22 M = 88 shapes at ~407s each:
    ~35000s of work in a 1680s window. The grouped batch spends the allowance on
    the first shapes and writes the rest as nothing, which is how a thorough run
    came back after 3606s having tuned zero.
    """

    def test_untouched_when_the_product_already_fits(self):
        m = [1, 8, 64, 512]
        assert sd._fit_m_values_to_budget(m, 4, 16) == m
        assert sd._fit_m_values_to_budget(m, 4, 999) == m

    def test_trims_m_not_nk(self):
        m = list(range(1, 23))
        out = sd._fit_m_values_to_budget(m, 4, 8)
        # 8 shapes across 4 NK pairs leaves 2 M values -- every matmul keeps an
        # entry, which dropping NK pairs instead would not give.
        assert len(out) == 2
        assert 4 * len(out) <= 8

    def test_keeps_both_ends_of_the_range(self):
        m = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        out = sd._fit_m_values_to_budget(m, 2, 8)
        assert out[0] == 1 and out[-1] == 512
        assert out == sorted(out)
        assert len(set(out)) == len(out)

    def test_samples_across_the_range_rather_than_truncating(self):
        m = list(range(1, 21))
        out = sd._fit_m_values_to_budget(m, 1, 5)
        # Evenly spaced over the whole list, not the first five.
        assert out == [1, 6, 11, 15, 20]
        assert out[:2] != m[:2]

    def test_one_m_per_nk_keeps_the_largest(self):
        # With room for a single M per matmul, prefill is the one that cannot be
        # served by a padded lookup from below.
        assert sd._fit_m_values_to_budget([1, 16, 128, 1024], 8, 8) == [1024]

    def test_degenerate_inputs_are_passed_through(self):
        m = [1, 2, 3]
        assert sd._fit_m_values_to_budget(m, 0, 4) == m
        assert sd._fit_m_values_to_budget(m, 2, 0) == m
        assert sd._fit_m_values_to_budget([], 4, 2) == []

    def test_generated_csv_row_count_matches_the_budget(self, tmp_path, monkeypatch):
        # The real generator's scale: the stub in _prep is too small to trim.
        nk = [(4096, 4096), (4096, 14336), (14336, 4096), (6144, 4096)]
        ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        monkeypatch.setattr(sd, "_compute_nk_shapes", lambda **kw: list(nk))
        monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(ms))

        _run(tmp_path, timeout_s=1_800, thorough=True)

        untuned = Path(cap["cmd"][cap["cmd"].index("-i") + 1])
        rows = untuned.read_text(encoding="utf-8").strip().splitlines()[1:]
        budget = (1_800 - sd._TIMEOUT_RESERVE_S) // sd._PER_SHAPE_COST_THOROUGH_S
        # Untrimmed this is 4 x 14 = 56 shapes, ~23000s of work in a 1680s window.
        assert len(nk) * len(ms) == 56
        # At most one M per NK pair may overshoot: keeping every matmul beats
        # covering more token counts on fewer of them.
        assert len(rows) <= max(budget, len(nk))
        # Every matmul still has an entry.
        assert len({(r.split(",")[1], r.split(",")[2]) for r in rows}) == len(nk)

    def test_fast_mode_keeps_its_wider_m_coverage(self, tmp_path, monkeypatch):
        nk = [(4096, 4096), (4096, 14336), (14336, 4096), (6144, 4096)]
        ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        cap = _prep(tmp_path, monkeypatch, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        monkeypatch.setattr(sd, "_compute_nk_shapes", lambda **kw: list(nk))
        monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(ms))

        _run(tmp_path, timeout_s=1_800, thorough=False)

        untuned = Path(cap["cmd"][cap["cmd"].index("-i") + 1])
        rows = untuned.read_text(encoding="utf-8").strip().splitlines()[1:]
        # 56 shapes at the fast cost is 5208s of work; the 1680s window pays for
        # 18, so fast mode trims too -- just far less aggressively than thorough.
        # (18, not 22: carrying torch for the baseline costs ~19s a shape, and
        # the budget has to charge for it or the batch is cut off part-way.)
        assert len(rows) == 4 * 4
        assert len(rows) > 4 * 1


# ── row count, not exit code, decides the outcome ────────────────────────────


class TestRowCountCriterion:
    def test_nonzero_rc_with_all_rows_is_ok(self, tmp_path, monkeypatch):
        # gemm_a16w16_tune.py exits 1 even when every shape tuned. Failing on
        # rc != 0 threw away complete, usable results.
        _prep(
            tmp_path,
            monkeypatch,
            rc=1,
            tuned_rows=[
                _row(1, 4096, 4096, "hipblaslt", 9.36),
                _row(512, 4096, 4096, "hipblaslt", 27.44),
            ],
        )
        res = _run(tmp_path)
        assert res.status == "ok"
        assert res.total_shapes == 2 and res.expected_shapes == 2

    def test_zero_rc_with_no_rows_is_empty_output(self, tmp_path, monkeypatch):
        # The shim rewrites the tuner's 1 into a 0, so rc==0 says nothing about
        # whether anything was written. This must not read as no_improvement.
        _prep(tmp_path, monkeypatch, rc=0, tuned_rows=[])
        res = _run(tmp_path)
        assert res.status == "empty_output"
        assert res.total_shapes == 0 and res.expected_shapes == 2
        assert res.candidate is False

    def test_missing_rows_are_partial_output(self, tmp_path, monkeypatch):
        # The grouped batch budget ran out after the first shape.
        _prep(tmp_path, monkeypatch, rc=1, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        res = _run(tmp_path)
        assert res.status == "partial_output"
        assert res.total_shapes == 1 and res.expected_shapes == 2
        assert res.to_dict()["missing_shapes"] == 1

    def test_partial_output_still_reaches_e2e(self, tmp_path, monkeypatch):
        _prep(tmp_path, monkeypatch, rc=1, tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)])
        res = _run(tmp_path)
        report = build_report(
            results=[res],
            skipped=[],
            profile=_ctx(tmp_path).profile,
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi355x",
            tp=1,
            conc=64,
            tokens=[1, 512],
            started_at="2026-01-01T00:00:00Z",
            total_elapsed_s=1.0,
        )
        assert report.micro_decision == "candidate"
        assert report.requires_e2e_validation is True


class TestOuterTimeoutKeepsWhatWasWritten:
    """A kill by the outer timeout must not discard rows already on disk.

    The tuner writes as it goes. Returning "failed" without looking at the CSV
    throws away completed shapes and reports nothing about how far it got --
    the same mistake as judging by exit code, one level up.
    """

    def test_partial_rows_survive_a_timeout(self, tmp_path, monkeypatch):
        _prep(
            tmp_path,
            monkeypatch,
            rc=124,
            tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)],
        )
        res = _run(tmp_path)
        assert res.status == "partial_output"
        assert res.total_shapes == 1 and res.expected_shapes == 2
        assert res.to_dict()["missing_shapes"] == 1
        assert res.candidate is True  # still worth validating end to end

    def test_timeout_with_no_rows_is_still_failed(self, tmp_path, monkeypatch):
        _prep(tmp_path, monkeypatch, rc=124, tuned_rows=[])
        res = _run(tmp_path)
        assert res.status == "failed" and res.error_class == "timeout"
        assert res.expected_shapes == 2

    def test_timeout_with_no_csv_at_all_is_failed(self, tmp_path, monkeypatch):
        _prep(tmp_path, monkeypatch, rc=124, tuned_rows=None)
        res = _run(tmp_path)
        assert res.status == "failed" and res.error_class == "timeout"


class TestHelpProbeGate:
    """The probe must refuse the run *before* it costs minutes, and must never
    veto a run just because it could not read --help."""

    def test_missing_with_hipblaslt_fails_before_running(self, tmp_path, monkeypatch):
        ran: list = []
        cap = _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)],
            surface=lambda s: ScriptSurface(str(s), frozenset({"-i", "-o", "-o2", "--libtype"}), True),
        )
        monkeypatch.setattr(sd, "run_subprocess", lambda cmd, **k: ran.append(cmd) or (0, "", ""))
        res = _run(tmp_path)
        assert res.status == "failed"
        assert res.error_class == "unsupported_argument"
        assert "--with-hipblaslt" in res.error
        assert ran == [], "tuner was launched despite an empty candidate set"
        assert "cmd" not in cap

    def test_droppable_flag_is_removed_not_fatal(self, tmp_path, monkeypatch):
        # -v only affects log verbosity, so a script that does not take it should
        # still be run -- without it.
        accepted = {
            "-i",
            "-o",
            "-o2",
            "--indtype",
            "--outdtype",
            "--mp",
            "--iters",
            "--warmup",
            "--timeout",
            "--shape_grouped",
            "--libtype",
            "--with-hipblaslt",
        }
        cap = _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[
                _row(1, 4096, 4096, "hipblaslt", 9.36),
                _row(512, 4096, 4096, "hipblaslt", 27.44),
            ],
            surface=lambda s: ScriptSurface(str(s), frozenset(accepted), True),
        )
        res = _run(tmp_path)
        assert res.status == "ok"
        assert "-v" not in cap["cmd"]
        assert "--with-hipblaslt" in cap["cmd"]

    def test_unreadable_help_does_not_block_the_run(self, tmp_path, monkeypatch):
        # Permissive surface (probed=False) is what _prep installs by default.
        cap = _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[
                _row(1, 4096, 4096, "hipblaslt", 9.36),
                _row(512, 4096, 4096, "hipblaslt", 27.44),
            ],
        )
        res = _run(tmp_path)
        assert res.status == "ok"
        assert "--with-hipblaslt" in cap["cmd"] and "-v" in cap["cmd"]


class TestRejectedArgument:
    """A rejected flag is a failure, not an empty run.

    The original breakage was 14 calls dying on
    ``unrecognized arguments: --libtype hipblaslt``. Anything that lets a
    rejected argument surface as "ran, nothing to report" recreates the exact
    illusion the row-count criterion exists to remove: the run looks complete
    and gainless when in fact the search space was never what was requested.
    """

    _STDERR = (
        "usage: gemm_a16w16_tune.py [-h] ...\ngemm_a16w16_tune.py: error: unrecognized arguments: --with-hipblaslt\n"
    )

    def test_rejected_argument_is_failed_not_empty_output(self, tmp_path, monkeypatch):
        _prep(tmp_path, monkeypatch, tuned_rows=None, rc=2, stderr=self._STDERR)
        res = _run(tmp_path)
        assert res.status == "failed"
        assert res.error_class == "unsupported_argument"

    def test_error_names_the_rejected_argument(self, tmp_path, monkeypatch):
        _prep(tmp_path, monkeypatch, tuned_rows=None, rc=2, stderr=self._STDERR)
        res = _run(tmp_path)
        assert "--with-hipblaslt" in res.error
        assert "gemm_a16w16_tune.py" in res.error

    def test_rejection_outranks_the_row_count(self, tmp_path, monkeypatch):
        # Even if a stale CSV from an earlier run is lying around, a rejected
        # argument means this invocation searched the wrong space.
        _prep(
            tmp_path,
            monkeypatch,
            rc=2,
            stderr=self._STDERR,
            tuned_rows=[_row(1, 4096, 4096, "hipblaslt", 9.36)],
        )
        res = _run(tmp_path)
        assert res.status == "failed" and res.error_class == "unsupported_argument"


# ── "no baseline" is not "no gain" ───────────────────────────────────────────


class TestUnverifiedShapes:
    def test_a_profile_without_torch_rows_has_no_baseline(self, tmp_path, monkeypatch):
        # What a torch-less profile CSV does downstream. Fast mode no longer
        # produces one -- it asks for `hipblaslt,torch` -- but the parser still
        # has to say "unverified" rather than "no gain" if torch is missing for
        # any other reason (an aiter build without it, a candidate that never
        # ran inside the batch budget).
        _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[
                _row(1, 4096, 4096, "hipblaslt", 9.36),
                _row(512, 4096, 4096, "hipblaslt", 27.44),
            ],
            profile_rows=[
                _row(1, 4096, 4096, "hipblaslt", 9.36),
                _row(512, 4096, 4096, "hipblaslt", 27.44),
            ],
        )
        res = _run(tmp_path)
        assert res.improved_shapes == 0
        assert res.unverified_shapes == 2
        assert res.best_micro_speedup == 1.0  # nothing fabricated from TFLOPS
        # Forced to e2e rather than dropped as no_improvement.
        assert res.candidate is True and res.status == "ok"
        assert all(r["tuned_unverified"] for r in res.shape_results)

    def test_torch_candidate_gives_a_real_speedup(self, tmp_path, monkeypatch):
        # --libtype all does time torch, which is exactly the kernel serving
        # falls back to, so the comparison is meaningful.
        _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[
                _row(1, 4096, 4096, "flydsl", 8.0),
                _row(512, 4096, 4096, "flydsl", 20.0),
            ],
            profile_rows=[
                _row(1, 4096, 4096, "torch", 10.0),
                _row(1, 4096, 4096, "flydsl", 8.0),
                _row(512, 4096, 4096, "torch", 10.0),
                _row(512, 4096, 4096, "flydsl", 20.0),
            ],
        )
        res = _run(tmp_path, thorough=True)
        assert res.improved_shapes == 1  # only M=1 beat torch
        assert res.unverified_shapes == 0
        assert res.best_micro_speedup == 1.25

    def test_fast_mode_reports_a_measured_speedup_not_unverified(self, tmp_path, monkeypatch):
        # The whole point of carrying torch in fast mode. Kimi-K3 on vLLM lands
        # here: 38600 misses in bf16_tuned_gemm.csv, every one of them falling
        # back to torch, and the run still reported best_micro_speedup=1.0
        # because nothing timed the kernel it was falling back to.
        _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[
                _row(1, 4096, 4096, "hipblaslt", 8.0),
                _row(512, 4096, 4096, "hipblaslt", 25.0),
            ],
            profile_rows=[
                _row(1, 4096, 4096, "torch", 10.0),
                _row(1, 4096, 4096, "hipblaslt", 8.0),
                _row(512, 4096, 4096, "torch", 20.0),
                _row(512, 4096, 4096, "hipblaslt", 25.0),
            ],
        )
        res = _run(tmp_path)  # fast, not thorough
        assert res.unverified_shapes == 0
        assert res.improved_shapes == 1
        assert res.best_micro_speedup == 1.25
        assert res.status == "ok"

    def test_infinite_torch_time_is_not_a_baseline(self, tmp_path, monkeypatch):
        # aiter writes `inf` for a candidate that never ran inside the budget.
        _prep(
            tmp_path,
            monkeypatch,
            tuned_rows=[_row(1, 4096, 4096, "flydsl", 8.0)],
            profile_rows=[_row(1, 4096, 4096, "torch", "inf")],
        )
        res = _run(tmp_path)
        assert res.shape_results[0]["tuned_unverified"] is True
        assert res.shape_results[0]["default_us"] is None

    def test_profile_defaults_ignore_non_torch_rows(self, tmp_path):
        p = tmp_path / "p.csv"
        p.write_text(
            _csv(
                [
                    _row(1, 4096, 4096, "hipblaslt", 9.0),
                    _row(1, 4096, 4096, "torch", 12.0),
                    _row(1, 4096, 4096, "torch", 11.0),
                ]
            ),
            encoding="utf-8",
        )
        # Only torch rows count, and the best of them wins.
        assert sd._parse_profile_defaults(p) == {(1, 4096, 4096): 11.0}

    def test_parse_survives_a_missing_file(self, tmp_path):
        assert sd._parse_profile_defaults(tmp_path / "nope.csv") == {}
        assert sd._parse_tuner_results(tmp_path / "nope.csv") == []
