# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TunableOp can consume the demand the router selected it on.

The router counts a demand file as a shape source, which is what unblocks this
tuner for a model with no recorded TunableOp trace. On real hardware that
change alone made things worse rather than better: the tuner was selected and
then failed in 0.0s with "No valid input file found", because it only knew
about --tunableop-input and --shapes-json. An honest skip had been replaced
with a silent failure.
"""

from __future__ import annotations

import json

from kernelforge.gemm_tune.tuners.vllm_dense_tunableop import (
    tunableop_untuned_line,
)


class TestTheRecordFormat:
    """Pinned to what torch itself wrote on an MI355X box.

    Enabling ``record_untuned`` and running bf16 ``a @ b.t()`` produced these
    exact lines. An inferred format would not fail loudly -- it would tune
    shapes nobody asked for.
    """

    OBSERVED = {
        (16, 1536, 7168): "tn_1536_16_7168_ld_7168_7168_1536",
        (1024, 4096, 7168): "tn_4096_1024_7168_ld_7168_7168_4096",
        (32, 2048, 4096): "tn_2048_32_4096_ld_4096_4096_2048",
    }

    def test_matches_what_torch_recorded(self):
        for (m, n, k), tail in self.OBSERVED.items():
            line = tunableop_untuned_line(m, n, k, "GemmTunableOp_BFloat16_TN")
            assert line == f"GemmTunableOp_BFloat16_TN,{tail}"

    def test_n_comes_before_m(self):
        # The one thing easy to get backwards, and it would silently tune the
        # transpose of every requested shape.
        line = tunableop_untuned_line(16, 1536, 7168, "op")
        assert line.startswith("op,tn_1536_16_7168")


def _demand_file(tmp_path, shapes):
    from kernelforge.gemm_tune.evidence import SCHEMA_VERSION

    path = tmp_path / "demand.json"
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "demands": [
                    {
                        "table": "tunableop",
                        "tuner": "vllm_dense_tunableop",
                        "env_var": "PYTORCH_TUNABLEOP_FILENAME",
                        "key_schema": ["M", "N", "K"],
                        "logged_fields": ["M", "N", "K"],
                        "miss_count": len(shapes),
                        "keys": [{"M": m, "N": n, "K": k, "requests": 1} for m, n, k in shapes],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _ctx(tmp_path, **overrides):
    from kernelforge.gemm_tune.model_analyzer import ModelProfile
    from kernelforge.gemm_tune.tuners.base import TuneContext

    base = dict(
        profile=ModelProfile(model_path="/fake", hidden_size=4096),
        framework="vllm",
        precision="bf16",
        quant_type="none",
        gpu_type="mi355x",
        tp=1,
        conc=8,
        tokens=[8],
        mp=1,
        output_dir=tmp_path,
        iters=20,
        warmup=5,
        min_improvement_pct=1.0,
        timeout_s=3600,
    )
    base.update(overrides)
    return TuneContext(**base)


class TestDemandBecomesAnInput:
    def _tuner(self, tmp_path, demand=None, precision="bf16"):
        from kernelforge.gemm_tune.tuners.vllm_dense_tunableop import (
            VllmDenseTunableopTuner,
        )

        return VllmDenseTunableopTuner(_ctx(tmp_path, precision=precision, demand_json=demand))

    def test_a_demand_file_alone_is_enough_to_validate(self, tmp_path):
        t = self._tuner(tmp_path, _demand_file(tmp_path, [(16, 1536, 7168)]))
        assert t.validate() is None

    def test_nothing_at_all_still_refuses_and_names_demand(self, tmp_path):
        reason = self._tuner(tmp_path).validate()
        assert reason and "--demand" in reason

    def test_demand_shapes_become_untuned_records(self, tmp_path):
        demand = _demand_file(tmp_path, [(16, 1536, 7168), (1024, 4096, 7168)])
        t = self._tuner(tmp_path, demand)

        path = t._resolve_input()

        assert path is not None
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines == [
            "GemmTunableOp_BFloat16_TN,tn_1536_16_7168_ld_7168_7168_1536",
            "GemmTunableOp_BFloat16_TN,tn_4096_1024_7168_ld_7168_7168_4096",
        ]

    def test_an_unknown_precision_produces_nothing_rather_than_a_guess(self, tmp_path):
        demand = _demand_file(tmp_path, [(16, 1536, 7168)])
        t = self._tuner(tmp_path, demand, precision="fp8")
        assert t._resolve_input() is None

    def test_the_shape_count_is_capped(self, tmp_path):
        from kernelforge.gemm_tune.tuners import vllm_dense_tunableop as vt

        many = [(m, 1536, 7168) for m in range(1, 400)]
        t = self._tuner(tmp_path, _demand_file(tmp_path, many))

        lines = t._resolve_input().read_text(encoding="utf-8").strip().splitlines()
        assert 0 < len(lines) <= vt._DEMAND_SHAPE_LIMIT

    def test_dense_demand_owned_by_another_tuner_is_still_usable(self, tmp_path):
        # The normal case on a real box: the runtime logs misses against
        # aiter's bf16 table, and TunableOp has no table of its own to miss.
        # The router still selects it off that demand, so it has to be able to
        # use it -- otherwise selection is followed by "no input file".
        from kernelforge.gemm_tune.evidence import SCHEMA_VERSION

        path = tmp_path / "demand.json"
        path.write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "demands": [
                        {
                            "table": "bf16_tuned_gemm.csv",
                            "tuner": "sglang_dense_bf16",
                            "key_schema": ["M", "N", "K"],
                            "miss_count": 2,
                            "keys": [
                                {"M": 16, "N": 1536, "K": 7168, "requests": 9},
                                {"M": 1024, "N": 4096, "K": 7168, "requests": 3},
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        lines = self._tuner(tmp_path, path)._resolve_input().read_text(encoding="utf-8").strip().splitlines()

        assert lines == [
            "GemmTunableOp_BFloat16_TN,tn_1536_16_7168_ld_7168_7168_1536",
            "GemmTunableOp_BFloat16_TN,tn_4096_1024_7168_ld_7168_7168_4096",
        ]

    def test_borrowed_demand_keeps_the_exact_m_the_runtime_asked_for(self, tmp_path):
        # demand_shapes() buckets to the padded M by default, which is right for
        # the aiter tuners only because aiter retries a failed lookup at the
        # padded M. TunableOp keys on the exact shape, so a record written at
        # M=512 does nothing for a request at M=464. The direct path already
        # passed bucket=False; the borrow path did not, so the fallback that
        # exists to make selection usable produced records nothing can hit.
        from kernelforge.gemm_tune.evidence import SCHEMA_VERSION

        path = tmp_path / "demand.json"
        path.write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "demands": [
                        {
                            "table": "bf16_tuned_gemm.csv",
                            "tuner": "sglang_dense_bf16",
                            "key_schema": ["M", "N", "K"],
                            "miss_count": 1,
                            "keys": [{"M": 464, "N": 4096, "K": 4096, "requests": 7}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        lines = self._tuner(tmp_path, path)._resolve_input().read_text(encoding="utf-8").strip().splitlines()

        assert lines == [
            "GemmTunableOp_BFloat16_TN,tn_4096_464_4096_ld_4096_4096_4096",
        ]

    def test_moe_demand_is_not_borrowed(self, tmp_path):
        # A MoE miss is not a dense (M, N, K) and tuning it here would be
        # tuning something the runtime never asked this path for.
        from kernelforge.gemm_tune.evidence import SCHEMA_VERSION

        path = tmp_path / "demand.json"
        path.write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "demands": [
                        {
                            "table": "tuned_fmoe.csv",
                            "tuner": "fmoe_ck",
                            "key_schema": ["token", "model_dim"],
                            "miss_count": 1,
                            "keys": [{"M": 16, "N": 1536, "K": 7168, "requests": 1}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        assert self._tuner(tmp_path, path)._resolve_input() is None

    def test_an_explicit_input_still_wins(self, tmp_path):
        explicit = tmp_path / "explicit.csv"
        explicit.write_text("GemmTunableOp_BFloat16_TN,tn_1_1_1_ld_1_1_1\n", encoding="utf-8")
        t = self._tuner(tmp_path, _demand_file(tmp_path, [(16, 1536, 7168)]))
        t.ctx.tunableop_input = explicit

        assert t._resolve_input() == explicit


class TestDemandReplacesTheModelConfig:
    """Demand is a *better* shape source, so it must not be gated on a worse one.

    A pure-MoE checkpoint has no dense intermediate_size, and the bf16 tuner
    refused to run on that basis even when the serving log had named 122 dense
    bf16 keys it had missed. That is precisely the case demand exists for.
    """

    def _tuner(self, tmp_path, **over):
        from kernelforge.gemm_tune.model_analyzer import ModelProfile
        from kernelforge.gemm_tune.tuners.base import TuneContext
        from kernelforge.gemm_tune.tuners.sglang_dense_bf16 import SglangDenseBf16Tuner

        base = dict(
            profile=ModelProfile(
                model_path="/fake",
                hidden_size=4096,
                intermediate_size=0,
            ),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi355x",
            tp=1,
            conc=8,
            tokens=[8],
            mp=1,
            output_dir=tmp_path,
            iters=20,
            warmup=5,
            min_improvement_pct=1.0,
            timeout_s=3600,
        )
        base.update(over)
        return SglangDenseBf16Tuner(TuneContext(**base))

    def test_a_moe_only_config_is_fine_when_demand_supplies_the_shapes(self, tmp_path, monkeypatch):
        from kernelforge.gemm_tune.tuners import sglang_dense_bf16 as sd

        root = tmp_path / "aiter"
        script = root / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py"
        script.parent.mkdir(parents=True)
        script.write_text("# tuner", encoding="utf-8")
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: root)

        t = self._tuner(tmp_path, demand_json=tmp_path / "demand.json")
        assert t.validate() is None

    def test_a_moe_only_config_alone_is_also_fine_on_its_attention_shapes(self, tmp_path, monkeypatch):
        """No demand, but the config still yields the attention projections.

        Missing ``intermediate_size`` only costs the FFN pair; QKV and O derive
        from ``hidden_size`` and the head counts, and their keys are correct. The
        earlier refusal threw those away too.
        """
        from kernelforge.gemm_tune.tuners import sglang_dense_bf16 as sd

        root = tmp_path / "aiter"
        script = root / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py"
        script.parent.mkdir(parents=True)
        script.write_text("# tuner", encoding="utf-8")
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: root)

        assert self._tuner(tmp_path).validate() is None

    def test_without_any_shape_source_it_still_refuses_and_says_why(self, tmp_path, monkeypatch):
        """Nothing derivable and no demand: refuse, and name the way out."""
        from kernelforge.gemm_tune.model_analyzer import ModelProfile
        from kernelforge.gemm_tune.tuners import sglang_dense_bf16 as sd

        root = tmp_path / "aiter"
        script = root / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py"
        script.parent.mkdir(parents=True)
        script.write_text("# tuner", encoding="utf-8")
        monkeypatch.setattr(sd, "resolve_aiter_root", lambda: root)

        barren = ModelProfile(model_path="/fake", hidden_size=0, intermediate_size=0)
        reason = self._tuner(tmp_path, profile=barren).validate()
        assert reason and "--demand" in reason


class TestTheFailureSaysWhichSourceWasMissing:
    def test_the_error_names_all_three_sources(self, tmp_path):
        from kernelforge.gemm_tune.tuners.vllm_dense_tunableop import (
            VllmDenseTunableopTuner,
        )

        result = VllmDenseTunableopTuner(_ctx(tmp_path)).run()

        assert result.status == "failed"
        # "No valid input file found" on its own cost a real run an hour of
        # guessing which of the three inputs was the missing one.
        for source in ("tunableop_input", "shapes_json", "demand_json"):
            assert source in (result.error or "")
