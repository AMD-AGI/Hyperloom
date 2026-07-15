# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the standalone baseline-arm roofline ceiling backup.

Verifies ``SharedState.record_baseline_roofline_ceiling`` computes a full
ceiling off baseline params alone (no profile trace), and that
``breakdown.collectors.collect_baseline`` passes it through unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown import collectors
from hyperloom.orchestrator.state.shared_state import SharedState


def _write_synthetic_model(model_dir: Path, *, total_size: int) -> None:
    """Lay down a minimal HF-shaped MoE model dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
        "torch_dtype": "bfloat16",
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "vocab_size": 151936,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}}),
        encoding="utf-8",
    )


def _make_baseline_state(tmp_path: Path) -> SharedState:
    """Build a SharedState whose baseline yaml + model dir resolve a ceiling."""
    _write_synthetic_model(tmp_path / "a3b", total_size=61_000_000_000)
    yaml_path = tmp_path / "bl.yaml"
    yaml_path.write_text(
        "benchmark:\n  envs:\n    CONC: 64\n", encoding="utf-8",
    )
    s = SharedState()
    s.model_path = str(tmp_path / "a3b")
    s.gpu_type = "mi300x"
    s.tp = 4
    s.precision = "bf16"
    s.conc = 64
    s.isl = 256
    s.osl = 256
    s.baseline_tput = 1707.9
    s.last_baseline = {"extras": {"materialized_config": str(yaml_path)}}
    return s


class TestRecordBaselineRooflineCeiling:
    def test_computes_full_ceiling_off_baseline_params(self, tmp_path):
        s = _make_baseline_state(tmp_path)
        ceiling = s.record_baseline_roofline_ceiling()

        # Returned dict and the cached field agree.
        assert ceiling is s.baseline_roofline_ceiling

        # Core numeric ceiling fields are present and positive.
        peak = ceiling["theoretical_peak_tok_per_sec"]
        assert isinstance(peak, float) and peak > 0
        assert ceiling["roofline_mem_ceiling_tok_per_sec"] > 0
        assert ceiling["roofline_cmp_ceiling_tok_per_sec"] > 0
        assert ceiling["roofline_bound_kind"] in ("memory", "compute")

        # Achieved + within/gap derived from baseline tput.
        assert ceiling["achieved_tok_per_sec"] == 1707.9

        assert ceiling["within_roofline_pct"] is not None
        assert ceiling["gap_to_roofline_pct"] is not None

        # Provenance marks the baseline arm.
        assert ceiling["ceiling_arm"] == "baseline"
        assert ceiling["roofline_provenance"]["runtime_tp"] == 4
        assert ceiling["roofline_provenance"]["effective_concurrency"] == 64
        # Compute-peak convention is surfaced.
        assert ceiling["roofline_provenance"]["compute_peak_convention"] == "achievable"
        assert ceiling["roofline_provenance"]["compute_peak_tflops"] > 0

        # Full per-op PerfModel breakdown is attached.
        pm = ceiling["perfmodel_breakdown"]
        assert pm["decode_tok_per_s"] > 0
        assert any(op["name"] == "moe_fused" for op in pm["ops"])

        # Trace-only fields stay empty/None (no profile ran).
        assert ceiling["kernel_roofline_path"] == ""
        assert ceiling["top_kernel"] is None
        assert ceiling["top_bottleneck"] is None

    def test_empty_when_no_baseline_yaml(self):
        s = SharedState()
        s.gpu_type = "mi300x"
        assert s.record_baseline_roofline_ceiling() == {}
        assert s.baseline_roofline_ceiling == {}


class TestCollectBaselinePassthrough:
    def test_collect_baseline_surfaces_ceiling(self, tmp_path):
        s = _make_baseline_state(tmp_path)
        s.record_baseline_roofline_ceiling()
        state_dict = {
            "baseline_tput": s.baseline_tput,
            "baseline_roofline_ceiling": s.baseline_roofline_ceiling,
        }
        warnings: list[str] = []
        out = collectors.collect_baseline(tmp_path, state_dict, warnings)
        rc = out["roofline_ceiling"]
        assert rc["theoretical_peak_tok_per_sec"] > 0
        assert rc["ceiling_arm"] == "baseline"

    def test_collect_baseline_surfaces_total_failures(self, tmp_path):
        warnings: list[str] = []
        out = collectors.collect_baseline(
            tmp_path,
            {"baseline_tput": 0.0, "baseline_failure_streak": 2,
             "baseline_total_failures": 3},
            warnings,
        )
        # Combined backstop count is surfaced alongside the per-class streak.
        assert out["failure_streak"] == 2
        assert out["total_failures"] == 3

    def test_collect_baseline_total_failures_defaults_zero(self, tmp_path):
        warnings: list[str] = []
        out = collectors.collect_baseline(tmp_path, {"baseline_tput": 100.0}, warnings)
        assert out["total_failures"] == 0

    def test_collect_baseline_empty_ceiling_when_absent(self, tmp_path):
        warnings: list[str] = []
        out = collectors.collect_baseline(tmp_path, {"baseline_tput": 100.0}, warnings)
        assert out["roofline_ceiling"] == {}
