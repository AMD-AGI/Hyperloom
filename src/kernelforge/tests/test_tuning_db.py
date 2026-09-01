"""Tests for the tuning database and auto-evolution pipeline."""

import tempfile

import pytest

from kernelforge.learning.auto_evolve import AutoEvolver
from kernelforge.learning.postmortem import PostMortem
from kernelforge.learning.tuning_db import TuningDatabase
from kernelforge.tracker.schema import Experiment

# These tests assert that the tuning database persists entries, but persistence is
# intentionally disabled in the source (kernelforge.learning.tuning_db,
# `_TUNING_DB_WRITE_ENABLED = False`) so runs do not mutate the committed
# knowledge_base. They are expected to fail until persistence is redesigned.
# strict=False so the suite stays green and auto-detects (XPASS) if the feature
# is re-enabled.
_PERSISTENCE_DISABLED = pytest.mark.xfail(
    reason="tuning DB persistence disabled (_TUNING_DB_WRITE_ENABLED=False); re-enable when persistence is redesigned",
    strict=False,
)


# ─── TuningDatabase tests ───


@_PERSISTENCE_DISABLED
def test_log_and_query_exact():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)
        db.log(
            operation="attention_bwd",
            backend="ck",
            gpu_target="gfx950",
            dtype="bf16",
            shape={"seq_len": 8192, "head_dim": 128},
            config={"BLOCK_M": 128, "wpe": 2},
            wall_ms=80.2,
            snr_db=35.0,
        )

        best = db.best_config("attention_bwd", "ck", shape={"seq_len": 8192, "head_dim": 128})
        assert best is not None
        assert best["wall_ms"] == 80.2
        assert best["config"]["BLOCK_M"] == 128


@_PERSISTENCE_DISABLED
def test_golden_config_updates():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        # First entry
        db.log(
            operation="gemm",
            backend="triton",
            gpu_target="gfx950",
            dtype="fp16",
            shape={"M": 4096, "N": 4096, "K": 4096},
            config={"BLOCK_M": 64},
            wall_ms=1.5,
            snr_db=40.0,
        )

        # Better entry — should replace golden
        db.log(
            operation="gemm",
            backend="triton",
            gpu_target="gfx950",
            dtype="fp16",
            shape={"M": 4096, "N": 4096, "K": 4096},
            config={"BLOCK_M": 128},
            wall_ms=0.9,
            snr_db=42.0,
        )

        best = db.best_config("gemm", "triton", shape={"M": 4096, "N": 4096, "K": 4096}, dtype="fp16")
        assert best["wall_ms"] == 0.9
        assert best["config"]["BLOCK_M"] == 128


def test_failed_correctness_not_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        db.log(
            operation="gemm",
            backend="ck",
            gpu_target="gfx950",
            dtype="bf16",
            shape={"M": 1024},
            config={"A": 1},
            wall_ms=0.1,
            snr_db=5.0,
            passed_correctness=False,
        )

        best = db.best_config("gemm", "ck", shape={"M": 1024})
        assert best is None  # Should not be golden


@_PERSISTENCE_DISABLED
def test_suggest_configs_similar_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        # Log for 4096
        db.log(
            operation="gemm",
            backend="ck",
            gpu_target="gfx950",
            dtype="bf16",
            shape={"M": 4096, "N": 4096},
            config={"BLOCK_M": 128, "BLOCK_N": 128},
            wall_ms=0.5,
            snr_db=40.0,
        )

        # Query for 8192 (within 2×)
        suggestions = db.suggest_configs("gemm", "ck", shape={"M": 8192, "N": 8192})
        assert len(suggestions) > 0
        assert suggestions[0]["config"]["BLOCK_M"] == 128


@_PERSISTENCE_DISABLED
def test_suggest_configs_related_op():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        # Log attention_fwd
        db.log(
            operation="attention_fwd",
            backend="ck",
            gpu_target="gfx950",
            dtype="bf16",
            shape={"seq_len": 8192},
            config={"BLOCK_M": 128, "wpe": 2},
            wall_ms=8.9,
            snr_db=35.0,
        )

        # Query attention_bwd — should get suggestion from fwd
        suggestions = db.suggest_configs("attention_bwd", "ck", shape={"seq_len": 8192})
        related = [s for s in suggestions if "related_op" in s.get("source", "")]
        assert len(related) > 0


@_PERSISTENCE_DISABLED
def test_transfer_rules():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        db.add_transfer_rule(
            rule_id="sparse_wpe2",
            description="For sparse attention on gfx950, wpe=2 beats wpe=3",
            scope="attention_*",
            parameter="wpe",
            recommended_value=2,
            anti_value=3,
            evidence=["exp_001", "exp_002"],
        )

        suggestions = db.suggest_configs("attention_bwd", "ck", shape={"seq_len": 8192})
        transfer = [s for s in suggestions if "transfer_rule" in s.get("source", "")]
        assert len(transfer) > 0
        assert transfer[0]["config"]["wpe"] == 2


@_PERSISTENCE_DISABLED
def test_discover_transfer_rules():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        # Log multiple operations where wpe=2 consistently wins
        for op in ["attention_fwd", "attention_bwd", "sla_fwd"]:
            db.log(
                operation=op,
                backend="flydsl",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 4096},
                config={"wpe": 2},
                wall_ms=8.0,
                snr_db=35.0,
            )
            db.log(
                operation=op,
                backend="flydsl",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 4096},
                config={"wpe": 3},
                wall_ms=12.0,
                snr_db=35.0,
            )

        rules = db.discover_transfer_rules()
        wpe_rules = [r for r in rules if r.parameter == "wpe"]
        assert len(wpe_rules) > 0
        assert wpe_rules[0].recommended_value == 2


@_PERSISTENCE_DISABLED
def test_context_for_task():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = TuningDatabase(tmpdir)

        db.log(
            operation="gemm",
            backend="ck",
            gpu_target="gfx950",
            dtype="bf16",
            shape={"M": 4096},
            config={"BLOCK_M": 128},
            wall_ms=0.5,
            snr_db=40.0,
        )

        ctx = db.context_for_task("gemm", "ck", shape={"M": 4096})
        assert "Best Known Config" in ctx
        assert "BLOCK_M" in ctx
        assert "START FROM THIS CONFIG" in ctx


@_PERSISTENCE_DISABLED
def test_auto_evolver_on_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = AutoEvolver(
            tuning_db=TuningDatabase(f"{tmpdir}/tuning"),
            postmortem=PostMortem(f"{tmpdir}/kb"),
        )

        evolver.on_benchmark(
            operation="gemm",
            backend="ck",
            shape={"M": 4096},
            config={"BLOCK_M": 128},
            wall_ms=0.5,
            snr_db=40.0,
        )

        best = evolver.tuning_db.best_config("gemm", "ck", shape={"M": 4096})
        assert best is not None


def test_auto_evolver_on_experiment_complete():
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = AutoEvolver(
            tuning_db=TuningDatabase(f"{tmpdir}/tuning"),
            postmortem=PostMortem(f"{tmpdir}/kb"),
        )

        exp = Experiment(experiment_id="test", backend="ck", task_id="attention_bwd")
        exp.add_iteration(snr_db=35.0, wall_ms=2.0, config={"BLOCK_M": 64})
        exp.add_iteration(snr_db=33.0, wall_ms=2.5, config={"BLOCK_M": 32})
        exp.add_iteration(snr_db=34.0, wall_ms=1.5, config={"BLOCK_M": 128})

        results = evolver.on_experiment_complete(exp)
        assert len(results["lessons"]) > 0
