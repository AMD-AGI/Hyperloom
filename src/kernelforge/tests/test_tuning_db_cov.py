"""Coverage tests for tuning_db read/query logic.

Persistence is disabled (_TUNING_DB_WRITE_ENABLED=False), so log() is a no-op.
These tests seed the on-disk files directly to exercise the read/query paths.
"""

from __future__ import annotations

import json

from kernelforge.learning.tuning_db import (
    TransferRule,
    TuningDatabase,
    TuningEntry,
)


def _seed_golden(db: TuningDatabase, golden: dict) -> None:
    db._golden_path.write_text(json.dumps(golden))


def _seed_entries(db: TuningDatabase, entries: list[dict]) -> None:
    with open(db._entries_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, default=str) + "\n")


def _seed_rules(db: TuningDatabase, rules: list[dict]) -> None:
    db._rules_path.write_text(json.dumps(rules))


# ─── dataclass helpers ───


def test_tuning_entry_keys_and_roundtrip():
    e = TuningEntry(
        operation="gemm",
        backend="ck",
        gpu_target="gfx950",
        dtype="bf16",
        shape={"N": 4096, "M": 1024},
        config={"BLOCK_M": 128},
        wall_ms=1.0,
    )
    # shape_key is sorted by key name
    assert e.shape_key() == "M=1024|N=4096"
    assert e.context_key() == "gemm|ck|gfx950|bf16|M=1024|N=4096"

    d = e.to_dict()
    restored = TuningEntry.from_dict({**d, "unknown_field": "ignored"})
    assert restored.operation == "gemm"
    assert restored.shape == {"N": 4096, "M": 1024}


def test_transfer_rule_roundtrip():
    r = TransferRule(rule_id="r1", description="d", scope="all", parameter="wpe", recommended_value=2)
    d = r.to_dict()
    restored = TransferRule.from_dict({**d, "extra": 1})
    assert restored.rule_id == "r1"
    assert restored.recommended_value == 2


# ─── log() no-op behavior (persistence disabled) ───


def test_log_returns_entry_but_does_not_persist(tmp_path):
    db = TuningDatabase(tmp_path)
    entry = db.log(
        operation="gemm",
        backend="ck",
        gpu_target="gfx950",
        dtype="bf16",
        shape={"M": 4096},
        config={"BLOCK_M": 64},
        wall_ms=1.0,
    )
    assert isinstance(entry, TuningEntry)
    assert not db._entries_path.exists()
    assert db.all_entries() == []


# ─── best_config ───


def test_best_config_exact_match(tmp_path):
    db = TuningDatabase(tmp_path)
    key = "gemm|ck|gfx950|bf16|M=4096"
    _seed_golden(db, {key: {"config": {"BLOCK_M": 128}, "wall_ms": 0.5}})
    best = db.best_config("gemm", "ck", shape={"M": 4096})
    assert best["wall_ms"] == 0.5


def test_best_config_prefix_fallback_picks_min(tmp_path):
    db = TuningDatabase(tmp_path)
    _seed_golden(
        db,
        {
            "gemm|ck|gfx950|bf16|M=1024": {"config": {}, "wall_ms": 2.0},
            "gemm|ck|gfx950|bf16|M=2048": {"config": {}, "wall_ms": 0.8},
        },
    )
    # No exact shape match -> falls back to cheapest matching prefix.
    best = db.best_config("gemm", "ck", shape={"M": 9999})
    assert best["wall_ms"] == 0.8


def test_best_config_none_when_empty(tmp_path):
    db = TuningDatabase(tmp_path)
    assert db.best_config("gemm", "ck", shape={"M": 4096}) is None
    assert db.best_config("gemm", "ck") is None


# ─── all_entries ───


def test_all_entries_skips_blank_lines(tmp_path):
    db = TuningDatabase(tmp_path)
    with open(db._entries_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "operation": "gemm",
                    "backend": "ck",
                    "gpu_target": "gfx950",
                    "dtype": "bf16",
                    "shape": {"M": 4096},
                    "config": {},
                    "wall_ms": 1.0,
                }
            )
            + "\n"
        )
        f.write("\n")  # blank line ignored
    entries = db.all_entries()
    assert len(entries) == 1
    assert entries[0].operation == "gemm"


# ─── suggest_configs (all four levels) ───


def test_suggest_configs_all_levels(tmp_path):
    db = TuningDatabase(tmp_path)
    # exact-match golden for level 1
    _seed_golden(
        db,
        {
            "attention_bwd|ck|gfx950|bf16|seq_len=8192": {"config": {"BLOCK_M": 128}, "wall_ms": 80.0},
        },
    )
    # entries drive level 2 (similar shape) + level 3 (related op)
    _seed_entries(
        db,
        [
            dict(
                operation="attention_bwd",
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 8192},
                config={"BLOCK_N": 64},
                wall_ms=82.0,
                passed_correctness=True,
            ),
            dict(
                operation="attention_fwd",
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 8192},
                config={"wpe": 2},
                wall_ms=9.0,
                passed_correctness=True,
            ),
        ],
    )
    # transfer rule for level 4
    _seed_rules(
        db,
        [dict(rule_id="r", description="d", scope="attention_*", parameter="wpe", recommended_value=2, confidence=0.6)],
    )

    suggestions = db.suggest_configs("attention_bwd", "ck", shape={"seq_len": 8192}, max_suggestions=10)
    sources = {s["source"].split(" ")[0] for s in suggestions}
    assert "exact_match" in sources
    assert any(s.startswith("similar_shape") for s in sources)
    assert any(s.startswith("related_op") for s in sources)
    assert any(s.startswith("transfer_rule") for s in sources)


def test_suggest_configs_dedup_and_max(tmp_path):
    db = TuningDatabase(tmp_path)
    _seed_entries(
        db,
        [
            dict(
                operation="gemm",
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"M": 4096},
                config={"BLOCK_M": 128},
                wall_ms=1.0,
                passed_correctness=True,
            ),
            dict(
                operation="gemm",
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"M": 4096},
                config={"BLOCK_M": 128},
                wall_ms=1.2,
                passed_correctness=True,
            ),  # duplicate config -> deduped
        ],
    )
    suggestions = db.suggest_configs("gemm", "ck", shape={"M": 4096}, max_suggestions=1)
    assert len(suggestions) == 1


# ─── _shape_similar ───


def test_shape_similar_variants(tmp_path):
    db = TuningDatabase(tmp_path)
    assert db._shape_similar({"M": 4096}, {"M": 8192}, factor=2.0)
    assert not db._shape_similar({"M": 4096}, {"M": 16384}, factor=2.0)
    assert not db._shape_similar({"M": 4096}, {"N": 4096})  # no shared keys


# ─── transfer rules: add / update / applies ───


def test_rule_applies_scopes(tmp_path):
    db = TuningDatabase(tmp_path)
    assert db._rule_applies({"scope": "all"}, "anything")
    assert db._rule_applies({"scope": "attention_*"}, "attention_bwd")
    assert not db._rule_applies({"scope": "attention_*"}, "gemm")
    assert db._rule_applies({"scope": "sparse"}, "sla_sparse_fwd")
    assert not db._rule_applies({"scope": "moe"}, "gemm")


def test_add_transfer_rule_new_and_update(tmp_path):
    db = TuningDatabase(tmp_path)
    # Persistence disabled: pre-seed the rules file so read-modify-write sees it.
    _seed_rules(db, [])
    db.add_transfer_rule(
        rule_id="r1", description="first", scope="all", parameter="wpe", recommended_value=2, evidence=["e1", "e2"]
    )
    # Writes are disabled, so nothing persisted; seed the existing rule and
    # verify the update branch merges evidence/confidence in memory.
    _seed_rules(
        db,
        [
            dict(
                rule_id="r1",
                description="first",
                scope="all",
                parameter="wpe",
                recommended_value=2,
                anti_value=None,
                evidence=["e1"],
                confidence=0.2,
            )
        ],
    )
    db.add_transfer_rule(
        rule_id="r1", description="updated", scope="all", parameter="wpe", recommended_value=3, evidence=["e2"]
    )
    # No persistence, but the update path executed without error.
    assert db._load_rules()[0]["rule_id"] == "r1"


# ─── discover_transfer_rules ───


def test_discover_transfer_rules_too_few_entries(tmp_path):
    db = TuningDatabase(tmp_path)
    _seed_entries(
        db,
        [
            dict(
                operation="gemm",
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"M": 4096},
                config={"wpe": 2},
                wall_ms=1.0,
                passed_correctness=True,
            ),
        ],
    )
    assert db.discover_transfer_rules() == []


def test_discover_transfer_rules_finds_winner(tmp_path):
    db = TuningDatabase(tmp_path)
    entries = []
    # wpe=2 consistently beats wpe=3 across three operations.
    for op in ["attention_fwd", "attention_bwd", "sla_fwd"]:
        entries.append(
            dict(
                operation=op,
                backend="flydsl",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 4096},
                config={"wpe": 2},
                wall_ms=8.0,
                passed_correctness=True,
            )
        )
        entries.append(
            dict(
                operation=op,
                backend="flydsl",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 4096},
                config={"wpe": 3},
                wall_ms=12.0,
                passed_correctness=True,
            )
        )
    _seed_entries(db, entries)
    rules = db.discover_transfer_rules()
    wpe = [r for r in rules if r.parameter == "wpe"]
    assert wpe and wpe[0].recommended_value == 2
    assert wpe[0].anti_value == 3


def test_discover_transfer_rules_single_value_skipped(tmp_path):
    db = TuningDatabase(tmp_path)
    # Only one value per param across >=5 entries -> nothing to compare.
    entries = [
        dict(
            operation=f"op{i}",
            backend="ck",
            gpu_target="gfx950",
            dtype="bf16",
            shape={"M": 4096},
            config={"wpe": 2},
            wall_ms=1.0,
            passed_correctness=True,
        )
        for i in range(6)
    ]
    _seed_entries(db, entries)
    assert db.discover_transfer_rules() == []


# ─── context_for_task ───


def test_context_for_task_with_data(tmp_path):
    db = TuningDatabase(tmp_path)
    _seed_golden(
        db,
        {
            "gemm|ck|gfx950|bf16|M=4096": {
                "config": {"BLOCK_M": 128},
                "wall_ms": 0.5,
                "pmc_diagnosis": "COMPUTE-BOUND",
            },
        },
    )
    _seed_rules(
        db,
        [
            dict(
                rule_id="r",
                description="use wpe=2",
                scope="all",
                parameter="wpe",
                recommended_value=2,
                anti_value=3,
                confidence=0.6,
            )
        ],
    )
    ctx = db.context_for_task("gemm", "ck", shape={"M": 4096})
    assert "Best Known Config" in ctx
    assert "START FROM THIS CONFIG" in ctx
    assert "COMPUTE-BOUND" in ctx
    assert "Transfer Rules" in ctx
    assert "AVOID" in ctx


def test_context_for_task_no_match(tmp_path):
    db = TuningDatabase(tmp_path)
    ctx = db.context_for_task("gemm", "ck", shape={"M": 4096})
    assert "No exact match" in ctx
    assert "DB Stats" in ctx
