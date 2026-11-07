# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hyperloom.common.benchmark_config import BenchmarkPointError
from hyperloom.measure.surface import (
    POINT_ENV_KEYS,
    REQUEST_ENV_KEYS,
    SURFACE_ENV_KEYS,
    SurfaceViolation,
    check_launch_envs,
    check_point,
    check_surface,
    extract_surface,
    hash_model_files,
    point_values,
)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"hidden": 1}')
    (root / "tokenizer.json").write_text("{}")
    (root / "model-00001.safetensors").write_bytes(b"weights")
    return root


@pytest.fixture()
def inferencex(tmp_path: Path) -> Path:
    scripts = tmp_path / "ix" / "benchmarks"
    scripts.mkdir(parents=True)
    (scripts / "sglang_mi355x.sh").write_text("run --isl $ISL\n")
    return tmp_path / "ix"


@pytest.fixture()
def config(model_dir: Path, inferencex: Path) -> dict:
    return {
        "benchmark": {
            "framework": "sglang",
            "model": str(model_dir),
            "benchmark_script": "sglang_mi355x.sh",
            "inferencex_path": str(inferencex),
            "envs": {
                "ISL": 1024,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1.0,
                "MAGPIE_EVAL_LIMIT": 250,
                "CONC": "64",
                "RUN_EVAL": "true",
                "NUM_PROMPTS": 320,
                "NUM_WARMUPS": 8,
            },
        }
    }


def changed(config: dict, **overrides) -> dict:
    other = copy.deepcopy(config)
    for key, value in overrides.items():
        other["benchmark"]["envs"][key] = value
    return other


def test_surface_hash_is_stable_and_ignores_what_the_request_decides(config):
    baseline = extract_surface(config).sha256

    assert extract_surface(copy.deepcopy(config)).sha256 == baseline
    assert extract_surface(changed(config, CONC="128", RUN_EVAL="false", NUM_PROMPTS=640)).sha256 == baseline
    assert extract_surface(changed(config, RANDOM_RANGE_RATIO=1)).sha256 == baseline


@pytest.mark.parametrize("name", [*SURFACE_ENV_KEYS])
def test_every_workload_and_eval_field_changes_the_hash(config, name):
    baseline = extract_surface(config)
    value = "7" if name != "MAGPIE_EVAL_TASKS" else "easy_task"

    assert extract_surface(changed(config, **{name: value})).sha256 != baseline.sha256


def test_check_surface_names_the_field_that_differs(config):
    canonical = extract_surface(config)

    check_surface(changed(config, CONC="128"), canonical)
    with pytest.raises(SurfaceViolation, match=r"envs\.MAGPIE_EVAL_LIMIT: canonical '250', test '5'"):
        check_surface(changed(config, MAGPIE_EVAL_LIMIT=5), canonical)
    with pytest.raises(SurfaceViolation, match=r"envs\.OSL: canonical '1024', test '16'"):
        check_surface(changed(config, OSL=16), canonical)


def test_benchmark_script_content_and_model_files_are_part_of_the_surface(config, inferencex, model_dir):
    canonical = extract_surface(config)

    (inferencex / "benchmarks" / "sglang_mi355x.sh").write_text("run --isl 1\n")
    with pytest.raises(SurfaceViolation, match="benchmark_script_sha256"):
        check_surface(config, canonical)

    (inferencex / "benchmarks" / "sglang_mi355x.sh").write_text("run --isl $ISL\n")
    check_surface(config, canonical)
    (model_dir / "tokenizer.json").write_text('{"chat_template": "x"}')
    with pytest.raises(SurfaceViolation, match="model_files"):
        check_surface(config, canonical)
    (model_dir / "model-00001.safetensors").write_bytes(b"other weights")


def test_other_script_name_is_a_surface_change(config):
    other = copy.deepcopy(config)
    other["benchmark"]["benchmark_script"] = "sglang_easy.sh"

    with pytest.raises(SurfaceViolation, match=r"benchmark\.benchmark_script"):
        check_surface(other, extract_surface(config))


def test_model_hashes_cover_configuration_and_tokenizer_but_not_weights(model_dir):
    assert set(hash_model_files(model_dir)) == {"config.json", "tokenizer.json"}


@pytest.mark.parametrize("name", [*SURFACE_ENV_KEYS, *POINT_ENV_KEYS, *REQUEST_ENV_KEYS])
def test_launch_environment_may_not_set_guarded_variables(name):
    with pytest.raises(SurfaceViolation, match=name.lower()):
        check_launch_envs(["HSA_X", name.lower()])  # the check is case-insensitive and reports the name as given


def test_launch_environment_with_ordinary_variables_passes():
    check_launch_envs(["HSA_FORCE_FINE_GRAIN_PCIE", "AITER_CONFIG_GEMM_BF16"])


def test_point_is_held_to_concurrency_run_eval_and_canonical_counts(config):
    reference = point_values(config)
    check_point(config, concurrency=64, accuracy=True, reference=reference)

    with pytest.raises(SurfaceViolation, match=r"CONC: expected '32', point has '64'"):
        check_point(config, concurrency=32, accuracy=True, reference=reference)
    with pytest.raises(SurfaceViolation, match=r"RUN_EVAL: expected 'false', point has 'true'"):
        check_point(config, concurrency=64, accuracy=False, reference=reference)
    with pytest.raises(SurfaceViolation, match=r"NUM_PROMPTS: expected '320', point has '8'"):
        check_point(changed(config, NUM_PROMPTS=8), concurrency=64, accuracy=True, reference=reference)


def test_a_field_the_point_leaves_out_is_none_in_the_surface(config: dict) -> None:
    del config["benchmark"]["benchmark_script"]

    fields = extract_surface(config).fields
    assert fields["benchmark.benchmark_script"] is None
    assert fields["benchmark_script_sha256"] is None


def test_a_configuration_with_no_benchmark_block_is_refused() -> None:
    for read in (extract_surface, point_values, lambda c: check_point(c, concurrency=1, accuracy=False, reference={})):
        with pytest.raises(BenchmarkPointError, match="benchmark mapping"):
            read({"model": "m"})


#: Each spelling as the surface spelled it before every reader moved onto BenchmarkPoint (commit a438e0dbd^).
_PARENT_SPELLINGS = [
    ("framework", "", "", "577c55411f01506f10525b8d1a9b050668e3b298a8de34bf547ffc5ed7b6a3a4"),
    ("framework", "  ", "", "577c55411f01506f10525b8d1a9b050668e3b298a8de34bf547ffc5ed7b6a3a4"),
    ("framework", 0, "0", "434c122728b7b7a59ef944686e0ab0d410adbc524c06676961cf90256225f39c"),
    ("framework", False, "false", "29210b5ffafefd3b7c82688d67e5fafa772fd249f2482478acad037416b55eb9"),
    ("framework", [], "[]", "65890228c5b66b68faa71adfb36921cf2f779826382f0710f3c6fed18f7d2394"),
    ("framework", {}, "{}", "10c877d4a398c85fd44fbae1412efbb2ff28d15ebb27b304face9fee1ce3af10"),
    ("framework", True, "true", "02944ee3def8ab4f356e8d0ab207f5f87c57fbb2d3a1f92d6922c5296b44a456"),
    ("framework", 1.0, "1", "70d4c70d735a301fdaaaa5e42024fbbcbc332558193d37ac7d62f24d7b32bc09"),
    ("model", "", "", "51050b23e9b2aa7fd49fd5571b78706c8d132ac7857009ed548d2134e401f1b2"),
    ("model", "  ", "", "51050b23e9b2aa7fd49fd5571b78706c8d132ac7857009ed548d2134e401f1b2"),
    ("model", 0, "0", "2c4f142b681595b7d32be4992977be33891492c181b641a5f49faf13c609880d"),
    ("model", False, "false", "9f2ffff89146754d400232e6b7a0a16e1f0ccfcacd9a441efefac57d8a1cf74f"),
    ("model", [], "[]", "c041373ebc4dd700e34d6925a5009532655b7e45065166de091320b082a2d8ac"),
    ("model", {}, "{}", "40ed54ea5b94d4b64af8a8e0a76487a93d00ef77755ac3f3981719bfea23677c"),
    ("model", True, "true", "3fe1d7f5ba7ae338d1c03e71e0ef87a9890c11fd9c0234a0d3fde426a9659866"),
    ("model", 1.0, "1", "ca21f71fb851711acbdb74d4f8248208adabed1fad76eabff817ae498c75bb7f"),
    ("benchmark_script", "", "", "7bc9ca4b4f1368a77726cdbe821d17a806670608b90e19422a46496b65f4e990"),
    ("benchmark_script", "  ", "", "7bc9ca4b4f1368a77726cdbe821d17a806670608b90e19422a46496b65f4e990"),
    ("benchmark_script", 0, "0", "4704821def878f1ef1bd6059317db456afa88b7fd9f0597cc4ef52459c0431c6"),
    ("benchmark_script", False, "false", "a3ed3ae44b649952bc6010470689eefe5098e98fbe9f9f002855f304267893d2"),
    ("benchmark_script", [], "[]", "d81bca751aa43da8e2f77684dfdb4fd4451160d40377b1b85eb76955024f8b0e"),
    ("benchmark_script", {}, "{}", "30f1c1c9c50ad40bd20bc6131f8b0a70c77ef220d3cc0e147e32be2b5f968220"),
    ("benchmark_script", True, "true", "548e4b89efb3740b6612268696f11ef4d05433f15670bb65da9e3802f21f7bba"),
    ("benchmark_script", 1.0, "1", "0164e567a990c556fe1cc2189e733677c8b2aca5cf0690c1b3af4822b4f4c0d3"),
]


@pytest.mark.parametrize(("key", "value", "spelled", "sha256"), _PARENT_SPELLINGS)
def test_a_falsy_or_non_text_field_keeps_the_surface_hash_it_had(
    key: str, value: object, spelled: str, sha256: str
) -> None:
    benchmark = {
        "framework": "sglang",
        "model": "/nonexistent/model",
        "benchmark_script": "bench.sh",
        "envs": {"ISL": 1024},
    }
    surface = extract_surface({"benchmark": {**benchmark, key: value}})

    assert (surface.fields[f"benchmark.{key}"], surface.sha256) == (spelled, sha256)
