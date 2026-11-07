# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hyperloom.common.benchmark_config import BenchmarkPointError
from hyperloom.measure.surface import (
    MeasurementSurface,
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
                "HYPERLOOM_EVAL_LIMIT": 250,
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
    value = "7" if name != "HYPERLOOM_EVAL_TASKS" else "easy_task"

    assert extract_surface(changed(config, **{name: value})).sha256 != baseline.sha256


def test_a_point_still_spelling_the_pre_switch_eval_names_has_another_surface_and_its_names_no_reader(config):
    """The eval names were ``MAGPIE_EVAL_*`` before the switch: a surface recorded then hashes another way, so whatever is
    keyed by it misses, and nothing reads the old names, so a point that still carries them evaluates the default."""
    envs = config["benchmark"]["envs"]
    old = copy.deepcopy(config)
    old["benchmark"]["envs"] = {**envs, "MAGPIE_EVAL_LIMIT": envs["HYPERLOOM_EVAL_LIMIT"]}
    del old["benchmark"]["envs"]["HYPERLOOM_EVAL_LIMIT"]
    pre_switch = MeasurementSurface({**extract_surface(old).fields, "envs.MAGPIE_EVAL_LIMIT": "250"})

    assert extract_surface(old).fields["envs.HYPERLOOM_EVAL_LIMIT"] is None
    assert pre_switch.sha256 != extract_surface(config).sha256
    assert set(pre_switch.differences(extract_surface(config))) == {
        "envs.MAGPIE_EVAL_LIMIT",
        "envs.HYPERLOOM_EVAL_LIMIT",
    }


def test_check_surface_names_the_field_that_differs(config):
    canonical = extract_surface(config)

    check_surface(changed(config, CONC="128"), canonical)
    with pytest.raises(SurfaceViolation, match=r"envs\.HYPERLOOM_EVAL_LIMIT: canonical '250', test '5'"):
        check_surface(changed(config, HYPERLOOM_EVAL_LIMIT=5), canonical)
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


#: Each spelling as the surface spelled it before every reader moved onto BenchmarkPoint (commit a438e0dbd^), and the
#: hash it gives over the evaluation names the surface holds.
_PARENT_SPELLINGS = [
    ("framework", "", "", "0f36829473e7d90acd0d3607e8ab3506261b3b4c62c7e79dd58ff43e61ab5227"),
    ("framework", "  ", "", "0f36829473e7d90acd0d3607e8ab3506261b3b4c62c7e79dd58ff43e61ab5227"),
    ("framework", 0, "0", "08ee7a6b2aa75bfb37cef74f8a3ec61fd2b6f788c28ebf65a18e5e54da497ee9"),
    ("framework", False, "false", "278088ece0781d74c7c79aa54b05e51ff3e1cc346a0734287c8a591d5597aed5"),
    ("framework", [], "[]", "539bf35d9f596fbd9a2e504be4491323fb066af083d67a898510b953c9500b64"),
    ("framework", {}, "{}", "1a619ca23a2bba7610172e8e691d07ff4700785c1495081fffb3a85259837a41"),
    ("framework", True, "true", "013294f3b31f543771f2ad1cd2766fca44f46f9e7a9a624ff94f5ea2b4b05b58"),
    ("framework", 1.0, "1", "94d376407f78bab85976162df1b12a3b63c97e3352f437099d8af67fb60c916f"),
    ("model", "", "", "069d56636bf0e46fbaab04f31dc1a64f65139e72324194b1dde6c7e9964fa17d"),
    ("model", "  ", "", "069d56636bf0e46fbaab04f31dc1a64f65139e72324194b1dde6c7e9964fa17d"),
    ("model", 0, "0", "5f8775a6b197b47fdd7695f5fdcd7bd2a0fc89d897b46321b7945ec9e4a192de"),
    ("model", False, "false", "852cf5333bd3a6feffd15bf641ec4fd28065b5d636c2ac917e91e2eca0b616c4"),
    ("model", [], "[]", "2a674250a564fbff69a7dff5820ef821ef578a082293d9f9ff8f5065e002c890"),
    ("model", {}, "{}", "659bb0305993abf5534c5fd1b8341a9cb2e3500f55259c70776c0097acbe0330"),
    ("model", True, "true", "45f382a2f7aa542f91e5179d30ce1afb99b64056bf1f0d0219d704b6dd2b1d3a"),
    ("model", 1.0, "1", "96240f28557d28931c20388dcb9bd7cd86e2591000927d1e7d8b9752a03f0c7b"),
    ("benchmark_script", "", "", "1d1f2b617beec5f5a0249d286fda973e24cf5655a6318ee1e03b063824b267c9"),
    ("benchmark_script", "  ", "", "1d1f2b617beec5f5a0249d286fda973e24cf5655a6318ee1e03b063824b267c9"),
    ("benchmark_script", 0, "0", "dcf55a733cc34bfd279ddac8214665a32b0397b737e15657bef9ec4387dcc691"),
    ("benchmark_script", False, "false", "de5b6c32f04103df70651693ef7f5ce5b4887e3347bdf27a0605a65f8fc4c034"),
    ("benchmark_script", [], "[]", "1bdc23ef30d4d2390a0f3a31b2de4bf2f0f7809fa84ec93dc75e3006438629c6"),
    ("benchmark_script", {}, "{}", "86155e39055b5bc0d3baca3e505cd7394e6d7a48a5e0ac15ac9b5a99a088e3a0"),
    ("benchmark_script", True, "true", "df3c6fc3fd22f086e43822381d9f23bf610a67f4cc627e9625d91b0ea7ca529f"),
    ("benchmark_script", 1.0, "1", "ab245f56a4c49d9a366a5e58f63ecc7cb81bd865becdd598ee3756b9fa7d6657"),
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
