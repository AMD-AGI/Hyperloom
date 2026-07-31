"""Unit tests for the opt-in candidate filter.

The filter's whole purpose is to remove benchmark slots, so every one of its
failure modes has to end in "keep everything" rather than "keep nothing" or
"raise". These tests pin that, plus the version guard, which exists because a
ranker from a neighbouring sglang version scores below chance.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _delta_filter as df


def _variant(name: str, args: str = "", envs: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, extra_server_args=args, extra_envs=envs or {})


@pytest.fixture
def grid():
    return [_variant("kv", "--kv-cache-dtype fp8_e4m3"),
            _variant("cons", "--schedule-conservativeness 0.3"),
            _variant("both", "--kv-cache-dtype fp8_e4m3 --enable-torch-compile")]


@pytest.fixture
def ctx():
    return {"framework": "sglang", "framework_version": "0.5.12",
            "model_path": "/nonexistent/model", "precision": "fp8",
            "hardware": "mi300x"}


def _fake_artifact(tmp_path, version="0.5.12", drop_flag="__schedule_conservativeness"):
    """A minimal artifact whose model is a real booster over two features."""
    lgb = pytest.importorskip("lightgbm")
    np = pytest.importorskip("numpy")

    cols = ["flag__kv_cache_dtype", "flag%s" % drop_flag]
    # Label 1 whenever the kv flag is present, 0 whenever the drop flag is, so
    # the fitted model separates the two candidates the tests care about.
    rows, labels = [], []
    for _ in range(80):
        rows += [[1, 0], [0, 1]]
        labels += [1, 0]
    ds = lgb.Dataset(np.array(rows, dtype=np.float32), label=np.array(labels),
                     feature_name=cols)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 2,
                         "min_data_in_leaf": 1, "min_data_in_bin": 1},
                        ds, num_boost_round=20)
    d = tmp_path / "ranker"
    d.mkdir()
    booster.save_model(str(d / "ranker.txt"))
    (d / "schema.json").write_text(json.dumps(
        {"columns": cols, "categorical": [], "vocab": {}}))
    (d / "manifest.json").write_text(json.dumps(
        {"valid_only_for_version": version, "framework": "sglang"}))
    return d


def test_disabled_without_env(monkeypatch, grid, ctx):
    monkeypatch.delenv(df.ENV_DIR, raising=False)
    kept, info = df.filter_variants(grid, **ctx)
    assert kept == grid
    assert info["reason"] == "disabled"
    assert info["applied"] is False


def test_non_sglang_is_untouched(monkeypatch, tmp_path, grid, ctx):
    monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
    kept, info = df.filter_variants(grid, **{**ctx, "framework": "vllm"})
    assert kept == grid
    assert info["reason"] == "framework_not_sglang"


def test_version_mismatch_disables_filtering(monkeypatch, tmp_path, grid, ctx):
    """A ranker from another version is worse than none, so it must not be used."""
    monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path, version="0.5.11")))
    kept, info = df.filter_variants(grid, **ctx)
    assert kept == grid
    assert info["reason"] == "version_mismatch"


def test_unreadable_artifact_keeps_everything(monkeypatch, tmp_path, grid, ctx):
    """Naming lightgbm's exception type here once made the handler itself raise."""
    monkeypatch.setenv(df.ENV_DIR, str(tmp_path / "does_not_exist"))
    kept, info = df.filter_variants(grid, **ctx)
    assert kept == grid
    assert info["reason"] == "artifact_unreadable"


def test_drops_the_low_scoring_variant(monkeypatch, tmp_path, grid, ctx):
    monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
    monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
    kept, info = df.filter_variants(grid, **ctx)
    names = {v.name for v in kept}
    assert "cons" not in names, "the flag trained as harmful should be dropped"
    assert "kv" in names
    assert info["reason"] == "ok"
    assert info["threshold"] == 0.5


def test_never_empties_the_grid(monkeypatch, tmp_path, grid, ctx):
    """An empty grid turns a slow search into no search."""
    monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
    monkeypatch.setenv(df.ENV_THRESHOLD, "0.999999")
    kept, info = df.filter_variants(grid, **ctx)
    assert len(kept) == 1
    assert info["reason"] == "all_below_threshold_kept_best"


def test_bad_threshold_falls_back_to_default(monkeypatch, tmp_path, grid, ctx):
    monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
    monkeypatch.setenv(df.ENV_THRESHOLD, "not-a-number")
    _, info = df.filter_variants(grid, **ctx)
    assert info["threshold"] == df.DEFAULT_THRESHOLD


def test_empty_input_is_returned_unchanged(monkeypatch, tmp_path, ctx):
    monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
    kept, info = df.filter_variants([], **ctx)
    assert kept == []
    assert info["reason"] == "empty_input"


def test_delta_features_capture_flags_and_envs():
    feats = df._delta_features("--kv-cache-dtype fp8_e4m3 --page-size 16",
                               {"SGLANG_USE_AITER": "1"})
    assert feats["flag__kv_cache_dtype"] == 1
    assert feats["flag__page_size"] == 1
    assert feats["env_SGLANG_USE_AITER"] == 1
    assert feats["shape_n_flags"] == 2
    assert feats["shape_max_num"] == 16.0
