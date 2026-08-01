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


@pytest.fixture
def checkpoint(tmp_path):
    """A readable config.json, which the LLM backend requires before ranking."""
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 64,
        "num_key_value_heads": 8, "vocab_size": 151936, "intermediate_size": 25600,
    }))
    return str(d)


class TestLLMBackend:
    """The served-ranker path, which prunes by keeping a fraction of the order."""

    @staticmethod
    def _reply(order: str):
        """Patch the ranker call to return a fixed ordering."""
        return lambda variants, identity_text: (
            [int(t) - 1 for t in order.split(",")] if order else None)

    def test_keeps_the_top_fraction(self, monkeypatch, grid, ctx, checkpoint):
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_KEEP_FRACTION, "0.3")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("3,1,2"))
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert info["reason"] == "ok_llm"
        # ceil(3 * 0.3) == 1, and the reply ranked candidate 3 first.
        assert [v.name for v in kept] == ["both"]

    def test_survivors_keep_their_original_order(self, monkeypatch, grid, ctx, checkpoint):
        """The ranking selects; it does not reorder what the caller then measures."""
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_KEEP_FRACTION, "0.6")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("3,1,2"))
        kept, _ = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        # ceil(3 * 0.6) == 2: candidates 3 and 1 survive, and they come back in
        # grid order rather than rank order.
        assert [v.name for v in kept] == ["kv", "both"]

    def test_version_mismatch_disables_the_llm_path(self, monkeypatch, grid, ctx, checkpoint):
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_LLM_VERSION, "0.5.11")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("3,1,2"))
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert kept == grid
        assert info["reason"] == "version_mismatch"

    def test_unreachable_ranker_keeps_everything(self, monkeypatch, grid, ctx, checkpoint):
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply(""))
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert kept == grid
        assert info["reason"] == "llm_unavailable"

    def test_never_empties_the_grid(self, monkeypatch, grid, ctx, checkpoint):
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_KEEP_FRACTION, "0.0")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("2,3,1"))
        kept, _ = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert len(kept) == 1

    def test_unreadable_checkpoint_refuses_rather_than_ranking(
            self, monkeypatch, grid, ctx):
        """A hollow identity still yields a confident order, so refuse instead."""
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        called = []
        monkeypatch.setattr(df, "_rank_with_llm",
                            lambda *a, **k: called.append(1) or [0, 1, 2])
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": "/nope"})
        assert kept == grid
        assert info["reason"] == "identity_unavailable"
        assert not called, "the ranker must not be asked without an identity"

    def test_token_budget_scales_with_candidate_count(self, monkeypatch):
        """A fixed budget truncates long orderings, and truncation is invisible."""
        seen = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"1"}}]}'

        def fake_urlopen(req, timeout=None):
            seen.update(json.loads(req.data))
            return _Resp()

        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        df._rank_with_llm([_variant("v%d" % i, "--flag-%d" % i) for i in range(24)], "x")
        assert seen["max_tokens"] >= 24 * 4, "24 candidates need room for 24 numbers"
        assert seen["chat_template_kwargs"] == {"enable_thinking": False}

    def test_delta_text_matches_the_training_rendering(self):
        """A paraphrase is a distribution shift; the ranker saw this exact form."""
        v = _variant("x", "--kv-cache-dtype fp8_e4m3", {"SGLANG_USE_AITER": "1"})
        assert df._delta_text(v) == \
            "--kv-cache-dtype fp8_e4m3 | env: SGLANG_USE_AITER=1"
        assert df._delta_text(_variant("y", "--page-size 16")) == "--page-size 16"


def test_delta_features_capture_flags_and_envs():
    feats = df._delta_features("--kv-cache-dtype fp8_e4m3 --page-size 16",
                               {"SGLANG_USE_AITER": "1"})
    assert feats["flag__kv_cache_dtype"] == 1
    assert feats["flag__page_size"] == 1
    assert feats["env_SGLANG_USE_AITER"] == 1
    assert feats["shape_n_flags"] == 2
    assert feats["shape_max_num"] == 16.0
