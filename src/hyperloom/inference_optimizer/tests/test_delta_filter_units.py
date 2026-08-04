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


def _fake_artifact(tmp_path, version="0.5.12", drop_flag="__schedule_conservativeness",
                   dest=None, hardware=None):
    """A minimal artifact whose model is a real booster over two features.

    ``hardware`` adds the categorical column and its vocabulary, which is what
    the hardware guard reads; leaving it None reproduces an artifact from before
    the guard existed.
    """
    lgb = pytest.importorskip("lightgbm")
    np = pytest.importorskip("numpy")

    cols = ["flag__kv_cache_dtype", "flag%s" % drop_flag]
    # Label 1 whenever the kv flag is present, 0 whenever the drop flag is, so
    # the fitted model separates the two candidates the tests care about.
    rows, labels = [], []
    for _ in range(80):
        rows += [[1, 0], [0, 1]]
        labels += [1, 0]
    if hardware:
        cols = cols + ["hardware"]
        rows = [r + [0] for r in rows]
    ds = lgb.Dataset(np.array(rows, dtype=np.float32), label=np.array(labels),
                     feature_name=cols)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 2,
                         "min_data_in_leaf": 1, "min_data_in_bin": 1},
                        ds, num_boost_round=20)
    d = dest if dest is not None else tmp_path / "ranker"
    d.mkdir(parents=True)
    booster.save_model(str(d / "ranker.txt"))
    (d / "schema.json").write_text(json.dumps({
        "columns": cols,
        "categorical": ["hardware"] if hardware else [],
        "vocab": {"hardware": {hardware: 0}} if hardware else {},
    }))
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
        return lambda variants, identity_text, lora="": (
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


class TestUnionBackend:
    """Both backends configured: threshold, plus the ranker's top picks.

    The two models disagree in useful ways -- the GBDT thresholds an absolute
    score, the ranker orders better -- so the union was measured to lose the
    best candidate on 1.1% of identities where the threshold alone loses 1.1
    times more, for 0.1pp of extra candidates kept.
    """

    @staticmethod
    def _reply(order: str):
        """Patch the ranker call to return a fixed ordering."""
        return lambda variants, identity_text, lora="": [int(t) - 1 for t in order.split(",")]

    def test_ranker_top_pick_survives_a_failing_score(
            self, monkeypatch, tmp_path, grid, ctx, checkpoint):
        """The whole point: a candidate the threshold drops is rescued by rank 1."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        # "cons" carries the flag trained as harmful, so it scores below 0.5 and
        # the threshold alone drops it -- but the ranker puts it first.
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("2,1,3"))
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert info["reason"] == "ok_union"
        assert info["backend"] == "union"
        assert "cons" in {v.name for v in kept}
        assert info["backstop_added"] == ["cons"]

    def test_backstop_zero_reduces_to_the_threshold(
            self, monkeypatch, tmp_path, grid, ctx, checkpoint):
        """Operators who want the old behaviour can turn the rescue off."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        monkeypatch.setenv(df.ENV_BACKSTOP, "0")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("2,1,3"))
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert "cons" not in {v.name for v in kept}
        assert info["backstop_added"] == []

    def test_survivors_keep_their_original_order(
            self, monkeypatch, tmp_path, grid, ctx, checkpoint):
        """The union selects; it does not reorder what the caller then measures."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("2,1,3"))
        kept, _ = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        names = [v.name for v in kept]
        assert names == sorted(names, key=[v.name for v in grid].index)

    def test_falls_back_to_the_gbdt_when_the_ranker_is_unreachable(
            self, monkeypatch, tmp_path, grid, ctx, checkpoint):
        """Losing one backend degrades to the other, never to no filtering."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        monkeypatch.setattr(df, "_rank_with_llm", lambda variants, identity, lora="": None)
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert info["reason"] == "ok"
        assert info["backend"] == "gbdt"
        assert info["llm_reason"] == "llm_unavailable"
        assert "cons" not in {v.name for v in kept}

    def test_falls_back_to_the_ranker_when_the_artifact_is_missing(
            self, monkeypatch, tmp_path, grid, ctx, checkpoint):
        monkeypatch.setenv(df.ENV_DIR, str(tmp_path / "absent"))
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_KEEP_FRACTION, "0.3")
        monkeypatch.setattr(df, "_rank_with_llm", self._reply("2,1,3"))
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert info["reason"] == "ok_llm"
        assert info["gbdt_reason"] == "artifact_unreadable"
        assert [v.name for v in kept] == ["cons"]

    def test_both_unavailable_keeps_everything(
            self, monkeypatch, tmp_path, grid, ctx, checkpoint):
        monkeypatch.setenv(df.ENV_DIR, str(tmp_path / "absent"))
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setattr(df, "_rank_with_llm", lambda variants, identity, lora="": None)
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert kept == grid
        assert info["applied"] is False

    def test_bad_backstop_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(df.ENV_BACKSTOP, "not-a-number")
        assert df._backstop() == df.DEFAULT_BACKSTOP


class TestContextFeatures:
    """Features describing a candidate's standing within its own grid.

    An artifact trained with them scores wrong rather than failing if the caller
    computes them differently, so the arithmetic is pinned here against the
    training-side definition in
    claw-dev/docs-zh/post-training/recipe-rank/try_context_features.py.
    """

    def test_relative_position_spans_the_field(self):
        rows = [{"shape_n_flags": 1}, {"shape_n_flags": 3}, {"shape_n_flags": 5}]
        df._add_context_features(rows)
        assert [r["ctx_shape_n_flags_rel"] for r in rows] == [0.0, 0.5, 1.0]
        assert [r["ctx_shape_n_flags_dev"] for r in rows] == [-2.0, 0.0, 2.0]
        assert all(r["ctx_n_candidates"] == 3 for r in rows)

    def test_identical_candidates_do_not_divide_by_zero(self):
        rows = [{"shape_n_flags": 2}, {"shape_n_flags": 2}]
        df._add_context_features(rows)
        assert all(r["ctx_shape_n_flags_rel"] == 0.0 for r in rows)

    def test_rarity_falls_as_a_flag_becomes_common(self):
        """A flag every candidate carries distinguishes none of them."""
        shared = [{"flag__kv_cache_dtype": 1} for _ in range(4)]
        df._add_context_features(shared)
        assert shared[0]["ctx_flag_rarity"] == 0.0

        mixed = [{"flag__kv_cache_dtype": 1}, {"flag__page_size": 1},
                 {"flag__page_size": 1}, {"flag__page_size": 1}]
        df._add_context_features(mixed)
        assert mixed[0]["ctx_flag_rarity"] > mixed[1]["ctx_flag_rarity"]

    def test_flagless_candidate_gets_zero_rarity(self):
        rows = [{"shape_n_flags": 0}, {"flag__page_size": 1}]
        df._add_context_features(rows)
        assert rows[0]["ctx_flag_rarity"] == 0.0

    def test_empty_grid_is_a_no_op(self):
        rows: list = []
        df._add_context_features(rows)
        assert rows == []

    def test_artifacts_without_context_columns_skip_the_pass(
            self, monkeypatch, tmp_path, grid, ctx):
        """The shipped model has no ctx_ columns and must be unaffected."""
        called = {"n": 0}
        real = df._add_context_features
        monkeypatch.setattr(df, "_add_context_features",
                            lambda rows: (called.__setitem__("n", called["n"] + 1),
                                          real(rows))[1])
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        df.filter_variants(grid, **ctx)
        assert called["n"] == 0


class TestRelativeCut:
    """Cutting at a share of the identity's own top score rather than a constant.

    The fixed cut is calibrated across the corpus but not within an identity, so
    it drops the best candidate of a low-scoring identity and keeps junk from a
    high-scoring one. On both held-out splits the relative cut dominates it:
    at max*0.08 it saves 20pp more candidates for the same 1.5% loss rate.
    """

    def test_scales_with_the_identity_top_score(self, monkeypatch):
        monkeypatch.setenv(df.ENV_RELATIVE, "0.08")
        cut, label = df._cut_for([0.9, 0.5, 0.01])
        assert cut == pytest.approx(0.072)
        assert label == "max*0.080"

    def test_top_scorer_always_clears_the_cut(self, monkeypatch, tmp_path, grid, ctx):
        """A relative cut cannot empty the grid, however low the scores are."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_RELATIVE, "0.99")
        kept, info = df.filter_variants(grid, **ctx)
        assert kept, "the highest-scoring candidate must survive"
        assert info["reason"] == "ok", "the empty-grid fallback should not fire"

    def test_unset_keeps_the_fixed_cut(self, monkeypatch):
        monkeypatch.delenv(df.ENV_RELATIVE, raising=False)
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.25")
        cut, label = df._cut_for([0.9, 0.5, 0.01])
        assert cut == 0.25
        assert label == "0.250"

    @pytest.mark.parametrize("bad", ["not-a-number", "0", "1", "1.5", "-0.2"])
    def test_out_of_range_falls_back_to_the_fixed_cut(self, monkeypatch, bad):
        """A factor at or past the ends is not a cut anyone meant to ask for."""
        monkeypatch.setenv(df.ENV_RELATIVE, bad)
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.030")
        assert df._cut_for([0.9, 0.5])[0] == 0.030

    def test_all_zero_scores_fall_back_to_the_fixed_cut(self, monkeypatch):
        """Scaling by a zero top score would keep everything; the constant won't."""
        monkeypatch.setenv(df.ENV_RELATIVE, "0.08")
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.030")
        assert df._cut_for([0.0, 0.0])[0] == 0.030


class TestHardwareAxis:
    """The GPU guard, and the runner-label normalisation it depends on.

    Version got a guard because a stale model was measured below chance. Hardware
    is the same shape of mistake with none of the evidence yet: an unseen GPU
    encodes to -1, which no tree ever split on, so the scores come out confident
    and meaningless.
    """

    @pytest.mark.parametrize("gpu,label", [
        ("mi325x", "mi300x"), ("mi308x", "mi300x"), ("MI325X", "mi300x"),
        ("mi300x", "mi300x"), ("mi355x", "mi355x"), ("", ""),
    ])
    def test_collapses_to_the_corpus_runner_label(self, gpu, label):
        """The corpus spells gfx942 cards mi300x, so mi325x must not reach it raw."""
        assert df._runner_label(gpu) == label

    def test_mi325x_session_uses_the_mi300x_artifact(self, monkeypatch, tmp_path, grid, ctx):
        """Without normalisation this session would score against an OOV sentinel."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path, hardware="mi300x")))
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        kept, info = df.filter_variants(grid, **{**ctx, "hardware": "mi325x"})
        assert info["reason"] == "ok"
        assert info["cell"] == "0.5.12/mi300x"
        assert "cons" not in {v.name for v in kept}

    def test_uncovered_gpu_disables_filtering(self, monkeypatch, tmp_path, grid, ctx):
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path, hardware="mi300x")))
        kept, info = df.filter_variants(grid, **{**ctx, "hardware": "mi355x"})
        assert kept == grid
        assert info["reason"] == "hardware_mismatch"

    def test_artifact_without_the_column_is_not_gated(self, monkeypatch, tmp_path, grid, ctx):
        """An artifact predating the vocabulary gives no basis to refuse."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        _, info = df.filter_variants(grid, **{**ctx, "hardware": "mi355x"})
        assert info["reason"] == "ok"

    def test_unknown_running_gpu_is_not_gated(self, monkeypatch, tmp_path, grid, ctx):
        """Mirrors the version guard: only compare when both sides are known."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path, hardware="mi300x")))
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        _, info = df.filter_variants(grid, **{**ctx, "hardware": ""})
        assert info["reason"] == "ok"


class TestArtifactRegistry:
    """Selecting one artifact per (version, GPU) cell in a mixed deployment.

    Preferences do not transfer across an sglang bump -- kappa 0.15 over 1,467
    paired models -- so a deployment running several versions needs one artifact
    each rather than one model that saw them all.
    """

    @staticmethod
    def _registry(tmp_path):
        root = tmp_path / "rankers"
        _fake_artifact(tmp_path, version="0.5.12", hardware="mi300x",
                       dest=root / "sglang-0.5.12-mi300x")
        _fake_artifact(tmp_path, version="0.5.11", hardware="mi300x",
                       dest=root / "sglang-0.5.11-mi300x")
        _fake_artifact(tmp_path, version="0.5.12", hardware="mi355x",
                       dest=root / "sglang-0.5.12-mi355x")
        return root

    @pytest.mark.parametrize("version,hardware,expected", [
        ("0.5.12", "mi300x", "sglang-0.5.12-mi300x"),
        ("0.5.11", "mi300x", "sglang-0.5.11-mi300x"),
        ("0.5.12", "mi355x", "sglang-0.5.12-mi355x"),
        # mi325x is a gfx942 card, so it resolves to the mi300x cell.
        ("0.5.12", "mi325x", "sglang-0.5.12-mi300x"),
    ])
    def test_each_cell_picks_its_own_artifact(self, monkeypatch, tmp_path, grid, ctx,
                                             version, hardware, expected):
        monkeypatch.setenv(df.ENV_DIR, str(self._registry(tmp_path)))
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        _, info = df.filter_variants(
            grid, **{**ctx, "framework_version": version, "hardware": hardware})
        assert info["gbdt_artifact"] == expected
        assert info["reason"] == "ok"

    def test_uncovered_cell_keeps_everything(self, monkeypatch, tmp_path, grid, ctx):
        """A version with no artifact yet must not borrow a neighbour's."""
        monkeypatch.setenv(df.ENV_DIR, str(self._registry(tmp_path)))
        kept, info = df.filter_variants(grid, **{**ctx, "framework_version": "0.5.13"})
        assert kept == grid
        assert info["reason"] == "registry_miss"

    def test_two_matches_refuse_rather_than_choose(self, monkeypatch, tmp_path, grid, ctx):
        """Picking one silently is the bug this whole guard exists to prevent."""
        root = tmp_path / "rankers"
        _fake_artifact(tmp_path, version="0.5.12", hardware="mi300x", dest=root / "a")
        _fake_artifact(tmp_path, version="0.5.12", hardware="mi300x", dest=root / "b")
        monkeypatch.setenv(df.ENV_DIR, str(root))
        kept, info = df.filter_variants(grid, **ctx)
        assert kept == grid
        assert info["reason"] == "registry_ambiguous"

    def test_single_artifact_root_is_unchanged(self, monkeypatch, tmp_path, grid, ctx):
        """An existing single-cell deployment must behave exactly as before."""
        monkeypatch.setenv(df.ENV_DIR, str(_fake_artifact(tmp_path)))
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        kept, info = df.filter_variants(grid, **ctx)
        assert info["reason"] == "ok"
        assert info["gbdt_artifact"] == "ranker"
        assert "cons" not in {v.name for v in kept}

    def test_malformed_entry_is_skipped_not_fatal(self, monkeypatch, tmp_path, grid, ctx):
        root = self._registry(tmp_path)
        broken = root / "half-written"
        broken.mkdir()
        (broken / "manifest.json").write_text("{not json")
        monkeypatch.setenv(df.ENV_DIR, str(root))
        monkeypatch.setenv(df.ENV_THRESHOLD, "0.5")
        _, info = df.filter_variants(grid, **ctx)
        assert info["gbdt_artifact"] == "sglang-0.5.12-mi300x"


class TestServedAdapterRegistry:
    """One endpoint holds every cell's adapter; the cell picks which to request."""

    def test_maps_a_cell_to_its_adapter(self, monkeypatch):
        monkeypatch.setenv(df.ENV_LLM_ADAPTERS,
                           "0.5.12/mi300x=ranker-0512, 0.5.11/mi300x=ranker-0511")
        assert df._llm_adapter_for("0.5.12", "mi300x") == ("ranker-0512", "")
        assert df._llm_adapter_for("0.5.11", "mi300x") == ("ranker-0511", "")

    def test_version_only_key_covers_every_gpu(self, monkeypatch):
        monkeypatch.setenv(df.ENV_LLM_ADAPTERS, "0.5.12=ranker-0512")
        assert df._llm_adapter_for("0.5.12", "mi355x") == ("ranker-0512", "")

    def test_uncovered_cell_reports_a_miss(self, monkeypatch):
        monkeypatch.setenv(df.ENV_LLM_ADAPTERS, "0.5.12/mi300x=ranker-0512")
        assert df._llm_adapter_for("0.5.13", "mi300x") == ("", "adapter_cell_miss")

    def test_single_cell_form_still_works(self, monkeypatch):
        """The pre-registry pair stays valid for a one-version deployment."""
        monkeypatch.delenv(df.ENV_LLM_ADAPTERS, raising=False)
        monkeypatch.setenv(df.ENV_LLM_LORA, "ranker")
        monkeypatch.setenv(df.ENV_LLM_VERSION, "0.5.12")
        assert df._llm_adapter_for("0.5.12", "mi300x") == ("ranker", "")
        assert df._llm_adapter_for("0.5.11", "mi300x") == ("", "version_mismatch")

    def test_resolved_adapter_reaches_the_request(self, monkeypatch, grid, ctx, checkpoint):
        """The adapter has to land in lora_path, or every cell serves the base model."""
        seen: dict = {}

        class _Resp:
            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "1, 2, 3"}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            seen.update(json.loads(req.data))
            return _Resp()

        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_LLM_ADAPTERS, "0.5.12/mi300x=ranker-0512")
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert seen["lora_path"] == "ranker-0512"

    def test_cell_miss_skips_ranking_entirely(self, monkeypatch, grid, ctx, checkpoint):
        called = []
        monkeypatch.setenv(df.ENV_LLM_URL, "http://ranker.invalid")
        monkeypatch.setenv(df.ENV_LLM_ADAPTERS, "0.5.11/mi300x=ranker-0511")
        monkeypatch.setattr(df, "_rank_with_llm",
                            lambda *a, **k: called.append(1) or [0, 1, 2])
        kept, info = df.filter_variants(grid, **{**ctx, "model_path": checkpoint})
        assert kept == grid
        assert info["reason"] == "adapter_cell_miss"
        assert not called, "an uncovered cell must not reach the endpoint"
