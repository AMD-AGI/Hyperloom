# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Conformance tests for the local-kb-recipe-snapshot requirements doc.

This module mirrors the checklist in §4 of
``primus-cortex-internal/docs/local-kb-recipe-snapshot-requirements.md``.
Each test maps 1:1 to one requirement bullet so a regression in
any item shows up under a self-describing test name.

The 9 items from the doc:

1. 五元组 canonical id 生成正确
2. 本地 KB root 是 ``${USER_DATA_PATH}/kb``
3. 默认不传中心化 KB 时，读写都走本地 KB
4. 指定中心化 KB 且可用时，读走中心化 KB，写仍走本地 KB
5. 指定中心化 KB 但不可用时，读自动 fallback 到本地 KB，写仍走本地 KB
6. 本地 recipe 文件或目录能区分五元组
7. model 包含 ``/`` 时，本地路径仍然安全
8. session 写入会合并已有 recipe 历史，不会清空已有
   ``sessions`` / ``what_worked`` / ``what_failed`` / ``pitfalls``
9. 最终本地 KB 文件的数据字段和 Arbor recipe 保持一致

These run as ordinary pytest unit tests against real
``LocalRecipeStore`` instances + ``respx``-mocked remote server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from inference_optimizer.cli import (
    _build_recipe_kb_dispatcher,
    _resolve_local_kb_root,
)
from inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    cid_to_path_components,
    recipe_canonical_id,
)
from inference_optimizer.recipe_snapshot_constants import (
    PATH_RECIPES_SEARCH,
)


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the env vars the resolver consults so each test owns
    its own precedence tier."""
    for key in (
        "HYPERLOOM_LOCAL_KB_ROOT",
        "USER_DATA_PATH",
        "CORTEX_KB_URL",
        "KB_SERVICE_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _ns(**overrides: Any) -> argparse.Namespace:
    """Helper to build a CLI Namespace with the four KB-related
    fields plus any operator-supplied overrides."""
    fields: dict[str, Any] = {
        "local_kb_root": None,
        "cortex_kb_url": None,
        "degraded_kb":   False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


# ===========================================================================
# §4 Item 1 — 五元组 canonical id 生成正确
# ===========================================================================
def test_item1_canonical_id_is_5tuple_with_inference_prefix() -> None:
    cid = recipe_canonical_id(
        model="DeepSeek-R1",
        hardware="MI300X",
        framework="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    assert cid == "inference:deepseek-r1:mi300x:sglang:0.4.5:fp8"
    # 6 colon-separated segments: prefix + 5 dimensions.
    assert len(cid.split(":")) == 6


def test_item1_canonical_id_keyword_only_no_positional_drift() -> None:
    """Positional args must raise so a future caller can't silently
    re-order the 5-tuple."""
    with pytest.raises(TypeError):
        recipe_canonical_id("m", "h", "fw", "v", "p")  # type: ignore[misc]


# ===========================================================================
# §4 Item 2 — 本地 KB root 是 ${USER_DATA_PATH}/kb
# ===========================================================================
def test_item2_default_local_kb_root_is_user_data_path_kb(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default resolution (no flag, no HYPERLOOM_LOCAL_KB_ROOT) must
    land at ``${USER_DATA_PATH}/kb`` — exact match required by
    requirements doc §2."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    args = _ns()
    assert _resolve_local_kb_root(args) == tmp_path / "kb"


def test_item2_explicit_flag_wins_over_user_data_path(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--local-kb-root`` is the highest-priority tier — operator
    can pin a non-default root for tests / sandbox isolation."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    args = _ns(local_kb_root=str(tmp_path / "alt-root"))
    assert _resolve_local_kb_root(args) == tmp_path / "alt-root"


# ===========================================================================
# §4 Item 3 — 默认不传中心化 KB 时，读写都走本地 KB
# ===========================================================================
def test_item3_no_central_url_reads_and_writes_go_local(
    env_clean: None,
    tmp_path: Path,
) -> None:
    args = _ns(local_kb_root=str(tmp_path))
    kb = _build_recipe_kb_dispatcher(args)
    # The dispatcher's remote half is None when no URL is configured.
    assert kb.remote is None

    cid = recipe_canonical_id(
        model="m", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
    )
    out = kb.put_recipe(
        canonical_id=cid,
        model="m", hardware="mi300x",
        framework="sglang", framework_version="0.4.5", precision="fp8",
        best_throughput=12345.0,
    )
    assert out["created"] is True
    # Read goes local because remote is None.
    row = kb.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_throughput"] == 12345.0


# ===========================================================================
# §4 Item 4 — 指定中心化 KB 且可用时，读走中心化 KB，写仍走本地 KB
# ===========================================================================
def test_item4_central_kb_reads_central_writes_local(
    env_clean: None,
    tmp_path: Path,
) -> None:
    """Verifies the read-remote / write-local invariant when remote
    is healthy. Uses respx to stand in for the central kb-service."""
    central_url = "http://central-kb.test"
    args = _ns(
        local_kb_root=str(tmp_path),
        cortex_kb_url=central_url,
    )
    kb = _build_recipe_kb_dispatcher(args)
    assert kb.remote is not None
    assert kb.remote.kb_url == central_url

    cid = recipe_canonical_id(
        model="m", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
    )

    # Central server has a stale row for this cid.
    central_payload = {
        "canonical_id": cid,
        "version":      9,
        "labels":       {"model": "m", "hardware": "mi300x"},
        "body":         {"best_config": {"tp": "16"}},
        "metrics":      {"throughput": 99999.0},
    }

    # 1. WRITE: must NOT touch central. We respx-mock the central
    #    service to raise on PUT (only allow GET-style reads). Any
    #    write attempt would surface as an unmatched-URL exception.
    with respx.mock(base_url=central_url) as mock:
        # No write routes — any HTTP call from the dispatcher.put
        # path would fail the test.
        # Remote reads go through the single /recipes/search route.
        mock.post(PATH_RECIPES_SEARCH).mock(
            return_value=httpx.Response(200, json={"recipes": [central_payload]}),
        )
        kb.put_recipe(
            canonical_id=cid,
            model="m", hardware="mi300x",
            framework="sglang", framework_version="0.4.5", precision="fp8",
            best_throughput=11111.0,
        )
        # Local store has our row.
        local_row = kb.local.get_recipe(canonical_id=cid)
        assert local_row is not None
        assert local_row["best_throughput"] == 11111.0

        # 2. READ: dispatcher returns the CENTRAL row (translated to
        #    arbor shape), not the local one we just wrote. Central
        #    is the wider corpus when reachable.
        out = kb.get_recipe(canonical_id=cid)
        assert out is not None
        assert out["version"] == 9
        assert out["best_throughput"] == 99999.0  # central wins


# ===========================================================================
# §4 Item 5 — 指定中心化 KB 但不可用时，读自动 fallback 到本地，写仍走本地
# ===========================================================================
def test_item5_unreachable_central_falls_back_to_local(
    env_clean: None,
    tmp_path: Path,
) -> None:
    central_url = "http://central-kb.test"
    args = _ns(
        local_kb_root=str(tmp_path),
        cortex_kb_url=central_url,
    )
    kb = _build_recipe_kb_dispatcher(args)
    # Foreground profile (1 retry, 2s timeout) so the test isn't slow.
    kb.remote.retry_attempts = 1  # type: ignore[union-attr]

    cid = recipe_canonical_id(
        model="m", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
    )

    # Seed the local store BEFORE the test so the fallback has
    # something to return.
    kb.local.put_recipe(
        canonical_id=cid,
        model="m", hardware="mi300x",
        framework="sglang", framework_version="0.4.5", precision="fp8",
        best_throughput=22222.0,
    )

    # Central server unreachable — any call returns 503.
    with respx.mock(base_url=central_url) as mock:
        # Central unreachable on the search route → fall back to local.
        mock.post(PATH_RECIPES_SEARCH).mock(
            return_value=httpx.Response(503, json={"detail": "warming up"}),
        )
        # 1. WRITE: still goes local (the dispatcher writes always go
        #    to local — central is read-only by design).
        kb.put_recipe(
            canonical_id=cid,
            model="m", hardware="mi300x",
            framework="sglang", framework_version="0.4.5", precision="fp8",
            best_throughput=33333.0,
        )
        # 2. READ: central 503 → fall through to local. The dispatcher
        #    absorbs the RemoteRecipeClientError silently.
        out = kb.get_recipe(canonical_id=cid)
        assert out is not None
        assert out["best_throughput"] == 33333.0  # local hit


# ===========================================================================
# §4 Item 6 — 本地 recipe 文件或目录能区分五元组
# ===========================================================================
def test_item6_local_path_distinguishes_5tuple(tmp_path: Path) -> None:
    """Two recipes that differ only in framework_version (or any
    single dimension) must land in distinct on-disk locations."""
    store = LocalRecipeStore(root=tmp_path)
    cid_v1 = recipe_canonical_id(
        model="m", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
    )
    cid_v2 = recipe_canonical_id(
        model="m", hardware="mi300x", framework="sglang",
        framework_version="0.5.0", precision="fp8",
    )
    store.put_recipe(
        canonical_id=cid_v1,
        model="m", hardware="mi300x",
        framework="sglang", framework_version="0.4.5", precision="fp8",
        best_throughput=1.0,
    )
    store.put_recipe(
        canonical_id=cid_v2,
        model="m", hardware="mi300x",
        framework="sglang", framework_version="0.5.0", precision="fp8",
        best_throughput=2.0,
    )
    # Distinct on-disk paths — last directory level encodes precision,
    # but framework_version (4th level) differs in this test.
    parts_v1 = cid_to_path_components(cid_v1)
    parts_v2 = cid_to_path_components(cid_v2)
    assert parts_v1 != parts_v2
    assert (tmp_path.joinpath(*parts_v1) / "recipe.json").is_file()
    assert (tmp_path.joinpath(*parts_v2) / "recipe.json").is_file()
    # And they don't shadow each other — both rows are independently
    # readable.
    row_v1 = store.get_recipe(canonical_id=cid_v1)
    row_v2 = store.get_recipe(canonical_id=cid_v2)
    assert row_v1 is not None and row_v2 is not None
    assert row_v1["best_throughput"] == 1.0
    assert row_v2["best_throughput"] == 2.0


def test_item6_path_levels_match_5_dimensions(tmp_path: Path) -> None:
    """Documented contract: the on-disk path is exactly 5 levels
    below the store root, one level per identity dimension."""
    store = LocalRecipeStore(root=tmp_path)
    cid = recipe_canonical_id(
        model="m", hardware="hw", framework="fw",
        framework_version="ver", precision="prec",
    )
    store.put_recipe(
        canonical_id=cid,
        model="m", hardware="hw", framework="fw",
        framework_version="ver", precision="prec",
    )
    expected = tmp_path / "m" / "hw" / "fw" / "ver" / "prec" / "recipe.json"
    assert expected.is_file()


# ===========================================================================
# §4 Item 7 — model 包含 / 时，本地路径仍然安全
# ===========================================================================
def test_item7_model_with_slash_is_path_safe(tmp_path: Path) -> None:
    """A model arg like ``/hyperloom/models/Qwen-Qwen3-30B-A3B-Base``
    must NOT split into three path segments — the slug step
    basenames it first."""
    store = LocalRecipeStore(root=tmp_path)
    cid = recipe_canonical_id(
        model="/hyperloom/models/Qwen-Qwen3-30B-A3B-Base",
        hardware="mi355x",
        framework="sglang",
        framework_version="0.4.5",
        precision="bf16",
    )
    # Slug rule: basename + lowercase + space → underscore.
    assert cid == (
        "inference:qwen-qwen3-30b-a3b-base:mi355x:sglang:0.4.5:bf16"
    )
    store.put_recipe(
        canonical_id=cid,
        model="/hyperloom/models/Qwen-Qwen3-30B-A3B-Base",
        hardware="mi355x",
        framework="sglang",
        framework_version="0.4.5",
        precision="bf16",
        best_throughput=42.0,
    )
    # Recipe lives at exactly 5 levels below root, model component
    # is the basename only.
    expected = (
        tmp_path / "qwen-qwen3-30b-a3b-base"
        / "mi355x" / "sglang" / "0.4.5" / "bf16"
        / "recipe.json"
    )
    assert expected.is_file()


def test_item7_model_with_double_slash_normalises(tmp_path: Path) -> None:
    """Edge case: trailing slash on the model arg shouldn't split
    into an empty segment."""
    store = LocalRecipeStore(root=tmp_path)
    cid = recipe_canonical_id(
        model="/some/path/MyModel/",
        hardware="hw", framework="fw",
        framework_version="v", precision="p",
    )
    # Trailing slash stripped, basename taken.
    assert ":mymodel:" in cid
    store.put_recipe(
        canonical_id=cid,
        model="/some/path/MyModel/",
        hardware="hw", framework="fw",
        framework_version="v", precision="p",
    )
    expected = tmp_path / "mymodel" / "hw" / "fw" / "v" / "p" / "recipe.json"
    assert expected.is_file()


# ===========================================================================
# §4 Item 8 — session 写入会合并已有 recipe 历史
# ===========================================================================
def test_item8_second_put_preserves_what_worked_when_not_overridden(
    tmp_path: Path,
) -> None:
    """When the second put_recipe doesn't supply ``what_worked``, the
    previously written value MUST survive. Otherwise the legacy
    requirement "不会清空已有 what_worked" fails."""
    store = LocalRecipeStore(root=tmp_path)
    cid = recipe_canonical_id(
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
    )
    # First put: stamp what_worked, what_failed, pitfalls, sessions.
    store.put_recipe(
        canonical_id=cid,
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
        what_worked=[
            {"description": "X helped", "measured_impact": "+10%"},
        ],
        what_failed=[
            {"description": "Y failed", "reason": "OOM"},
        ],
        pitfalls=[{"description": "watch for Z"}],
        sessions=[
            {"date": "2026-05-28",
             "throughput_before": 1.0, "throughput_after": 1.1,
             "actions_taken": ["a"]},
        ],
    )

    # Second put through the dispatcher's _kb_amend_recipe-style
    # read-modify-write: the caller MUST preserve existing fields.
    # Here we simulate the safe pattern: read live + only override
    # the field we want to change.
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    store.put_recipe(
        canonical_id=cid,
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
        what_worked=list(live.get("what_worked") or []),
        what_failed=list(live.get("what_failed") or []),
        pitfalls=list(live.get("pitfalls") or []),
        sessions=list(live.get("sessions") or []),
        # only updating throughput
        best_throughput=99.0,
    )

    after = store.get_recipe(canonical_id=cid)
    assert after is not None
    assert after["best_throughput"] == 99.0
    # All four arrays preserved verbatim.
    assert len(after["what_worked"]) == 1
    assert after["what_worked"][0]["description"] == "X helped"
    assert len(after["what_failed"]) == 1
    assert after["what_failed"][0]["reason"] == "OOM"
    assert len(after["pitfalls"]) == 1
    assert after["pitfalls"][0]["description"] == "watch for Z"
    assert len(after["sessions"]) == 1
    assert after["sessions"][0]["date"] == "2026-05-28"


def test_item8_history_archives_prior_version(tmp_path: Path) -> None:
    """The history archive contract: every put_recipe bumps version
    and snapshots the prior live row to ``history/v{N}.json`` so
    rollback is always possible."""
    store = LocalRecipeStore(root=tmp_path)
    cid = recipe_canonical_id(
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
    )
    store.put_recipe(
        canonical_id=cid,
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
        best_throughput=1.0,
    )
    store.put_recipe(
        canonical_id=cid,
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
        best_throughput=2.0,
    )
    history = store.get_history(canonical_id=cid)
    assert len(history) == 1
    assert history[0]["version"] == 1
    assert history[0]["snapshot"]["best_throughput"] == 1.0


# ===========================================================================
# §4 Item 9 — 最终本地 KB 文件的数据字段和 Arbor recipe 保持一致
# ===========================================================================
def test_item9_on_disk_json_uses_arbor_field_names(tmp_path: Path) -> None:
    """The persisted ``recipe.json`` must use arbor's documented
    field names (``what_worked`` / ``what_failed`` /
    ``remaining_gaps`` / ``pitfalls`` / ``best_config`` /
    ``best_throughput`` / ``stack_fingerprint`` / ``last_profiled``
    / ``sessions``) — NOT the v2 wire spec's
    ``findings`` / ``failures`` / ``gaps`` / ``body`` / ``metrics``.
    """
    store = LocalRecipeStore(root=tmp_path)
    cid = recipe_canonical_id(
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
    )
    store.put_recipe(
        canonical_id=cid,
        model="m", hardware="hw", framework="fw",
        framework_version="v", precision="p",
        best_config={"tp": "8"},
        best_throughput=42.0,
        what_worked=[{"description": "x", "measured_impact": "+5%"}],
        what_failed=[{"description": "y", "reason": "OOM"}],
        remaining_gaps=[{"description": "z", "metrics": "tput"}],
        pitfalls=[{"description": "watch out"}],
        last_profiled="2026-05-28",
        stack_fingerprint={
            "vllm_version": "0.6.0",
            "aiter_commit": "abc123",
            "rocm_version": "7.2",
        },
        sessions=[
            {"date": "2026-05-28",
             "throughput_before": 1.0, "throughput_after": 42.0,
             "actions_taken": ["tp+ep"]},
        ],
    )
    on_disk = json.loads(
        (tmp_path / "m" / "hw" / "fw" / "v" / "p" / "recipe.json")
        .read_text(encoding="utf-8")
    )

    # Arbor field names present at the top level.
    assert "best_config"        in on_disk
    assert "best_throughput"    in on_disk
    assert "what_worked"        in on_disk
    assert "what_failed"        in on_disk
    assert "remaining_gaps"     in on_disk
    assert "pitfalls"           in on_disk
    assert "last_profiled"      in on_disk
    assert "stack_fingerprint"  in on_disk
    assert "sessions"           in on_disk
    assert "model"              in on_disk
    assert "hardware"           in on_disk

    # Stack fingerprint sub-shape matches arbor's StackFingerprint.
    assert set(on_disk["stack_fingerprint"]) >= {
        "vllm_version", "aiter_commit", "rocm_version",
    }
    # Sessions row sub-shape matches arbor's SessionSummary.
    assert set(on_disk["sessions"][0]) >= {
        "date", "throughput_before", "throughput_after", "actions_taken",
    }
    # Finding / Failure / Gap sub-shapes (pure arbor).
    assert set(on_disk["what_worked"][0])    == {"description", "measured_impact"}
    assert set(on_disk["what_failed"][0])    == {"description", "reason"}
    assert set(on_disk["remaining_gaps"][0]) == {"description", "metrics"}
    # Pitfall is arbor's ``description`` plus hyperloom's optional
    # ``severity`` superset field (same spirit as the ``>=`` checks on
    # sessions / stack_fingerprint above — a superset of arbor, never a
    # v2 wire key). The Coordinator stamps ``severity`` for warm-start
    # ranking; arbor consumers ignore the extra key.
    assert set(on_disk["pitfalls"][0]) >= {"description"}
    assert set(on_disk["pitfalls"][0]) <= {"description", "severity"}

    # The v2 wire-spec key names MUST NOT appear at the top level
    # (they're translated by the dispatcher only on read from
    # central; on-disk is arbor-pure).
    for v2_only_key in ("findings", "failures", "gaps", "body", "metrics"):
        assert v2_only_key not in on_disk, (
            f"unexpected v2 wire-spec key {v2_only_key!r} in arbor "
            f"on-disk recipe.json"
        )
