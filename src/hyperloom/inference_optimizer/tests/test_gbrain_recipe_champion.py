# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Champion-level coverage for the gbrain recipe-KB backend on the 5-tuple.

Locks that a champion survives the dispatcher's ``_v2_to_arbor`` projection,
stays warm-replay consumable, and degrades to the local store on a gbrain
failure. No network: gbrain gets a fake MCP.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB
from hyperloom.orchestrator.knowledge.recipe_kb.canonical_id import (
    cid_to_path_components,
    recipe_canonical_id,
)
from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import (
    GbrainRemoteRecipeClient,
)

_ARGS_KEY = "extra_server_args"
_RECIPE_SLUG_PREFIX = "hyperloom-recipe-kb"


class _Spec:
    """A single logical recipe expressed once, rendered into the gbrain shape."""

    def __init__(
        self,
        *,
        model: str,
        hardware: str,
        framework_name: str,
        framework_version: str,
        precision: str,
        args: str,
        envs: dict[str, str],
        throughput: float,
        what_worked: list[Any],
        pitfalls: list[Any],
        lessons: list[Any],
    ) -> None:
        self.model = model
        self.hardware = hardware
        self.framework_name = framework_name
        self.framework_version = framework_version
        self.precision = precision
        self.args = args
        self.envs = envs
        self.throughput = throughput
        self.what_worked = what_worked
        self.pitfalls = pitfalls
        self.lessons = lessons

    @property
    def cid(self) -> str:
        return recipe_canonical_id(
            model=self.model,
            hardware=self.hardware,
            framework_name=self.framework_name,
            framework_version=self.framework_version,
            precision=self.precision,
        )

    @property
    def best_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.args:
            cfg[_ARGS_KEY] = self.args
        if self.envs:
            cfg["extra_envs"] = {k: str(v) for k, v in self.envs.items()}
        return cfg

    def gbrain_page(self) -> dict[str, Any]:
        """Render as a gbrain recipe-page frontmatter (flat attrs)."""
        attrs: dict[str, Any] = {
            "model": self.model,
            "hardware": self.hardware,
            "framework_name": self.framework_name,
            "framework_version": self.framework_version,
            "precision": self.precision,
            "best_throughput": self.throughput,
            "what_worked": json.dumps(self.what_worked),
            "pitfalls": json.dumps(self.pitfalls),
            "lessons": json.dumps(self.lessons),
        }
        if self.args:
            attrs["best_config_args"] = self.args
        if self.envs:
            attrs["best_config_envs"] = dict(self.envs)
        return {
            "attrs": attrs,
            "authority": "EXPERIENTIAL",
            "confidence": 0.9,
            "updated_at": "2026-06-03T14:00:00Z",
        }

    def cortex_v2_row(self) -> dict[str, Any]:
        """Render as a central kb-service v2-nested recipe row."""
        model_s, hw_s, fw_s, mt_s, arch_s, fwv_s, prec_s = cid_to_path_components(self.cid)
        return {
            "canonical_id": self.cid,
            "version": 1,
            "labels": {
                "model": model_s,
                "hardware": hw_s,
                "framework_name": fw_s,
                "framework_version": fwv_s,
                "precision": prec_s,
                "model_type": mt_s,
                "architectures": arch_s,
            },
            "body": {
                "best_config": self.best_config,
                "best_throughput": self.throughput,
            },
            "metrics": {"throughput": self.throughput},
            "findings": self.what_worked,
            "failures": [],
            "gaps": [],
            "pitfalls": self.pitfalls,
            "lessons": self.lessons,
            "authority": "EXPERIENTIAL",
            "confidence": 0.9,
            "created_at": "",
            "updated_at": "2026-06-03T14:00:00Z",
        }


SPECS = [
    _Spec(
        model="Qwen/Qwen3-32B",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.5.11",
        precision="fp8",
        args="--cuda-graph-max-bs 256",
        envs={"SGLANG_X": "1"},
        throughput=5430.9,
        what_worked=[{"id": "w1"}],
        pitfalls=[{"id": "p1"}],
        lessons=[{"id": "l1"}],
    ),
    _Spec(
        model="meta-llama/Llama-3-70B",
        hardware="mi300x",
        framework_name="vllm",
        framework_version="0.6.0",
        precision="fp16",
        args="--max-num-seqs 512",
        envs={},
        throughput=3200.0,
        what_worked=[],
        pitfalls=[{"id": "p2"}],
        lessons=[],
    ),
    _Spec(
        model="Qwen/Qwen3-32B",
        hardware="mi355x",
        framework_name="sglang",
        framework_version="0.5.11",
        precision="fp8",
        args="--attention-backend fa3",
        envs={"FOO": "bar"},
        throughput=6100.0,
        what_worked=[{"id": "w3"}],
        pitfalls=[],
        lessons=[{"id": "l3"}],
    ),
]


class _FakeMcp:
    """Stand-in for the gbrain MCP: canned list_pages / get_page."""

    def __init__(self, pages: dict[str, dict[str, Any]]) -> None:
        self.pages = pages

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "list_pages":
            return [
                {"slug": s, "type": "recipe", "updated_at": fm.get("updated_at", "")} for s, fm in self.pages.items()
            ]
        if tool == "get_page":
            fm = self.pages.get(args.get("slug"))
            return {"frontmatter": fm} if fm is not None else {}
        return {}


def _gbrain_dispatcher(local: LocalRecipeStore) -> RecipeKB:
    client = GbrainRemoteRecipeClient(
        base_url="http://gbrain.test",
        token="tok",
        enabled=True,
    )
    client._mcp = _FakeMcp(  # type: ignore[assignment]
        {f"{_RECIPE_SLUG_PREFIX}/recipe/{i}": s.gbrain_page() for i, s in enumerate(SPECS)}
    )
    return RecipeKB(local=local, remote=client)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.cid)
def test_get_recipe_champion(spec: _Spec, tmp_path) -> None:
    """An exact 5-tuple resolves to its champion through the gbrain dispatcher."""
    local = LocalRecipeStore(root=tmp_path)
    row = _gbrain_dispatcher(local).get_recipe(canonical_id=spec.cid)

    assert row is not None, "gbrain dispatcher missed an exact 5-tuple"
    assert row.get("canonical_id") == spec.cid
    assert dict(row.get("best_config") or {}) == spec.best_config
    assert round(float(row.get("best_throughput") or 0.0), 6) == round(spec.throughput, 6)
    assert bool(row.get("what_worked")) == bool(spec.what_worked)
    assert bool(row.get("pitfalls")) == bool(spec.pitfalls)
    assert bool(row.get("lessons")) == bool(spec.lessons)


def test_search_subset_label(tmp_path) -> None:
    """A model-only label filter resolves to the expected 5-tuples."""
    local = LocalRecipeStore(root=tmp_path)
    kb = _gbrain_dispatcher(local)

    model_slug = cid_to_path_components(SPECS[0].cid)[0]
    rows = kb.search(label_match={"model": model_slug})

    cids = sorted(r["canonical_id"] for r in rows)
    expected = sorted(s.cid for s in SPECS if cid_to_path_components(s.cid)[0] == model_slug)
    assert cids == expected


def test_miss_returns_none(tmp_path) -> None:
    """An unknown 5-tuple is a miss (no local seed)."""
    local = LocalRecipeStore(root=tmp_path)
    kb = _gbrain_dispatcher(local)

    unknown = "inference:does-not-exist:mi325x:trtllm:unknown_model_type:unknown_arch:9.9.9:int4"
    assert kb.get_recipe(canonical_id=unknown) is None


def test_gbrain_dispatcher_preserves_champion_regression(tmp_path) -> None:
    """Regression lock for the _v2_to_arbor double-translation bug: the champion config survives the dispatcher."""
    spec = SPECS[0]
    local = LocalRecipeStore(root=tmp_path)
    row = _gbrain_dispatcher(local).get_recipe(canonical_id=spec.cid)

    assert row is not None
    assert row.get("canonical_id") == spec.cid
    assert row.get("best_config") == spec.best_config
    assert row["best_config"], "champion best_config was wiped by the dispatcher"
    assert float(row.get("best_throughput") or 0.0) == pytest.approx(spec.throughput)
    assert row.get("what_worked"), "champion what_worked was wiped by the dispatcher"


def test_gbrain_best_config_is_warm_replay_consumable(tmp_path) -> None:
    """The gbrain round-trip champion survives ``_maybe_enqueue_warm_replay``'s extraction."""
    spec = SPECS[0]
    local = LocalRecipeStore(root=tmp_path)
    row = _gbrain_dispatcher(local).get_recipe(canonical_id=spec.cid)
    assert row is not None
    best_config = row["best_config"]

    bc_args = str(best_config.get("extra_server_args") or best_config.get("args") or "").strip()
    bc_envs = best_config.get("extra_envs") or best_config.get("envs") or {}

    assert bc_args == spec.args
    assert isinstance(bc_envs, dict)
    assert bc_envs == {k: str(v) for k, v in spec.envs.items()}
    assert bc_args or bc_envs, "warm recipe would be skipped as best_config_empty"


def test_gbrain_transport_error_falls_back_to_local(tmp_path) -> None:
    """A gbrain MCP failure degrades to the local store via the ``GbrainRemoteError -> RemoteRecipeClientError`` subclassing."""
    from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import (
        GbrainRemoteError,
        GbrainRemoteRecipeClient,
    )

    spec = SPECS[0]
    model_s, hw_s, fw_s, _mt, _arch, fwv_s, prec_s = cid_to_path_components(spec.cid)
    local = LocalRecipeStore(root=tmp_path)
    local.put_recipe(
        canonical_id=spec.cid,
        model=model_s,
        hardware=hw_s,
        framework_name=fw_s,
        framework_version=fwv_s,
        precision=prec_s,
        best_config=spec.best_config,
        best_throughput=spec.throughput,
    )

    class _BoomMcp:
        def call(self, *a, **k):
            raise GbrainRemoteError("gbrain down")

    client = GbrainRemoteRecipeClient(
        base_url="http://gbrain.test",
        token="tok",
        enabled=True,
    )
    client._mcp = _BoomMcp()  # type: ignore[assignment]
    kb = RecipeKB(local=local, remote=client)

    row = kb.get_recipe(canonical_id=spec.cid)
    assert row is not None
    assert row["canonical_id"] == spec.cid
