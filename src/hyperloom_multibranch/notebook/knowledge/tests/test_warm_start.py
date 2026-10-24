# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE reads the Recipe KB champions of this session's task and scope as leads, whichever flow wrote them."""

from __future__ import annotations

import copy
import hashlib
import logging
import urllib.error
from dataclasses import dataclass, field, replace
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError, RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.models import RecipeScope
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
)
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.knowledge.recipe_kb import overlay_ref
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import FinalRecipe, knowledge_document, recipe_identity
from hyperloom_multibranch.notebook.knowledge.tests.fakes import FakeRecipeStore
from hyperloom_multibranch.contracts.notebook.records import REPLAY_TOOL
from hyperloom_multibranch.notebook.knowledge import warm_start
from hyperloom_multibranch.notebook.knowledge.warm_start import (
    IDENTITY_LABELS,
    PreparedReplay,
    ReplayRefusal,
    ReplayTarget,
    WarmLead,
    identity_labels,
    labels_of,
    replay,
    replay_leads,
    warm_leads,
)
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

PROFILE = profile()
EXACT_ID, SCOPE = recipe_identity(PROFILE)
OLDER_ID, _ = recipe_identity(profile(framework_version="0.10.0"))
LABELS = identity_labels(PROFILE)


def test_a_sessions_labels_are_the_slugs_of_its_canonical_id() -> None:
    """The KB searches identities by these labels, so each must be the slug the canonical id holds for it."""

    scheme, *slugs = EXACT_ID.split(":")
    assert (scheme, list(LABELS)) == (SCOPE.identity_scheme, list(IDENTITY_LABELS))
    assert list(LABELS.values()) == slugs
    assert LABELS["model"] == "deepseek-r1" and LABELS["hardware"] == "mi355x"


def test_an_identity_without_one_slug_per_dimension_has_no_labels() -> None:
    with pytest.raises(ValueError, match="does not hold one slug"):
        labels_of("inference:model:with:a:colon:too:many:slugs:here")
    assert warm_leads(FakeRecipeStore(), profile(model="org/model:latest")) == ()


IMAGE = "sha256:" + "a" * 64
VLLM, VLLM_AT = "https://github.com/vllm-project/vllm", "1" * 40
PATCH_BYTES, TUNED_BYTES = b"--- a/vllm/moe.py\n+++ b/vllm/moe.py\n", b"M,N,K,config\n"
PATCH = PatchRef("patch/vllm/0001-moe.patch", hashlib.sha256(PATCH_BYTES).hexdigest())
TUNED = KernelArtifact(
    KernelKind.GEMM, "bf16 4096x14336", "kernel/gemm/tuned.csv", hashlib.sha256(TUNED_BYTES).hexdigest()
)
ROOT = Recipe(IMAGE, LaunchConfig("--max-num-seqs 128"), (ComponentPatches("vllm", VLLM, VLLM_AT),))
CHAMPION = Recipe(
    IMAGE,
    LaunchConfig("--max-num-seqs 256", {"VLLM_USE_AITER": "1"}),
    (ComponentPatches("vllm", VLLM, VLLM_AT, (PATCH,)),),
    (TUNED,),
)
LEAD = WarmLead(EXACT_ID, "sess-1", SCOPE.as_dict(), 4.0, 1040.0, "--max-num-seqs 256", {"VLLM_USE_AITER": "1"}, True)


def _envelope(canonical_id: str, session_id: str, knowledge: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "canonical_id": canonical_id,
        "session_id": session_id,
        "record_id": f"record-{session_id}",
        "revision": 1,
        "knowledge": knowledge,
        "view": {"source": "current", "replayable": True},
        "scope": SCOPE.as_dict(),
        "artifacts": {"files": [], "file_count": 0},
        **overrides,
    }


def _single_lineage_knowledge() -> dict[str, Any]:
    return {
        "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
        "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
        "optimized_throughput": 1100.0,
        "validated_e2e_gain": 10.0,
        "workload_shape": {"tp": 8, "conc": 64, "isl": 1024, "osl": 1024},
        "value": {
            "config": {"extra_server_args": "--enable-expert-parallel", "extra_envs": {"VLLM_USE_AITER": "1"}},
            "patch": {"patches": [], "provenance": []},
            "kernel": {},
        },
        "lessons": ["expert parallelism pays at conc 64"],
        "pitfalls": ["fp8 kv cache fails accuracy"],
        "what_failed": ["block size 32 regressed"],
        "provenance": {"kernel_optimizer": "forge"},
    }


@pytest.fixture
def store(snapshot: NotebookSnapshot, tmp_path: Path) -> FakeRecipeStore:
    ours = knowledge_document(
        snapshot,
        PROFILE,
        FinalRecipe("i1", Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 256")), tmp_path, METRIC),
        SCOPE,
        {"version": "1", "digest": "sha256:" + "d" * 64},
    )
    malformed = _envelope("inference:malformed", "bad", ours, schema_version=1)
    other_scope = _envelope("inference:other-scope", "wide", ours, scope={**SCOPE.as_dict(), "conc": 128})
    return FakeRecipeStore(
        identities=[
            {"canonical_id": OLDER_ID, "dimensions": {**LABELS, "framework_version": "0.10.0"}},
            {"canonical_id": EXACT_ID, "dimensions": LABELS},
            {"canonical_id": "agentx:elsewhere"},
            {"canonical_id": "inference:malformed"},
            {"canonical_id": "inference:other-scope"},
            {"canonical_id": "inference:nothing-in-scope"},
        ],
        views={
            EXACT_ID: _envelope(EXACT_ID, "sess-1", ours),
            OLDER_ID: _envelope(OLDER_ID, "older", _single_lineage_knowledge()),
            "inference:malformed": malformed,
            "inference:other-scope": other_scope,
        },
    )


def test_leads_come_from_either_flow_exact_identity_first(
    store: FakeRecipeStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.knowledge.warm_start"):
        leads = warm_leads(RemoteRecipeClient(store), PROFILE)

    assert [lead.canonical_id for lead in leads] == [EXACT_ID, OLDER_ID]
    exact, older = leads
    assert (exact.session_id, exact.server_args, exact.replayable, exact.differing) == (
        "sess-1",
        "--max-num-seqs 256",
        True,
        {},
    )
    assert exact.gain_pct == pytest.approx(4.1)
    assert exact.what_failed[0] == "attention / split kv on c0: regressed at -2.00%"
    assert (older.gain_pct, older.best_throughput, dict(older.envs)) == (10.0, 1100.0, {"VLLM_USE_AITER": "1"})
    assert (older.lessons, older.pitfalls, older.what_failed) == (
        ("expert parallelism pays at conc 64",),
        ("fp8 kv cache fails accuracy",),
        ("block size 32 regressed",),
    )
    assert dict(older.differing) == {"framework_version": ("0.10.0", LABELS["framework_version"])}
    assert older.render().startswith(f"- {OLDER_ID}, framework_version 0.10.0 (here ")
    assert "inference:malformed" in caplog.text and "inference:other-scope" in caplog.text


def test_the_search_matches_six_dimensions_page_by_page(store: FakeRecipeStore) -> None:
    warm_leads(RemoteRecipeClient(store), PROFILE)

    assert [search["offset"] for search in store.searches] == [0, 2, 4]
    assert store.searches[0]["scheme"] == "inference"
    assert store.searches[0]["match"] == {key: value for key, value in LABELS.items() if key != "framework_version"}
    assert len(warm_leads(RemoteRecipeClient(store), PROFILE, limit=1)) == 1


def test_a_scope_the_kb_cannot_file_reads_nothing(store: FakeRecipeStore) -> None:
    assert warm_leads(RemoteRecipeClient(store), profile(kernel_optimizer="")) == ()
    assert store.searches == []


def test_a_lead_without_values_renders_what_it_has(store: FakeRecipeStore) -> None:
    store.views[OLDER_ID]["knowledge"]["value"]["config"] = {"extra_envs": "not a mapping"}
    store.views[OLDER_ID]["knowledge"].pop("validated_e2e_gain")

    older = warm_leads(RemoteRecipeClient(store), PROFILE)[1]

    assert (older.server_args, dict(older.envs)) == ("", {})
    assert older.gain_pct == 0.0 and older.render().endswith(": +0.00% [older]")


def test_a_lead_lists_its_launch_arguments_without_the_context_length_and_watchdog_it_was_measured_with(
    store: FakeRecipeStore,
) -> None:
    config = store.views[OLDER_ID]["knowledge"]["value"]["config"]
    config["extra_server_args"] = "--context-length 6144 --watchdog-timeout 1800 --enable-expert-parallel"

    older = warm_leads(RemoteRecipeClient(store), PROFILE)[1]

    assert older.server_args == "--enable-expert-parallel"


def _failed(cause: Exception) -> KBStoreError:
    error = KBStoreError(f"POST /v1/kb/search failed: {cause!r}")
    error.__cause__ = cause
    return error


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://kb.example/v1/kb/search", code, "error", Message(), None)


@dataclass
class _Flaky(FakeRecipeStore):
    """A store whose next searches fail with ``failures``, one each, before it answers."""

    failures: list[KBStoreError] = field(default_factory=list)
    asks: int = 0

    def search_identities(self, **query: Any) -> dict[str, Any]:
        self.asks += 1
        if self.failures:
            raise self.failures.pop(0)
        return super().search_identities(**query)


@pytest.fixture
def no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(warm_start, "KB_RETRY_SECONDS", (0.0, 0.0, 0.0))


@pytest.mark.usefixtures("no_waits")
@pytest.mark.parametrize("cause", [_http(503), urllib.error.URLError(ConnectionResetError("reset by peer"))])
def test_a_server_error_or_a_dropped_connection_is_asked_again(store: FakeRecipeStore, cause: Exception) -> None:
    flaky = _Flaky(identities=store.identities, views=store.views, failures=[_failed(cause), _failed(cause)])

    leads = warm_leads(RemoteRecipeClient(flaky), PROFILE)

    assert [lead.canonical_id for lead in leads] == [EXACT_ID, OLDER_ID]
    assert flaky.asks == len(flaky.searches) + 2


@pytest.mark.usefixtures("no_waits")
def test_a_refusal_is_not_asked_again_and_a_kb_that_keeps_failing_fails_the_read(store: FakeRecipeStore) -> None:
    refused = _Flaky(identities=store.identities, views=store.views, failures=[_failed(_http(403))])
    down = _Flaky(identities=store.identities, views=store.views, failures=[_failed(_http(503)) for _ in range(4)])

    for kb in (refused, down):
        with pytest.raises(KBStoreError):
            warm_leads(RemoteRecipeClient(kb), PROFILE)

    assert (refused.asks, down.asks) == (1, 4)


@dataclass
class _Downloads:
    """One champion's View, and its files laid out under ``files`` as ``RemoteRecipeClient.read`` lays them."""

    view: dict[str, Any]
    files: dict[str, bytes]
    destinations: list[Path] = field(default_factory=list)

    def search_identities(self, **_: Any) -> dict[str, Any]:
        return {"items": []}

    def get_view(self, canonical_id: str, scope: RecipeScope) -> dict[str, Any] | None:
        return copy.deepcopy(self.view)

    def read(
        self, canonical_id: str, destination: str | Path, *, scope: RecipeScope, envelope: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.destinations.append(Path(destination))
        for ref, data in self.files.items():
            path = Path(destination) / "files" / ref
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {}


def _downloads(snapshot: NotebookSnapshot, tmp_path: Path, champion: Recipe = CHAMPION) -> _Downloads:
    knowledge = knowledge_document(
        snapshot,
        PROFILE,
        FinalRecipe("i1", champion, tmp_path, METRIC),
        SCOPE,
        {"version": "1", "digest": "sha256:" + "d" * 64},
    )
    files = {overlay_ref(0, 0, PATCH.path): PATCH_BYTES, TUNED.ref: TUNED_BYTES}
    return _Downloads(_envelope(EXACT_ID, "sess-1", knowledge), files)


def _target(tmp_path: Path) -> ReplayTarget:
    (tmp_path / "blobs").mkdir(exist_ok=True)
    return ReplayTarget(ROOT, LocalBlobStore(tmp_path / "blobs"), tmp_path / "warm")


def test_a_champion_on_this_image_replays_its_whole_recipe_on_any_head_its_files_in_the_blob_store(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    downloads, target = _downloads(snapshot, tmp_path), _target(tmp_path)
    moved = replace(ROOT, launch=LaunchConfig("--max-num-seqs 192 --enable-chunked-prefill"))

    found = replay(downloads, LEAD, SCOPE, target)

    assert found is not None and (found.recipe, found.reason) == (CHAMPION, "")
    assert (target.blobs.get(PATCH.sha256), target.blobs.get(TUNED.sha256)) == (PATCH_BYTES, TUNED_BYTES)
    assert ":" not in downloads.destinations[0].name
    assert found.describe() == f"replay on your head with {REPLAY_TOOL}: the whole recipe"
    for head, commit in ((ROOT, "c0"), (moved, "c3")):
        on_head = found.on(head, commit)
        assert on_head is not None and (on_head.parent_commit, on_head.layer) == (commit, IntentLayer.KERNEL)
        made = apply(head, on_head.delta)
        assert (made.launch, made.components, made.kernels) == (CHAMPION.launch, CHAMPION.components, CHAMPION.kernels)
    assert found.on(CHAMPION, "c9") is None


def test_a_replay_stacks_the_champions_table_over_the_heads_instead_of_replacing_it(tmp_path: Path) -> None:
    from hyperloom_multibranch.repo.merge import wider_table
    from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
    from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import B09_NEW, ENV, SHARED, csv_text

    narrow = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED))
    wide = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED, B09_NEW))
    store = {narrow.sha256: csv_text(SHARED), wide.sha256: csv_text(SHARED, B09_NEW)}
    head = replace(ROOT, kernels=(narrow,))
    champion = replace(CHAMPION, kernels=(wide,), components=ROOT.components)
    found = PreparedReplay(EXACT_ID, champion, champion.launch)

    on_head = found.on(head, "c3", lambda wanted, current: wider_table(wanted, current, lambda a: store[a.sha256]))

    assert on_head is not None
    assert on_head.delta.add_kernels == (wide,)
    assert apply(head, on_head.delta).kernels == (wide,)


def test_a_replay_keeps_the_heads_upstream_for_a_component_the_head_has_and_brings_the_rest(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    found = replay(_downloads(snapshot, tmp_path), LEAD, SCOPE, _target(tmp_path))
    assert found is not None and found.recipe is not None
    aiter = ComponentPatches("aiter", "https://github.com/ROCm/aiter", "3" * 40)
    labelled = (ComponentPatches("vllm", "file:///workspace/vllm", "2" * 40, (PATCH,)), aiter)

    on_root = replace(found, recipe=replace(CHAMPION, components=labelled)).on(ROOT, "c0")

    assert on_root is not None
    assert apply(ROOT, on_root.delta).components == (ComponentPatches("vllm", VLLM, VLLM_AT, (PATCH,)), aiter)


def test_a_replay_keeps_the_heads_context_length_and_watchdog_and_drops_the_champions() -> None:
    launch = LaunchConfig("--max-num-seqs 256 --context-length 6144 --watchdog-timeout 1800", CHAMPION.launch.envs)
    capped = replace(CHAMPION, launch=launch)
    found = PreparedReplay(EXACT_ID, capped, capped.launch)
    served = replace(ROOT, launch=LaunchConfig("--max-num-seqs 128 --max-model-len 9216 --watchdog-timeout 600"))
    reordered = replace(CHAMPION, launch=LaunchConfig("--context-length=8192 --max-num-seqs 256", CHAMPION.launch.envs))

    on_root, on_served = found.on(ROOT, "c0"), found.on(served, "c1")

    assert on_root is not None and on_served is not None
    assert apply(ROOT, on_root.delta).launch == CHAMPION.launch and found.is_on(apply(ROOT, on_root.delta))
    made = apply(served, on_served.delta).launch.server_args
    assert made == "--max-num-seqs 256 --max-model-len 9216 --watchdog-timeout 600"
    assert found.is_on(reordered) and found.on(reordered, "c2") is None
    assert not found.is_on(ROOT) and not found.is_on(replace(reordered, kernels=()))


@pytest.mark.parametrize(
    ("spoil", "reason"),
    [
        ("another image", "built on another image"),
        ("no image", "names no base image"),
        ("tampered file", "the record pins"),
        ("fusion kernel", "measurement loads no fusion kernel artifacts yet"),
        ("no files", "holds no patch or kernel files"),
    ],
)
def test_a_champion_whose_recipe_cannot_come_along_replays_its_launch_config_only(
    snapshot: NotebookSnapshot, tmp_path: Path, spoil: str, reason: str
) -> None:
    fused = replace(CHAMPION, kernels=(replace(TUNED, kind=KernelKind.FUSION),))
    downloads = _downloads(snapshot, tmp_path, fused if spoil == "fusion kernel" else CHAMPION)
    provenance = downloads.view["knowledge"]["provenance"]
    if spoil == "another image":
        provenance["base_image_digest"] = "sha256:" + "b" * 64
    elif spoil == "no image":
        provenance.pop("base_image_digest")
    elif spoil == "tampered file":
        downloads.files[TUNED.ref] = b"tampered"
    lead = replace(LEAD, replayable=spoil != "no files")

    found = replay(downloads, lead, SCOPE, _target(tmp_path))

    assert found is not None and found.recipe is None and reason in found.reason
    on_root = found.on(ROOT, "c0")
    assert on_root is not None and on_root.layer is IntentLayer.CONFIG and reason in on_root.reason
    made = apply(ROOT, on_root.delta)
    assert (made.launch, made.components) == (LaunchConfig(LEAD.server_args, LEAD.envs), ROOT.components)
    assert "its launch config only" in found.describe()


def test_only_the_best_leads_get_a_replay_and_one_that_changes_nothing_gets_none(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    same = replace(LEAD, canonical_id="inference:same", server_args=ROOT.launch.server_args, envs={}, replayable=False)
    leads = (LEAD, same, replace(LEAD, canonical_id="inference:third"))

    replayed = replay_leads(_downloads(snapshot, tmp_path), PROFILE, leads, _target(tmp_path), limit=2)

    assert [lead.replay is not None for lead in replayed] == [True, False, False]
    assert replayed[0].render().endswith(f"; replay on your head with {REPLAY_TOOL}: the whole recipe")
    assert replay_leads(_downloads(snapshot, tmp_path), profile(kernel_optimizer=""), leads, _target(tmp_path)) == leads


def test_a_replay_acceptance_refuses_is_named_with_its_code_in_place_of_the_replay(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    found = replay(_downloads(snapshot, tmp_path), LEAD, SCOPE, _target(tmp_path))
    refusal = ReplayRefusal("online_tuning_unsupported", "TunableOp tunes every GEMM shape " * 10)

    described = replace(LEAD, replay=None, refusal=refusal).describe()
    unchecked = replace(LEAD, replay=found, unchecked="the time ran out").describe()

    assert described.startswith(f"{LEAD.summary()}; not replayable here: online_tuning_unsupported (TunableOp")
    assert REPLAY_TOOL not in described and described.endswith("...)")
    assert (
        replace(refusal, rule_pack="2026.10.2")
        .describe()
        .startswith("not replayable here under rule pack 2026.10.2: online_tuning_unsupported (TunableOp")
    )
    assert unchecked == f"{LEAD.summary()}; {found.describe()}; not checked against acceptance: the time ran out"
