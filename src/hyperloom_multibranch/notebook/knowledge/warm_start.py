# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE warm start: what the Recipe KB says about this session's task, as leads for the first specialists.

The read takes the single-lineage PRELUDE's two steps:

1. ``search_identities`` finds the canonical ids that match the session on six of the seven serving dimensions,
   labelled with the slugs of the session's own canonical id; the framework version is left free, so a recipe measured
   on the previous release still counts.
2. ``get_view`` reads each one's champion in exactly this session's scope.

Only exact scopes are read: the KB Store does not document whether ``get_rollup`` without a scope lists a record's
other scopes, so finding the nearest concurrency and sequence lengths waits until it does.

Every record goes through ``knowledge_to_warm_recipe``, so records the single-lineage CLOSE wrote read the same way.
A lead names the dimensions where it differs from this session. It is a lead, not a starting point: specialists still
measure it on this session's root.

PRELUDE reads the KB once per session, and the KB Store answers a server error now and then, so each read is asked
again after a server error or a dropped connection, waiting :data:`KB_RETRY_SECONDS`; a refusal is not asked again.

:func:`replay_leads` gives the first leads a :class:`PreparedReplay`: the champion's recipe, patches and kernel
artifacts too, their files downloaded from the KB into the session's blob store. A champion built on another image
replays its launch config and patches, each component on the head's upstream as a merge takes it, but not its kernel
artifacts, tuned against the other image's libraries: they are listed as references, and its gain is said to be
measured there. One whose files the KB does not hold replays its launch config only. The specialist's
``notebook_replay`` makes the change for its own branch head, since a gain lands only on the head it was measured
against.

A replay leaves the flags of :data:`HEAD_FLAGS` to the head it is made for: the champion's context length and
scheduler watchdog are dropped, and the head's, if it sets them, kept. A record can carry the ones the launcher set for
the session that measured it, and the launcher sets both again for this one when a recipe leaves them out. The cap a
champion was measured with is no optimization, and one below what this session's accuracy evaluation runs in is refused
as ``context_below_eval``. A lead's launch arguments are listed without them for the same reason.

PRELUDE checks each replay against the toolbox's acceptance on the root. A replay acceptance refuses keeps its lead, its
:class:`ReplayRefusal` named in place of the replay, so no specialist spends a submit on it; one acceptance did not
answer for in PRELUDE's time stays offered, its lead saying it was not checked. A cross-image replay whose patches
acceptance refuses, as a stack that does not apply in order on this image, is checked again as
:meth:`PreparedReplay.launch_only`: its launch config without the environment its patches read, the patches listed
as references.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeDelta, RecipeError, apply, delta_between
from hyperloom.inference_optimizer.grid_server_args import (
    INJECTED_LAUNCH_FLAGS,
    merge_server_args,
    remove_server_args,
    tokenize_server_args_preserving_json,
)
from hyperloom.inference_optimizer.recipe_snapshot_constants import recipe_canonical_id
from hyperloom.measure.compile import LOADABLE_KERNEL_KINDS, BlobError, BlobStore
from hyperloom.measure.eval_context import CONTEXT_FLAGS
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.models import (
    CONFIG_SECTION,
    RecipeScope,
    RemoteRecipeValidationError,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import knowledge_to_warm_recipe
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import REPLAY_TOOL, SessionProfile, WarmReplay
from hyperloom_multibranch.notebook.changes import same_recipe
from hyperloom_multibranch.notebook.knowledge.lessons import listed
from hyperloom_multibranch.notebook.knowledge.recipe_kb import from_recipe_kb_value, overlay_ref
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import recipe_identity, serving_identity
from hyperloom_multibranch.notebook.knowledge.replay_tables import StackSlot, stack_added_kernels

log = logging.getLogger(__name__)

MAX_CANDIDATES = 20
PAGE_SIZE = 50
#: The flags a replay leaves to the head: the context length a server serves (``--context-length`` on SGLang,
#: ``--max-model-len`` on vLLM), and what the launch guards add for the run they launch.
HEAD_FLAGS = tuple(dict.fromkeys((*CONTEXT_FLAGS, *INJECTED_LAUNCH_FLAGS)))
#: How many of the best leads get a replay; each whole one downloads the champion's files.
REPLAY_LEADS = 3
#: The waits before each new ask of a read that failed transiently; one more ask follows the last wait.
KB_RETRY_SECONDS = (2.0, 5.0, 15.0)
#: How much of an acceptance refusal's reason a lead's line keeps.
REFUSAL_CHARS = 160
#: Why a cross-image replay fell back to its launch config, by default.
LAUNCH_ONLY_REASON = "acceptance refused its patches on this session's image"
#: The kind a patch is listed under among a replay's references; a kernel artifact is listed under its own kind.
PATCH_REFERENCE = "patch"
#: How much of an image digest a lead's line shows.
IMAGE_CHARS = 19
#: How much of a blob's sha256 or an upstream commit a lead's line shows.
SHORT_REF = 12
MODEL_LABEL = "model"
#: The one label a lead may differ in and still be read: a recipe measured on another framework release.
FRAMEWORK_VERSION_LABEL = "framework_version"
#: The KB labels of a recipe identity, in the order :func:`recipe_canonical_id` writes their slugs after the scheme.
IDENTITY_LABELS = (
    MODEL_LABEL,
    "hardware",
    "framework_name",
    "model_type",
    "architectures",
    FRAMEWORK_VERSION_LABEL,
    "precision",
)
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_REPLAY_ERRORS = (KBStoreError, RemoteRecipeValidationError, RecipeError, BlobError, OSError, ValueError)
_Read = TypeVar("_Read")


def _transient(error: KBStoreError) -> bool:
    """Whether ``error`` is a server error or a dropped connection, as the KB Store client raises them."""

    cause = error.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code >= 500
    return isinstance(cause, OSError)


def _asked(read: Callable[[], _Read]) -> _Read:
    """``read()``, asked again after each wait of :data:`KB_RETRY_SECONDS` while it fails transiently."""

    for wait in KB_RETRY_SECONDS:
        try:
            return read()
        except KBStoreError as exc:
            if not _transient(exc):
                raise
            log.warning("Recipe KB read failed, asking again in %gs: %s", wait, exc)
            time.sleep(wait)
    return read()


class RecipeReader(Protocol):
    """What the warm start needs of a Recipe KB client, such as ``RemoteRecipeClient``."""

    def search_identities(
        self,
        *,
        scheme: str,
        match: dict[str, str],
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]: ...

    def get_view(self, canonical_id: str, scope: RecipeScope) -> dict[str, Any] | None: ...


class RecipeDownloader(RecipeReader, Protocol):
    """What a replay needs of a Recipe KB client besides reading: a champion's verified files, as ``RemoteRecipeClient``
    downloads them under ``destination/files``."""

    def read(
        self, canonical_id: str, destination: str | Path, *, scope: RecipeScope, envelope: dict[str, Any] | None = None
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class ReplayTarget:
    """Where a replay is prepared: the session's root recipe, whose image a champion must share for its patches and
    kernels to apply, the session's blob store, and a directory to download into."""

    root: Recipe
    blobs: BlobStore
    workdir: Path


@dataclass(frozen=True)
class ReplayReference:
    """A file of a champion's its replay leaves out: its ``kind`` (:data:`PATCH_REFERENCE` or a kernel artifact's
    kind), the path or ref and the sha256 the recipe pins, the blob it is in the session's store under, the image the
    champion was built on, and for a patch the component and the upstream commit it was made on, and its ``text`` as
    PRELUDE read it into the store."""

    kind: str
    ref: str
    sha256: str
    image: str
    component: str = ""
    upstream_ref: str = ""
    text: str = ""

    def describe(self) -> str:
        made = f" on {self.component} {self.upstream_ref[:SHORT_REF]}" if self.component else ""
        return f"{self.kind} {self.ref} (blob {self.sha256[:SHORT_REF]}{made}, built on {self.image[:IMAGE_CHARS]})"


@dataclass(frozen=True)
class PreparedReplay:
    """A champion's recipe ready to replay on any head of the session.

    ``recipe`` is its whole recipe, its patch and kernel files already in the session's blob store; ``None`` when only
    its launch config ``launch`` replays, ``reason`` saying why. ``source_image`` is the image the champion was built
    on when it is not the session's: its kernel artifacts stay out, each head keeping its own, and its gain was
    measured there. ``references`` are the files the replay leaves out, ``patch_envs`` the variables of the launch
    environment its patches read, and ``patch_texts`` each patch's text by its sha256, which a launch-only replay
    keeps for the specialist to port.
    """

    canonical_id: str
    recipe: Recipe | None
    launch: LaunchConfig
    reason: str = ""
    references: tuple[ReplayReference, ...] = ()
    source_image: str = ""
    patch_envs: tuple[str, ...] = ()
    patch_texts: Mapping[str, str] = field(default_factory=dict)

    def launch_only(self, reason: str = LAUNCH_ONLY_REASON) -> PreparedReplay:
        """The champion's launch config alone, for ``reason``: without the variables its patches read, which mean
        nothing without them, and with every patch and kernel artifact of the recipe among the references."""

        if self.recipe is None:
            return self
        image = self.source_image or self.recipe.base_image_digest
        envs = {name: value for name, value in self.launch.envs.items() if name not in self.patch_envs}
        return replace(
            self,
            recipe=None,
            launch=replace(self.launch, envs=envs),
            reason=reason,
            references=(
                *_patch_references(self.recipe, image, self.patch_texts),
                *self.references,
                *_kernel_references(self.recipe, image),
            ),
        )

    def reference_text(self, ref: str) -> str | None:
        """The text of the patch ``ref`` the replay leaves out; ``None`` when it leaves out no patch of that path
        whose text PRELUDE read."""

        found = (item.text for item in self.references if item.kind == PATCH_REFERENCE and item.ref == ref)
        return next((text for text in found if text), None)

    def on(self, head: Recipe, parent_commit: str, stack_slot: StackSlot | None = None) -> WarmReplay | None:
        """The change that makes ``head``, the recipe of ``parent_commit``, into the champion's; ``None`` when ``head``
        is the champion's already (:meth:`is_on`).

        A component the head has keeps the head's upstream repository and commit, as a merge keeps it: the session
        builds it from its image, so the champion's upstream label names nothing to rebuild. The head keeps its flags of
        :data:`HEAD_FLAGS` as well. With ``stack_slot``, each kernel the champion adds in a slot the head already fills
        is stacked over the head's table (:func:`stack_added_kernels`) instead of replacing it.
        """

        if self.is_on(head):
            return None
        delta = delta_between(head, self._wanted(head))
        delta, _replaces = stack_added_kernels(delta, head, stack_slot)
        if apply(head, delta) == head:
            return None
        references = tuple(item.describe() for item in self.references)
        layer = delta_layer(delta)
        return WarmReplay(self.canonical_id, parent_commit, delta, layer, self.reason, self.source_image, references)

    def is_on(self, recipe: Recipe) -> bool:
        """Whether ``recipe`` is the champion's already (:func:`same_recipe`), its flags of :data:`HEAD_FLAGS` aside."""

        return same_recipe(recipe, self._wanted(recipe))

    def describe(self) -> str:
        what = "the whole recipe" if self.recipe is not None else f"its launch config only, {self.reason}"
        if self.source_image:
            what += (
                f"; built on {self.source_image[:IMAGE_CHARS]}, so its gain was measured there, not on this session's "
                "image"
            )
        if self.references:
            what += f"; left out: {', '.join(item.describe() for item in self.references)}"
        return f"replay on your head with {REPLAY_TOOL}: {what}"

    def _wanted(self, head: Recipe) -> Recipe:
        champion = self.recipe if self.recipe is not None else replace(head, launch=self.launch)
        if self.source_image:
            champion = replace(champion, kernels=head.kernels)
        args = _with_head_flags(champion.launch.server_args, head.launch.server_args)
        return replace(_on_head_upstreams(champion, head), launch=replace(champion.launch, server_args=args))


def _patch_references(recipe: Recipe, image: str, texts: Mapping[str, str]) -> tuple[ReplayReference, ...]:
    return tuple(
        ReplayReference(
            PATCH_REFERENCE,
            patch.path,
            patch.sha256,
            image,
            component.component,
            component.upstream_ref,
            texts.get(patch.sha256, ""),
        )
        for component in recipe.components
        for patch in component.patches
    )


def _kernel_references(recipe: Recipe, image: str) -> tuple[ReplayReference, ...]:
    return tuple(ReplayReference(item.kind.value, item.ref, item.sha256, image) for item in recipe.kernels)


def _read_envs(patches: Sequence[bytes], envs: Iterable[str]) -> tuple[str, ...]:
    """The variables of ``envs`` a line ``patches`` add names: environment the patched code reads."""

    added = [
        line
        for text in patches
        for line in text.decode("utf-8", "replace").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return tuple(name for name in envs if any(re.search(rf"\b{re.escape(name)}\b", line) for line in added))


def _head_args(server_args: str) -> tuple[str, ...]:
    """The arguments of ``server_args`` that set a flag of :data:`HEAD_FLAGS`, each flag with its value."""

    parsed = tokenize_server_args_preserving_json(server_args)
    tokens = parsed[1] if parsed is not None else []
    found: list[str] = []
    for index, token in enumerate(tokens):
        if token.partition("=")[0] not in HEAD_FLAGS:
            continue
        value = tokens[index + 1] if "=" not in token and index + 1 < len(tokens) else ""
        found.append(f"{token} {value}" if value and not value.startswith("-") else token)
    return tuple(found)


def _with_head_flags(server_args: str, head_args: str) -> str:
    """``server_args`` with the flags of :data:`HEAD_FLAGS` ``head_args`` sets: its own dropped, the head's kept."""

    own, kept = _head_args(server_args), _head_args(head_args)
    if own == kept:
        return server_args
    args = remove_server_args(server_args, HEAD_FLAGS) if own else server_args
    return merge_server_args(args, *kept)


def _on_head_upstreams(wanted: Recipe, head: Recipe) -> Recipe:
    components = tuple(
        component
        if (current := head.component(component.component)) is None
        else replace(component, upstream_repo=current.upstream_repo, upstream_ref=current.upstream_ref)
        for component in wanted.components
    )
    return replace(wanted, components=components)


@dataclass(frozen=True)
class ReplayRefusal:
    """Why the toolbox's acceptance refuses a champion's replay on the session's root: its code, its reason, and the
    version of the rule pack it was refused under, which a later pack may judge otherwise."""

    code: str
    reason: str
    rule_pack: str = ""

    def describe(self) -> str:
        reason = self.reason if len(self.reason) <= REFUSAL_CHARS else self.reason[: REFUSAL_CHARS - 3] + "..."
        under = f" under rule pack {self.rule_pack}" if self.rule_pack else ""
        return f"not replayable here{under}: {self.code} ({reason})"


@dataclass(frozen=True)
class WarmLead:
    """One Recipe KB champion in this session's scope; ``differing`` maps a dimension to (theirs, ours)."""

    canonical_id: str
    session_id: str
    scope: Mapping[str, object]
    gain_pct: float | None
    best_throughput: float | None
    server_args: str
    envs: Mapping[str, str]
    replayable: bool
    lessons: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    what_failed: tuple[str, ...] = ()
    differing: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    replay: PreparedReplay | None = None
    refusal: ReplayRefusal | None = None
    unchecked: str = ""

    def summary(self) -> str:
        """The champion: its id, where its scope differs from this session's, its gain, launch arguments and session."""

        gain = "n/a" if self.gain_pct is None else f"{self.gain_pct:+.2f}%"
        where = "".join(f", {key} {theirs} (here {ours})" for key, (theirs, ours) in self.differing.items())
        args = f" `{self.server_args}`" if self.server_args else ""
        return f"{self.canonical_id}{where}: {gain}{args} [{self.session_id}]"

    def describe(self) -> str:
        if self.refusal is not None:
            return f"{self.summary()}; {self.refusal.describe()}"
        replay = f"; {self.replay.describe()}" if self.replay is not None else ""
        unchecked = f"; not checked against acceptance: {self.unchecked}" if self.unchecked else ""
        return f"{self.summary()}{replay}{unchecked}"

    def render(self) -> str:
        return f"- {self.describe()}"


def labels_of(canonical_id: str) -> dict[str, str]:
    """The KB labels of ``canonical_id``: the slug of each serving dimension, as :func:`recipe_canonical_id` writes
    them after its scheme. Raises ``ValueError`` when the id does not hold one slug per dimension."""

    _, *slugs = canonical_id.split(":")
    if len(slugs) != len(IDENTITY_LABELS):
        raise ValueError(f"{canonical_id!r} does not hold one slug for each of {', '.join(IDENTITY_LABELS)}")
    return dict(zip(IDENTITY_LABELS, slugs))


def identity_labels(profile: SessionProfile) -> dict[str, str]:
    """The session's seven serving dimensions, as the KB labels them; ``ValueError`` as :func:`labels_of` raises it."""

    return labels_of(recipe_canonical_id(serving_identity(profile)))


def _config(envelope: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """The launch arguments and environment of a record ``knowledge_to_warm_recipe`` accepted."""

    config = envelope["knowledge"]["value"][CONFIG_SECTION]
    envs = config.get("extra_envs")
    return str(config.get("extra_server_args") or ""), (
        {str(key): str(item) for key, item in envs.items()} if isinstance(envs, Mapping) else {}
    )


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def lead_of(
    envelope: dict[str, Any], scope: RecipeScope, labels: Mapping[str, str], dimensions: Mapping[str, Any]
) -> WarmLead:
    """The lead one champion View gives; raises ``RemoteRecipeValidationError`` on a record of another contract."""

    row = knowledge_to_warm_recipe(envelope)
    server_args, envs = _config(envelope)
    return WarmLead(
        canonical_id=str(envelope.get("canonical_id") or ""),
        session_id=str(row.get("remote_session_id") or ""),
        scope=scope.as_dict(),
        gain_pct=_number(row.get("validated_gain_pct")),
        best_throughput=_number(row.get("best_throughput")),
        server_args=_with_head_flags(server_args, ""),
        envs=envs,
        replayable=row.get("replay_material_available") is True,
        lessons=listed(map(str, row.get("lessons") or ())),
        pitfalls=listed(map(str, row.get("pitfalls") or ())),
        what_failed=listed(map(str, row.get("what_failed") or ())),
        differing={
            key: (str(dimensions[key]), ours)
            for key, ours in labels.items()
            if key in dimensions and str(dimensions[key]) != ours
        },
    )


def candidates(reader: RecipeReader, scheme: str, match: dict[str, str], limit: int) -> list[Mapping[str, Any]]:
    """The identity search results for ``match``, page by page: at most ``limit`` distinct ids of ``scheme``."""

    found: dict[str, Mapping[str, Any]] = {}
    offset: int | None = 0
    while offset is not None and len(found) < limit:
        page = _asked(partial(reader.search_identities, scheme=scheme, match=match, offset=offset, limit=PAGE_SIZE))
        items = [item for item in page.get("items") or () if isinstance(item, Mapping)]
        for item in items:
            canonical_id = str(item.get("canonical_id") or "")
            if canonical_id.startswith(f"{scheme}:") and len(found) < limit:
                found.setdefault(canonical_id, item)
        next_offset = page.get("next_offset")
        offset = next_offset if isinstance(next_offset, int) and items and next_offset > offset else None
    return list(found.values())


def _read_lead(
    reader: RecipeReader, item: Mapping[str, Any], scope: RecipeScope, labels: Mapping[str, str]
) -> WarmLead | None:
    canonical_id = str(item["canonical_id"])
    try:
        envelope = _asked(partial(reader.get_view, canonical_id, scope))
        dimensions = item.get("dimensions")
        return (
            lead_of(envelope, scope, labels, dimensions if isinstance(dimensions, Mapping) else {})
            if envelope is not None
            else None
        )
    except (KBStoreError, RemoteRecipeValidationError, ValueError) as exc:
        log.warning("Recipe KB record %s is no warm-start lead: %s", canonical_id, exc)
        return None


def warm_leads(reader: RecipeReader, profile: SessionProfile, *, limit: int = MAX_CANDIDATES) -> tuple[WarmLead, ...]:
    """Champions of this session's scope on its task, exact identity first, then best gain first.

    None when the session's scope is not one the KB files, or its identity names no slug per dimension.
    """

    try:
        _, scope = recipe_identity(profile)
        labels = identity_labels(profile)
    except (RemoteRecipeValidationError, ValueError):
        return ()
    match = {key: value for key, value in labels.items() if key != FRAMEWORK_VERSION_LABEL}
    found = (
        _read_lead(reader, item, scope, labels) for item in candidates(reader, scope.identity_scheme, match, limit)
    )
    leads = [lead for lead in found if lead is not None]
    return tuple(sorted(leads, key=lambda lead: (bool(lead.differing), -(lead.gain_pct or 0.0))))


def replay(client: RecipeDownloader, lead: WarmLead, scope: RecipeScope, target: ReplayTarget) -> PreparedReplay | None:
    """``lead``'s champion ready to replay on any head of the session; ``None`` when it changes nothing on the root.

    The whole recipe needs the record's files, kernel artifacts of kinds measurement loads, and components the root
    takes from the same upstream repositories. Its files go into the session's blob store, each checked against the
    hash the recipe pins. A champion built on another image than the root's replays without its kernel artifacts.
    Otherwise only the launch config replays, and the replay says why.
    """

    try:
        launch = LaunchConfig(server_args=lead.server_args, envs=dict(lead.envs))
    except RecipeError as exc:
        log.warning("Recipe KB champion %s has no launch config to replay: %s", lead.canonical_id, exc)
        return None
    config_only = PreparedReplay(lead.canonical_id, None, launch, "the record holds no patch or kernel files")
    if lead.replayable:
        try:
            whole = replace(_champion(client, lead, scope, target), launch=launch)
            return whole if whole.on(target.root, "") is not None else None
        except _REPLAY_ERRORS as exc:
            log.warning("Recipe KB champion %s replays its launch config only: %s", lead.canonical_id, exc)
            config_only = replace(config_only, reason=str(exc))
    return config_only if config_only.on(target.root, "") is not None else None


def replay_leads(
    client: RecipeDownloader,
    profile: SessionProfile,
    leads: Sequence[WarmLead],
    target: ReplayTarget,
    *,
    limit: int = REPLAY_LEADS,
) -> tuple[WarmLead, ...]:
    """``leads`` with the replay of each of the first ``limit``, best first as :func:`warm_leads` orders them."""

    try:
        _, scope = recipe_identity(profile)
    except RemoteRecipeValidationError:
        return tuple(leads)
    return tuple(
        replace(lead, replay=replay(client, lead, scope, target)) if index < limit else lead
        for index, lead in enumerate(leads)
    )


def _champion(client: RecipeDownloader, lead: WarmLead, scope: RecipeScope, target: ReplayTarget) -> PreparedReplay:
    """The champion's whole replay, its files put into the session's blob store: on another image than this
    session's, without its kernel artifacts, which it lists as references."""

    envelope = _asked(partial(client.get_view, lead.canonical_id, scope))
    if envelope is None:
        raise RecipeError("the record is gone")
    champion = _recorded_recipe(envelope)
    source, references = "", ()
    if (champion.base_image_digest, champion.runtime_snapshot) != (
        target.root.base_image_digest,
        target.root.runtime_snapshot,
    ):
        log.info("Recipe KB champion %s was built on %s", lead.canonical_id, champion.base_image_digest)
        source, references = champion.base_image_digest, _kernel_references(champion, champion.base_image_digest)
        champion = replace(champion, kernels=())
    unloadable = sorted({item.kind.value for item in champion.kernels if item.kind not in LOADABLE_KERNEL_KINDS})
    if unloadable:
        raise RecipeError(f"measurement loads no {', '.join(unloadable)} kernel artifacts yet")
    destination = target.workdir / _UNSAFE_NAME.sub("-", lead.canonical_id)
    _asked(partial(client.read, lead.canonical_id, destination, scope=scope, envelope=envelope))
    files = destination / "files"
    patches: dict[str, bytes] = {}
    for stack, component in enumerate(champion.components):
        for member, patch in enumerate(component.patches):
            patches[patch.sha256] = _keep(target.blobs, files / overlay_ref(stack, member, patch.path), patch.sha256)
    for artifact in champion.kernels:
        _keep(target.blobs, files / artifact.ref, artifact.sha256)
    return PreparedReplay(
        lead.canonical_id,
        champion,
        champion.launch,
        references=references,
        source_image=source,
        patch_envs=_read_envs(tuple(patches.values()), lead.envs),
        patch_texts={sha256: data.decode(errors="replace") for sha256, data in patches.items()},
    )


def _recorded_recipe(envelope: Mapping[str, Any]) -> Recipe:
    knowledge = _object(envelope.get("knowledge"))
    provenance = _object(knowledge.get("provenance"))
    image = provenance.get("base_image_digest")
    if not isinstance(image, str) or not image:
        raise RecipeError("the record names no base image")
    return from_recipe_kb_value(
        _object(knowledge.get("value")), base_image_digest=image, runtime_snapshot=provenance.get("runtime_snapshot")
    )


def _object(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _keep(blobs: BlobStore, path: Path, sha256: str) -> bytes:
    """The file at ``path``, put into ``blobs``; raises ``RecipeError`` unless its sha256 is ``sha256``."""

    data = path.read_bytes()
    found = blobs.put(data)
    if found != sha256:
        raise RecipeError(f"{path.name} has sha256 {found}; the record pins {sha256}")
    return data


def delta_layer(delta: RecipeDelta) -> IntentLayer:
    """The deepest layer ``delta`` reaches, the one a replay of it claims."""

    if delta.add_kernels or delta.remove_kernels:
        return IntentLayer.KERNEL
    return IntentLayer.SOURCE if delta.components else IntentLayer.CONFIG


__all__ = [
    "FRAMEWORK_VERSION_LABEL",
    "HEAD_FLAGS",
    "IDENTITY_LABELS",
    "IMAGE_CHARS",
    "KB_RETRY_SECONDS",
    "LAUNCH_ONLY_REASON",
    "MAX_CANDIDATES",
    "MODEL_LABEL",
    "PATCH_REFERENCE",
    "REPLAY_LEADS",
    "SHORT_REF",
    "PreparedReplay",
    "RecipeDownloader",
    "RecipeReader",
    "ReplayReference",
    "ReplayRefusal",
    "ReplayTarget",
    "WarmLead",
    "candidates",
    "delta_layer",
    "identity_labels",
    "labels_of",
    "lead_of",
    "replay",
    "replay_leads",
    "warm_leads",
]
