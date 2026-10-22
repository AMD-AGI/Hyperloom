# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE warm start: what the Recipe KB says about this session's task, as leads for the first specialists.

The read takes the single-lineage PRELUDE's two steps:

1. ``search_identities`` finds the canonical ids that match the session on six of the seven serving dimensions,
   labelled as ``canonical_labels`` labels them; the framework version is left free, so a recipe measured on the
   previous release still counts.
2. ``get_view`` reads each one's champion in exactly this session's scope.

Only exact scopes are read: the KB Store does not document whether ``get_rollup`` without a scope lists a record's
other scopes, so finding the nearest concurrency and sequence lengths waits until it does.

Every record goes through ``knowledge_to_warm_recipe``, so records the single-lineage CLOSE wrote read the same way.
A lead names the dimensions where it differs from this session. It is a lead, not a starting point: specialists still
measure it on this session's root.

PRELUDE reads the KB once per session, and the KB Store answers a server error now and then, so each read is asked
again after a server error or a dropped connection, waiting :data:`KB_RETRY_SECONDS`; a refusal is not asked again.

:func:`replay_leads` gives the first leads a :class:`PreparedReplay`: the champion's recipe, patches and kernel
artifacts too, their files downloaded from the KB into the session's blob store. A champion built on another image, or
one whose files the KB does not hold, replays its launch config only. The specialist's ``notebook_replay`` makes the
change for its own branch head, since a gain lands only on the head it was measured against.

A replay leaves the flags of :data:`HEAD_FLAGS` to the head it is made for: the champion's context length and
scheduler watchdog are dropped, and the head's, if it sets them, kept. A record can carry the ones the launcher set for
the session that measured it, and the launcher sets both again for this one when a recipe leaves them out. The cap a
champion was measured with is no optimization, and one below what this session's accuracy evaluation runs in is refused
as ``context_below_eval``. A lead's launch arguments are listed without them for the same reason.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

from hyperloom.contracts.recipe import (
    ConfigDelta,
    LaunchConfig,
    Recipe,
    RecipeDelta,
    RecipeError,
    delta_between,
    diff,
)
from hyperloom.inference_optimizer.grid_server_args import (
    merge_server_args,
    remove_server_args,
    tokenize_server_args_preserving_json,
)
from hyperloom.inference_optimizer.recipe_snapshot_constants import F_LABEL_FRAMEWORK_VERSION, canonical_labels
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
from hyperloom_multibranch.notebook.knowledge.lessons import listed
from hyperloom_multibranch.notebook.knowledge.recipe_kb import from_recipe_kb_value, overlay_ref
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import recipe_identity

log = logging.getLogger(__name__)

MAX_CANDIDATES = 20
PAGE_SIZE = 50
#: The flags a replay leaves to the head: the context length a server serves (``--context-length`` on SGLang,
#: ``--max-model-len`` on vLLM), and SGLang's scheduler watchdog.
HEAD_FLAGS = (*CONTEXT_FLAGS, "--watchdog-timeout")
#: How many of the best leads get a replay; each whole one downloads the champion's files.
REPLAY_LEADS = 3
#: The waits before each new ask of a read that failed transiently; one more ask follows the last wait.
KB_RETRY_SECONDS = (2.0, 5.0, 15.0)
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
class PreparedReplay:
    """A champion's recipe ready to replay on any head of the session.

    ``recipe`` is its whole recipe, its patch and kernel files already in the session's blob store; ``None`` when only
    its launch config ``launch`` replays, ``reason`` saying why.
    """

    canonical_id: str
    recipe: Recipe | None
    launch: LaunchConfig
    reason: str = ""

    def on(self, head: Recipe, parent_commit: str) -> WarmReplay | None:
        """The change that makes ``head``, the recipe of ``parent_commit``, into the champion's; ``None`` when ``head``
        is the champion's already (:meth:`is_on`).

        A component the head has keeps the head's upstream repository and commit, as a merge keeps it: the session
        builds it from its image, so the champion's upstream label names nothing to rebuild. The head keeps its flags of
        :data:`HEAD_FLAGS` as well.
        """

        if self.is_on(head):
            return None
        delta = delta_between(head, self._wanted(head))
        return WarmReplay(self.canonical_id, parent_commit, delta, delta_layer(delta), self.reason)

    def is_on(self, recipe: Recipe) -> bool:
        """Whether ``recipe`` is the champion's already: the same components and kernel artifacts, and the same server
        arguments and environment whatever their order, its flags of :data:`HEAD_FLAGS` aside."""

        wanted = self._wanted(recipe)
        change = diff(recipe, wanted)
        rest = replace(delta_between(recipe, wanted), config=ConfigDelta())
        return not (change.args or change.envs) and rest == RecipeDelta()

    def describe(self) -> str:
        what = "the whole recipe" if self.recipe is not None else f"its launch config only, {self.reason}"
        return f"replay on your head with {REPLAY_TOOL}: {what}"

    def _wanted(self, head: Recipe) -> Recipe:
        champion = self.recipe if self.recipe is not None else replace(head, launch=self.launch)
        args = _with_head_flags(champion.launch.server_args, head.launch.server_args)
        return replace(_on_head_upstreams(champion, head), launch=replace(champion.launch, server_args=args))


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

    def summary(self) -> str:
        """The champion: its id, where its scope differs from this session's, its gain, launch arguments and session."""

        gain = "n/a" if self.gain_pct is None else f"{self.gain_pct:+.2f}%"
        where = "".join(f", {key} {theirs} (here {ours})" for key, (theirs, ours) in self.differing.items())
        args = f" `{self.server_args}`" if self.server_args else ""
        return f"{self.canonical_id}{where}: {gain}{args} [{self.session_id}]"

    def describe(self) -> str:
        replay = f"; {self.replay.describe()}" if self.replay is not None else ""
        return f"{self.summary()}{replay}"

    def render(self) -> str:
        return f"- {self.describe()}"


def identity_labels(profile: SessionProfile) -> dict[str, str]:
    """The session's seven serving dimensions, as the KB labels them."""

    return canonical_labels(
        model=profile.model,
        hardware=profile.hardware,
        framework_name=profile.framework,
        model_type=profile.model_type,
        architectures=list(profile.architectures),
        framework_version=profile.framework_version,
        precision=profile.precision,
    )


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

    None when the session's scope is not one the KB files.
    """

    try:
        _, scope = recipe_identity(profile)
    except RemoteRecipeValidationError:
        return ()
    labels = identity_labels(profile)
    match = {key: value for key, value in labels.items() if key != F_LABEL_FRAMEWORK_VERSION}
    found = (
        _read_lead(reader, item, scope, labels) for item in candidates(reader, scope.identity_scheme, match, limit)
    )
    leads = [lead for lead in found if lead is not None]
    return tuple(sorted(leads, key=lambda lead: (bool(lead.differing), -(lead.gain_pct or 0.0))))


def replay(client: RecipeDownloader, lead: WarmLead, scope: RecipeScope, target: ReplayTarget) -> PreparedReplay | None:
    """``lead``'s champion ready to replay on any head of the session; ``None`` when it changes nothing on the root.

    The whole recipe needs the record's files, a base image and runtime snapshot the same as the root's, kernel
    artifacts of kinds measurement loads, and components the root takes from the same upstream repositories. Its files
    go into the session's blob store, each checked against the hash the recipe pins. Otherwise only the launch config
    replays, and the replay says why.
    """

    try:
        launch = LaunchConfig(server_args=lead.server_args, envs=dict(lead.envs))
    except RecipeError as exc:
        log.warning("Recipe KB champion %s has no launch config to replay: %s", lead.canonical_id, exc)
        return None
    config_only = PreparedReplay(lead.canonical_id, None, launch, "the record holds no patch or kernel files")
    if lead.replayable:
        try:
            whole = PreparedReplay(lead.canonical_id, _champion(client, lead, scope, target), launch)
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


def _champion(client: RecipeDownloader, lead: WarmLead, scope: RecipeScope, target: ReplayTarget) -> Recipe:
    """The champion's recipe on this session's image, its files put into the session's blob store."""

    envelope = _asked(partial(client.get_view, lead.canonical_id, scope))
    if envelope is None:
        raise RecipeError("the record is gone")
    champion = _recorded_recipe(envelope)
    if (champion.base_image_digest, champion.runtime_snapshot) != (
        target.root.base_image_digest,
        target.root.runtime_snapshot,
    ):
        log.info("Recipe KB champion %s was built on %s", lead.canonical_id, champion.base_image_digest)
        raise RecipeError("it was built on another image than this session's")
    unloadable = sorted({item.kind.value for item in champion.kernels if item.kind not in LOADABLE_KERNEL_KINDS})
    if unloadable:
        raise RecipeError(f"measurement loads no {', '.join(unloadable)} kernel artifacts yet")
    destination = target.workdir / _UNSAFE_NAME.sub("-", lead.canonical_id)
    _asked(partial(client.read, lead.canonical_id, destination, scope=scope, envelope=envelope))
    files = destination / "files"
    for stack, component in enumerate(champion.components):
        for member, patch in enumerate(component.patches):
            _keep(target.blobs, files / overlay_ref(stack, member, patch.path), patch.sha256)
    for artifact in champion.kernels:
        _keep(target.blobs, files / artifact.ref, artifact.sha256)
    return champion


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


def _keep(blobs: BlobStore, path: Path, sha256: str) -> None:
    found = blobs.put(path.read_bytes())
    if found != sha256:
        raise RecipeError(f"{path.name} has sha256 {found}; the record pins {sha256}")


def delta_layer(delta: RecipeDelta) -> IntentLayer:
    """The deepest layer ``delta`` reaches, the one a replay of it claims."""

    if delta.add_kernels or delta.remove_kernels:
        return IntentLayer.KERNEL
    return IntentLayer.SOURCE if delta.components else IntentLayer.CONFIG


__all__ = [
    "HEAD_FLAGS",
    "KB_RETRY_SECONDS",
    "MAX_CANDIDATES",
    "REPLAY_LEADS",
    "PreparedReplay",
    "RecipeDownloader",
    "RecipeReader",
    "ReplayTarget",
    "WarmLead",
    "candidates",
    "delta_layer",
    "identity_labels",
    "lead_of",
    "replay",
    "replay_leads",
    "warm_leads",
]
