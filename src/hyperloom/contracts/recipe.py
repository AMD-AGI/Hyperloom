# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declarative recipe a serving environment is rebuilt from, and the changes made to it.

Each commit of a multi-branch experiment repository stores one :class:`Recipe`; hosted measurement compiles a test
environment from a recipe and the :class:`RecipeDelta` a test request carries. A recipe has the three parts of the
Recipe KB columns, so a final recipe is written back without translation:

* ``launch``: server arguments and extra environment variables (the ``config`` column);
* ``components``: per component, an ordered patch stack against a fixed upstream commit (the ``patch`` column);
* ``kernels``: the kernel artifacts it loads (the ``kernel`` column).

``base_image_digest`` names the image everything starts from. ``runtime_snapshot``, when set, names the image the
enablement flow built on it, with its setup commands and component builds done, so a recipe never replays them.
``unexpressed`` names what the measured environment had that the recipe cannot say (a kept patch outside a component
checkout, setup commands, ...); a recipe that names anything there does not rebuild what was measured.

The configuration part of a delta has the ``recipe_delta`` shape the single-lineage flow publishes, and :func:`apply`
composes server arguments with the same functions that flow launches with.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import takewhile
from pathlib import PurePosixPath

from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
    dedupe_extra_server_args,
    removal_option_names,
    tokenize_server_args_preserving_json,
    validate_server_args_shell_safe,
)

RECIPE_SCHEMA_VERSION = 1
PATCH_ROOT = "patch"
KERNEL_ROOT = "kernel"
DIGEST_PREFIX = "sha256:"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_IMAGE = re.compile(r"(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Framework arguments travel in the server arguments; the launcher derives these variables from them.
_FRAMEWORK_ARGS_ENV = re.compile(r"EXTRA_\w*_ARGS")
_FLAG = "--"
_OPTION = re.compile(r"--[A-Za-z0-9][\w.-]*")
_REMOVE_ARGS = "remove_args"
_LAUNCH_REMOVE_ARGS = "launch_remove_args"
_UNSET_ENVS = "unset_envs"
#: How a recipe diff names an option a launch removes from the resolved launch.
REMOVED = "<removed>"
_COMPONENTS = "components"
#: What a recipe may name as left out of it; an entry is ``<kind>`` or ``<kind>:<detail>``.
UNEXPRESSED_KINDS = frozenset(
    {
        "patch",
        "setup_commands",
        "kept_artifacts",
        "active_runtime",
        "environment_closure",
        _REMOVE_ARGS,
        _UNSET_ENVS,
        "args_mode",
    }
)
_UNEXPRESSED = re.compile(r"(?P<kind>[a-z_]+)(?::\S+)?")


class RecipeError(ValueError):
    """A recipe, a delta or a recipe document breaks the contract."""


class ArgsMode(str, Enum):
    """How a delta's server arguments combine with the recipe's, as the single-lineage ``args_mode`` does."""

    APPEND = "append"
    REPLACE = "replace"


class KernelKind(str, Enum):
    """The sub-column of the Recipe KB ``kernel`` column an artifact belongs to."""

    GEMM = "gemm"
    FUSION = "fusion"
    REWRITE = "rewrite"


# --- validation --------------------------------------------------------------------------------------------------


def _matching(pattern: re.Pattern[str], value: object, what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise RecipeError(f"{what} must match {pattern.pattern}, got {value!r}")
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RecipeError(f"{what} must be a non-empty string without surrounding whitespace, got {value!r}")
    return value


def _path(value: object, root: str) -> str:
    path = _text(value, f"a path under {root}/")
    parts = PurePosixPath(path).parts
    if "\\" in path or str(PurePosixPath(path)) != path or len(parts) < 2 or parts[0] != root or ".." in parts:
        raise RecipeError(f"{path!r} is not a relative path under {root}/")
    return path


def _server_args(value: object) -> str:
    if not isinstance(value, str):
        raise RecipeError(f"server arguments must be a string, got {value!r}")
    try:
        args = validate_server_args_shell_safe(value)
    except ValueError as exc:
        raise RecipeError(f"server arguments {value!r}: {exc}") from exc
    if args != value or tokenize_server_args_preserving_json(args) is None:
        raise RecipeError(f"server arguments must be trimmed and split into whole arguments, got {value!r}")
    return args


def _env_names(names: Iterable[object]) -> None:
    for name in names:
        _matching(_ENV_NAME, name, "an environment variable name")
        if _FRAMEWORK_ARGS_ENV.fullmatch(str(name)):
            raise RecipeError(f"{name} carries framework arguments; put them in the server arguments")


def _string_map(value: object, what: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise RecipeError(f"{what} must map strings to strings, got {value!r}")
    return value


def _strings(value: object, what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not all(isinstance(item, str) for item in value):
        raise RecipeError(f"{what} must be a list of strings, got {value!r}")
    return tuple(value)


def _option_names(names: Sequence[str], what: str) -> tuple[str, ...]:
    """``names`` as bare long options, sorted and each once, so one launch has one digest."""

    for name in names:
        if not isinstance(name, str) or not _OPTION.fullmatch(name):
            raise RecipeError(f"{what} must be bare option names (--name), got {name!r}")
    if list(names) != sorted(set(names)):
        raise RecipeError(f"{what} must be sorted and unique, got {list(names)!r}")
    return tuple(names)


def _items(value: object, what: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecipeError(f"{what} must be a list, got {value!r}")
    return tuple(value)


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecipeError(f"{what} must be an object, got {value!r}")
    return value


def _known(value: object, keys: tuple[str, ...], what: str) -> Mapping[str, object]:
    """``value`` as a mapping of ``keys`` only: a key a writer misspelt would otherwise drop its change unseen."""

    fields = _mapping(value, what)
    unknown = sorted(str(key) for key in fields if key not in keys)
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        named = ", ".join(repr(key) for key in unknown)
        raise RecipeError(f"unknown key{plural} {named} in {what}; known: {', '.join(keys)}")
    return fields


def _unique(values: Sequence[object], what: str) -> None:
    duplicates = sorted({str(value) for value in values if values.count(value) > 1})
    if duplicates:
        raise RecipeError(f"duplicate {what}: {', '.join(duplicates)}")


def _unexpressed(entries: tuple[str, ...]) -> None:
    for entry in entries:
        match = _UNEXPRESSED.fullmatch(entry) if isinstance(entry, str) else None
        if match is None or match["kind"] not in UNEXPRESSED_KINDS:
            raise RecipeError(
                f"an unexpressed entry is <kind> or <kind>:<detail>, kind one of "
                f"{sorted(UNEXPRESSED_KINDS)}, got {entry!r}"
            )
    _unique(entries, "unexpressed entries")
    if list(entries) != sorted(entries):
        raise RecipeError(f"unexpressed entries must be sorted, got {list(entries)}")


def _canonical(document: Mapping[str, object]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_object(value: object, what: str) -> Mapping[str, object]:
    document = _mapping(value, what)
    try:
        _canonical(document)
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"{what} is not a JSON object: {exc}") from exc
    return document


def _digest(document: Mapping[str, object]) -> str:
    return DIGEST_PREFIX + hashlib.sha256(_canonical(document).encode()).hexdigest()


# --- the recipe --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchConfig:
    """Server arguments in the shell form the launcher takes, the extra environment variables, and ``remove_args``:
    the long options, by name, removed from the launch the framework resolves before these arguments are added (a
    Magpie AgentX recipe's, from InferenceX), so a change that turns off an option of that launch stays off. Its
    document carries ``remove_args`` only when there are any, so a launch without them keeps its digest."""

    server_args: str = ""
    envs: Mapping[str, str] = field(default_factory=dict)
    remove_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _server_args(self.server_args)
        _env_names(_string_map(self.envs, "environment variables"))
        object.__setattr__(self, "remove_args", _option_names(self.remove_args, "a launch's remove_args"))
        both = sorted(set(self.remove_args) & set(_flag_values(self.server_args)))
        if both:
            raise RecipeError(f"a launch both sets and removes {', '.join(both)}")

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "extra_server_args": self.server_args,
            "extra_envs": dict(sorted(self.envs.items())),
        }
        if self.remove_args:
            document[_REMOVE_ARGS] = list(self.remove_args)
        return document

    @classmethod
    def from_document(cls, document: object) -> LaunchConfig:
        fields = _known(document, ("extra_server_args", "extra_envs", _REMOVE_ARGS), "a launch config")
        return cls(
            server_args=fields.get("extra_server_args", ""),
            envs=fields.get("extra_envs", {}),
            remove_args=_strings(fields.get(_REMOVE_ARGS, ()), _REMOVE_ARGS),
        )


@dataclass(frozen=True)
class PatchRef:
    """One patch of a component's stack: the file in the recipe tree and its content hash."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _path(self.path, PATCH_ROOT)
        _matching(_SHA256, self.sha256, f"the sha256 of {self.path}")

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_document(cls, document: object) -> PatchRef:
        fields = _known(document, ("path", "sha256"), "a patch")
        return cls(path=fields.get("path"), sha256=fields.get("sha256"))


@dataclass(frozen=True)
class ComponentPatches:
    """One component pinned to a fixed upstream commit, and the patches applied on it, in order."""

    component: str
    upstream_repo: str
    upstream_ref: str
    patches: tuple[PatchRef, ...] = ()

    def __post_init__(self) -> None:
        _matching(_COMPONENT, self.component, "a component name")
        _text(self.upstream_repo, f"the upstream repository of {self.component}")
        _matching(_COMMIT, self.upstream_ref, f"the upstream commit of {self.component}")
        _unique([patch.path for patch in self.patches], f"patches of {self.component}")

    def to_document(self) -> dict[str, object]:
        return {
            "component": self.component,
            "upstream_repo": self.upstream_repo,
            "upstream_ref": self.upstream_ref,
            "patches": [patch.to_document() for patch in self.patches],
        }

    @classmethod
    def from_document(cls, document: object) -> ComponentPatches:
        fields = _mapping(document, "a component")
        return cls(
            component=fields.get("component"),
            upstream_repo=fields.get("upstream_repo"),
            upstream_ref=fields.get("upstream_ref"),
            patches=tuple(map(PatchRef.from_document, _items(fields.get("patches", ()), "patches"))),
        )


@dataclass(frozen=True)
class KernelArtifact:
    """A kernel artifact the recipe loads; ``target`` (the operator or shape it serves) is unique per kind.

    ``metadata`` is a free JSON object; its ``load`` entry says how a test loads the artifact.
    ``{"load": {"via": "env_file", "env": "<ENV_VAR_NAME>"}}`` attaches the file to the test and sets the variable to
    its path in the sandbox; the toolbox refuses a ``via`` it does not know. ``metadata`` is part of the recipe
    document, so the load method counts in ``Recipe.digest()`` and ``RecipeDelta.fingerprint()``: the same file loaded
    another way is another artifact.
    """

    kind: KernelKind
    target: str
    ref: str
    sha256: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, KernelKind):
            raise RecipeError(f"a kernel kind must be one of {[kind.value for kind in KernelKind]}, got {self.kind!r}")
        _text(self.target, "a kernel target")
        _path(self.ref, KERNEL_ROOT)
        _matching(_SHA256, self.sha256, f"the sha256 of {self.ref}")
        _json_object(self.metadata, f"the metadata of {self.ref}")

    @property
    def slot(self) -> tuple[KernelKind, str]:
        """What a newer artifact replaces: the one of the same kind serving the same target."""

        return self.kind, self.target

    def to_document(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "target": self.target,
            "ref": self.ref,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_document(cls, document: object) -> KernelArtifact:
        fields = _known(document, ("kind", "target", "ref", "sha256", "metadata"), "a kernel artifact")
        try:
            kind = KernelKind(fields.get("kind"))
        except ValueError as exc:
            raise RecipeError(f"unknown kernel kind {fields.get('kind')!r}") from exc
        return cls(
            kind=kind,
            target=fields.get("target"),
            ref=fields.get("ref"),
            sha256=fields.get("sha256"),
            metadata=fields.get("metadata", {}),
        )


@dataclass(frozen=True)
class Recipe:
    """Everything one serving environment is rebuilt from; :meth:`digest` is its identity."""

    base_image_digest: str
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    components: tuple[ComponentPatches, ...] = ()
    kernels: tuple[KernelArtifact, ...] = ()
    runtime_snapshot: str | None = None
    schema_version: int = RECIPE_SCHEMA_VERSION
    unexpressed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _matching(_IMAGE, self.base_image_digest, "the base image digest")
        if self.runtime_snapshot is not None:
            _matching(_IMAGE, self.runtime_snapshot, "the runtime snapshot")
        if isinstance(self.schema_version, bool) or self.schema_version != RECIPE_SCHEMA_VERSION:
            raise RecipeError(f"recipe schema version {self.schema_version!r} is not {RECIPE_SCHEMA_VERSION}")
        _unique([component.component for component in self.components], "component names")
        _unique([artifact.slot for artifact in self.kernels], "kernel targets")
        _unique([artifact.ref for artifact in self.kernels], "kernel artifacts")
        _unexpressed(self.unexpressed)

    @property
    def complete(self) -> bool:
        """Whether the recipe rebuilds everything that was measured: it names nothing as unexpressed."""

        return not self.unexpressed

    def component(self, name: str) -> ComponentPatches | None:
        return next((component for component in self.components if component.component == name), None)

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "base_image_digest": self.base_image_digest,
            "runtime_snapshot": self.runtime_snapshot,
            "launch": self.launch.to_document(),
            _COMPONENTS: [component.to_document() for component in self.components],
            "kernels": [artifact.to_document() for artifact in self.kernels],
        }
        if self.unexpressed:
            document["unexpressed"] = list(self.unexpressed)
        return document

    @classmethod
    def from_document(cls, document: object) -> Recipe:
        fields = _mapping(document, "a recipe")
        return cls(
            schema_version=fields.get("schema_version"),
            base_image_digest=fields.get("base_image_digest"),
            runtime_snapshot=fields.get("runtime_snapshot"),
            launch=LaunchConfig.from_document(fields.get("launch", {})),
            components=tuple(map(ComponentPatches.from_document, _items(fields.get(_COMPONENTS, ()), _COMPONENTS))),
            kernels=tuple(map(KernelArtifact.from_document, _items(fields.get("kernels", ()), "kernels"))),
            unexpressed=_strings(fields.get("unexpressed", ()), "unexpressed"),
        )

    def digest(self) -> str:
        return _digest(self.to_document())


def with_unexpressed(recipe: Recipe, entries: Iterable[str]) -> Recipe:
    """``recipe`` that also names ``entries`` as unexpressed; a merge passes the other side's entries."""

    return replace(recipe, unexpressed=tuple(sorted({*recipe.unexpressed, *entries})))


# --- changes -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigDelta:
    """A change to the launch config; its document is the single-lineage ``recipe_delta``.

    ``launch_remove_args``, when not ``None``, is the launch's ``remove_args`` after the change, as a whole: how a delta
    stops removing an option of the resolved launch, which a removal cannot say. Its document carries the field only
    then, so every other delta keeps its fingerprint."""

    server_args: str = ""
    args_mode: ArgsMode = ArgsMode.APPEND
    remove_args: tuple[str, ...] = ()
    envs: Mapping[str, str] = field(default_factory=dict)
    unset_envs: tuple[str, ...] = ()
    launch_remove_args: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _server_args(self.server_args)
        if not isinstance(self.args_mode, ArgsMode):
            raise RecipeError(f"args_mode must be one of {[mode.value for mode in ArgsMode]}, got {self.args_mode!r}")
        for spec in self.remove_args:
            if not _server_args(spec).startswith(_FLAG):
                raise RecipeError(f"a removed argument must name a flag, got {spec!r}")
        _env_names([*_string_map(self.envs, "environment variables"), *self.unset_envs])
        _unique([*self.envs, *self.unset_envs], "environment variables set and unset at once")
        if self.launch_remove_args is not None:
            names = _option_names(self.launch_remove_args, _LAUNCH_REMOVE_ARGS)
            object.__setattr__(self, _LAUNCH_REMOVE_ARGS, names)

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "extra_server_args": self.server_args,
            "args_mode": self.args_mode.value,
            _REMOVE_ARGS: list(self.remove_args),
            "extra_envs": dict(sorted(self.envs.items())),
            _UNSET_ENVS: list(self.unset_envs),
        }
        if self.launch_remove_args is not None:
            document[_LAUNCH_REMOVE_ARGS] = list(self.launch_remove_args)
        return document

    @classmethod
    def from_document(cls, document: object) -> ConfigDelta:
        keys = ("extra_server_args", "args_mode", _REMOVE_ARGS, "extra_envs", _UNSET_ENVS, _LAUNCH_REMOVE_ARGS)
        fields = _known(document, keys, "a config delta")
        try:
            mode = ArgsMode(fields.get("args_mode", ArgsMode.APPEND.value))
        except ValueError as exc:
            raise RecipeError(f"unknown args_mode {fields.get('args_mode')!r}") from exc
        launch_removals = fields.get(_LAUNCH_REMOVE_ARGS)
        return cls(
            server_args=fields.get("extra_server_args", ""),
            args_mode=mode,
            remove_args=_strings(fields.get(_REMOVE_ARGS, ()), _REMOVE_ARGS),
            envs=fields.get("extra_envs", {}),
            unset_envs=_strings(fields.get(_UNSET_ENVS, ()), _UNSET_ENVS),
            launch_remove_args=None if launch_removals is None else _strings(launch_removals, _LAUNCH_REMOVE_ARGS),
        )


@dataclass(frozen=True)
class ComponentChange:
    """A change to one component: move it to another upstream commit, add or drop patches, or add it.

    ``upstream_repo`` is given only for a component the recipe does not have yet.
    """

    component: str
    upstream_ref: str | None = None
    upstream_repo: str | None = None
    add_patches: tuple[PatchRef, ...] = ()
    remove_patches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _matching(_COMPONENT, self.component, "a component name")
        if self.upstream_ref is not None:
            _matching(_COMMIT, self.upstream_ref, f"the upstream commit of {self.component}")
        if self.upstream_repo is not None:
            _text(self.upstream_repo, f"the upstream repository of {self.component}")
        for path in self.remove_patches:
            _path(path, PATCH_ROOT)
        _unique([patch.path for patch in self.add_patches], f"patches added to {self.component}")

    def to_document(self) -> dict[str, object]:
        return {
            "component": self.component,
            "upstream_ref": self.upstream_ref,
            "upstream_repo": self.upstream_repo,
            "add_patches": [patch.to_document() for patch in self.add_patches],
            "remove_patches": list(self.remove_patches),
        }

    @classmethod
    def from_document(cls, document: object) -> ComponentChange:
        keys = ("component", "upstream_ref", "upstream_repo", "add_patches", "remove_patches")
        fields = _known(document, keys, "a component change")
        return cls(
            component=fields.get("component"),
            upstream_ref=fields.get("upstream_ref"),
            upstream_repo=fields.get("upstream_repo"),
            add_patches=tuple(map(PatchRef.from_document, _items(fields.get("add_patches", ()), "add_patches"))),
            remove_patches=_strings(fields.get("remove_patches", ()), "remove_patches"),
        )


@dataclass(frozen=True)
class RecipeDelta:
    """One change to a recipe: the action a test request carries; :meth:`fingerprint` identifies it."""

    config: ConfigDelta = field(default_factory=ConfigDelta)
    components: tuple[ComponentChange, ...] = ()
    add_kernels: tuple[KernelArtifact, ...] = ()
    remove_kernels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _unique([change.component for change in self.components], "component changes")
        _unique([artifact.slot for artifact in self.add_kernels], "added kernel targets")
        for ref in self.remove_kernels:
            _path(ref, KERNEL_ROOT)

    def to_document(self) -> dict[str, object]:
        return {
            "config": self.config.to_document(),
            _COMPONENTS: [change.to_document() for change in self.components],
            "add_kernels": [artifact.to_document() for artifact in self.add_kernels],
            "remove_kernels": list(self.remove_kernels),
        }

    @classmethod
    def from_document(cls, document: object) -> RecipeDelta:
        fields = _known(document, ("config", _COMPONENTS, "add_kernels", "remove_kernels"), "a recipe delta")
        return cls(
            config=ConfigDelta.from_document(fields.get("config", {})),
            components=tuple(map(ComponentChange.from_document, _items(fields.get(_COMPONENTS, ()), _COMPONENTS))),
            add_kernels=tuple(map(KernelArtifact.from_document, _items(fields.get("add_kernels", ()), "add_kernels"))),
            remove_kernels=_strings(fields.get("remove_kernels", ()), "remove_kernels"),
        )

    def fingerprint(self) -> str:
        """The action fingerprint: the same change gets the same fingerprint wherever it is made."""

        return _digest(self.to_document())


def apply(recipe: Recipe, delta: RecipeDelta) -> Recipe:
    """The recipe ``delta`` makes of ``recipe``; raises :class:`RecipeError` when the change does not fit it.

    Whether a patch applies to the component's source is checked when the environment is compiled, not here.
    """

    return replace(
        recipe,
        launch=_apply_config(recipe.launch, delta.config),
        components=_apply_components(recipe.components, delta.components),
        kernels=_apply_kernels(recipe.kernels, delta),
    )


def _apply_config(launch: LaunchConfig, delta: ConfigDelta) -> LaunchConfig:
    """``launch`` changed by ``delta``. A removal of an option the launch's own arguments lack is one of the resolved
    launch's, kept in ``remove_args``, unless the delta gives ``launch_remove_args`` whole; an option the new arguments
    set again leaves it."""

    args = dedupe_extra_server_args(
        compose_server_args(
            base_extra_args=launch.server_args,
            variant_extra_args=delta.server_args,
            remove_args=list(delta.remove_args),
            args_mode=delta.args_mode.value,
        )
    )
    envs = {name: value for name, value in launch.envs.items() if name not in delta.unset_envs}
    envs.update(delta.envs)
    if delta.launch_remove_args is not None:
        removed = set(delta.launch_remove_args)
    else:
        own = _flag_values(launch.server_args)
        resolved = {name for name in removal_option_names(delta.remove_args) if name not in own}
        removed = set(launch.remove_args) | resolved
    return LaunchConfig(server_args=args, envs=envs, remove_args=tuple(sorted(removed - set(_flag_values(args)))))


def _apply_components(
    components: tuple[ComponentPatches, ...], changes: tuple[ComponentChange, ...]
) -> tuple[ComponentPatches, ...]:
    by_name = {component.component: component for component in components}
    for change in changes:
        by_name[change.component] = _apply_component(by_name.get(change.component), change)
    return tuple(by_name.values())


def _apply_component(current: ComponentPatches | None, change: ComponentChange) -> ComponentPatches:
    if current is None:
        if change.upstream_repo is None or change.upstream_ref is None:
            raise RecipeError(f"{change.component} is a new component; give its upstream repository and commit")
        current = ComponentPatches(change.component, change.upstream_repo, change.upstream_ref)
    elif change.upstream_repo not in (None, current.upstream_repo):
        raise RecipeError(f"{change.component} already comes from {current.upstream_repo}")
    missing = sorted(set(change.remove_patches) - {patch.path for patch in current.patches})
    if missing:
        raise RecipeError(f"{change.component} has no patches {', '.join(missing)} to remove")
    kept = tuple(patch for patch in current.patches if patch.path not in change.remove_patches)
    clashing = sorted({patch.path for patch in kept} & {patch.path for patch in change.add_patches})
    if clashing:
        raise RecipeError(f"{change.component} already has patches {', '.join(clashing)}")
    return replace(
        current,
        upstream_ref=change.upstream_ref or current.upstream_ref,
        patches=(*kept, *change.add_patches),
    )


def _apply_kernels(kernels: tuple[KernelArtifact, ...], delta: RecipeDelta) -> tuple[KernelArtifact, ...]:
    missing = sorted(set(delta.remove_kernels) - {artifact.ref for artifact in kernels})
    if missing:
        raise RecipeError(f"the recipe loads no kernel artifacts {', '.join(missing)} to remove")
    slots = {artifact.slot: artifact for artifact in kernels if artifact.ref not in delta.remove_kernels}
    for artifact in delta.add_kernels:
        slots[artifact.slot] = artifact
    return tuple(slots.values())


def delta_between(first: Recipe, second: Recipe) -> RecipeDelta:
    """The change that makes ``first`` into ``second`` on ``first``'s image.

    ``apply(first, delta)`` launches with ``second``'s server arguments and environment and loads ``second``'s kernel
    artifacts. Each of ``second``'s components carries ``second``'s upstream commit and patch stack; a component only
    ``first`` has keeps its upstream commit, without patches. Raises :class:`RecipeError` for a component the two take
    from different upstream repositories, which no change can move.
    """

    changes = [_component_delta(first.component(wanted.component), wanted) for wanted in second.components]
    changes += [
        ComponentChange(component.component, remove_patches=tuple(patch.path for patch in component.patches))
        for component in first.components
        if second.component(component.component) is None and component.patches
    ]
    return RecipeDelta(
        config=_config_delta(first.launch, second.launch),
        components=tuple(change for change in changes if change is not None),
        add_kernels=tuple(artifact for artifact in second.kernels if artifact not in first.kernels),
        remove_kernels=tuple(
            sorted({artifact.ref for artifact in first.kernels} - {artifact.ref for artifact in second.kernels})
        ),
    )


def _config_delta(first: LaunchConfig, second: LaunchConfig) -> ConfigDelta:
    """The change from ``first`` to ``second``: each option ``second`` newly removes from the resolved launch is
    removed again, so :func:`apply` remembers it. Where that does not make ``second``'s removals -- one it stops
    removing without setting it, or one ``first``'s own arguments set -- the delta gives them whole."""

    changed_args = first.server_args != second.server_args
    dropped = set(_flag_values(first.server_args)) - set(_flag_values(second.server_args)) - {""}
    removals = (dropped if changed_args else set()) | (set(second.remove_args) - set(first.remove_args))
    delta = ConfigDelta(
        server_args=second.server_args if changed_args else "",
        remove_args=tuple(sorted(removals)),
        envs={name: value for name, value in second.envs.items() if first.envs.get(name) != value},
        unset_envs=tuple(sorted(set(first.envs) - set(second.envs))),
    )
    if _apply_config(first, delta).remove_args == second.remove_args:
        return delta
    return replace(delta, launch_remove_args=second.remove_args)


def _component_delta(current: ComponentPatches | None, wanted: ComponentPatches) -> ComponentChange | None:
    if current is None:
        return ComponentChange(wanted.component, wanted.upstream_ref, wanted.upstream_repo, wanted.patches)
    if current.upstream_repo != wanted.upstream_repo:
        raise RecipeError(
            f"{wanted.component} comes from {current.upstream_repo} in one recipe and {wanted.upstream_repo} in the other"
        )
    shared = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(current.patches, wanted.patches)))
    if current.upstream_ref == wanted.upstream_ref and shared == len(current.patches) == len(wanted.patches):
        return None
    return ComponentChange(
        wanted.component,
        upstream_ref=wanted.upstream_ref if wanted.upstream_ref != current.upstream_ref else None,
        add_patches=wanted.patches[shared:],
        remove_patches=tuple(patch.path for patch in current.patches[shared:]),
    )


# --- differences -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueChange:
    """A server argument or environment variable that differs; ``None`` is a value one side does not set.

    A flag without a value has the value ``""``.
    """

    name: str
    before: str | None
    after: str | None

    @property
    def conflicting(self) -> bool:
        """Both sides set it, to different values: what a merge has to choose between."""

        return self.before is not None and self.after is not None


@dataclass(frozen=True)
class RefChange:
    """A component pinned to different upstream commits; ``None`` is a side that lacks the component."""

    component: str
    before: str | None
    after: str | None


@dataclass(frozen=True)
class RecipeDiff:
    """How a second recipe differs from a first; patches and kernel artifacts compare by content hash."""

    args: tuple[ValueChange, ...]
    envs: tuple[ValueChange, ...]
    refs: tuple[RefChange, ...]
    shared_patches: frozenset[str]
    patches_only_in_first: frozenset[str]
    patches_only_in_second: frozenset[str]
    kernels_only_in_first: frozenset[str]
    kernels_only_in_second: frozenset[str]
    base_image_changed: bool
    unexpressed_only_in_first: frozenset[str] = frozenset()
    unexpressed_only_in_second: frozenset[str] = frozenset()

    @property
    def config_conflicts(self) -> tuple[ValueChange, ...]:
        return tuple(change for change in (*self.args, *self.envs) if change.conflicting)

    @property
    def patch_overlap(self) -> float:
        """Jaccard similarity of the two patch sets; ``1.0`` when neither recipe has patches."""

        union = len(self.shared_patches) + len(self.patches_only_in_first) + len(self.patches_only_in_second)
        return len(self.shared_patches) / union if union else 1.0

    @property
    def is_empty(self) -> bool:
        return not (
            self.args
            or self.envs
            or self.refs
            or self.patches_only_in_first
            or self.patches_only_in_second
            or self.kernels_only_in_first
            or self.kernels_only_in_second
            or self.base_image_changed
            or self.unexpressed_only_in_first
            or self.unexpressed_only_in_second
        )


def diff(first: Recipe, second: Recipe) -> RecipeDiff:
    """What changes from ``first`` to ``second``, for similarity checks and merges."""

    first_patches, second_patches = _patch_hashes(first), _patch_hashes(second)
    first_kernels = {artifact.sha256 for artifact in first.kernels}
    second_kernels = {artifact.sha256 for artifact in second.kernels}
    return RecipeDiff(
        args=_value_changes(_launch_values(first.launch), _launch_values(second.launch)),
        envs=_value_changes(first.launch.envs, second.launch.envs),
        refs=_ref_changes(first, second),
        shared_patches=frozenset(first_patches & second_patches),
        patches_only_in_first=frozenset(first_patches - second_patches),
        patches_only_in_second=frozenset(second_patches - first_patches),
        kernels_only_in_first=frozenset(first_kernels - second_kernels),
        kernels_only_in_second=frozenset(second_kernels - first_kernels),
        base_image_changed=(first.base_image_digest, first.runtime_snapshot)
        != (second.base_image_digest, second.runtime_snapshot),
        unexpressed_only_in_first=frozenset(first.unexpressed) - frozenset(second.unexpressed),
        unexpressed_only_in_second=frozenset(second.unexpressed) - frozenset(first.unexpressed),
    )


def _patch_hashes(recipe: Recipe) -> set[str]:
    return {patch.sha256 for component in recipe.components for patch in component.patches}


def _launch_values(launch: LaunchConfig) -> dict[str, str]:
    """Each option the launch sets with its value, and each it removes from the resolved launch as removed."""

    return {**dict.fromkeys(launch.remove_args, REMOVED), **_flag_values(launch.server_args)}


def _flag_values(server_args: str) -> dict[str, str]:
    """Each flag's value; a later occurrence wins, as it does when the server parses its arguments."""

    parsed = tokenize_server_args_preserving_json(server_args)
    values: dict[str, list[str]] = {}
    flag = ""
    for token in parsed[1] if parsed is not None else []:
        if token.startswith(_FLAG):
            flag, _, inline = token.partition("=")
            values[flag] = [inline] if inline else []
        else:
            values.setdefault(flag, []).append(token)
    return {name: " ".join(parts) for name, parts in values.items()}


def _value_changes(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[ValueChange, ...]:
    return tuple(
        ValueChange(name, before.get(name), after.get(name))
        for name in sorted({*before, *after})
        if before.get(name) != after.get(name)
    )


def _ref_changes(first: Recipe, second: Recipe) -> tuple[RefChange, ...]:
    names = sorted({component.component for component in (*first.components, *second.components)})
    changes = []
    for name in names:
        before, after = first.component(name), second.component(name)
        refs = (before.upstream_ref if before else None, after.upstream_ref if after else None)
        if refs[0] != refs[1]:
            changes.append(RefChange(name, *refs))
    return tuple(changes)


__all__ = [
    "DIGEST_PREFIX",
    "KERNEL_ROOT",
    "PATCH_ROOT",
    "REMOVED",
    "RECIPE_SCHEMA_VERSION",
    "UNEXPRESSED_KINDS",
    "ArgsMode",
    "ComponentChange",
    "ComponentPatches",
    "ConfigDelta",
    "KernelArtifact",
    "KernelKind",
    "LaunchConfig",
    "PatchRef",
    "Recipe",
    "RecipeDelta",
    "RecipeDiff",
    "RecipeError",
    "RefChange",
    "ValueChange",
    "apply",
    "delta_between",
    "diff",
    "with_unexpressed",
]
