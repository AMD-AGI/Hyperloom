# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile a recipe and the change a test request carries into the environment overlay the test runs in.

The overlay is the difference between the original package in the image (the base) and the package the recipe
describes: for each component with patches, the files its patch stack adds, changes or deletes, stored by content in
a blob store. Kernel artifacts become attachments plus the environment variable that makes the service load them. A
runner needs neither git nor the patches: it takes the base from its own image and lays the entries over it.

Every way the overlay could silently differ from the recipe is a refusal here, as a :class:`CompileError` (a
:class:`~hyperloom.contracts.recipe.RecipeError`, so acceptance rejects the request): a patch that does not apply to
the installed package, a patch that reaches outside the package, a blob that is missing or does not match its hash,
a kernel artifact with no way to load it, and a base image other than the active one.

The blob store here is a minimal local content-addressed store, ``<root>/<sha256>``, written once and read-only
afterwards. Where the session keeps it and who writes the new files is settled with the tool layer; the
:class:`BlobStore` protocol is all this module depends on.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Protocol, TypeVar

from hyperloom.common.aiter_layout import image_aiter_meta
from hyperloom.common.outside_facts import (
    AITER_META_WHEEL_DIR,
    SGLANG_IMAGE_WORKSPACE,
    TUNABLEOP_TABLE_ENV,
    component_layouts,
)
from hyperloom.common.outside_facts.component_layout import META_BASE, PACKAGE_BASE, ComponentLayoutSpec, LayoutSection
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.contracts.recipe import (
    DIGEST_PREFIX,
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    Recipe,
    RecipeDelta,
    RecipeError,
    apply,
)
from hyperloom.measure.jit.native.needs import (
    BuildNeed,
    BuildNotReady,
    BuildRef,
    EnvEntry,
    EnvRoot,
    ModuleBuild,
    ModulePlan,
    NativePlan,
    NativeStack,
    check_runtime_sources,
)
from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER
from hyperloom.measure.jit.modules import (
    CONFIG_FILE,
    ESTIMATED_MODULE_SECONDS,
    MAX_REBUILD_SECONDS,
    SourceTree,
    Touched,
    config_path,
    prebuilt_names,
    read_config,
    touched_entries,
    touched_modules,
)
from hyperloom.measure.framework.table_modules import named_table_modules
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX


ATTACH_PLACEHOLDER = "${HL_ATTACH:%s}"
#: ``KernelArtifact.metadata["load"] = {"via": ..., ...}`` says how the service picks an artifact up. Only one form is supported:
#: the file is attached and the named environment variable points at it (gemm tuning tables).
KERNEL_LOAD_KEY = "load"
LOAD_VIA_ENV_FILE = "env_file"
#: ``metadata["load"]["rows_only"] = true``: the file holds only the specialist's tuned rows, and the variable names the
#: image's default table and then the file. Only an aiter table variable can list tables, so only one carries it.
LOAD_ROWS_ONLY_KEY = "rows_only"
#: Kernel kinds whose artifacts are loaded as files through a variable; others have no loading convention yet.
LOADABLE_KERNEL_KINDS = frozenset({KernelKind.GEMM})
#: Variables whose value PyTorch turns into one file per device: TunableOp reads ``<stem><ordinal><extension>`` (``x.csv``
#: is read as ``x0.csv`` on device 0, ``x1.csv`` on device 1; the ordinal counts the devices the process sees, so a test
#: on N cards reads 0..N-1). A table loaded through one of them is placed under the name the variable holds and under the
#: name of each of the ``DEVICE_ORDINALS`` of a node, so a test on any number of its cards finds its file.
PER_DEVICE_TABLE_ENVS = frozenset({TUNABLEOP_TABLE_ENV})
DEVICE_ORDINALS = range(8)
TABLE_EXTENSION = ".csv"
#: Directories an editable-installed source tree may live in and still count as part of the image.
IMAGE_SOURCE_ROOTS = (SGLANG_IMAGE_WORKSPACE, "/opt", "/app", "/usr")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIFF_HEADER = re.compile(r"^diff --git a/(?P<old>\S+) b/(?P<new>\S+)$")
_MODE_LINE = re.compile(r"^(?:(?:new file|deleted file|old|new) mode|index [0-9a-f]+\.\.[0-9a-f]+) (?P<mode>\d{6})$")
_SYMLINK_MODE = "120000"
_PACKAGE_DIST_DIRS = ("site-packages", "dist-packages")
_GIT_APPLY_TIMEOUT_SEC = 120
_UPSERT = "upsert"


class CompileError(RecipeError):
    """The environment a recipe describes cannot be built exactly; the request is refused."""


class PatchApplyError(CompileError):
    """A patch does not apply to the package in the image, or is not a patch this module can apply faithfully."""


class BaseImageMismatchError(CompileError):
    """The recipe starts from an image other than the active framework image."""


class UnsupportedComponentError(CompileError):
    """A component has no package in the image that a runner could take as the base of its patches."""


class TooManyModulesError(CompileError):
    """A patch stack touches so many of aiter's modules that its first test would spend most of its time compiling."""


class BlobError(CompileError):
    """A blob is missing, or its content does not hash to its address."""


class KernelLoadError(CompileError):
    """A kernel artifact has no supported way to be loaded by the service."""


# --- blob store --------------------------------------------------------------------------------------------------


class BlobStore(Protocol):
    """A content-addressed store: the address of a blob is the hex sha256 of its bytes."""

    root: Path

    def put(self, data: bytes) -> str:
        """Store ``data`` and return its address."""

    def get(self, sha256: str) -> bytes:
        """The bytes stored at ``sha256``; a :class:`BlobError` when they are missing or do not hash to it."""


class LocalBlobStore:
    """``<root>/<sha256>``: written once, read-only afterwards, verified on every read."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        sha256 = hashlib.sha256(data).hexdigest()
        target = self.root / sha256
        if target.exists():
            return sha256
        scratch = self.root / f".{sha256}.{uuid.uuid4().hex}.tmp"
        scratch.write_bytes(data)
        scratch.chmod(0o444)
        os.replace(scratch, target)
        return sha256

    def get(self, sha256: str) -> bytes:
        if not _SHA256.fullmatch(sha256):
            raise BlobError(f"{sha256!r} is not a sha256 blob address")
        try:
            data = (self.root / sha256).read_bytes()
        except FileNotFoundError as exc:
            raise BlobError(f"blob {sha256} is not in the blob store") from exc
        if hashlib.sha256(data).hexdigest() != sha256:
            raise BlobError(f"blob {sha256} does not hash to its address")
        return data


# --- where a component lives ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentLayout:
    """How a component's upstream repository maps onto the package installed in the image.

    ``sections`` are the layout's parts (an upstream prefix, the base it lies over and where in the base it starts); a
    layout without any is the one package section at ``upstream_prefix``. ``refuse`` are the patterns of the file names
    whose change this layout cannot measure, and ``generated`` the package directories the runtime writes builds into.
    The rule pack owns all of it (``component_layout`` entries).
    """

    import_name: str
    #: The package directory's path inside the upstream repository; patches are relative to the repository root.
    upstream_prefix: str
    sections: tuple[LayoutSection, ...] = ()
    refuse: tuple[str, ...] = ()
    generated: tuple[str, ...] = ()

    @classmethod
    def from_spec(cls, spec: ComponentLayoutSpec) -> ComponentLayout:
        return cls(spec.import_name, spec.sections[0].prefix, spec.sections, spec.refuse, spec.generated)

    @property
    def spec(self) -> ComponentLayoutSpec:
        sections = self.sections or (LayoutSection(self.upstream_prefix),)
        return ComponentLayoutSpec(self.import_name, sections, self.refuse, self.generated)


COMPONENT_LAYOUTS: Mapping[str, ComponentLayout] = {
    name: ComponentLayout.from_spec(spec) for name, spec in component_layouts().items()
}


@dataclass(frozen=True)
class PackageBase:
    """The original package in the image: the directory a root's entries are laid over."""

    import_name: str
    path: Path
    dist_name: str | None
    dist_version: str | None


PackageLocator = Callable[[ComponentLayout], PackageBase]


def _editable_source_dir(dist: importlib.metadata.Distribution) -> Path | None:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    url = info.get("url", "") if isinstance(info, dict) else ""
    editable = isinstance(info, dict) and bool(info.get("dir_info", {}).get("editable"))
    return Path(url[len("file://") :]) if editable and url.startswith("file://") else None


def locate_installed_package(layout: ComponentLayout, image_roots: tuple[str, ...] = IMAGE_SOURCE_ROOTS) -> PackageBase:
    """Find the package in the running image, without importing it.

    Accepts a package under ``site-packages`` or ``dist-packages``, and an editable-installed source tree inside the
    image (the SGLang checkout installed in place). An editable tree anywhere else, such as shared storage, is not
    part of the image and cannot be a base.
    """

    name = layout.import_name
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    locations = list(spec.submodule_search_locations or []) if spec else []
    if not locations:
        raise UnsupportedComponentError(f"package {name} is not installed in this image")
    path = Path(locations[0]).resolve()
    dist_names = importlib.metadata.packages_distributions().get(name, [])
    dist = importlib.metadata.distribution(dist_names[0]) if dist_names else None
    in_site = any(part in _PACKAGE_DIST_DIRS for part in path.parts)
    editable = _editable_source_dir(dist) if dist else None
    in_image = any(path == Path(root) or Path(root) in path.parents for root in image_roots)
    if not in_site and not (editable is not None and in_image):
        raise UnsupportedComponentError(
            f"package {name} at {path} is neither installed in site-packages nor an editable tree inside the image; "
            "hosted measurement cannot take it as a base"
        )
    return PackageBase(name, path, dist.metadata["Name"] if dist else None, dist.version if dist else None)


# --- the compiled overlay --------------------------------------------------------------------------------------------


def attachment_placeholder(attachment: AttachmentRef) -> str:
    """What the test YAML carries where the attachment's path would be; the runner substitutes the real path."""

    return ATTACH_PLACEHOLDER % attachment.id


@dataclass(frozen=True)
class CompiledEnvironment:
    """The environment of one test: the recipe it was compiled from, and what a runner needs to rebuild it.

    ``envs`` are the variables the test YAML must carry so the service loads the kernel artifacts; their values are
    attachment placeholders. ``recipe`` is the recipe after the change, whose ``launch`` the measurement point
    carries. A ``runtime`` root is a package root whose native outputs come from a build: ``builds`` names the build of
    each, and the sheet names it only once it is ready (:meth:`with_builds`), by the manifest it was built to.
    """

    recipe: Recipe
    base_image_digest: str
    roots: tuple[EnvRoot, ...]
    attachments: tuple[AttachmentRef, ...]
    envs: Mapping[str, str]
    blob_root: Path
    builds: tuple[BuildNeed, ...] = ()
    build_refs: tuple[BuildRef, ...] = ()

    def with_builds(self, refs: tuple[BuildRef, ...]) -> CompiledEnvironment:
        """This environment with the ready builds ``refs``, which must be exactly the builds it needs."""
        wanted = {(need.root_id, need.inputs.key) for need in self.builds}
        if {(ref.root_id, ref.key) for ref in refs} != wanted or len(refs) != len(wanted):
            raise BuildNotReady(f"the builds {sorted(wanted)} are not the ready ones {[ref.key for ref in refs]}")
        return replace(self, build_refs=refs)

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "roots": [root.to_document() for root in self.roots],
            "runtime": None,
            "blob_root": str(self.blob_root),
        }
        if self.builds:
            if len(self.build_refs) != len(self.builds):
                raise BuildNotReady(f"{len(self.builds)} builds are needed and {len(self.build_refs)} are ready")
            document["builds"] = [ref.to_document() for ref in self.build_refs]
        return document

    @property
    def is_base(self) -> bool:
        """No patches and no kernel artifacts: the test runs the image as it is."""

        return not self.roots and not self.attachments


# --- base image ------------------------------------------------------------------------------------------------------


def _image_digest(image: str) -> str:
    digest = image.rsplit("@", 1)[-1]
    if not digest.startswith(DIGEST_PREFIX):
        raise BaseImageMismatchError(f"{image!r} does not name an image by digest")
    return digest


def check_base_image(recipe: Recipe, active_image_digest: str) -> None:
    """Refuse a recipe built on another image: its patches and kernels were made for different bits.

    Only the base image is compared: the runtime is the base image plus signed native builds, which no image digest
    names, so a ``runtime_snapshot`` is not an image to compare (:func:`check_runtime_sources` refuses it beside native
    builds)."""

    active = _image_digest(active_image_digest)
    if _image_digest(recipe.base_image_digest) != active:
        raise BaseImageMismatchError(
            f"the recipe's base image {recipe.base_image_digest} is not the active framework image {active}"
        )


# --- patches ---------------------------------------------------------------------------------------------------------


class PlacedPath(NamedTuple):
    """A path of a patch placed in the image: ``upstream`` in the component's repository, ``relative`` in the base ``base``."""

    upstream: str
    base: str
    relative: str


_Placed = TypeVar("_Placed")


def _safe_parts(path: str, patch: str) -> tuple[str, ...]:
    parts = PurePosixPath(path).parts
    if path.startswith("/") or ".." in parts:
        raise PatchApplyError(f"{patch}: unsafe path {path!r}")
    return parts


def _package_path(path: str, prefix: str, patch: str) -> str:
    parts = _safe_parts(path, patch)
    prefix_parts = PurePosixPath(prefix).parts
    if parts[: len(prefix_parts)] != prefix_parts or len(parts) == len(prefix_parts):
        raise PatchApplyError(
            f"{patch}: {path!r} is outside the installed package ({prefix}/); a runner cannot apply it to the image"
        )
    return str(PurePosixPath(*parts[len(prefix_parts) :]))


def _place(path: str, layout: ComponentLayout, patch: str) -> PlacedPath:
    _safe_parts(path, patch)
    pattern = layout.spec.refused_by(path)
    if pattern is not None:
        raise PatchApplyError(
            f"{patch}: {path!r} is a native or binary file ({pattern}) of {layout.import_name}; a patch changes its "
            "Python only, so the compiled code in the image would stay as it is"
        )
    placed = layout.spec.place(path)
    if placed is None:
        prefixes = ", ".join(f"{section.prefix}/" for section in layout.spec.sections)
        raise PatchApplyError(
            f"{patch}: {path!r} is outside the installed package ({prefixes}); a runner cannot apply it to the image"
        )
    section, relative = placed
    return PlacedPath(path, section.base, relative)


def _scan_patch(text: str, patch: str, place: Callable[[str], _Placed]) -> list[_Placed]:
    targets: list[_Placed] = []
    for line in text.splitlines():
        header = _DIFF_HEADER.match(line)
        if header:
            targets += [place(header["old"]), place(header["new"])]
        elif line.startswith(("GIT binary patch", "Binary files ")):
            raise PatchApplyError(f"{patch}: binary patches are not supported")
        elif line.startswith("diff --git "):
            raise PatchApplyError(f"{patch}: cannot read the file names of {line!r}")
        elif (mode := _MODE_LINE.match(line)) and mode["mode"] == _SYMLINK_MODE:
            raise PatchApplyError(f"{patch}: symbolic links are not supported")
    if not targets:
        raise PatchApplyError(f"{patch}: not a git patch (no 'diff --git' header)")
    return list(dict.fromkeys(targets))


def patch_targets(text: str, prefix: str, patch: str) -> list[str]:
    """The package-relative files a git patch reads or writes; refuses what cannot be applied faithfully."""

    return _scan_patch(text, patch, lambda path: _package_path(path, prefix, patch))


def layout_targets(text: str, layout: ComponentLayout, patch: str) -> list[PlacedPath]:
    """The files a git patch reads or writes, each placed by ``layout``; refuses a file outside its sections and one it
    cannot measure, by path."""

    return _scan_patch(text, patch, lambda path: _place(path, layout, patch))


def _git_apply(tree: Path, patch_file: Path, strip: int, patch: str) -> None:
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent), "LC_ALL": "C"}
    cmd = ["git", "apply", f"-p{strip}", "--whitespace=nowarn", str(patch_file)]
    try:
        done = subprocess.run(
            cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=_GIT_APPLY_TIMEOUT_SEC, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PatchApplyError(f"{patch}: git apply could not run: {exc}") from exc
    if done.returncode != 0:
        raise PatchApplyError(f"{patch} does not apply to the installed package: {done.stderr.strip()[:500]}")


def _installed(bases: Mapping[str, PackageBase], placed: PlacedPath) -> Path:
    base = bases.get(placed.base)
    if base is None:
        raise UnsupportedComponentError(
            f"{placed.upstream} lies over the {placed.base} of the image, which this compile does not lay yet"
        )
    return base.path / placed.relative


def _stage(
    bases: Mapping[str, PackageBase], tree: Path, placed: PlacedPath, staged: dict[tuple[str, str], str]
) -> None:
    """Copy a file the next patch touches from the image into the working tree, at its upstream path, the first time."""

    key = (placed.base, placed.relative)
    if key in staged:
        if staged[key] != placed.upstream:
            raise PatchApplyError(
                f"{placed.upstream} and {staged[key]} are one file of the image ({placed.relative}); "
                "a patch stack may name it by one path only"
            )
        return
    staged[key] = placed.upstream
    source = _installed(bases, placed)
    if source.is_symlink():
        raise PatchApplyError(f"{placed.relative} is a symbolic link in the installed package")
    if source.is_file():
        (tree / placed.upstream).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, tree / placed.upstream)


def _mode(path: Path) -> str:
    return "0755" if path.stat().st_mode & stat.S_IXUSR else "0644"


def _entries(
    bases: Mapping[str, PackageBase], tree: Path, staged: dict[tuple[str, str], str], blobs: BlobStore
) -> dict[str, tuple[EnvEntry, ...]]:
    """What the patches made different from the image, by base."""

    entries: dict[str, list[EnvEntry]] = {}
    for (base, rel), upstream in sorted(staged.items()):
        old, new = _installed(bases, PlacedPath(upstream, base, rel)), tree / upstream
        found = entries.setdefault(base, [])
        if new.is_file():
            data = new.read_bytes()
            if old.is_file() and old.read_bytes() == data and _mode(old) == _mode(new):
                continue
            found.append(EnvEntry(rel, _UPSERT, blobs.put(data), _mode(new)))
        elif old.is_file():
            found.append(EnvEntry(rel, "delete"))
    return {base: tuple(found) for base, found in entries.items()}


def _apply_stack(
    component: ComponentPatches, layout: ComponentLayout, bases: Mapping[str, PackageBase], blobs: BlobStore
) -> dict[str, tuple[EnvEntry, ...]]:
    """Apply the component's patches, each to the upstream paths it names, over copies of the files they touch."""

    with tempfile.TemporaryDirectory(prefix="hl-compile-") as scratch:
        tree = Path(scratch) / "tree"
        tree.mkdir()
        staged: dict[tuple[str, str], str] = {}
        for ref in component.patches:
            data = blobs.get(ref.sha256)
            for placed in layout_targets(_decode(data, ref.path), layout, ref.path):
                _stage(bases, tree, placed, staged)
            patch_file = Path(scratch) / "current.patch"
            patch_file.write_bytes(data)
            _git_apply(tree, patch_file, 1, ref.path)
        return _entries(bases, tree, staged, blobs)


#: The id suffix of a root over a base other than the package: ``aiter.meta`` is the root of aiter's meta directory.
_ROOT_SUFFIX = {META_BASE: ".meta"}


def apply_component_stack(
    component: ComponentPatches, layout: ComponentLayout, bases: Mapping[str, PackageBase], blobs: BlobStore
) -> tuple[EnvRoot, ...]:
    """Lay the component's patches over copies of the files they touch; each root is what ended up different in one base.

    The package root comes first and is always there; a root over another base is there when a patch changed it.
    """

    entries = _apply_stack(component, layout, bases, blobs)
    package = bases[PACKAGE_BASE]
    package_entries = entries.get(PACKAGE_BASE, ())
    roots = [
        EnvRoot(component.component, package.import_name, package.dist_name, package.dist_version, package_entries)
    ]
    for base, changed in entries.items():
        if base != PACKAGE_BASE and changed:
            other = bases[base]
            root_id = f"{component.component}{_ROOT_SUFFIX[base]}"
            roots.append(EnvRoot(root_id, other.import_name, other.dist_name, other.dist_version, changed))
    return tuple(roots)


def _patched_config(entries: tuple[EnvEntry, ...], blobs: BlobStore) -> dict | None:
    """The module config the stack leaves, when it changes it; ``ValueError`` when it deletes or breaks it."""
    for entry in entries:
        if entry.path == f"jit/{CONFIG_FILE}":
            if entry.op != _UPSERT:
                raise ValueError("the patch deletes the module config")
            return read_config(blobs.get(entry.blob or ""))
    return None


@dataclass(frozen=True)
class _AiterTouch:
    """What a stack of aiter touches: the module config it leaves, the touched modules, and those whose tuned tables a
    variable the test sets names (``tabled``), which only its server can build from them."""

    config: dict
    found: Touched
    tabled: frozenset[str]


def _aiter_touch(
    roots: tuple[EnvRoot, ...], bases: Mapping[str, PackageBase], blobs: BlobStore, table_variables: Collection[str]
) -> _AiterTouch | None:
    package, meta = bases[PACKAGE_BASE], bases[META_BASE]
    by_id = {root.import_name: root for root in roots}
    meta_entries = by_id[meta.import_name].entries if meta.import_name in by_id else ()
    package_entries = by_id[package.import_name].entries if package.import_name in by_id else ()
    try:
        before = read_config(config_path(package.path))
        after = _patched_config(package_entries, blobs) or before
    except (OSError, ValueError) as exc:
        raise PatchApplyError(f"cannot tell which of aiter's modules the patch touches: {exc}") from exc
    tables = named_table_modules(after, package.path, table_variables, [entry.path for entry in package_entries])
    if not meta_entries and after == before and not tables:
        return None
    overlay = {entry.path: blobs.get(entry.blob or "") if entry.op == _UPSERT else None for entry in meta_entries}
    prebuilt = prebuilt_names(package.path)
    found = touched_modules(before, after, SourceTree(meta.path, overlay), overlay, prebuilt) | touched_entries(
        tables, after, prebuilt
    )
    tabled = named_table_modules(after, package.path, table_variables, ()) if table_variables else frozenset()
    return _AiterTouch(after, found, tabled)


def refuse_costly_rebuild(
    roots: tuple[EnvRoot, ...],
    bases: Mapping[str, PackageBase],
    blobs: BlobStore,
    table_variables: Collection[str] = (),
    modules: ModulePlan | None = None,
) -> ModuleBuild | None:
    """Refuse a stack that touches so many of aiter's modules that their first build would not fit the time given to it.

    The modules are the prebuilt ones the patch leaves stale (:func:`~hyperloom.measure.jit.modules.touched_modules`),
    with the ones generated from a tuned table that the patch (below ``configs``) or one of ``table_variables`` (the
    ``AITER_CONFIG_*`` the test sets) may change; the first test of the stack builds them on its card, when its server
    calls them. With ``modules`` (a host that builds natively, #125 stage 2b) a CPU build makes the ones it can before the
    test, and only the rest are built on the card; that build, returned, is refused when it would not fit its own time.
    A module generated from a table the test's variables name is left to the card: only its server reads those tables.
    """
    touch = _aiter_touch(roots, bases, blobs, table_variables)
    if touch is None:
        return None
    candidates = touch.found.rebuilt - touch.tabled
    built = modules(roots, touch.config, candidates) if modules is not None and candidates else None
    on_cpu = frozenset() if built is None else built.modules
    _refuse_slow_build(sorted(touch.found.rebuilt - on_cpu), ESTIMATED_MODULE_SECONDS * 1.0, "on the first test's card")
    if modules is not None and on_cpu:
        _refuse_slow_build(sorted(on_cpu), modules.seconds(on_cpu) / len(on_cpu), "on a CPU build runner")
    return built


def _refuse_slow_build(names: list[str], each: float, where: str) -> None:
    seconds = round(len(names) * each)
    if seconds > MAX_REBUILD_SECONDS:
        examples = ", ".join(names[:5])
        raise TooManyModulesError(
            f"the patch touches {len(names)} of aiter's prebuilt modules (for example {examples}), which take about "
            f"{seconds} s to build again {where}; more than {MAX_REBUILD_SECONDS} s is refused. "
            "Change fewer modules at a time: a header that many modules include touches all of them"
        )


def apply_patch_stack(
    component: ComponentPatches, layout: ComponentLayout, base: PackageBase, blobs: BlobStore
) -> EnvRoot:
    """Lay the component's patches over a copy of the files they touch; the root is what ended up different."""

    return apply_component_stack(component, layout, {PACKAGE_BASE: base}, blobs)[0]


def _decode(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchApplyError(f"{name} is not UTF-8 text") from exc


# --- kernel artifacts ----------------------------------------------------------------------------------------------------


def kernel_attachments(
    kernels: tuple[KernelArtifact, ...], launch_envs: Mapping[str, str], blobs: BlobStore
) -> tuple[tuple[AttachmentRef, ...], dict[str, str]]:
    """Attach each kernel artifact and name the variable that points the service at it."""

    attachments: dict[str, AttachmentRef] = {}
    envs: dict[str, str] = {}
    for artifact in kernels:
        env_name = kernel_load_env(artifact)
        if env_name in envs or env_name in launch_envs:
            raise KernelLoadError(f"{artifact.ref}: {env_name} is already set by another artifact or by the launch")
        blobs.get(artifact.sha256)
        placed = _table_attachments(artifact, env_name)
        for attachment in placed:
            attachments.setdefault(attachment.id, attachment)
        envs[env_name] = _table_value(artifact, placed[0])
    return tuple(attachments.values()), envs


def _table_value(artifact: KernelArtifact, table: AttachmentRef) -> str:
    """What the variable holds: the table's placeholder, behind the image's default table for a rows-only table."""

    placeholder = attachment_placeholder(table)
    return os.pathsep.join([DEFAULT_TABLE_PLACEHOLDER, placeholder]) if kernel_rows_only(artifact) else placeholder


def _table_attachments(artifact: KernelArtifact, env_name: str) -> tuple[AttachmentRef, ...]:
    """The files that carry the table of ``artifact``; the first is the one ``env_name`` points at."""

    stem = f"k-{artifact.sha256[:16]}"
    if env_name not in PER_DEVICE_TABLE_ENVS:
        return (AttachmentRef(stem, "file", artifact.sha256),)
    names = [stem + TABLE_EXTENSION, *(f"{stem}{ordinal}{TABLE_EXTENSION}" for ordinal in DEVICE_ORDINALS)]
    return tuple(AttachmentRef(name, "file", artifact.sha256) for name in names)


def kernel_load_env(artifact: KernelArtifact) -> str:
    """The variable that points the service at ``artifact``; :class:`KernelLoadError` when it has no supported way."""

    load = artifact.metadata.get(KERNEL_LOAD_KEY)
    if artifact.kind not in LOADABLE_KERNEL_KINDS:
        raise KernelLoadError(f"{artifact.ref}: no loading convention for {artifact.kind.value} kernel artifacts yet")
    if not isinstance(load, Mapping):
        raise KernelLoadError(
            f"{artifact.ref}: metadata.{KERNEL_LOAD_KEY} is missing; say how it is loaded, "
            f"e.g. {{'via': '{LOAD_VIA_ENV_FILE}', 'env': '<VARIABLE>'}}"
        )
    if load.get("via") != LOAD_VIA_ENV_FILE:
        raise KernelLoadError(f"{artifact.ref}: unknown load via {load.get('via')!r}; supported: {LOAD_VIA_ENV_FILE!r}")
    env_name = load.get("env")
    if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
        raise KernelLoadError(f"{artifact.ref}: metadata.{KERNEL_LOAD_KEY}.env must name an environment variable")
    return env_name


def kernel_rows_only(artifact: KernelArtifact) -> bool:
    """Whether ``artifact`` holds only tuned rows to load over the image's default table.

    Raises:
        KernelLoadError: When it says so for a variable that cannot list tables, or says it with a value that is no bool.
    """
    load = artifact.metadata.get(KERNEL_LOAD_KEY)
    rows_only = load.get(LOAD_ROWS_ONLY_KEY, False) if isinstance(load, Mapping) else False
    if not isinstance(rows_only, bool):
        raise KernelLoadError(f"{artifact.ref}: metadata.{KERNEL_LOAD_KEY}.{LOAD_ROWS_ONLY_KEY} must be true or false")
    if rows_only and not kernel_load_env(artifact).startswith(AITER_VARIABLE_PREFIX):
        raise KernelLoadError(
            f"{artifact.ref}: rows_only lays rows over the image's default table, which only an aiter table variable "
            f"({AITER_VARIABLE_PREFIX}*) has; {kernel_load_env(artifact)} loads one whole file"
        )
    return rows_only


# --- compile -------------------------------------------------------------------------------------------------------------


@dataclass
class EnvironmentCompiler:
    """Compiles environments for one active image and one blob store; results are cached per recipe and change."""

    blobs: BlobStore
    active_image_digest: str
    locator: PackageLocator = locate_installed_package
    layouts: Mapping[str, ComponentLayout] = field(default_factory=lambda: COMPONENT_LAYOUTS)
    #: The native builds a stack needs; without one a stack that changes a native source is refused by its layout.
    native: NativePlan | None = None
    #: The CPU build of aiter's touched modules; without one the test's server builds them on its card.
    modules: ModulePlan | None = None
    _cache: dict[tuple[str, str], CompiledEnvironment] = field(default_factory=dict, init=False, repr=False)

    def compile(self, recipe: Recipe, action: RecipeDelta) -> CompiledEnvironment:
        key = (recipe.digest(), action.fingerprint())
        if key not in self._cache:
            self._cache[key] = self._compile(recipe, action)
        return self._cache[key]

    def _compile(self, recipe: Recipe, action: RecipeDelta) -> CompiledEnvironment:
        check_base_image(recipe, self.active_image_digest)
        target = apply(recipe, action)
        attachments, envs = kernel_attachments(target.kernels, target.launch.envs, self.blobs)
        tables = sorted(
            name
            for name, value in {**target.launch.envs, **envs}.items()
            if name.startswith(AITER_VARIABLE_PREFIX) and str(value).strip()
        )
        roots: list[EnvRoot] = []
        builds: list[BuildNeed] = []
        for component in (component for component in target.components if component.patches):
            native = self.native(component) if self.native is not None else None
            if native is None:
                component_roots, needs = self._roots(component, tables)
                roots.extend(component_roots)
                builds.extend(needs)
            else:
                roots.append(self._runtime_root(component, native))
                builds.append(BuildNeed(component.component, native.inputs))
        check_runtime_sources(target, tuple(builds))
        return CompiledEnvironment(
            recipe=target,
            base_image_digest=target.base_image_digest,
            roots=tuple(roots),
            attachments=attachments,
            envs=envs,
            blob_root=self.blobs.root,
            builds=tuple(builds),
        )

    def _runtime_root(self, component: ComponentPatches, native: NativeStack) -> EnvRoot:
        """The package root of a native stack: the stack's own package files, its native outputs to come from a build."""
        layout = self.layouts.get(component.component)
        if layout is None:
            raise UnsupportedComponentError(f"{component.component} has no package layout; its build cannot be laid")
        package = self.locator(layout)
        return EnvRoot(
            component.component,
            package.import_name,
            package.dist_name,
            package.dist_version,
            native.entries,
            kind="runtime",
        )

    def _roots(
        self, component: ComponentPatches, table_variables: Collection[str] = ()
    ) -> tuple[tuple[EnvRoot, ...], tuple[BuildNeed, ...]]:
        """The component's roots, and the CPU build of the aiter modules they touch when one makes any."""
        layout = self.layouts.get(component.component)
        if layout is None:
            raise UnsupportedComponentError(
                f"{component.component} has no package layout; its patches cannot be applied"
            )
        package = self.locator(layout)
        bases = {PACKAGE_BASE: package}
        if any(section.base == META_BASE for section in layout.spec.sections):
            bases[META_BASE] = PackageBase(
                AITER_META_WHEEL_DIR, image_aiter_meta(package.path), package.dist_name, package.dist_version
            )
        roots = apply_component_stack(component, layout, bases, self.blobs)
        if META_BASE not in bases:
            return roots, ()
        built = refuse_costly_rebuild(roots, bases, self.blobs, table_variables, self.modules)
        return roots, () if built is None else (BuildNeed(component.component, built.inputs),)


def compile_environment(
    recipe: Recipe,
    action: RecipeDelta,
    *,
    blobs: BlobStore,
    active_image_digest: str,
    locator: PackageLocator = locate_installed_package,
) -> CompiledEnvironment:
    """The environment ``recipe`` plus ``action`` describes; raises :class:`CompileError` when it cannot be built."""

    return EnvironmentCompiler(blobs, active_image_digest, locator).compile(recipe, action)


__all__ = [
    "ATTACH_PLACEHOLDER",
    "COMPONENT_LAYOUTS",
    "BaseImageMismatchError",
    "BlobError",
    "BlobStore",
    "CompileError",
    "CompiledEnvironment",
    "ComponentLayout",
    "EnvEntry",
    "EnvRoot",
    "EnvironmentCompiler",
    "KernelLoadError",
    "LocalBlobStore",
    "PackageBase",
    "PatchApplyError",
    "TooManyModulesError",
    "UnsupportedComponentError",
    "apply_component_stack",
    "apply_patch_stack",
    "attachment_placeholder",
    "check_base_image",
    "compile_environment",
    "kernel_attachments",
    "kernel_load_env",
    "kernel_rows_only",
    "locate_installed_package",
    "patch_targets",
]
