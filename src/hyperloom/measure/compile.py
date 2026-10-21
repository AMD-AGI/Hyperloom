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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

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

ATTACH_PLACEHOLDER = "${HL_ATTACH:%s}"
#: ``KernelArtifact.metadata["load"] = {"via": ..., ...}`` says how the service picks an artifact up. Only one form is supported:
#: the file is attached and the named environment variable points at it (gemm tuning tables).
KERNEL_LOAD_KEY = "load"
LOAD_VIA_ENV_FILE = "env_file"
#: Kernel kinds whose artifacts are loaded as files through a variable; others have no loading convention yet.
LOADABLE_KERNEL_KINDS = frozenset({KernelKind.GEMM})
#: Variables whose value PyTorch turns into one file per device: TunableOp reads ``<stem><ordinal><extension>`` (``x.csv``
#: is read as ``x0.csv`` on device 0, ``x1.csv`` on device 1; the ordinal counts the devices the process sees, so a test
#: on N cards reads 0..N-1). A table loaded through one of them is placed under the name the variable holds and under the
#: name of each of the ``DEVICE_ORDINALS`` of a node, so a test on any number of its cards finds its file.
PER_DEVICE_TABLE_ENVS = frozenset({"PYTORCH_TUNABLEOP_FILENAME"})
DEVICE_ORDINALS = range(8)
TABLE_EXTENSION = ".csv"
#: Directories an editable-installed source tree may live in and still count as part of the image.
IMAGE_SOURCE_ROOTS = ("/sgl-workspace", "/opt", "/app", "/usr")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIFF_HEADER = re.compile(r"^diff --git a/(?P<old>\S+) b/(?P<new>\S+)$")
_MODE_LINE = re.compile(r"^(?:(?:new file|deleted file|old|new) mode|index [0-9a-f]+\.\.[0-9a-f]+) (?P<mode>\d{6})$")
_SYMLINK_MODE = "120000"
_PACKAGE_DIST_DIRS = ("site-packages", "dist-packages")
_GIT_APPLY_TIMEOUT_SEC = 120


class CompileError(RecipeError):
    """The environment a recipe describes cannot be built exactly; the request is refused."""


class PatchApplyError(CompileError):
    """A patch does not apply to the package in the image, or is not a patch this module can apply faithfully."""


class BaseImageMismatchError(CompileError):
    """The recipe starts from an image other than the active framework image."""


class UnsupportedComponentError(CompileError):
    """A component has no package in the image that a runner could take as the base of its patches."""


class BlobError(CompileError):
    """A blob is missing, or its content does not hash to its address."""


class KernelLoadError(CompileError):
    """A kernel artifact has no supported way to be loaded by the service."""


# --- blob store --------------------------------------------------------------------------------------------------


class BlobStore(Protocol):
    """A content-addressed store: the address of a blob is the hex sha256 of its bytes."""

    root: Path

    def put(self, data: bytes) -> str: ...

    def get(self, sha256: str) -> bytes: ...


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
    """How a component's upstream repository maps onto the package installed in the image."""

    import_name: str
    #: The package directory's path inside the upstream repository; patches are relative to the repository root.
    upstream_prefix: str


COMPONENT_LAYOUTS: Mapping[str, ComponentLayout] = {
    "sglang": ComponentLayout("sglang", "python/sglang"),
    "vllm": ComponentLayout("vllm", "vllm"),
    "aiter": ComponentLayout("aiter", "aiter"),
    "atom": ComponentLayout("atom", "atom"),
    "xdit": ComponentLayout("xfuser", "xfuser"),
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


@dataclass(frozen=True)
class EnvEntry:
    """One file of a root that differs from the base: laid over it (``upsert``) or removed (``delete``)."""

    path: str
    op: Literal["upsert", "delete"]
    blob: str | None = None
    mode: str | None = None

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"path": self.path, "op": self.op}
        if self.op == "upsert":
            document.update(blob=f"{DIGEST_PREFIX}{self.blob}", mode=self.mode)
        return document


@dataclass(frozen=True)
class EnvRoot:
    """A package in the image and everything that differs from it."""

    id: str
    import_name: str
    dist_name: str | None
    dist_version: str | None
    entries: tuple[EnvEntry, ...]
    kind: Literal["package"] = "package"

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "import_name": self.import_name,
            "dist": {"name": self.dist_name, "version": self.dist_version},
            "entries": [entry.to_document() for entry in self.entries],
        }


def attachment_placeholder(attachment: AttachmentRef) -> str:
    """What the test YAML carries where the attachment's path would be; the runner substitutes the real path."""

    return ATTACH_PLACEHOLDER % attachment.id


@dataclass(frozen=True)
class CompiledEnvironment:
    """The environment of one test: the recipe it was compiled from, and what a runner needs to rebuild it.

    ``envs`` are the variables the test YAML must carry so the service loads the kernel artifacts; their values are
    attachment placeholders. ``recipe`` is the recipe after the change, whose ``launch`` the measurement point
    carries. ``runtime`` roots are not produced yet: how the recipe's snapshot image relates to a runtime snapshot is
    still open.
    """

    recipe: Recipe
    base_image_digest: str
    roots: tuple[EnvRoot, ...]
    attachments: tuple[AttachmentRef, ...]
    envs: Mapping[str, str]
    blob_root: Path

    def to_document(self) -> dict[str, object]:
        return {
            "roots": [root.to_document() for root in self.roots],
            "runtime": None,
            "blob_root": str(self.blob_root),
        }

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
    """Refuse a recipe built on another image: its patches and kernels were made for different bits."""

    active = _image_digest(active_image_digest)
    for what, image in (("base image", recipe.base_image_digest), ("runtime snapshot", recipe.runtime_snapshot)):
        if image is not None and _image_digest(image) != active:
            raise BaseImageMismatchError(f"the recipe's {what} {image} is not the active framework image {active}")


# --- patches ---------------------------------------------------------------------------------------------------------


def _package_path(path: str, prefix: str, patch: str) -> str:
    parts = PurePosixPath(path).parts
    if path.startswith("/") or ".." in parts:
        raise PatchApplyError(f"{patch}: unsafe path {path!r}")
    prefix_parts = PurePosixPath(prefix).parts
    if parts[: len(prefix_parts)] != prefix_parts or len(parts) == len(prefix_parts):
        raise PatchApplyError(
            f"{patch}: {path!r} is outside the installed package ({prefix}/); a runner cannot apply it to the image"
        )
    return str(PurePosixPath(*parts[len(prefix_parts) :]))


def patch_targets(text: str, prefix: str, patch: str) -> list[str]:
    """The package-relative files a git patch reads or writes; refuses what cannot be applied faithfully."""

    targets: list[str] = []
    for line in text.splitlines():
        header = _DIFF_HEADER.match(line)
        if header:
            targets += [_package_path(header["old"], prefix, patch), _package_path(header["new"], prefix, patch)]
        elif line.startswith(("GIT binary patch", "Binary files ")):
            raise PatchApplyError(f"{patch}: binary patches are not supported")
        elif line.startswith("diff --git "):
            raise PatchApplyError(f"{patch}: cannot read the file names of {line!r}")
        elif (mode := _MODE_LINE.match(line)) and mode["mode"] == _SYMLINK_MODE:
            raise PatchApplyError(f"{patch}: symbolic links are not supported")
    if not targets:
        raise PatchApplyError(f"{patch}: not a git patch (no 'diff --git' header)")
    return list(dict.fromkeys(targets))


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


def _stage(base: Path, tree: Path, rel: str, staged: set[str]) -> None:
    """Copy a file the next patch touches from the base into the working tree, the first time it is touched."""

    if rel in staged:
        return
    staged.add(rel)
    source = base / rel
    if source.is_symlink():
        raise PatchApplyError(f"{rel} is a symbolic link in the installed package")
    if source.is_file():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, tree / rel)


def _mode(path: Path) -> str:
    return "0755" if path.stat().st_mode & stat.S_IXUSR else "0644"


def _entries(base: Path, tree: Path, staged: set[str], blobs: BlobStore) -> tuple[EnvEntry, ...]:
    entries = []
    for rel in sorted(staged):
        old, new = base / rel, tree / rel
        if new.is_file():
            data = new.read_bytes()
            if old.is_file() and old.read_bytes() == data and _mode(old) == _mode(new):
                continue
            entries.append(EnvEntry(rel, "upsert", blobs.put(data), _mode(new)))
        elif old.is_file():
            entries.append(EnvEntry(rel, "delete"))
    return tuple(entries)


def apply_patch_stack(
    component: ComponentPatches, layout: ComponentLayout, base: PackageBase, blobs: BlobStore
) -> EnvRoot:
    """Lay the component's patches over a copy of the files they touch; the root is what ended up different."""

    prefix_depth = 1 + len(PurePosixPath(layout.upstream_prefix).parts)
    with tempfile.TemporaryDirectory(prefix="hl-compile-") as scratch:
        tree = Path(scratch) / "tree"
        tree.mkdir()
        staged: set[str] = set()
        for ref in component.patches:
            data = blobs.get(ref.sha256)
            text = _decode(data, ref.path)
            for rel in patch_targets(text, layout.upstream_prefix, ref.path):
                _stage(base.path, tree, rel, staged)
            patch_file = Path(scratch) / "current.patch"
            patch_file.write_bytes(data)
            _git_apply(tree, patch_file, prefix_depth, ref.path)
        entries = _entries(base.path, tree, staged, blobs)
    return EnvRoot(component.component, base.import_name, base.dist_name, base.dist_version, entries)


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
        envs[env_name] = attachment_placeholder(placed[0])
    return tuple(attachments.values()), envs


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


# --- compile -------------------------------------------------------------------------------------------------------------


@dataclass
class EnvironmentCompiler:
    """Compiles environments for one active image and one blob store; results are cached per recipe and change."""

    blobs: BlobStore
    active_image_digest: str
    locator: PackageLocator = locate_installed_package
    layouts: Mapping[str, ComponentLayout] = field(default_factory=lambda: COMPONENT_LAYOUTS)
    _cache: dict[tuple[str, str], CompiledEnvironment] = field(default_factory=dict, init=False, repr=False)

    def compile(self, recipe: Recipe, action: RecipeDelta) -> CompiledEnvironment:
        key = (recipe.digest(), action.fingerprint())
        if key not in self._cache:
            self._cache[key] = self._compile(recipe, action)
        return self._cache[key]

    def _compile(self, recipe: Recipe, action: RecipeDelta) -> CompiledEnvironment:
        check_base_image(recipe, self.active_image_digest)
        target = apply(recipe, action)
        roots = tuple(self._root(component) for component in target.components if component.patches)
        attachments, envs = kernel_attachments(target.kernels, target.launch.envs, self.blobs)
        return CompiledEnvironment(
            recipe=target,
            base_image_digest=target.base_image_digest,
            roots=roots,
            attachments=attachments,
            envs=envs,
            blob_root=self.blobs.root,
        )

    def _root(self, component: ComponentPatches) -> EnvRoot:
        layout = self.layouts.get(component.component)
        if layout is None:
            raise UnsupportedComponentError(
                f"{component.component} has no package layout; its patches cannot be applied"
            )
        return apply_patch_stack(component, layout, self.locator(layout), self.blobs)


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
    "UnsupportedComponentError",
    "apply_patch_stack",
    "attachment_placeholder",
    "check_base_image",
    "compile_environment",
    "kernel_attachments",
    "kernel_load_env",
    "locate_installed_package",
    "patch_targets",
]
