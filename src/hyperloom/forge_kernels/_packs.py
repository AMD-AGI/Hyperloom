# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Loading and build-caching for installed KernelForge kernel packs.

An *installed* pack lives at ``<root>/<name>/`` and always holds three files:

- ``kernel.py``     — the KernelForge artifact, copied verbatim.
- ``pack.json``     — the manifest, normalized from KernelForge's ``pack.yaml``
                      by the orchestrator-side installer so nothing in the
                      serving process needs a YAML parser.
- ``preflight.json``— written by :mod:`hyperloom.forge_kernels.preflight` after
                      it built and score-gated every candidate shape ON THIS
                      MACHINE. Its ``verified`` list is the runtime allowlist.

The split matters: KernelForge's manifest says what the kernel is *meant* to
support, ``preflight.json`` says what this GPU + this FlyDSL build actually
delivered. Only the latter gates dispatch.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: csv of pack names to enable. Unset/empty => the whole feature is off and
#: every entry point returns ``None`` (strict no-op vs. upstream).
ENV_ENABLED = "HYPERLOOM_FORGE_KERNEL_PACKS"
#: Root the orchestrator installed packs into.
ENV_ROOT = "HYPERLOOM_FORGE_KERNEL_PACK_ROOT"
#: Compare the first launch of each new (M, N, dtype) against the framework
#: reference before trusting it. "0" disables.
ENV_VERIFY = "HYPERLOOM_FORGE_KERNEL_VERIFY"
#: Upper bound on distinct built (M, N, dtype) modules; past it we fall back
#: rather than thrash the FlyDSL JIT.
ENV_CACHE_MAX = "HYPERLOOM_FORGE_KERNEL_CACHE_MAX"

_DEFAULT_CACHE_MAX = 64

_TORCH_DTYPE_TAGS = {
    "torch.float32": "f32",
    "torch.float16": "f16",
    "torch.bfloat16": "bf16",
}


def dtype_tag(dtype: Any) -> str | None:
    """Map a ``torch.dtype`` to KernelForge's dtype tag (``f32``/``f16``/``bf16``)."""
    return _TORCH_DTYPE_TAGS.get(str(dtype))


def default_root() -> Path:
    """Installed-pack root: ``$HYPERLOOM_FORGE_KERNEL_PACK_ROOT`` else
    ``$USER_DATA_PATH/runtime/forge-kernel-packs`` (``USER_DATA_PATH`` itself
    defaulting to ``/workspace/hyperloom``, matching ``session/paths.py``)."""
    explicit = os.environ.get(ENV_ROOT, "").strip()
    if explicit:
        return Path(explicit)
    workspace = os.environ.get("USER_DATA_PATH", "").strip() or "/workspace/hyperloom"
    return Path(workspace) / "runtime" / "forge-kernel-packs"


def enabled_pack_names() -> tuple[str, ...]:
    """Pack names listed in ``$HYPERLOOM_FORGE_KERNEL_PACKS``, in order."""
    raw = os.environ.get(ENV_ENABLED, "").strip()
    if not raw:
        return ()
    return tuple(n.strip() for n in raw.split(",") if n.strip())


def cache_max() -> int:
    """Distinct-build ceiling from ``$HYPERLOOM_FORGE_KERNEL_CACHE_MAX``."""
    try:
        value = int(os.environ.get(ENV_CACHE_MAX, "") or _DEFAULT_CACHE_MAX)
    except ValueError:
        return _DEFAULT_CACHE_MAX
    return value if value > 0 else _DEFAULT_CACHE_MAX


def verify_enabled() -> bool:
    """Whether to reference-check the first launch of each new build key."""
    return os.environ.get(ENV_VERIFY, "1").strip() not in {"0", "false", "False"}


@dataclass
class Pack:
    """One installed, preflight-gated kernel pack."""

    name: str
    root: Path
    op: str
    builder: str
    manifest: dict[str, Any]
    preflight: dict[str, Any]
    #: ``(N, dtype_tag)`` pairs preflight actually verified on this machine.
    verified: frozenset[tuple[int, str]]
    module: Any = None
    _builds: dict[tuple[int, int, str], Any] = field(default_factory=dict)
    _dead: set[tuple[int, int, str]] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def supports(self, n: int, tag: str) -> bool:
        """Whether preflight verified this ``(N, dtype)`` on this machine."""
        return (n, tag) in self.verified

    def build(self, m: int, n: int, tag: str) -> Any | None:
        """Return the cached launcher for ``(M, N, dtype)``, building on miss.

        ``build_softmax_module`` bakes ``M`` into the grid on some of its
        internal paths, so the cache key has to include it even though the fast
        path derives the grid from the runtime ``m_rows`` argument.

        Returns:
            The FlyDSL launcher, or ``None`` when the build failed (the key is
            then blacklisted so a broken shape costs one attempt, not one per
            call) or the cache ceiling is reached.
        """
        key = (m, n, tag)
        cached = self._builds.get(key)
        if cached is not None:
            return cached
        if key in self._dead:
            return None
        with self._lock:
            cached = self._builds.get(key)
            if cached is not None:
                return cached
            if key in self._dead:
                return None
            if len(self._builds) >= cache_max():
                log.warning(
                    "forge_kernels: pack %r hit the %d-build ceiling (%s); "
                    "falling back to the framework op for %s. Raise $%s if the "
                    "workload legitimately needs more distinct shapes.",
                    self.name,
                    cache_max(),
                    ENV_CACHE_MAX,
                    key,
                    ENV_CACHE_MAX,
                )
                self._dead.add(key)
                return None
            try:
                builder = getattr(self.load_module(), self.builder)
                launcher = builder(m, n, tag)
            except Exception as e:  # noqa: BLE001 - any build failure => fall back
                log.warning(
                    "forge_kernels: pack %r failed to build %s (%s: %s); blacklisting the shape and falling back",
                    self.name,
                    key,
                    type(e).__name__,
                    e,
                )
                self._dead.add(key)
                return None
            self._builds[key] = launcher
            return launcher

    def build_if_cached(self, m: int, n: int, tag: str) -> Any | None:
        """Cache-only lookup, for use while a CUDA/HIP graph is capturing.

        Building runs the FlyDSL JIT (host compilation plus allocations), which
        is illegal mid-capture, so a cold shape has to fall back instead.
        """
        return self._builds.get((m, n, tag))

    def blacklist(self, m: int, n: int, tag: str) -> None:
        """Permanently fall back for ``(M, N, dtype)`` in this process."""
        with self._lock:
            self._dead.add((m, n, tag))
            self._builds.pop((m, n, tag), None)

    def load_module(self) -> Any:
        """Import ``kernel.py`` (once) with the FlyDSL compat shim applied."""
        if self.module is not None:
            return self.module
        from ._compat import install as install_compat

        install_compat()
        path = self.root / "kernel.py"
        mod_name = f"_hyperloom_forge_pack_{self.name}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load kernel module at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        self.module = module
        return module

    def build_count(self) -> int:
        """Number of distinct (M, N, dtype) modules built so far."""
        return len(self._builds)


_PACKS: dict[str, Pack | None] = {}
_PACKS_LOCK = threading.Lock()


def load_pack(name: str) -> Pack | None:
    """Load an installed pack by name; ``None`` when unusable (fail-soft).

    Unusable covers every reason a serving process should quietly keep using
    upstream code: the pack is not installed, has no preflight report, or
    preflight ran and rejected every shape.
    """
    if name in _PACKS:
        return _PACKS[name]
    with _PACKS_LOCK:
        if name in _PACKS:
            return _PACKS[name]
        pack = _load_pack_uncached(name)
        _PACKS[name] = pack
        return pack


def _load_pack_uncached(name: str) -> Pack | None:
    root = default_root() / name
    manifest = _read_json(root / "pack.json")
    preflight = _read_json(root / "preflight.json")
    if manifest is None:
        log.warning(
            "forge_kernels: pack %r requested via $%s but %s is missing; keeping the framework op",
            name,
            ENV_ENABLED,
            root / "pack.json",
        )
        return None
    if preflight is None:
        log.warning(
            "forge_kernels: pack %r has no preflight.json (never gated on this machine); keeping the framework op",
            name,
        )
        return None
    if not preflight.get("ok"):
        log.warning(
            "forge_kernels: pack %r failed preflight (%s); keeping the framework op",
            name,
            preflight.get("reason") or "no verified shapes",
        )
        return None

    verified: set[tuple[int, str]] = set()
    for entry in preflight.get("verified") or ():
        try:
            verified.add((int(entry["N"]), str(entry["dtype"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not verified:
        log.warning("forge_kernels: pack %r verified no shapes; keeping the framework op", name)
        return None

    log.info(
        "forge_kernels: pack %r active (op=%s, %d verified shape(s), root=%s)",
        name,
        manifest.get("op"),
        len(verified),
        root,
    )
    return Pack(
        name=name,
        root=root,
        op=str(manifest.get("op") or ""),
        builder=str(manifest.get("builder") or ""),
        manifest=manifest,
        preflight=preflight,
        verified=frozenset(verified),
    )


def packs_for_op(op: str) -> list[Pack]:
    """Enabled, loadable packs implementing ``op``, in ``$...PACKS`` order."""
    out: list[Pack] = []
    for name in enabled_pack_names():
        pack = load_pack(name)
        if pack is not None and pack.op == op:
            out.append(pack)
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def reset_for_tests() -> None:
    """Drop the process-wide pack cache (tests flip env between cases)."""
    with _PACKS_LOCK:
        _PACKS.clear()
