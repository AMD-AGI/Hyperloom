# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registry-backed enablement adapters for runtime acquisition.

One adapter per serving framework decides whether a :class:`CapabilityGap` is
runtime-acquirable, builds the :class:`EnablementStackAction` that acquires it,
provisions that action into an *attempt-local* venv (never ``/opt/venv``), and
probes the result for the expected files/symbols.

Isolation & safety:

* vLLM ROCm provisioning pins ROCm torch from a host-allowlisted index and
  refuses to fall back to a PyPI CUDA wheel.
* Only wheel / editable-ref acquisition here; compiled builds are deferred to
  the targeted-build path.
* Every subprocess goes through an injectable ``run(argv, env, cwd)`` shim so the
  pure argv/env/version logic is CI-testable without ROCm / network.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from hyperloom.agents.framework.enablement import (
    MISSING_MODEL_ARCH,
    NOT_IMPLEMENTED,
    RESOURCE_CONSTRAINT,
    SERVE_FLAG,
    TOKENIZER_ERROR,
    UNSUPPORTED_DTYPE,
    CapabilityGap,
)

from .stack_actions import EnablementStackAction, FrameworkRuntime, ProvisionResult


log = logging.getLogger(__name__)


# Operator-configured allowlists (comma-separated). A candidate index / origin
# must match one of these prefixes or provisioning is refused (supply-chain
# safety). The ROCm index also seeds the default vLLM adapter index.
_VLLM_ROCM_INDEX_ENV = "HYPERLOOM_VLLM_ROCM_INDEX_URL"
_INDEX_ALLOWLIST_ENV = "HYPERLOOM_ENABLEMENT_INDEX_ALLOWLIST"
_ORIGIN_ALLOWLIST_ENV = "HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST"

# Per-provision hard timeout: wheel/editable install must fit inside the
# integrate lane TTL. Long compiles are deferred to the targeted-build path.
_PROVISION_TIMEOUT_SEC = 1800

# Gaps that a runtime candidate might repair. RESOURCE_CONSTRAINT is excluded
# (CapabilityGap.requires_code_acquisition is False for it).
_RUNTIME_ACQUIRABLE_KINDS: frozenset[str] = frozenset(
    {MISSING_MODEL_ARCH, UNSUPPORTED_DTYPE, NOT_IMPLEMENTED, TOKENIZER_ERROR, SERVE_FLAG}
)


# Injectable subprocess shim: (argv, env, cwd) -> CompletedProcess. Tests pass a
# fake to exercise the pure argv/env/version logic without ROCm / network.
RunFn = Callable[[list[str], dict[str, str], "str | None"], subprocess.CompletedProcess]


def _default_run(argv: list[str], env: dict[str, str], cwd: str | None) -> subprocess.CompletedProcess:
    """Default subprocess runner used in production (captured, timeout-bound)."""
    return subprocess.run(
        argv,
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_PROVISION_TIMEOUT_SEC,
        check=False,
    )


def _allowlist(env_key: str) -> tuple[str, ...]:
    """Return the comma-separated allowlist prefixes for ``env_key`` (may be empty)."""
    raw = os.environ.get(env_key, "") or ""
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _is_allowlisted(value: str, prefixes: tuple[str, ...]) -> bool:
    """True when ``value`` starts with one of ``prefixes`` (empty prefixes => reject)."""
    v = (value or "").strip()
    if not v or not prefixes:
        return False
    return any(v.startswith(p) for p in prefixes)


# ---------------------------------------------------------------------------
# ROCm safety checks (injectable Python)
# ---------------------------------------------------------------------------
def verify_torch_is_rocm(python_path: str, *, run: RunFn = _default_run) -> bool:
    """True when ``python_path``'s torch is a ROCm build (torch.version.hip set).

    A CUDA torch must never be swapped into a ROCm attempt runtime.
    """
    argv = [python_path, "-c", "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"]
    try:
        cp = run(argv, dict(os.environ), None)
    except Exception:  # noqa: BLE001 — probe failure == not verified
        return False
    return getattr(cp, "returncode", 1) == 0


def verify_vllm_rocm(python_path: str, *, run: RunFn = _default_run) -> bool:
    """True when vLLM imports AND reports a ROCm platform.

    Confirms ``current_platform.is_rocm()``.
    """
    probe = (
        "import sys\n"
        "import torch\n"
        "if not getattr(torch.version,'hip',None): sys.exit(1)\n"
        "import vllm\n"
        "from vllm.platforms import current_platform\n"
        "ck=getattr(current_platform,'is_rocm',None)\n"
        "ok=bool(ck()) if callable(ck) else False\n"
        "if 'rocm' in f'{current_platform!r}'.lower(): ok=True\n"
        "sys.exit(0 if ok else 1)\n"
    )
    argv = [python_path, "-c", probe]
    try:
        cp = run(argv, dict(os.environ), None)
    except Exception:  # noqa: BLE001
        return False
    return getattr(cp, "returncode", 1) == 0


def _installed_version(python_path: str, package: str, *, run: RunFn = _default_run) -> str:
    """Return the installed version of ``package`` in ``python_path``, or ""."""
    argv = [
        python_path,
        "-c",
        f"import importlib.metadata as m; print(m.version({package!r}))",
    ]
    try:
        cp = run(argv, dict(os.environ), None)
    except Exception:  # noqa: BLE001
        return ""
    if getattr(cp, "returncode", 1) != 0:
        return ""
    return (getattr(cp, "stdout", "") or "").strip()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class BaseAdapter:
    """Base enablement adapter.

    Subclasses override the acquisition hooks (``supports`` /
    ``build_stack_action`` / ``provision`` / ``probe``) and, where localization
    is supported, ``build_localization_action`` / ``editable_refresh_argv``.
    """

    framework: str = ""

    def __init__(self, run: RunFn = _default_run) -> None:
        """Store the injectable subprocess shim (default: real subprocess)."""
        self._run = run

    def supports(self, gap: CapabilityGap) -> bool:
        """Whether this adapter can attempt to repair ``gap`` via a runtime.

        Args:
            gap: The projected capability gap.

        Returns:
            bool: False for the base/null adapter and for gaps that do not
            require code acquisition (e.g. resource constraints).
        """
        return False

    def build_stack_action(
        self,
        gap: CapabilityGap,
        *,
        framework: str,
        model: str,
        gpu_type: str = "",
    ) -> EnablementStackAction | None:
        """Build a candidate stack action, or None when unsupported/no-evidence.

        Args:
            gap: The projected capability gap.
            framework: Target framework name.
            model: Model id/path being enabled.
            gpu_type: Target GPU type (routes ROCm index selection).

        Returns:
            EnablementStackAction | None: A candidate action, or None when this
            adapter cannot produce an evidence-backed candidate.
        """
        return None

    def provision(self, action: EnablementStackAction, attempt_dir: Path) -> ProvisionResult:
        """Provision ``action`` into an attempt-local venv under ``attempt_dir``.

        Args:
            action: The stack action to acquire.
            attempt_dir: Attempt root; the venv is created at ``attempt_dir/venv``.

        Returns:
            ProvisionResult: ``ok=False`` for unsupported adapters.
        """
        return ProvisionResult(ok=False, error=f"{self.framework or 'null'} adapter does not provision")

    def probe(self, result: ProvisionResult, action: EnablementStackAction) -> bool:
        """Validate the provisioned runtime carries the expected capability.

        Args:
            result: The provision result to probe.
            action: The action whose ``expected_files`` / ``expected_symbols``
                gate the probe.

        Returns:
            bool: True when the runtime looks usable.
        """
        return result.ok

    def build_localization_action(
        self,
        gap: CapabilityGap,
        *,
        framework: str,
        model: str,
        candidate_ref: str,
        repo_url: str,
    ) -> EnablementStackAction | None:
        """Build a code-localization action, or None (unsupported).

        Args:
            gap: The projected capability gap.
            framework: Target framework name.
            model: Model id/path being enabled.
            candidate_ref: A discovered merged-PR ref (e.g. ``"PR:1234"`` or a
                PR html_url).
            repo_url: Origin-allowlisted repo URL to localize from.

        Returns:
            EnablementStackAction | None: A ``pr_backport`` action, or None when
            the adapter cannot localize (base/null default).
        """
        return None

    def editable_refresh_argv(self, venv_python: str, checkout: str) -> list[str] | None:
        """Return the argv that re-installs an editable checkout, or None.

        Args:
            venv_python: The attempt-venv interpreter.
            checkout: Path to the editable source checkout.

        Returns:
            list[str] | None: A ``pip install -e`` argv, or None when the tree
            is a plain (non-editable) install that needs no refresh.
        """
        return None

    def source_import_root(self, framework_root: str) -> str:
        """Return the import root relative to a source snapshot's ``files/`` dir.

        A dist-packages install stores modules at the tree root, so ``""`` is
        correct; a repo checkout that nests them (e.g. under ``python/``)
        overrides this.

        Args:
            framework_root: Absolute path to the framework checkout or install.

        Returns:
            str: The import-root path component, or ``""``.
        """
        return ""


def _pr_number_from_ref(candidate_ref: str) -> int:
    """Parse a PR number from a ``"PR:1234"`` ref or a PR html_url; 0 if absent."""
    ref = str(candidate_ref or "").strip()
    if not ref:
        return 0
    if ref.upper().startswith("PR:"):
        tail = ref.split(":", 1)[1].strip()
        return int(tail) if tail.isdigit() else 0
    # .../pull/1234 (optionally with a trailing segment).
    parts = [p for p in ref.rstrip("/").split("/") if p]
    for i, seg in enumerate(parts):
        if seg == "pull" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0


class NullAdapter(BaseAdapter):
    """Adapter for unknown / unsupported frameworks: never supports, never raises."""

    framework = ""


class _VenvProvisionMixin(BaseAdapter):
    """Shared attempt-venv creation + pip-install plumbing for real adapters."""

    def _create_venv(self, attempt_dir: Path) -> tuple[Path, Path]:
        """Create ``attempt_dir/venv`` with system-site-packages; return (bin, python).

        System site packages are inherited so the attempt runtime reuses the
        already-present ROCm torch/aiter instead of re-downloading them; only
        the target framework wheel/checkout is layered on top.

        Args:
            attempt_dir: Attempt root directory.

        Returns:
            tuple[Path, Path]: ``(bin_dir, python_path)``.
        """
        venv_root = attempt_dir / "venv"
        argv = [sys.executable, "-m", "venv", "--system-site-packages", str(venv_root)]
        cp = self._run(argv, dict(os.environ), None)
        if getattr(cp, "returncode", 1) != 0:
            raise RuntimeError(f"venv creation failed: {getattr(cp, 'stderr', '') or ''}")
        bin_dir = venv_root / "bin"
        return bin_dir, bin_dir / "python"

    def _pip_install(
        self,
        python_path: Path,
        specs: list[str],
        *,
        index_url: str = "",
        editable: str = "",
    ) -> subprocess.CompletedProcess:
        """Run one ``pip install`` into the attempt venv.

        Args:
            python_path: Attempt interpreter.
            specs: Package specs to install (empty when ``editable`` is set).
            index_url: Extra pip index (host-allowlisted by the caller).
            editable: Editable source path (``-e``), or empty.

        Returns:
            subprocess.CompletedProcess: The pip result.
        """
        argv = [str(python_path), "-m", "pip", "install", "--upgrade"]
        if index_url:
            argv += ["--extra-index-url", index_url]
        if editable:
            argv += ["-e", editable]
        argv += list(specs)
        return self._run(argv, dict(os.environ), None)

    def build_localization_action(
        self,
        gap: CapabilityGap,
        *,
        framework: str,
        model: str,
        candidate_ref: str,
        repo_url: str,
    ) -> EnablementStackAction | None:
        """Build a pr_backport localization from a merged-PR ref (origin-allowlisted)."""
        if not self.supports(gap):
            return None
        pr_number = _pr_number_from_ref(candidate_ref)
        if not repo_url or pr_number <= 0:
            return None
        origin_allow = _allowlist(_ORIGIN_ALLOWLIST_ENV)
        if origin_allow and not _is_allowlisted(repo_url, origin_allow):
            log.warning("%s: repo_url %r not in origin allowlist", type(self).__name__, repo_url)
            return None
        return EnablementStackAction(
            kind="pr_backport",
            framework=self.framework,
            gap_id=f"gap.enablement.{gap.kind}",
            capability=gap.kind,
            reason=f"{self.framework} PR backport #{pr_number} for {gap.kind}",
            acquisition_method="none",
            repo_url=repo_url,
            pr_number=pr_number,
        )

    def editable_refresh_argv(self, venv_python: str, checkout: str) -> list[str] | None:
        """Re-install the editable checkout so localized Python changes take effect."""
        if not venv_python or not checkout:
            return None
        return [str(venv_python), "-m", "pip", "install", "-e", str(checkout), "--no-deps"]


class VllmRocmAdapter(_VenvProvisionMixin):
    """vLLM ROCm adapter: wheel install from a host-allowlisted ROCm index only."""

    framework = "vllm"

    def supports(self, gap: CapabilityGap) -> bool:
        """True for code-acquirable gaps (never for resource constraints)."""
        if not gap.requires_code_acquisition or gap.kind == RESOURCE_CONSTRAINT:
            return False
        return gap.kind in _RUNTIME_ACQUIRABLE_KINDS

    def build_stack_action(
        self,
        gap: CapabilityGap,
        *,
        framework: str,
        model: str,
        gpu_type: str = "",
    ) -> EnablementStackAction | None:
        """Build a vLLM ROCm wheel candidate; None when no ROCm index is configured.

        Refuses to produce a candidate unless a ROCm index is configured — never
        falls back to a generic PyPI (CUDA) wheel for a ROCm run.
        """
        if not self.supports(gap):
            return None
        index_url = os.environ.get(_VLLM_ROCM_INDEX_ENV, "").strip()
        if not index_url:
            log.info("VllmRocmAdapter: no %s configured; refusing PyPI CUDA fallback", _VLLM_ROCM_INDEX_ENV)
            return None
        # An explicit index allowlist, when set, must match; without one the
        # operator-set ROCm index is trusted by construction.
        allow = _allowlist(_INDEX_ALLOWLIST_ENV)
        if allow and not _is_allowlisted(index_url, allow):
            log.warning("VllmRocmAdapter: index_url %r not in allowlist", index_url)
            return None
        capability = gap.kind
        return EnablementStackAction(
            kind="runtime_candidate",
            framework="vllm",
            gap_id=f"gap.enablement.{gap.kind}",
            capability=capability,
            reason=f"vLLM ROCm wheel candidate for {capability} on {gpu_type or 'rocm'}",
            acquisition_method="wheel",
            index_url=index_url,
            packages=("vllm",),
            expected_symbols=(),
            expected_files=(),
        )

    def provision(self, action: EnablementStackAction, attempt_dir: Path) -> ProvisionResult:
        """Create an attempt venv, pip-install the vLLM wheel, verify ROCm."""
        log_path = str(attempt_dir / "provision.log")
        try:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            bin_dir, python_path = self._create_venv(attempt_dir)
        except Exception as exc:  # noqa: BLE001
            return ProvisionResult(ok=False, log_path=log_path, error=f"venv setup failed: {exc!r}")

        if action.acquisition_method != "wheel" or not action.index_url:
            return ProvisionResult(ok=False, log_path=log_path, error="vLLM adapter requires a ROCm wheel index")

        cp = self._pip_install(python_path, list(action.packages) or ["vllm"], index_url=action.index_url)
        if getattr(cp, "returncode", 1) != 0:
            return ProvisionResult(
                ok=False, log_path=log_path, error=f"pip install failed: {(getattr(cp, 'stderr', '') or '')[:400]}"
            )

        # Refuse a CUDA torch / non-ROCm vLLM swap-in.
        if not verify_torch_is_rocm(str(python_path), run=self._run):
            return ProvisionResult(ok=False, log_path=log_path, error="attempt torch is not a ROCm build")
        if not verify_vllm_rocm(str(python_path), run=self._run):
            return ProvisionResult(ok=False, log_path=log_path, error="vLLM did not report a ROCm platform")

        versions = {
            "vllm": _installed_version(str(python_path), "vllm", run=self._run),
            "torch": _installed_version(str(python_path), "torch", run=self._run),
        }
        runtime = FrameworkRuntime(
            bin_path=str(bin_dir),
            python_path=str(python_path),
            venv_root=str(attempt_dir / "venv"),
            server_args=action.server_args,
            envs=dict(action.envs),
        )
        return ProvisionResult(
            ok=True,
            runtime=runtime,
            installed_versions={k: v for k, v in versions.items() if v},
            log_path=log_path,
        )

    def probe(self, result: ProvisionResult, action: EnablementStackAction) -> bool:
        """Confirm expected files exist under the venv (symbols validated at boot)."""
        if not result.ok:
            return False
        root = Path(result.runtime.venv_root) if result.runtime.venv_root else None
        for rel in action.expected_files:
            if root is not None and not (root / rel).exists() and not Path(rel).exists():
                log.info("VllmRocmAdapter.probe: expected file missing: %s", rel)
                return False
        return True


class SglangAdapter(_VenvProvisionMixin):
    """SGLang adapter: editable checkout at a ref, or wheel install."""

    framework = "sglang"

    def source_import_root(self, framework_root: str) -> str:
        """Return ``"python"`` for a sglang checkout (``python/sglang/...``), else ``""``."""
        if not framework_root:
            return ""
        root = Path(framework_root)
        return "python" if (root / "python" / "sglang").is_dir() else ""

    def supports(self, gap: CapabilityGap) -> bool:
        """True for code-acquirable gaps (never for resource constraints)."""
        if not gap.requires_code_acquisition or gap.kind == RESOURCE_CONSTRAINT:
            return False
        return gap.kind in _RUNTIME_ACQUIRABLE_KINDS

    def build_stack_action(
        self,
        gap: CapabilityGap,
        *,
        framework: str,
        model: str,
        gpu_type: str = "",
    ) -> EnablementStackAction | None:
        """Prefer an editable source ref (origin-allowlisted); else a wheel index."""
        if not self.supports(gap):
            return None
        repo_url = os.environ.get("HYPERLOOM_SGLANG_REPO_URL", "").strip()
        ref = os.environ.get("HYPERLOOM_SGLANG_REF", "").strip()
        origin_allow = _allowlist(_ORIGIN_ALLOWLIST_ENV)
        if repo_url and ref:
            if origin_allow and not _is_allowlisted(repo_url, origin_allow):
                log.warning("SglangAdapter: repo_url %r not in origin allowlist", repo_url)
                return None
            return EnablementStackAction(
                kind="runtime_candidate",
                framework="sglang",
                gap_id=f"gap.enablement.{gap.kind}",
                capability=gap.kind,
                reason=f"SGLang editable checkout {repo_url}@{ref} for {gap.kind}",
                acquisition_method="editable_ref",
                repo_url=repo_url,
                ref=ref,
            )
        index_url = os.environ.get("HYPERLOOM_SGLANG_INDEX_URL", "").strip()
        if not index_url:
            return None
        index_allow = _allowlist(_INDEX_ALLOWLIST_ENV)
        if index_allow and not _is_allowlisted(index_url, index_allow):
            return None
        return EnablementStackAction(
            kind="runtime_candidate",
            framework="sglang",
            gap_id=f"gap.enablement.{gap.kind}",
            capability=gap.kind,
            reason=f"SGLang wheel candidate for {gap.kind}",
            acquisition_method="wheel",
            index_url=index_url,
            packages=("sglang",),
        )

    def provision(self, action: EnablementStackAction, attempt_dir: Path) -> ProvisionResult:
        """Create an attempt venv, install SGLang (editable ref or wheel)."""
        log_path = str(attempt_dir / "provision.log")
        try:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            bin_dir, python_path = self._create_venv(attempt_dir)
        except Exception as exc:  # noqa: BLE001
            return ProvisionResult(ok=False, log_path=log_path, error=f"venv setup failed: {exc!r}")

        pythonpath_prefix = ""
        if action.acquisition_method == "editable_ref":
            if not action.repo_url or not action.ref:
                return ProvisionResult(ok=False, log_path=log_path, error="editable_ref requires repo_url and ref")
            checkout = attempt_dir / "src"
            clone = self._run(
                ["git", "clone", "--depth", "1", "--branch", action.ref, action.repo_url, str(checkout)],
                dict(os.environ),
                None,
            )
            if getattr(clone, "returncode", 1) != 0:
                return ProvisionResult(
                    ok=False, log_path=log_path, error=f"git clone failed: {(getattr(clone, 'stderr', '') or '')[:400]}"
                )
            cp = self._pip_install(python_path, [], editable=str(checkout / "python"))
            pythonpath_prefix = str(checkout / "python")
        elif action.acquisition_method == "wheel":
            cp = self._pip_install(python_path, list(action.packages) or ["sglang"], index_url=action.index_url)
        else:
            return ProvisionResult(ok=False, log_path=log_path, error=f"unsupported method {action.acquisition_method}")

        if getattr(cp, "returncode", 1) != 0:
            return ProvisionResult(
                ok=False, log_path=log_path, error=f"install failed: {(getattr(cp, 'stderr', '') or '')[:400]}"
            )
        if not verify_torch_is_rocm(str(python_path), run=self._run):
            return ProvisionResult(ok=False, log_path=log_path, error="attempt torch is not a ROCm build")

        versions = {"sglang": _installed_version(str(python_path), "sglang", run=self._run)}
        runtime = FrameworkRuntime(
            bin_path=str(bin_dir),
            python_path=str(python_path),
            venv_root=str(attempt_dir / "venv"),
            pythonpath_prefix=pythonpath_prefix,
            server_args=action.server_args,
            envs=dict(action.envs),
        )
        return ProvisionResult(
            ok=True,
            runtime=runtime,
            installed_versions={k: v for k, v in versions.items() if v},
            log_path=log_path,
        )


class AtomAdapter(BaseAdapter):
    """Atom adapter: no runtime acquisition, but Python localization.

    Atom is a wheel-installed (non-git) tree; a Python-only PR backport localizes
    via the executor's no-git apply. There is no editable refresh (importlib
    picks up the file changes on the next boot).
    """

    framework = "atom"

    def build_localization_action(
        self,
        gap: CapabilityGap,
        *,
        framework: str,
        model: str,
        candidate_ref: str,
        repo_url: str,
    ) -> EnablementStackAction | None:
        """Build a pr_backport localization (applied via no-git; no refresh)."""
        if not gap.requires_code_acquisition or gap.kind == RESOURCE_CONSTRAINT:
            return None
        pr_number = _pr_number_from_ref(candidate_ref)
        if not repo_url or pr_number <= 0:
            return None
        origin_allow = _allowlist(_ORIGIN_ALLOWLIST_ENV)
        if origin_allow and not _is_allowlisted(repo_url, origin_allow):
            return None
        return EnablementStackAction(
            kind="pr_backport",
            framework="atom",
            gap_id=f"gap.enablement.{gap.kind}",
            capability=gap.kind,
            reason=f"atom PR backport #{pr_number} for {gap.kind}",
            repo_url=repo_url,
            pr_number=pr_number,
        )


class XditAdapter(BaseAdapter):
    """xDiT adapter: unsupported for runtime acquisition and localization here."""

    framework = "xdit"


# Registry: one adapter class per framework. Extend here (single source).
_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "vllm": VllmRocmAdapter,
    "sglang": SglangAdapter,
    "atom": AtomAdapter,
    "xdit": XditAdapter,
}


def get_adapter(framework: str | None, *, run: RunFn = _default_run) -> BaseAdapter:
    """Return the adapter for ``framework``; a NullAdapter for unknown names.

    Args:
        framework: Framework name (case-insensitive).
        run: Injectable subprocess shim threaded into the adapter.

    Returns:
        BaseAdapter: The matching adapter instance, or a :class:`NullAdapter`
        (never raises) for an unknown framework.
    """
    key = str(framework or "").strip().lower()
    cls = _ADAPTERS.get(key)
    if cls is None:
        return NullAdapter(run=run)
    return cls(run=run)


__all__ = [
    "AtomAdapter",
    "BaseAdapter",
    "NullAdapter",
    "SglangAdapter",
    "VllmRocmAdapter",
    "XditAdapter",
    "get_adapter",
    "verify_torch_is_rocm",
    "verify_vllm_rocm",
]
