# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registry-backed enablement adapters for runtime acquisition."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Callable

from hyperloom.agents.framework.isolation import checkout_ref, create_venv
from hyperloom.common.failure_signature import FailureSignature
from hyperloom.inference_optimizer import framework_registry

from .build_actions import resolve_build_ref
from .build_utils import ROCM_CONSTRAINTS_FILE, sglang_rocm_install_target, write_rocm_constraints
from .stack_actions import EnablementStackAction, FrameworkRuntime, ProvisionResult, RuntimeRequest


log = logging.getLogger(__name__)


# Per-provision hard timeout: wheel/editable install must fit inside the integrate lane TTL.
_PROVISION_TIMEOUT_SEC = 1800

# pip/git put the error last; earlier lines are warnings.
_FAILURE_TAIL_CHARS = 1500

_PROVISION_LOG = "provision.log"


# Injectable subprocess shim: (argv, env, cwd) -> CompletedProcess.
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


# ROCm safety checks (injectable Python)
def verify_torch_is_rocm(python_path: str, *, run: RunFn = _default_run) -> bool:
    """True when ``python_path``'s torch is a ROCm build (torch.version.hip set)."""
    argv = [python_path, "-c", "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"]
    try:
        cp = run(argv, dict(os.environ), None)
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(cp, "returncode", 1) == 0


def verify_vllm_rocm(python_path: str, *, run: RunFn = _default_run) -> bool:
    """True when vLLM imports AND reports a ROCm platform."""
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
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(cp, "returncode", 1) == 0


def _stdout(argv: list[str], *, run: RunFn) -> str:
    """Return ``argv``'s stripped stdout, or ``""`` when it cannot run or exits non-zero."""
    try:
        cp = run(argv, dict(os.environ), None)
    except (OSError, subprocess.SubprocessError):
        return ""
    if getattr(cp, "returncode", 1) != 0:
        return ""
    return (getattr(cp, "stdout", "") or "").strip()


def _failure_tail(cp: subprocess.CompletedProcess) -> str:
    """The end of ``cp``'s stderr, which carries the error."""
    return (cp.stderr or "").strip()[-_FAILURE_TAIL_CHARS:]


def _resolved_clone_ref(checkout: str, *, run: RunFn = _default_run) -> str:
    """Return the commit a shallow clone landed on, or ``""``.

    The provisioner clones a branch or tag verbatim, so the action's own ``ref``
    names different bytes tomorrow; this is the identity it lacks.
    """
    return _stdout(["git", "-C", str(checkout), "rev-parse", "HEAD"], run=run)


def _resolved_packages(python_path: str, names: list[str], *, run: RunFn = _default_run) -> dict[str, dict[str, str]]:
    """Return ``{name: {version, artifact_digest}}`` for each installed name.

    The digest is a sha256 over the distribution's own ``RECORD`` manifest, read
    in the same interpreter: a version names the release the metadata claims,
    not the bytes installed. A distribution publishing no ``RECORD`` carries an
    empty digest, which is a judged condition rather than a default.
    """
    if not names:
        return {}
    # Source for the attempt interpreter, not this one: a name it cannot resolve
    # is skipped and a ``RECORD`` it cannot read yields the empty digest, while
    # anything else fails the probe's exit status and yields no packages.
    probe = (
        "import hashlib,json,re,sys\n"
        "import importlib.metadata as m\n"
        "out={}\n"
        # ``action.packages`` holds requirement strings -- ``vllm>=0.10``,
        # ``vllm[all]``, ``aiter @ https://...`` -- and only a bare name resolves;
        # everything else raised PackageNotFoundError, was skipped, and left the
        # map empty, which reads downstream as an acquisition that pinned nothing.
        "def _dist_name(spec):\n"
        "    head=re.split(r'[;@]', spec, 1)[0]\n"
        "    return re.split(r'[\\[<>=!~ ]', head.strip(), 1)[0].strip()\n"
        "for spec in sys.argv[1:]:\n"
        "    name=_dist_name(spec)\n"
        "    if not name:\n"
        "        continue\n"
        "    try:\n"
        "        dist=m.distribution(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        continue\n"
        "    digest=''\n"
        "    try:\n"
        "        record=dist.read_text('RECORD') or ''\n"
        "        digest='sha256:'+hashlib.sha256(record.encode()).hexdigest() if record else ''\n"
        "    except (OSError, ValueError):\n"
        "        digest=''\n"
        "    out[name]={'version': dist.version or '', 'artifact_digest': digest}\n"
        "print(json.dumps(out))\n"
    )
    try:
        parsed = json.loads(_stdout([python_path, "-c", probe, *names], run=run) or "{}")
    except ValueError:
        return {}
    return {str(k): {str(kk): str(vv) for kk, vv in v.items()} for k, v in parsed.items()}


def _site_packages(python_path: str, *, run: RunFn = _default_run) -> str:
    """Return the directory ``python_path`` installs distributions into, or ``""``.

    The attempt venv inherits the host's system site-packages, so a wheel
    installed into it occupies this tree alone -- which is therefore the only
    tree a patch against that wheel's sources can land in.
    """
    return _stdout([python_path, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], run=run)


def _installed_version(python_path: str, package: str, *, run: RunFn = _default_run) -> str:
    """Return the installed version of ``package`` in ``python_path``, or ""."""
    return _stdout([python_path, "-c", f"import importlib.metadata as m; print(m.version({package!r}))"], run=run)


# Adapters
class BaseAdapter:
    """Base enablement adapter."""

    framework: str = ""
    # The importable package's directory in the upstream repo; localization keeps only files under it.
    package_path: str = ""

    def __init__(self, run: RunFn = _default_run) -> None:
        """Store the injectable subprocess shim (default: real subprocess)."""
        self._run = run

    def build_stack_action(self, request: RuntimeRequest) -> EnablementStackAction | None:
        """Turn a specialist's runtime request into a stack action, or None when this framework cannot acquire it."""
        return None

    def provision(self, action: EnablementStackAction, attempt_dir: Path) -> ProvisionResult:
        """Provision ``action`` into an attempt-local venv under ``attempt_dir``.

        Blocking (venv/pip, 1800s cap); call via ``asyncio.to_thread``.
        Cancel of the await is fire-and-forget for an in-flight pip install.
        """
        return ProvisionResult(ok=False, error=f"{self.framework or 'null'} adapter does not provision")

    def probe(self, result: ProvisionResult, action: EnablementStackAction) -> bool:
        """Validate the provisioned runtime carries the expected capability."""
        return result.ok

    def build_localization_action(
        self,
        signature: FailureSignature,
        *,
        candidate_ref: str,
        repo_url: str,
    ) -> EnablementStackAction | None:
        """Build a pr_backport localization from a PR ref, or None when this framework has no package to localize into."""
        if not self.package_path:
            return None
        pr_number = _pr_number_from_ref(candidate_ref)
        if pr_number <= 0:
            return None
        return EnablementStackAction(
            kind="pr_backport",
            framework=self.framework,
            gap_id=f"gap.enablement.{signature.kind}",
            capability=signature.kind,
            reason=f"{self.framework} PR backport #{pr_number} for {signature.kind}",
            acquisition_method="none",
            repo_url=repo_url,
            pr_number=pr_number,
        )

    def source_import_root(self, framework_root: str) -> str:
        """Return the import root relative to a source snapshot's ``files/`` dir.

        The directory above the package when the checkout holds it, so a
        ``python/sglang`` layout gives ``python`` and a top-level package none.
        """
        if not framework_root or not (Path(framework_root) / self.package_path).is_dir():
            return ""
        parent = PurePosixPath(self.package_path).parent.as_posix()
        return "" if parent == "." else parent


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
    """Adapter for unknown / unsupported frameworks: acquires nothing, never raises."""

    framework = ""


class _VenvProvisionMixin(BaseAdapter):
    """Attempt-venv acquisition shared by the adapters that provision a runtime."""

    def build_stack_action(self, request: RuntimeRequest) -> EnablementStackAction | None:
        """An editable checkout of ``request.ref`` with its packages beside it, or the packages alone."""
        capability = request.capability or "runtime"
        action = EnablementStackAction(
            kind="runtime_candidate",
            framework=self.framework,
            gap_id=f"gap.enablement.{capability}",
            capability=capability,
            reason=request.reason or f"{self.framework} runtime with {', '.join(request.packages)}",
            acquisition_method="wheel",
            index_url=request.index_url,
            packages=request.packages,
        )
        if not request.ref:
            return action
        repo_url, ref, _ = resolve_build_ref(
            request.ref, request.repo_url or framework_registry.repo_url(self.framework)
        )
        if not repo_url or not ref:
            log.warning("%s: runtime ref %r names no checkout", type(self).__name__, request.ref)
            return None
        return replace(
            action,
            acquisition_method="editable_ref",
            repo_url=repo_url,
            ref=ref,
            reason=request.reason or f"{self.framework} editable checkout {repo_url}@{ref}",
        )

    def _source_root_in(self, purelib: str) -> str:
        """``purelib`` when the install put the framework there, else no tree: it is still imported from the shared one.

        A runtime's source root is where its patches land, so it must be the
        tree the server executes.
        """
        return purelib if (Path(purelib) / PurePosixPath(self.package_path).name).is_dir() else ""

    def _prepare_attempt_venv(self, attempt_dir: Path) -> tuple[Path, Path]:
        """Reset the provision log, create ``attempt_dir/venv`` and pin its ROCm stack; return (bin, python)."""
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / _PROVISION_LOG).write_bytes(b"")
        bin_dir = attempt_dir / "venv" / "bin"
        create_venv(bin_dir.parent)
        write_rocm_constraints(
            str(bin_dir / "python"),
            attempt_dir / ROCM_CONSTRAINTS_FILE,
            replacing=self.framework,
            run=lambda argv, **_kwargs: self._run(argv, dict(os.environ), None),
        )
        return bin_dir, bin_dir / "python"

    def _logged_run(self, argv: list[str], attempt_dir: Path) -> subprocess.CompletedProcess:
        """Run ``argv`` and append its command, exit code and full output to the attempt's provision log."""
        cp = self._run(argv, dict(os.environ), None)
        with (attempt_dir / _PROVISION_LOG).open("a", encoding="utf-8") as fh:
            fh.write(f"$ {shlex.join(argv)}\nrc={cp.returncode}\n")
            fh.write(f"{cp.stdout or ''}{cp.stderr or ''}\n")
        return cp

    def _pip_install(
        self,
        python_path: Path,
        specs: list[str],
        attempt_dir: Path,
        *,
        index_url: str = "",
        editable: str = "",
    ) -> subprocess.CompletedProcess:
        """Run one ``pip install`` into the attempt venv, held to the attempt's ROCm constraints."""
        argv = [str(python_path), "-m", "pip", "install", "--constraint", str(attempt_dir / ROCM_CONSTRAINTS_FILE)]
        if index_url:
            argv += ["--extra-index-url", index_url]
        if editable:
            argv += ["-e", editable]
        argv += list(specs)
        return self._logged_run(argv, attempt_dir)

    def _checkout(self, repo_url: str, ref: str, dest: Path, attempt_dir: Path) -> subprocess.CompletedProcess:
        """Shallow-fetch ``ref`` of ``repo_url`` into ``dest`` and detach on it; returns the first failing step."""
        dest.mkdir(parents=True, exist_ok=True)
        git = ["git", "-C", str(dest)]
        steps = (
            ["init", "-q"],
            ["fetch", "-q", "--depth", "1", repo_url, checkout_ref(ref)],
            ["checkout", "-q", "--detach", "FETCH_HEAD"],
        )
        for step in steps:
            cp = self._logged_run([*git, *step], attempt_dir)
            if getattr(cp, "returncode", 1) != 0:
                break
        return cp


class VllmRocmAdapter(_VenvProvisionMixin):
    """vLLM ROCm adapter: package installs over the image's ROCm vLLM."""

    framework = "vllm"
    package_path = "vllm"

    def build_stack_action(self, request: RuntimeRequest) -> EnablementStackAction | None:
        """Packages only: a vLLM checkout compiles, which is a targeted build."""
        if request.ref:
            log.warning(
                "VllmRocmAdapter: runtime ref %r refused; a vLLM checkout needs needs_targeted_build", request.ref
            )
            return None
        return super().build_stack_action(request)

    def provision(self, action: EnablementStackAction, attempt_dir: Path) -> ProvisionResult:
        """Create an attempt venv, pip-install the requested packages, verify ROCm.

        Blocking (venv/pip, 1800s cap); call via ``asyncio.to_thread``.
        """
        log_path = str(attempt_dir / _PROVISION_LOG)
        try:
            bin_dir, python_path = self._prepare_attempt_venv(attempt_dir)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return ProvisionResult(ok=False, log_path=log_path, error=f"venv setup failed: {exc!r}")

        if action.acquisition_method != "wheel":
            return ProvisionResult(ok=False, log_path=log_path, error=f"unsupported method {action.acquisition_method}")

        cp = self._pip_install(python_path, list(action.packages), attempt_dir, index_url=action.index_url)
        if getattr(cp, "returncode", 1) != 0:
            return ProvisionResult(ok=False, log_path=log_path, error=f"pip install failed: {_failure_tail(cp)}")

        # Refuse a CUDA torch / non-ROCm vLLM swap-in.
        if not verify_torch_is_rocm(str(python_path), run=self._run):
            return ProvisionResult(ok=False, log_path=log_path, error="attempt torch is not a ROCm build")
        if not verify_vllm_rocm(str(python_path), run=self._run):
            return ProvisionResult(ok=False, log_path=log_path, error="vLLM did not report a ROCm platform")
        purelib = _site_packages(str(python_path), run=self._run)
        if not purelib:
            return ProvisionResult(ok=False, log_path=log_path, error="attempt interpreter reports no site-packages")

        versions = {
            "vllm": _installed_version(str(python_path), "vllm", run=self._run),
            "torch": _installed_version(str(python_path), "torch", run=self._run),
        }
        resolved_packages = _resolved_packages(str(python_path), list(action.packages), run=self._run)
        runtime = FrameworkRuntime(
            bin_path=str(bin_dir),
            python_path=str(python_path),
            venv_root=str(attempt_dir / "venv"),
            server_args=action.server_args,
            envs=dict(action.envs),
            source_root=self._source_root_in(purelib),
        )
        return ProvisionResult(
            ok=True,
            runtime=runtime,
            installed_versions={k: v for k, v in versions.items() if v},
            log_path=log_path,
            resolved_packages=resolved_packages,
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
    """SGLang adapter: editable checkout at a ref, package installs, or both."""

    framework = "sglang"
    package_path = "python/sglang"

    def provision(self, action: EnablementStackAction, attempt_dir: Path) -> ProvisionResult:
        """Create an attempt venv, install SGLang from a checkout and/or the requested packages.

        Blocking (venv/pip, 1800s cap); call via ``asyncio.to_thread``.
        """
        log_path = str(attempt_dir / _PROVISION_LOG)
        try:
            bin_dir, python_path = self._prepare_attempt_venv(attempt_dir)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return ProvisionResult(ok=False, log_path=log_path, error=f"venv setup failed: {exc!r}")

        pythonpath_prefix = ""
        resolved_ref = ""
        if action.acquisition_method == "editable_ref":
            if not action.repo_url or not action.ref:
                return ProvisionResult(ok=False, log_path=log_path, error="editable_ref requires repo_url and ref")
            checkout = attempt_dir / "src"
            fetched = self._checkout(action.repo_url, action.ref, checkout, attempt_dir)
            if getattr(fetched, "returncode", 1) != 0:
                return ProvisionResult(
                    ok=False,
                    log_path=log_path,
                    error=f"git checkout of {action.ref} failed: {_failure_tail(fetched)}",
                )
            torch_version = _installed_version(str(python_path), "torch", run=self._run)
            try:
                with sglang_rocm_install_target(checkout / "python", torch_version) as target:
                    cp = self._pip_install(python_path, list(action.packages), attempt_dir, editable=target)
            except (OSError, ValueError) as exc:
                return ProvisionResult(ok=False, log_path=log_path, error=f"no ROCm install of {action.ref}: {exc}")
            pythonpath_prefix = str(checkout / "python")
            # The editable install imports straight out of the clone, and its
            # diffs are cut against the repo root that holds ``python/sglang``.
            source_root = str(checkout)
            resolved_ref = _resolved_clone_ref(str(checkout), run=self._run)
        elif action.acquisition_method == "wheel":
            cp = self._pip_install(python_path, list(action.packages), attempt_dir, index_url=action.index_url)
            purelib = _site_packages(str(python_path), run=self._run)
            if not purelib:
                return ProvisionResult(
                    ok=False, log_path=log_path, error="attempt interpreter reports no site-packages"
                )
            source_root = self._source_root_in(purelib)
        else:
            return ProvisionResult(ok=False, log_path=log_path, error=f"unsupported method {action.acquisition_method}")

        if getattr(cp, "returncode", 1) != 0:
            return ProvisionResult(ok=False, log_path=log_path, error=f"install failed: {_failure_tail(cp)}")
        if not verify_torch_is_rocm(str(python_path), run=self._run):
            return ProvisionResult(ok=False, log_path=log_path, error="attempt torch is not a ROCm build")
        resolved_packages = _resolved_packages(str(python_path), list(action.packages), run=self._run)

        versions = {"sglang": _installed_version(str(python_path), "sglang", run=self._run)}
        runtime = FrameworkRuntime(
            bin_path=str(bin_dir),
            python_path=str(python_path),
            venv_root=str(attempt_dir / "venv"),
            pythonpath_prefix=pythonpath_prefix,
            server_args=action.server_args,
            envs=dict(action.envs),
            source_root=source_root,
        )
        return ProvisionResult(
            ok=True,
            runtime=runtime,
            installed_versions={k: v for k, v in versions.items() if v},
            log_path=log_path,
            resolved_ref=resolved_ref,
            resolved_packages=resolved_packages,
        )


class AtomAdapter(BaseAdapter):
    """Atom adapter: no runtime acquisition, but Python localization."""

    framework = "atom"
    package_path = "atom"


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
    """Return the adapter for ``framework``; a NullAdapter for unknown names."""
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
