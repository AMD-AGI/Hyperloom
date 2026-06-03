"""Session manifest writer ().

The manifest is the **first** file written to a session directory after
``make_session_dir()`` runs and is the canonical session-resume tag.
Atomic write via tmp + ``os.replace``.

Why this lives in its own module:

* Separation of concerns — ``paths.py`` owns where things live;
  ``manifest.py`` owns what's in the resume tag.
* Avoids a hard dependency from ``paths`` (called by everything) on
  ``argparse`` and Python version helpers.

Schema v3 (dependencies added)::

    {
      "schema_version":    3,
      "session_id":        "<UTC_YYYYMMDDTHHMMSSZ>_<uuid8>",
      "claw_session_id":   "<uuid>" or null,   # Primus-Claw session UUID
      "sandbox_user_id":   "<str>"  or null,   # Primus-Claw sandbox user
      "created_at_utc":    "...",
      "host":              "...",
      "image":             "<registry>/<repo>:<tag>" or null,
      "model_path":        "...",
      "model_name":        "...",
      "framework":         "sglang|vllm|atom",
      "gpu_type":          "mi300x|mi325x|mi355x|''",
      "tp":                N or null,
      "workload":          {"isl":..., "osl":..., "max_model_len":...,
                            "precision":..., "conc":...},
      "objective":         {"kind":"gain_pct|tput|baseline|time_only",
                            "value":...},
      "max_minutes":       N,
      "code_revision":     "<git sha or empty>",
      "dependencies":      {
        "magpie":     {"path": "...", "commit": "<git sha or empty>",
                        "remote": "<origin url or empty>"},
        "inferencex": {"path": "...", "commit": "<git sha or empty>",
                        "remote": "<origin url or empty>"},
      },
      "pid":               N
    }

The ``dependencies`` block records the on-disk Magpie / InferenceX
checkout each session ran against. With install.sh now `git clone`-ing
a fresh InferenceX per install (rather than scanning a shared `wekafs`
mirror, see `bugs.md` §C #1 root-cause fix), the commit fields are the
only reliable provenance for "which upstream did this run actually
exercise". The block is best-effort: each subfield is empty when the
path is unset or the directory is not a git checkout. We do **not**
fail the manifest write on git lookup failures — provenance gaps must
not block session bring-up.

``claw_session_id`` / ``sandbox_user_id`` are read from the
``CLAW_SESSION_ID`` / ``SANDBOX_USER_ID`` env vars (set by the
Primus-Claw spawn path); they are ``null`` when Hyperloom runs standalone
outside the claw sandbox.

``image`` records the container image the run executed inside, for
later reproducibility / dashboard provenance. Detection priority:

1. ``HYPERLOOM_IMAGE`` env var (preferred — explicitly set by the spawn).
2. ``CONTAINER_IMAGE`` / ``IMAGE`` env vars (fallback for non-claw spawns).
3. ``/etc/podinfo/image`` (k8s downward API mount).
4. ``/etc/hyperloom-image`` (legacy spawn-script convention).
5. Best-effort parse of ``/proc/1/cgroup`` (extracts the container hash;
   reported as ``unknown@<sha256_short>``).
6. ``None`` when nothing matches; consumers warn rather than fabricate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .session_paths import manifest_path

log = logging.getLogger(__name__)

# Schema bumped to 3 in the legacy release to add ``stack_fingerprint`` (rocm / aiter /
# sglang / vllm versions, mandatory attrs for Cortex KB ``session begin``
# per KB_design §3.6.5.1 + §3.13 M1) plus the ``dependencies`` provenance
# block (Magpie / InferenceX commit + remote — bugs.md §C #1). Older v2
# readers stay compatible because all new fields are additive.
SCHEMA_VERSION = 3


def _utc_now_compact() -> str:
    """Format the current UTC time as a compact session-id timestamp.

    Returns:
        str: Timestamp in ``YYYYMMDDTHHMMSSZ`` form.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# Environment variables consulted by :func:`_detect_stack_fingerprint`.
# Operators may pin them ahead of launch when the auto-detect heuristic
# (importing the framework + reading a marker file) is too slow or runs
# inside a sandbox that doesn't expose the source tree.
_STACK_FINGERPRINT_ENVS: dict[str, tuple[str, ...]] = {
    "rocm":   ("ROCM_VERSION", "HIP_VERSION"),
    "aiter":  ("AITER_COMMIT", "AITER_VERSION"),
    "sglang": ("SGLANG_VERSION", "SGL_VERSION"),
    "vllm":   ("VLLM_VERSION",),
}


def _read_first_line(path: Path) -> str:
    """Return the first non-empty, stripped line of a file.

    Args:
        path (Path): File to read.

    Returns:
        str: First non-blank line stripped of surrounding whitespace, or an
        empty string when the file is missing, empty, or unreadable.
    """
    try:
        if not path.exists():
            return ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s:
                return s
    except OSError:
        return ""
    return ""


def _detect_stack_fingerprint() -> dict[str, str]:
    """Best-effort ``stack_fingerprint`` per KB_design §3.6.5.1.

    Resolution order per component (first non-empty wins):

    1. Explicit env var (operator pin / cluster spawn injection).
    2. ``/opt/rocm/.info/version`` (ROCm only).
    3. Importing the python package and reading ``__version__`` /
       ``__commit__`` (best-effort, swallows ImportError).

    Returns:
        dict[str, str]: Fixed-shape mapping of component name (``rocm``,
        ``aiter``, ``sglang``, ``vllm``) to a version/commit string, with
        ``"unknown"`` for components that could not be resolved.
    """
    out: dict[str, str] = {}
    for component, env_vars in _STACK_FINGERPRINT_ENVS.items():
        val = ""
        for var in env_vars:
            candidate = (os.environ.get(var) or "").strip()
            if candidate:
                val = candidate
                break
        if not val and component == "rocm":
            for marker in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-utils"):
                v = _read_first_line(Path(marker))
                if v:
                    val = v
                    break
        if not val:
            try:
                if component == "sglang":
                    import sglang as _mod  # type: ignore
                    val = str(getattr(_mod, "__version__", "")).strip()
                elif component == "vllm":
                    import vllm as _mod  # type: ignore
                    val = str(getattr(_mod, "__version__", "")).strip()
                elif component == "aiter":
                    import aiter as _mod  # type: ignore
                    val = str(
                        getattr(_mod, "__commit__", None)
                        or getattr(_mod, "__version__", "")
                    ).strip()
            except Exception:  # noqa: BLE001 — defensive, missing pkg is normal.
                val = ""
        out[component] = val or "unknown"
    return out


def _git_revision() -> str:
    """Best-effort short git SHA of the repo containing this package.

    Returns:
        str: Short HEAD SHA, or an empty string when the package directory is
        not a git checkout or the lookup fails.
    """
    here = Path(__file__).resolve().parent
    return _git_revision_at(here)


def _git_revision_at(path: Path) -> str:
    """Best-effort short git SHA at ``path``.

    Args:
        path (Path): Directory expected to be (within) a git checkout.

    Returns:
        str: Short HEAD SHA, or an empty string when ``path`` is not a checkout
        or the git invocation fails.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return ""
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return ""


def _git_remote_at(path: Path) -> str:
    """Best-effort ``origin`` remote URL at ``path``.

    Args:
        path (Path): Directory expected to be (within) a git checkout.

    Returns:
        str: ``remote.origin.url`` value, or an empty string when unset or the
        git invocation fails.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return ""
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return ""


def _describe_dep(env_var: str) -> dict[str, str]:
    """Build a `{path, commit, remote}` provenance dict for one dependency
    pointed at by ``$env_var``. All fields default to empty string when
    the env var is unset, the directory is missing, or git is unhappy —
    we never raise out of here.

    Args:
        env_var (str): Name of the environment variable holding the dependency
            checkout path.

    Returns:
        dict[str, str]: Mapping with ``path``, ``commit``, and ``remote`` keys;
        any unresolved field is an empty string.
    """
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return {"path": "", "commit": "", "remote": ""}
    path = Path(raw)
    if not path.is_dir():
        return {"path": raw, "commit": "", "remote": ""}
    return {
        "path":   raw,
        "commit": _git_revision_at(path),
        "remote": _git_remote_at(path),
    }


def _build_dependencies() -> dict[str, dict[str, str]]:
    """Provenance for the third-party trees this session executes against.

    Magpie / InferenceX are cloned per-install by ``install.sh``
    (bugs.md §C #1 root-cause fix — see ``ensure_inferencex`` there).
    Recording the commit SHA + remote URL is how downstream debuggers
    answer "which upstream did this run hit?" once the clones have
    moved on.

    Returns:
        dict[str, dict[str, str]]: Mapping of dependency name (``magpie``,
        ``inferencex``) to its ``{path, commit, remote}`` provenance dict.
    """
    return {
        "magpie":     _describe_dep("MAGPIE_DIR"),
        "inferencex": _describe_dep("INFERENCEX_PATH"),
    }


def _detect_image() -> str | None:
    """Best-effort container image detection.

    Tries env vars first (most reliable + easiest for operators to
    override), then well-known mount points, finally a best-effort
    cgroup probe. Returns ``None`` when nothing matches; the breakdown
    layer surfaces a warning rather than fabricating a value.

    Never raises — every disk / parse failure is swallowed.

    Returns:
        str | None: Detected container image reference (or
        ``unknown@<sha>`` from a cgroup probe), or ``None`` when no source
        yields a value.
    """
    for var in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    for marker in ("/etc/podinfo/image", "/etc/hyperloom-image"):
        try:
            p = Path(marker)
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    return txt
        except OSError:
            continue
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            for line in cgroup.read_text(encoding="utf-8", errors="replace").splitlines():
                if "docker" not in line and "containerd" not in line:
                    continue
                # Lines look like
                # ``12:devices:/docker/<sha256>`` — pull a 12+ hex token.
                import re as _re
                m = _re.search(r"([0-9a-f]{12,64})", line)
                if m:
                    short = m.group(1)[:12]
                    return f"unknown@{short}"
    except OSError as exc:
        # /proc/1/cgroup may be unreadable in restricted sandboxes,
        # non-Linux hosts, or stripped-down containers. Best-effort
        # source — fall through to None so the breakdown layer surfaces
        # an honest "image not detected" rather than fabricating one.
        log.debug("cgroup-based image detection failed: %r", exc)
    return None


def _objective_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Mirror cli._run_optimize's objective derivation, without importing it.

    Args:
        args (argparse.Namespace): Parsed CLI args; checked for
            ``target_gain``, ``target_tput``, and ``target_baseline_dir``.

    Returns:
        dict[str, Any]: Objective mapping with ``kind`` (one of ``gain_pct``,
        ``tput``, ``baseline``, ``time_only``) and an associated ``value``.
    """
    if getattr(args, "target_gain", None):
        return {"kind": "gain_pct", "value": float(args.target_gain)}
    if getattr(args, "target_tput", None):
        return {"kind": "tput", "value": float(args.target_tput)}
    if getattr(args, "target_baseline_dir", None):
        return {"kind": "baseline", "value": str(args.target_baseline_dir)}
    return {"kind": "time_only", "value": None}


def build_session_id(model_name: str = "") -> str:
    """Derive an internal session_id label.

    The label is **not** used for any path computation (paths are
    computed from :func:`paths.session_dir`); it only goes into
    manifest.json, SharedState.session_id, and log/report metadata so
    multiple archived sessions are distinguishable.

    Args:
        model_name (str): Optional model name used as the label stem; falls
            back to ``"session"`` when empty.

    Returns:
        str: A label of the form ``<stem>_<UTC_timestamp>_<uuid8>``.
    """
    stem = (model_name or "session").strip().replace("/", "_") or "session"
    return f"{stem}_{_utc_now_compact()}_{uuid.uuid4().hex[:8]}"


def build_manifest(
    session_dir: Path,
    *,
    args: argparse.Namespace | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the schema-v3 session manifest dictionary.

    Merges environment variables and (optional) parsed CLI args into the
    canonical resume tag, including workload, objective, dependency
    provenance, stack fingerprint, image, and warm-replay settings.

    Args:
        session_dir (Path): Session directory the manifest describes.
        args (argparse.Namespace | None): Parsed CLI args overriding env-based
            defaults; ``None`` uses environment/defaults only.
        session_id (str | None): Explicit session-id label; derived from the
            model name when ``None``.

    Returns:
        dict[str, Any]: JSON-serializable manifest mapping.
    """
    model_path = ""
    model_name = ""
    framework = os.environ.get("FRAMEWORK", "")
    gpu_type = os.environ.get("GPU_TYPE", "")
    workload: dict[str, Any] = {
        "isl": int(os.environ["ISL"]) if os.environ.get("ISL", "").strip().isdigit() else None,
        "osl": int(os.environ["OSL"]) if os.environ.get("OSL", "").strip().isdigit() else None,
        "max_model_len": int(os.environ["MAX_MODEL_LEN"])
            if os.environ.get("MAX_MODEL_LEN", "").strip().isdigit() else None,
        "precision": os.environ.get("PRECISION", "") or None,
        "conc": int(os.environ["CONC"]) if os.environ.get("CONC", "").strip().isdigit() else None,
    }
    tp = int(os.environ["TP"]) if os.environ.get("TP", "").strip().isdigit() else None
    if args is not None:
        if getattr(args, "model", None):
            model_path = str(args.model)
            model_name = Path(model_path).name
        if getattr(args, "framework", None):
            framework = str(args.framework)
        if getattr(args, "gpu_type", None):
            gpu_type = str(args.gpu_type)
        if getattr(args, "isl", None) is not None:
            workload["isl"] = int(args.isl)
        if getattr(args, "osl", None) is not None:
            workload["osl"] = int(args.osl)
        if getattr(args, "precision", None):
            workload["precision"] = str(args.precision)
    claw_session_id = (os.environ.get("CLAW_SESSION_ID") or "").strip() or None
    sandbox_user_id = (os.environ.get("SANDBOX_USER_ID") or "").strip() or None
    return {
        "schema_version":    SCHEMA_VERSION,
        "session_id":        session_id or build_session_id(model_name),
        "claw_session_id":   claw_session_id,
        "sandbox_user_id":   sandbox_user_id,
        "created_at_utc":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_dir":       str(session_dir),
        "model_path":        model_path,
        "model_name":        model_name,
        "framework":         framework or "sglang",
        "gpu_type":          gpu_type,
        "tp":                tp,
        "workload":          workload,
        "objective":         _objective_summary(args) if args is not None else {"kind": "time_only", "value": None},
        "max_minutes":       int((getattr(args, "max_hours", 0) or 0) * 60) if args is not None else 0,
        "code_revision":     _git_revision(),
        "dependencies":      _build_dependencies(),
        "pid":               os.getpid(),
        "host":              platform.node() or socket.gethostname() or "",
        "image":             _detect_image(),
        # Cortex KB ``session begin`` requires the stack
        # fingerprint as a mandatory attribute. We
        # snapshot it on manifest write so resume-after-redeploy can
        # detect drift (``--cortex-strict-fingerprint``).
        "stack_fingerprint": _detect_stack_fingerprint(),
        # research_lane capacity locked at session start
        #. Resume reads this back into SharedState
        # so a mid-session restart can't change concurrency semantics.
        "research_lane_capacity": int(
            getattr(args, "research_lane_capacity", 1) or 1
        ) if args is not None else 1,
        # IR-3 soft-degrade audit.
        "kb_degraded_reason": (
            getattr(args, "kb_degraded_reason", None) if args is not None else None
        ),
        "pr_degraded_reason": (
            getattr(args, "pr_degraded_reason", None) if args is not None else None
        ),
        # GAP 1 — Warm-recipe replay flags. Persisted into manifest so
        # robustness_monitor.sh resume / cross-machine resume picks up
        # the same gate thresholds rather than reverting to defaults.
        # The ``warm_replay_enabled`` field is the inverted form of
        # ``--no-warm-replay`` so the YAML reads more naturally.
        "warm_replay_enabled": (
            not bool(getattr(args, "no_warm_replay", False))
            if args is not None else True
        ),
        "warm_replay_min_confidence": (
            float(getattr(args, "warm_replay_min_confidence", 0.7) or 0.7)
            if args is not None else 0.7
        ),
        "warm_replay_min_reproduce_pct": (
            float(getattr(args, "warm_replay_min_reproduce_pct", 0.8) or 0.8)
            if args is not None else 0.8
        ),
    }


def write_manifest(
    session_dir: Path,
    *,
    args: argparse.Namespace | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Atomically write ``manifest.json`` under session_dir.

    The manifest is built via :func:`build_manifest` and written through a
    temp file + ``os.replace`` so readers never observe a partial file.

    Args:
        session_dir (Path): Session directory to write the manifest into.
        args (argparse.Namespace | None): Parsed CLI args passed to
            :func:`build_manifest`.
        session_id (str | None): Explicit session-id label, if any.

    Returns:
        dict[str, Any]: The manifest dict that was written, so the caller can
        echo it or reuse the derived session-id label.
    """
    sd = Path(session_dir)
    manifest = build_manifest(sd, args=args, session_id=session_id)
    target = manifest_path(sd)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, target)
    return manifest


def load_manifest(session_dir: Path) -> dict[str, Any]:
    """Read ``manifest.json`` for an existing session.

    Args:
        session_dir (Path): Session directory expected to contain
            ``manifest.json``.

    Returns:
        dict[str, Any]: The parsed manifest mapping.

    Raises:
        FileNotFoundError: When ``manifest.json`` is missing — the signal
            ``--resume`` uses to refuse a fresh sandbox.
    """
    p = manifest_path(Path(session_dir))
    if not p.exists():
        raise FileNotFoundError(
            f"manifest.json not found under {session_dir} — "
            f"the session was never initialised; cannot --resume"
        )
    with p.open(encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "SCHEMA_VERSION",
    "build_manifest",
    "build_session_id",
    "load_manifest",
    "write_manifest",
]
