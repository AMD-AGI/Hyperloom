# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Preflight: is the aiter that TUNES the same as the aiter that SERVES?

kernelforge.gemm_tune produces an aiter tuned CSV (per-shape kernel config incl.
split-K). vLLM serves with whatever ``import aiter`` resolves to. If the tuner's
aiter (``AITER_ROOT_DIR`` / the aiter whose csrc tuner scripts run) differs from
the serving aiter, a tuned CSV can carry split-K the serving dispatch cannot run
("This GEMM is not supported!" engine-init crash) or, after an aiter upgrade,
become silently stale (wrong/absent gain). The serve-safe split-K cap in
``_aiter_dense_common`` prevents the *crash*, but cannot detect a *drifted* CSV.

Portable (Docker or bare metal, single-tenant), dependency-light. By default it
only warns (misalignment can still work via the cap); ``--strict`` exits
non-zero on any hard problem, for use as a gate before tuning/deploying.

Run: ``python -m kernelforge.gemm_tune.aiter_preflight [--strict] [--check-gpu GPU]``
"""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404 - guarded rocm-smi probe only
import sys
from pathlib import Path
from typing import Mapping


def serve_aiter_path() -> str | None:
    """Realpath of the aiter package ``import aiter`` would resolve to, or None.

    Uses ``importlib.util.find_spec`` so the (potentially multi-second, .so-JIT)
    aiter import is NOT triggered just to read its location -- the preflight runs
    on every tuner CLI startup and must stay cheap.
    """
    try:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.find_spec("aiter")
    except Exception:  # noqa: BLE001 - unresolvable / broken package means "no serving aiter"
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.realpath(os.path.dirname(spec.origin))


def is_aligned(serve: str, root: str) -> bool:
    """True if the serving aiter and the tuner root come from one installation.

    Two layouts count as aligned:

    * source / editable -- the serving package sits under the root
      (``<root>/aiter`` next to ``<root>/csrc``).
    * wheel -- the distribution installs the importable ``aiter`` package and the
      tuner sources (``aiter_meta``, which owns ``csrc``) as SIBLINGS in
      site-packages, so the serving package is never *under* the root.
      ``resolve_aiter_root`` deliberately selects ``<site-packages>/aiter_meta``
      for this layout; treating that pair as misaligned made the check fire on
      every wheel install even though both halves ship in the same wheel and
      therefore cannot drift apart.

    Separator-normalized so the comparison is stable regardless of the host that
    runs the check (deployment is Linux; unit tests may run on Windows).
    """
    s = serve.replace("\\", "/").rstrip("/")
    r = root.replace("\\", "/").rstrip("/")
    if s == r or s.startswith(r + "/"):
        return True
    parent, _, name = r.rpartition("/")
    return bool(parent) and name == "aiter_meta" and s == f"{parent}/aiter"


def classify(serve: str | None, root: str | None, commit: str | None) -> tuple[list[str], list[str]]:
    """Pure decision logic -> (hard_problems, soft_warnings). No I/O."""
    hard: list[str] = []
    soft: list[str] = []
    if serve is None:
        hard.append("serving aiter is not importable (`import aiter` failed)")
    if not root:
        soft.append("AITER_ROOT_DIR unset -> tuner aiter is not pinned to the serving aiter")
    if serve and root and not is_aligned(serve, root):
        hard.append(
            f"MISALIGNED: serving aiter ({serve}) is not the tuner root ({root}); "
            "the tuned CSV may not be dispatchable / may be stale at serve time"
        )
    if not commit:
        soft.append(
            "AITER_COMMIT unset -> tuned-CSV provenance falls back to the installed "
            "aiter distribution version (coarser than a commit)"
        )
    return hard, soft


def _installed_aiter_version() -> str | None:
    """``<dist>==<version>`` for the installed aiter, or None.

    Provenance fallback when ``AITER_COMMIT`` is unset: a wheel version pins the
    tuned CSV to a release even though it cannot pin a commit, which beats
    recording nothing at all. Prefixed with the distribution name so a reader can
    never mistake the value for a commit sha.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - stdlib shape differs on exotic runtimes
        return None
    for dist in ("amd-aiter", "aiter"):
        try:
            found = version(dist)
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - a broken dist-info must not break preflight
            return None
        if found:
            return f"{dist}=={found}"
    return None


def _gpu_idle(gpu: str) -> bool:
    """Best-effort: True if rocm-smi shows GPU[gpu] <=5% (or is unavailable)."""
    try:
        out = subprocess.run(  # nosec B603 B607
            ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    for line in out.splitlines():
        if f"GPU[{gpu}]" in line and "use (%)" in line:
            try:
                return int(line.rsplit(":", 1)[1].strip()) <= 5
            except (ValueError, IndexError):
                continue
    return True


def _resolve_root(env: Mapping[str, str]) -> str | None:
    root = env.get("AITER_ROOT_DIR")
    if not root:
        return None
    rp = os.path.realpath(root)
    return rp if Path(rp).is_dir() else None


def collect(env: Mapping[str, str] | None = None) -> dict:
    """Structured alignment status for programmatic use (e.g. the tuner CLI).

    Best-effort and side-effect-free (no GPU probe); returns the serving aiter,
    tuner root, commit, alignment flag, and the hard/soft problem lists. The tuner
    CLI records this as an artifact and warns -- it never aborts on the result,
    since the serve-safe split-K cap keeps a misaligned CSV from crashing.
    """
    e = os.environ if env is None else env
    serve = serve_aiter_path()
    root = _resolve_root(e)
    # Classify on the env var alone -- an operator who wants exact provenance
    # still gets told to set AITER_COMMIT -- but record the package-version
    # fallback, so the audit artifact carries a real pin instead of null.
    commit_env = e.get("AITER_COMMIT")
    hard, soft = classify(serve, root, commit_env)
    commit = commit_env or _installed_aiter_version()
    return {
        "serve_aiter": serve,
        "tuner_root": root,
        "aiter_commit": commit,
        "aligned": bool(serve and root and is_aligned(serve, root)),
        "hard": hard,
        "soft": soft,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="aiter tune/serve alignment preflight")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any hard problem")
    ap.add_argument("--check-gpu", metavar="GPU", default=None, help="also assert this GPU id is idle")
    args = ap.parse_args(argv)

    st = collect(os.environ)
    serve, root, commit = st["serve_aiter"], st["tuner_root"], st["aiter_commit"]
    hard, soft = list(st["hard"]), list(st["soft"])

    print("== aiter alignment preflight ==")
    print(f"  serve aiter : {serve or '<not importable>'}")
    print(f"  tuner root  : {root or '<AITER_ROOT_DIR unset / not a dir>'}")
    print(f"  AITER_COMMIT: {commit or '<unset>'}")

    if st["aligned"]:
        print("  [ok] serve aiter == tuner root (aligned)")
    if args.check_gpu is not None and not _gpu_idle(args.check_gpu):
        hard.append(f"GPU[{args.check_gpu}] is busy")

    for m in soft:
        print(f"  [WARN] {m}")
    for m in hard:
        print(f"  [PROBLEM] {m}")

    if hard and args.strict:
        print("== FAIL (strict) ==")
        return 1
    print("== ok ==" if not hard else "== warnings only (non-strict) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
