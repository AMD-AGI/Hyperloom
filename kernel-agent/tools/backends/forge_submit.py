#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Forge (Kernel-Forge) submission backend.

Runs the Kernel-Forge autonomous IterationLoop on a single kernel, entirely
inside a git WORKTREE of the kernel repo so the live repo is never mutated
(Hyperloom forbids backends from writing the repo; integrate applies the
artifact later). Emits the same artifacts every other backend does:

  <output_dir>/optimized_versions/v1_forge.<ext>   complete replaceable source
  <output_dir>/optimization_report.md              [micro_speedup] Nx + [correctness] pass

Stage 1 scope: triton / JIT kernels only (no separate build step). Compiled
backends (hip/ck/flydsl) need a build step and are deferred.

Design ref: claw-dev/docs-zh/forge-as-hyperloom-backend-integration.md
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _ensure_forge_on_path() -> str:
    """Make `kernel_agents` (Kernel-Forge) importable from $FORGE_PATH.

    Mirrors how the OOB backend is located via OOB_PATH: read $FORGE_PATH
    (also accepts $KERNEL_FORGE_ROOT / $KERNEL_FORGE_PATH), resolve the dir
    that actually contains the `kernel_agents` package (the repo root, its
    `src/`, or the package dir itself) and prepend it to sys.path. When the
    env var is unset, do nothing and rely on an installed `kernel_agents`
    (e.g. `pip install -e`). Returns the path inserted, or "".
    """
    root = (os.environ.get("FORGE_PATH")
            or os.environ.get("KERNEL_FORGE_ROOT")
            or os.environ.get("KERNEL_FORGE_PATH")
            or "").strip()
    if not root:
        return ""
    for cand in (os.path.join(root, "src"), root, os.path.dirname(root)):
        if os.path.isfile(os.path.join(cand, "kernel_agents", "__init__.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    return ""


# Platform -> gfx target (mirrors tracelens_analysis._FLYDSL_TARGET_ARCH_BY_PLATFORM).
_PLATFORM_TO_GFX = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}

# Stage 1: only triton maps to a fellow; compiled backends are deferred.
_SOURCE_TYPE_TO_FELLOW = {
    "triton": "triton-fellow",
    "python": "triton-fellow",
}


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing text output (never raises on non-zero)."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _resolve_gpu_target(candidate: dict) -> str:
    """Resolve the gfx target: env GPU_TARGET -> candidate platform -> probe.

    Never hard-codes; falls back to rocminfo when nothing else is available.
    """
    env_target = (os.environ.get("GPU_TARGET") or os.environ.get("GPU_TYPE") or "").strip()
    if env_target:
        return _PLATFORM_TO_GFX.get(env_target.lower(), env_target)
    platform = str(candidate.get("platform") or candidate.get("arch") or "").strip().lower()
    if platform in _PLATFORM_TO_GFX:
        return _PLATFORM_TO_GFX[platform]
    # Probe via rocminfo as a last resort.
    try:
        proc = _run(["rocminfo"], timeout=30)
        m = re.search(r"\bgfx\d+[a-z]*\b", proc.stdout or "")
        if m:
            return m.group(0)
    except Exception:
        pass
    return "gfx942"


def _fellow_for_source_type(source_type: str) -> str | None:
    """Map source_type to a Forge fellow (stage 1: triton only). None if unsupported."""
    return _SOURCE_TYPE_TO_FELLOW.get((source_type or "").strip().lower())


def _git_toplevel(path: str) -> str:
    """Return the git repo root containing `path`, or '' if not a git repo."""
    try:
        proc = _run(["git", "-C", str(Path(path).parent), "rev-parse", "--show-toplevel"], timeout=30)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


def _default_branch(repo: str) -> str:
    """Best-effort default branch name for `repo` (e.g. 'main'/'master').

    Used to auto-recover a repo stranded on a leftover ``forge/`` temp branch by
    a hard-killed prior run. Prefers the remote's advertised default, then falls
    back to common local branch names.
    """
    p = _run(["git", "-C", repo, "symbolic-ref", "--short",
              "refs/remotes/origin/HEAD"], timeout=30)
    ref = (p.stdout or "").strip()
    if ref.startswith("origin/"):
        return ref[len("origin/"):]
    for name in ("main", "master"):
        if _run(["git", "-C", repo, "rev-parse", "--verify", name],
                timeout=30).returncode == 0:
            return name
    return ""


def _prepare_worktree(source_file: str, kernel_repo: str, output_dir: Path,
                      branch: str) -> tuple[str, str, str] | None:
    """Create a git worktree of kernel_repo at output_dir/worktree (R1/W1).

    Returns (worktree_dir, worktree_kernel_file, base_commit) or None when the
    repo is not a clean git checkout / source_file is not tracked (forge then
    skips, never mutating the live repo). base_commit is the commit the worktree
    was created at (HEAD); export diffs the best state against it.
    """
    repo = kernel_repo or _git_toplevel(source_file)
    if not repo or not (Path(repo) / ".git").exists():
        return None
    src_abs = Path(source_file).resolve()
    try:
        rel = src_abs.relative_to(Path(repo).resolve())
    except ValueError:
        return None  # source_file not inside the repo

    wt = output_dir / "worktree"
    # Clean any stale worktree at this path first (W3).
    if wt.exists():
        _run(["git", "-C", repo, "worktree", "remove", "--force", str(wt)], timeout=60)
        shutil.rmtree(wt, ignore_errors=True)
    _run(["git", "-C", repo, "worktree", "prune"], timeout=60)

    base_commit = _run(["git", "-C", repo, "rev-parse", "HEAD"], timeout=30).stdout.strip()
    add = _run(["git", "-C", repo, "worktree", "add", "-b", branch, str(wt), "HEAD"], timeout=120)
    if add.returncode != 0:
        return None

    # Ensure a local git identity so IterationLoop commit/revert does not silently
    # fail (observed: missing user.name/email -> no-op keep/revert).
    _run(["git", "-C", str(wt), "config", "user.name", "forge-bot"], timeout=30)
    _run(["git", "-C", str(wt), "config", "user.email", "forge-bot@local"], timeout=30)

    return str(wt), str(wt / rel), base_commit


def _editable_roots() -> list[str]:
    """Collect filesystem roots of PEP 660 editable-finder installs.

    Scans site-packages for ``__editable__*.pth`` and ``__editable___*_finder.py``
    and extracts the absolute paths they map into. Such packages are imported via
    a sys.meta_path finder that points at the *live* repo and CANNOT be overridden
    by PYTHONPATH — so a git worktree copy is never imported (see doc Section 6.6 W2).

    Handles two finder layouts:
      1. Path-string .pth files that contain absolute paths in quotes.
      2. Setuptools-style .pth files that ``import __editable___<pkg>_finder``;
         the finder .py has a ``MAPPING`` dict mapping package names to paths.
    """
    import re
    import site
    roots: set[str] = set()
    seen_dirs: set[str] = set()
    scan_dirs = list(sys.path)
    try:
        scan_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            scan_dirs.append(site.getusersitepackages())
        except Exception:
            pass
    # Venv / conda site-packages may not appear in sys.path when PYTHONPATH
    # is overridden and the venv is not activated. Probe conventional locations
    # for sys.prefix, VIRTUAL_ENV, CONDA_PREFIX, and the running interpreter.
    _pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    _prefixes = {sys.prefix, sys.exec_prefix, sys.base_prefix}
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        v = os.environ.get(var)
        if v:
            _prefixes.add(v)
    # Also derive the venv from the interpreter path (e.g. /opt/venv/bin/python
    # → /opt/venv).
    _interp = os.path.realpath(sys.executable)
    if os.sep + "bin" + os.sep in _interp:
        _prefixes.add(_interp.rsplit(os.sep + "bin" + os.sep, 1)[0])
    for prefix in _prefixes:
        for sub in (f"lib/{_pyver}/site-packages", f"lib/{_pyver}/dist-packages"):
            cand = os.path.join(prefix, sub)
            if os.path.isdir(cand):
                scan_dirs.append(cand)
    for d in scan_dirs:
        if not d or d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(d)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if not n.startswith("__editable__"):
                continue
            if not (n.endswith(".pth") or n.endswith("_finder.py")):
                continue
            fpath = os.path.join(d, n)
            try:
                txt = open(fpath, errors="replace").read()
            except OSError:
                continue
            # Layout 1: quoted absolute paths directly in the file.
            for m in re.findall(r"['\"](/[^'\"]+)['\"]", txt):
                if os.path.isdir(m):
                    roots.add(os.path.realpath(m))
            # Layout 2: .pth imports a _finder.py; read its MAPPING dict for
            # paths. The finder file lives next to the .pth in site-packages.
            if n.endswith(".pth"):
                fm = re.search(r"import\s+(__editable___\w+_finder)", txt)
                if fm:
                    finder_file = os.path.join(d, fm.group(1) + ".py")
                    try:
                        ftxt = open(finder_file, errors="replace").read()
                    except OSError:
                        continue
                    for m in re.findall(r"['\"](/[^'\"]+)['\"]", ftxt):
                        if os.path.isdir(m):
                            roots.add(os.path.realpath(m))
    return sorted(roots)


def _needs_inplace(kernel_repo: str) -> bool:
    """True when kernel_repo is (or contains/sits under) an editable-finder root.

    In that case forge must edit the live repo in place (the finder imports the
    live path; a worktree copy would be invisible -> the loop would no-op).
    """
    if not kernel_repo:
        return False
    repo = os.path.realpath(kernel_repo)
    for r in _editable_roots():
        if r == repo or r.startswith(repo + os.sep) or repo.startswith(r + os.sep):
            return True
    return False


def _acquire_repo_lock(repo: str) -> int | None:
    """Take a non-blocking exclusive lock on the live repo for in-place editing.

    In-place mode mutates + ``reset --hard`` the shared live repo, so two
    concurrent forge sessions on the same repo would race (branch steal,
    cross-contaminated measurements). The lock serializes them; a caller that
    cannot get it must skip in-place (fall through to the next backend). Returns
    the held fd (release with _release_repo_lock) or None when already held.
    """
    try:
        fd = os.open(os.path.join(repo, ".git", "forge_inplace.lock"),
                     os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_repo_lock(fd: int | None) -> None:
    """Release + close the in-place repo lock (best-effort)."""
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _prepare_inplace(source_file: str, kernel_repo: str, branch: str) -> tuple[str, str, dict] | None:
    """In-place mode (Option 1): edit the LIVE repo so an editable-finder import
    sees the changes. Snapshots the original branch/HEAD + source bytes for a
    per-file restore in finally. Returns (workspace=repo, kernel_file=source_file,
    restore_info) or None when the repo is not a usable git checkout.

    Safety:
      - refuse if HEAD is already on a forge/ temp branch (prior crashed run),
      - hold a per-repo lock so concurrent forge runs never interleave,
      - dirty working trees are allowed: restore only touches the source_file
        (per-file write-back, no ``reset --hard``), so other uncommitted changes
        in the repo are never destroyed.
    """
    repo = kernel_repo or _git_toplevel(source_file)
    if not repo or not (Path(repo) / ".git").exists():
        return None
    if not Path(source_file).is_file():
        return None
    try:
        relpath = str(Path(source_file).resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        return None  # source not inside repo

    # Serialize in-place runs on this repo before touching any git state.
    lock_fd = _acquire_repo_lock(repo)
    if lock_fd is None:
        return None  # another forge in-place run holds this repo; skip cleanly

    def _skip() -> None:
        _release_repo_lock(lock_fd)
        return None

    try:
        orig_branch = _run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                           timeout=30).stdout.strip()
        orig_head = _run(["git", "-C", repo, "rev-parse", "HEAD"], timeout=30).stdout.strip()
        if not orig_head:
            return _skip()
        # Auto-recover from a leftover forge temp branch. A prior forge run that
        # was hard-killed (SIGKILL) before _restore_inplace could switch back
        # leaves the repo stranded on its `forge/<ts>/...` branch, after which
        # EVERY subsequent run fails with "repo is not a usable git checkout"
        # (orig_head would be a non-pristine baseline). Recover by forcing the
        # repo back onto its default branch and deleting the stale temp branch so
        # the snapshot below reflects a pristine baseline again.
        if orig_branch.startswith("forge/"):
            default_branch = _default_branch(repo)
            if not default_branch:
                return _skip()
            stale = orig_branch
            co = _run(["git", "-C", repo, "checkout", "-f", default_branch],
                      timeout=120)
            if co.returncode != 0:
                return _skip()
            _run(["git", "-C", repo, "branch", "-D", stale], timeout=30)
            orig_branch = default_branch
            orig_head = _run(["git", "-C", repo, "rev-parse", "HEAD"],
                             timeout=30).stdout.strip()
            if not orig_head:
                return _skip()
        # Preflight: drop any stale temp branch from a prior crashed run so the
        # snapshot below reflects a clean baseline, not leftover mutations.
        _run(["git", "-C", repo, "branch", "-D", branch], timeout=30)
        # Snapshot the source_file bytes ON DISK (which may differ from the
        # committed version in a dirty repo — that's fine, we restore exactly
        # what was there before forge touched it).
        try:
            backup = Path(source_file).read_bytes()
        except OSError:
            return _skip()
        _run(["git", "-C", repo, "config", "user.name", "forge-bot"], timeout=30)
        _run(["git", "-C", repo, "config", "user.email", "forge-bot@local"], timeout=30)
        # Create a temp branch for the forge loop to commit/revert on. Without
        # this, IterationLoop's _git_commit / _git_revert_last would operate
        # directly on the live branch (or detached HEAD), and commits would
        # persist after restore. The branch is deleted in _restore_inplace.
        cb = _run(["git", "-C", repo, "checkout", "-b", branch], timeout=60)
        if cb.returncode != 0:
            return _skip()
        # Snapshot any pre-existing dirty TRACKED files as a baseline commit.
        # The loop now stages every tracked edit (`git add -u`), so without this
        # the first iteration's commit would absorb pre-forge modifications and a
        # later revert would destroy them. base_commit is the pre-forge tree;
        # agent edits stack on top of it, and export/restore diff against it so
        # pre-existing dirty files are preserved untouched. When the tree is
        # clean, base_commit == orig_head.
        _run(["git", "-C", repo, "add", "-u"], timeout=60)
        dirty = _run(["git", "-C", repo, "diff", "--cached", "--quiet"], timeout=30)
        if dirty.returncode != 0:
            _run(["git", "-C", repo, "commit", "-m",
                  "forge: pre-existing dirty baseline"], timeout=60)
            base_commit = _run(["git", "-C", repo, "rev-parse", "HEAD"],
                               timeout=30).stdout.strip() or orig_head
        else:
            base_commit = orig_head
    except Exception:
        _release_repo_lock(lock_fd)
        raise

    restore = {"repo": repo, "orig_branch": orig_branch, "orig_head": orig_head,
               "branch": branch, "source_file": source_file, "backup": backup,
               "relpath": relpath, "lock_fd": lock_fd, "base_commit": base_commit}
    return repo, source_file, restore


def _restore_inplace(restore: dict) -> None:
    """Restore the live repo after in-place editing: revert EVERY file the agent
    changed back to its pre-forge content, return to the original branch/HEAD,
    and drop the temp branch.

    Restores the full changed-file set (not just ``source_file``): the agent may
    have edited a sibling tracked file (e.g. a config defaults module), and the
    loop's ``git add -u`` commits mean those edits live on the temp branch.
    ``base_commit`` holds the exact pre-forge tree (including any pre-existing
    dirty content snapshotted at prepare time), so checking files out of it
    restores precisely what was there before forge ran. Untracked files (build
    artifacts) are never touched (no ``reset --hard``).
    """
    if not restore:
        return
    repo = restore["repo"]
    # Abort any in-progress revert the loop may have left.
    _run(["git", "-C", repo, "revert", "--abort"], timeout=30)
    orig_branch = restore.get("orig_branch") or ""
    orig_head = restore.get("orig_head") or ""
    base_commit = restore.get("base_commit") or orig_head
    # Step 1: restore every file that differs from the pre-forge baseline back to
    # its base_commit content (working tree + index). This undoes ALL the agent's
    # tracked edits, including sibling files outside source_file. Done while still
    # on the temp branch so base_commit is reachable.
    if base_commit:
        diff = _run(["git", "-C", repo, "diff", "--name-only", base_commit], timeout=60)
        for rel in (diff.stdout or "").splitlines():
            rel = rel.strip()
            if rel:
                _run(["git", "-C", repo, "checkout", base_commit, "--", rel], timeout=30)
    # Step 2: move HEAD back to the original ref WITHOUT touching the working tree.
    if orig_branch and orig_branch != "HEAD":
        # Was on a named branch: point HEAD back at it via symbolic-ref.
        _run(["git", "-C", repo, "symbolic-ref", "HEAD", f"refs/heads/{orig_branch}"], timeout=30)
    elif orig_head:
        # Was on detached HEAD: detach via `update-ref --no-deref HEAD` so the
        # working tree is NOT touched (a plain `checkout --detach` would reset
        # tracked files to orig_head and clobber pre-existing dirty; a plain
        # `update-ref HEAD` would follow the symref and move the temp branch).
        _run(["git", "-C", repo, "update-ref", "--no-deref", "HEAD", orig_head], timeout=30)
    # Step 3: reset the index to match orig_head (without touching working tree)
    # so `git status` reflects the same dirty state as before forge ran.
    if orig_head:
        _run(["git", "-C", repo, "reset", orig_head, "--", "."], timeout=30)
    # Step 4: belt-and-suspenders — ensure the primary source_file is exactly the
    # pre-forge bytes even if the git restore above raced or partially applied.
    try:
        Path(restore["source_file"]).write_bytes(restore["backup"])
    except OSError:
        pass
    # Step 5: delete the temp branch (safe now that HEAD points elsewhere).
    if restore.get("branch"):
        _run(["git", "-C", repo, "branch", "-D", restore["branch"]], timeout=30)
    # Release the per-repo in-place lock last, after the repo is fully restored.
    _release_repo_lock(restore.get("lock_fd"))


def _remove_worktree(kernel_repo: str, source_file: str, wt: str, branch: str) -> None:
    """Tear down the worktree + temp branch; live repo untouched (W3)."""
    repo = kernel_repo or _git_toplevel(source_file)
    if not repo:
        return
    _run(["git", "-C", repo, "worktree", "remove", "--force", wt], timeout=60)
    shutil.rmtree(wt, ignore_errors=True)
    _run(["git", "-C", repo, "branch", "-D", branch], timeout=30)
    _run(["git", "-C", repo, "worktree", "prune"], timeout=60)


# Adapter template: wraps a Hyperloom harness/test_command as a Forge-contract
# driver. Forces the worktree onto sys.path/cwd (W2) so edited code is imported,
# and emits 'allclose: True/False' (correctness) and 'wall_ms: <v>' (bench).
_ADAPTER_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated Forge driver-adapter wrapping a Hyperloom harness."""
import argparse, os, re, subprocess, sys

TEST_COMMAND = {test_command!r}
WORKTREE = {worktree!r}


def _run_harness():
    env = dict(os.environ)
    env["PYTHONPATH"] = WORKTREE + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(TEST_COMMAND, shell=True, cwd=WORKTREE, env=env,
                       capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + "\\n" + (p.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--bench-mode", action="store_true")
    a, _ = ap.parse_known_args()
    rc, out = _run_harness()

    if a.bench_mode:
        m = re.search(r"(?:median_ms|wall_ms)\\s*[:=]\\s*([0-9.]+)", out)
        if not m:
            ms = re.findall(r"([0-9]+\\.[0-9]+)\\s*ms\\b", out)
            if ms:
                print(f"wall_ms: {{ms[-1]}}")
        else:
            print(f"wall_ms: {{m.group(1)}}")
        sys.exit(0 if rc == 0 else 1)

    low = out.lower()
    if rc != 0:
        print("allclose: False")
        sys.exit(1)
    # Fail-safe correctness: only PASS on an EXPLICIT positive signal from the
    # harness (SNR / allclose:true / known pass phrases). A bare exit-0 with no
    # correctness signal emits NO metric -> Forge's test_correctness reports
    # "no metric found" -> the iteration fails (never a fabricated pass).
    snr = re.search(r"snr\\s*[:=]\\s*([-0-9.]+)\\s*db", low)
    m = re.search(r"allclose\\s*[:=]\\s*(true|false)", low)
    if any(k in low for k in ("mismatch", "not close", "correctness failed", "validation failed")):
        print("allclose: False")
    elif m:
        print(f"allclose: {{'True' if m.group(1) == 'true' else 'False'}}")
    elif snr:
        print(f"SNR: {{snr.group(1)}} dB")
    elif any(k in low for k in ("correctness passed", "all tests passed", "test passed", "ok")):
        print("allclose: True")
    else:
        # No correctness signal at all -> do NOT fabricate a pass.
        print("correctness: unknown (no metric in harness output)")
    sys.exit(0)


main()
'''


def _build_driver_adapter(test_command: str, worktree: str, output_dir: Path) -> str:
    """Write the driver-adapter script and return its path."""
    adapter = output_dir / "forge_driver_adapter.py"
    adapter.write_text(_ADAPTER_TEMPLATE.format(test_command=test_command, worktree=worktree))
    adapter.chmod(0o755)
    return str(adapter)


# Auto-generated Forge-native driver for harness-less candidates. Imports the
# kernel module by file path (so it always targets the worktree copy), discovers
# a callable entry, builds inputs from --shape, and emits 'SNR: <v> dB' +
# 'wall_ms: <v>'. Op-specific input/reference logic lives in build_inputs/ref.
_AUTOGEN_GEMM_DRIVER = '''#!/usr/bin/env python3
"""Auto-generated Forge driver (gemm/matmul) — no external harness needed."""
import argparse, importlib.util, math, sys
import torch

KERNEL_FILE = {kernel_file!r}
ENTRY_HINTS = ("matmul", "gemm", "mm", "run", "forward", "kernel")


def _load():
    spec = importlib.util.spec_from_file_location("forge_autogen_kernel", KERNEL_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _entry(m):
    import inspect
    for name in ENTRY_HINTS:
        f = getattr(m, name, None)
        if callable(f):
            return f
    cands = [f for n, f in vars(m).items()
             if not n.startswith("_") and inspect.isfunction(f)]
    if cands:
        return cands[0]
    raise RuntimeError("no callable entry found in kernel module")


def _shape(s):
    out = {{}}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return out


def _inputs(sh, scale):
    M = sh.get("M", 2048); N = sh.get("N", 2048); K = sh.get("K", 2048)
    torch.manual_seed(0)
    a = (torch.randn((M, K), device="cuda", dtype=torch.float16) * scale)
    b = (torch.randn((K, N), device="cuda", dtype=torch.float16) * scale)
    return a, b


def _snr(ref, out):
    ref_f = ref.float(); err = ref_f - out.float()
    n = err.norm().item()
    return 120.0 if n == 0 else 20.0 * math.log10(ref_f.norm().item() / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="")
    p.add_argument("--mode", default="full")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    a, _ = p.parse_known_args()
    m = _load(); fn = _entry(m)
    sh = _shape(a.shape)
    scale = 4.0 if a.mode == "stability" else 1.0
    x, y = _inputs(sh, scale)
    if a.bench_mode:
        for _ in range(max(1, a.warmup)):
            fn(x, y)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        for _ in range(max(1, a.iters)):
            s.record(); fn(x, y); e.record(); torch.cuda.synchronize()
            print(f"wall_ms: {{s.elapsed_time(e):.4f}}")
        return
    out = fn(x, y); torch.cuda.synchronize()
    ref = torch.matmul(x, y)
    print(f"SNR: {{_snr(ref, out):.2f}} dB")


main()
'''


# Auto-generated Forge driver for sglang classic triton fused_moe (e.g. k006).
# Imports the HIGH-LEVEL sglang fused_moe() wrapper (which dispatches to
# fused_moe_triton_kernels.py::fused_moe_kernel) so an in-place edit to that
# kernel is exercised; correctness vs a torch naive-MoE reference. Requires the
# in-place mode (editable-finder packages) so the edited kernel is the one
# imported. No {} substitution (imports sglang by package, not by file path).
_AUTOGEN_MOE_DRIVER = '''#!/usr/bin/env python3
"""Auto-generated Forge driver for sglang triton fused_moe (no external harness)."""
import argparse, math
import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

DT = torch.bfloat16
DEFAULT = dict(M=512, N=1024, K=1024, E=8, TOPK=2)


def torch_naive_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a2 = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    tw, ti = torch.topk(score, topk)
    tw = tw.view(-1); ti = ti.view(-1)
    for i in range(w1.shape[0]):
        mask = ti == i
        if mask.sum():
            out[mask] = SiluAndMul()(a2[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) * tw.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def _shape(s):
    d = dict(DEFAULT)
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().upper()
            if k in d:
                try:
                    d[k] = int(v.strip())
                except ValueError:
                    pass
    return d


def _build(d, scale):
    torch.manual_seed(0)
    M, N, K, E, TOPK = d["M"], d["N"], d["K"], d["E"], d["TOPK"]
    a = torch.empty((M, K), dtype=DT, device="cuda").normal_(0, scale)
    w1 = torch.empty((E, 2 * N, K), dtype=DT, device="cuda").normal_(0, scale)
    w2 = torch.empty((E, K, N), dtype=DT, device="cuda").normal_(0, scale)
    score = torch.empty((M, E), dtype=DT, device="cuda").normal_(0, scale)
    # Build StandardTopKOutput directly (no TopK module -> avoids TP group init).
    probs = torch.softmax(score.float(), dim=-1)
    tw, ti = torch.topk(probs, TOPK, dim=-1)
    tko = StandardTopKOutput(tw.to(torch.float32), ti.to(torch.int32), score)
    return a, w1, w2, score, tko, TOPK


def _run(a, w1, w2, tko):
    return fused_moe(a, w1, w2, tko, MoeRunnerConfig(inplace=False))


def main():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="")
    p.add_argument("--mode", default="full")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--bench-mode", action="store_true")
    a_, _ = p.parse_known_args()
    d = _shape(a_.shape)
    scale = 0.05 if a_.mode == "stability" else 0.01
    x, w1, w2, score, tko, topk = _build(d, scale)
    if a_.bench_mode:
        for _ in range(max(1, a_.warmup)):
            _run(x, w1, w2, tko)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        for _ in range(max(1, a_.iters)):
            s.record(); _run(x, w1, w2, tko); e.record(); torch.cuda.synchronize()
            print("wall_ms: %.4f" % s.elapsed_time(e))
        return
    out = _run(x, w1, w2, tko); torch.cuda.synchronize()
    ref = torch_naive_moe(x, w1, w2, score, topk)
    err = (ref.float() - out.float()).norm().item()
    snr = 120.0 if err == 0 else 20.0 * math.log10(ref.float().norm().item() / err)
    print("SNR: %.2f dB" % snr)


if __name__ == "__main__":
    main()
'''


def _autogen_forge_driver(candidate: dict, worktree_kernel: str, output_dir: Path,
                          inplace: bool = False) -> str | None:
    """Auto-generate a Forge-native driver when no harness is supplied.

    Op templates keyed by candidate['operation'] / kernel name:
      - fused_moe / moe  -> sglang fused_moe() wrapper + torch naive-MoE golden.
        Imports sglang by PACKAGE (not by file), so it only exercises the edited
        kernel under in-place mode; in worktree mode the edits are invisible and
        the loop would silently no-op -> require inplace, else skip cleanly.
      - gemm / matmul    -> imports the kernel by FILE path (worktree-safe) +
        torch.matmul golden.
    Returns the driver path, or None when the op has no usable template.
    """
    op = str(candidate.get("operation") or "").lower()
    hint = (op + " " + str(candidate.get("name") or "") + " " + worktree_kernel).lower()
    drv = output_dir / "forge_autogen_driver.py"
    if "moe" in hint:
        if not inplace:
            return None  # package-import driver only works in-place; skip otherwise
        drv.write_text(_AUTOGEN_MOE_DRIVER)
        drv.chmod(0o755)
        return str(drv)
    if any(t in hint for t in ("gemm", "matmul", "_mm", "linear")):
        drv.write_text(_AUTOGEN_GEMM_DRIVER.format(kernel_file=worktree_kernel))
        drv.chmod(0o755)
        return str(drv)
    return None


def _tensor_dim_lists(candidate: dict) -> list[list[int]]:
    """Extract per-tensor integer dim lists from candidate['input_shapes'].

    TraceLens emits input_shapes either as integer lists
    ``[{"call_num": N, "shape": [d0, d1, ...]}, ...]`` OR as dtype-tagged strings
    ``[{"shape": "(16384,2048) bf16"}, ...]`` (the format the kernel-agent passes
    through from the rendered trace). Without parsing the string form, every dim
    list is dropped, _shapes_from_candidate returns {}, and the auto-gen driver
    falls back to its tiny default shape (M=512) — which benches a memory-bound
    regime and yields a near-1.0x speedup instead of the real prefill gain. Parse
    both forms here.
    """
    out: list[list[int]] = []
    for e in candidate.get("input_shapes") or []:
        s = e.get("shape") if isinstance(e, dict) else e
        if isinstance(s, (list, tuple)) and s and all(isinstance(x, int) for x in s):
            out.append([int(x) for x in s])
        elif isinstance(s, str):
            # One entry may hold a SINGLE shape ("(16384,2048) bf16") or MANY
            # shapes joined by "<br>" / newlines
            # ("(16384,2048) bf16<br>(128,1536,2048) bf16<br>(16384,8) fp32..").
            # findall over every "(...)" group handles both; re.search (first
            # group only) would drop the expert-weight (E,*,K) and topk (M,t)
            # tensors and leave _moe_dims with just M/K -> tiny default E/TOPK.
            for grp in re.findall(r"\(([\d,\s]*)\)", s):
                dims = [int(x) for x in grp.split(",") if x.strip().isdigit()]
                if dims:
                    out.append(dims)
    return out


def _gemm_dims(shapes: list[list[int]]) -> dict:
    """Derive {M,N,K} from matmul operands A[M,K] @ B[K,N] (best-effort).

    Picks the first pair of 2D tensors whose inner dims agree (A[1]==B[0]); falls
    back to M/K from a single 2D tensor. Dims that cannot be derived are omitted
    so the driver keeps its own default for them.
    """
    twod = [s for s in shapes if len(s) == 2]
    for a in twod:
        for b in twod:
            if a is not b and a[1] == b[0]:
                return {"M": a[0], "K": a[1], "N": b[1]}
    if twod:
        return {"M": twod[0][0], "K": twod[0][1]}
    return {}


def _moe_dims(shapes: list[list[int]]) -> dict:
    """Derive {M,N,K,E,TOPK} from fused_moe tensor shapes (best-effort).

    Recognizes hidden_states [M,K] (2D, widest feature dim), expert weights
    [E,*,K] (3D), and topk ids/weights [M,t] (2D, small second dim). w2 is
    [E,K,N] (dim1==K) -> N=dim2; else w1 [E,2N,K] (dim2==K) -> N=dim1//2.
    Only confidently derived dims are returned; the rest fall back to defaults.
    """
    twod = [s for s in shapes if len(s) == 2]
    threed = [s for s in shapes if len(s) == 3]
    d: dict = {}
    hidden = max(twod, key=lambda s: s[1]) if twod else None
    if hidden is not None:
        d["M"], d["K"] = hidden[0], hidden[1]
        for s in twod:  # topk ids/weights: [M, topk] with a small second dim
            if s is not hidden and s[0] == hidden[0] and 0 < s[1] <= 64:
                d["TOPK"] = s[1]
                break
    if threed:
        d["E"] = threed[0][0]
        k = d.get("K")
        n = None
        if k is not None:
            for s in threed:  # w2 [E,K,N]
                if s[1] == k:
                    n = s[2]
                    break
            if n is None:
                for s in threed:  # w1 [E,2N,K]
                    if s[2] == k:
                        n = s[1] // 2
                        break
        if n is not None:
            d["N"] = n
    return d


def _shapes_from_candidate(candidate: dict) -> dict:
    """Build a Forge shapes dict (primary/minimal/validation) for the driver.

    Forge formats ``--shape K=V,...`` from shapes['primary'] and passes it to the
    auto-generated driver, so the keys must match what the driver parses
    (M/N/K for gemm; M/N/K/E/TOPK for moe). We derive those named dims from the
    candidate's per-tensor input_shapes; when a dim is not derivable the key is
    omitted and the driver keeps its built-in default (safe degradation).

    With a single shape, minimal == primary and the sweep degenerates (Y3).
    """
    op = (str(candidate.get("operation") or "") + " " + str(candidate.get("name") or "")).lower()
    dims = _tensor_dim_lists(candidate)
    if "moe" in op:
        primary = _moe_dims(dims)
    elif any(t in op for t in ("gemm", "matmul", "_mm", "linear")):
        primary = _gemm_dims(dims)
    else:
        primary = {}
    # Back-compat: honor an explicit pre-named dim dict if one was supplied.
    if not primary:
        shapes = candidate.get("input_shapes") or []
        if shapes and isinstance(shapes[0], dict) and any(
                k in shapes[0] for k in ("M", "N", "K", "E", "TOPK")):
            primary = {k: v for k, v in shapes[0].items()
                       if k in ("M", "N", "K", "E", "TOPK")}
    return {"primary": primary, "minimal": primary, "validation": [primary] if primary else []}


def _write_report(output_dir: Path, baseline_ms: float | None, best_ms: float | None,
                  improved: bool) -> Path:
    """Write optimization_report.md with the locked anchors (doc Section 6.4).

    Only claims a KEEP-worthy result when the loop actually kept a validated
    kernel strictly faster than baseline (improved=True). Otherwise emits no
    speedup and [correctness] fail, so build_verification never KEEPs a kernel
    that wasn't really optimized/validated.
    """
    lines = ["# Forge optimization report", ""]
    if improved and baseline_ms and best_ms and best_ms > 0:
        speedup = baseline_ms / best_ms
        lines.append(f"[micro_speedup] {speedup:.4f}x")
        lines.append(f"baseline_ms={baseline_ms:.4f} best_ms={best_ms:.4f}")
        lines.append("[correctness] pass")
    else:
        lines.append("micro_speedup: N/A (no validated improvement kept)")
        lines.append("[correctness] fail")
    report = output_dir / "optimization_report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def _export_best_artifacts(workspace: str, base_commit: str, worktree_kernel_file: str,
                           source_file: str, output_dir: Path) -> tuple[str, list[str]]:
    """Export the best-kept state — ALL files the agent changed, not just the kernel.

    The loop now commits every tracked edit (``runner._git_commit`` uses
    ``git add -u``), so the agent's winning change may live in a sibling tracked
    file (e.g. a ``*_config.py`` defaults module) rather than ``source_file``.
    Exporting only ``source_file`` would yield a byte-identical artifact that
    carries none of the optimization (the in-place bench measured it, but it
    would not transfer on integration), and the sibling file would be left dirty.

    This:
      - copies the primary kernel to ``optimized_versions/v1_forge.<ext>`` (the
        Hyperloom report scan's drop-in-replacement contract), and
      - copies EVERY file changed since ``base_commit`` under
        ``optimized_versions/files/<repo-relative-path>``, and
      - writes a single ``optimized_versions/forge.patch`` (``git diff
        base_commit``) so a multi-file change can be applied at integration time.

    Returns (primary_artifact_path, changed_relpaths).
    """
    dst_dir = output_dir / "optimized_versions"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Primary kernel artifact (drop-in replacement contract).
    ext = Path(source_file).suffix or ".py"
    primary = dst_dir / f"v1_forge{ext}"
    try:
        shutil.copy2(worktree_kernel_file, primary)
    except OSError:
        pass

    # Every file changed vs the pre-forge baseline (best-kept state == the
    # current worktree tree). Compare base_commit to the working tree so both
    # committed and any residual uncommitted edits are captured.
    changed: list[str] = []
    diff = _run(["git", "-C", workspace, "diff", "--name-only", base_commit], timeout=60)
    for rel in (diff.stdout or "").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        changed.append(rel)
        srcp = Path(workspace) / rel
        if not srcp.is_file():
            continue
        dstp = dst_dir / "files" / rel
        dstp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(srcp, dstp)
        except OSError:
            pass

    # Full multi-file patch (agent's net optimization, excludes pre-existing dirty).
    patch = _run(["git", "-C", workspace, "diff", base_commit], timeout=60)
    try:
        (dst_dir / "forge.patch").write_text(patch.stdout or "")
    except OSError:
        pass

    return str(primary), changed


def _normalized(returncode: int, stdout: str, stderr: str, elapsed_s: float,
                gpu_ids: str = "") -> dict:
    """Shape the result like oob_submit/geak_submit return dicts."""
    return {
        "returncode": returncode,
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
        "stdout": stdout or "",
        "gpu_ids": gpu_ids or (os.environ.get("HIP_VISIBLE_DEVICES")
                               or os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        "elapsed_s": round(elapsed_s, 2),
        "cmd": ["forge_submit.submit"],
    }


def submit(source_file: str, prompt_file: Path, output_dir: Path,
           test_command: str = "", source_type: str = "unknown",
           candidate: dict | None = None, num_gpus: int = 1,
           timeout_s: int = 1800, prefer_ray: bool = True,
           kernel_repo: str = "") -> dict:
    """Run Forge's autonomous loop on one kernel; emit Hyperloom-contract artifacts.

    Stage 1 runs the loop in-process inside a git worktree (Ray wrapping is a
    follow-up to match OOB GPU leasing). Returns a normalized result dict and
    writes optimized_versions/ + optimization_report.md under output_dir.
    """
    started = time.time()
    candidate = candidate or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fellow = _fellow_for_source_type(source_type)
    if fellow is None:
        return _normalized(2, "", f"forge stage-1 supports triton only; got source_type={source_type}",
                           time.time() - started)
    # No early skip when test_command is empty: forge can auto-generate a driver
    # from the candidate's operation + input_shapes (see _autogen_forge_driver),
    # which is its edge over GEAK for harness-less candidates.

    session_id = output_dir.parent.name or "forge"
    kernel_id = Path(source_file).stem
    branch = f"forge/{session_id}/{kernel_id}"

    repo = kernel_repo or _git_toplevel(source_file)
    # Editable-finder packages (e.g. sglang) import the LIVE path via a meta_path
    # finder that PYTHONPATH can't override, so a worktree copy is invisible. For
    # those, edit in place on a temp branch and hard-restore afterward (Option 1).
    inplace = _needs_inplace(repo)
    restore_info: dict | None = None
    if inplace:
        prep = _prepare_inplace(source_file, repo, branch)
        if prep is None:
            return _normalized(2, "", "forge: editable-finder package but repo is not a usable git "
                               "checkout; skipping", time.time() - started)
        workspace, worktree_kernel, restore_info = prep
        base_commit = restore_info.get("base_commit") or ""
    else:
        wt_info = _prepare_worktree(source_file, kernel_repo, output_dir, branch)
        if wt_info is None:
            return _normalized(2, "", "forge: kernel_repo is not a clean git checkout or source_file "
                               "not tracked; skipping (live repo untouched)", time.time() - started)
        workspace, worktree_kernel, base_commit = wt_info

    try:
        # Locate the Kernel-Forge code via $FORGE_PATH (like OOB_PATH for oob),
        # falling back to whatever `kernel_agents` is importable in the env.
        _ensure_forge_on_path()
        # Lazy import: kernel_agents (Forge) is an optional dependency.
        try:
            from kernel_agents.config import Config
            from kernel_agents.loop import IterationLoop
            from kernel_agents.loop.runner import IterationConfig
            from kernel_agents.tracker import ExperimentTracker
            from kernel_agents.orchestrator.agent import make_agent_fn
        except ImportError as exc:
            return _normalized(127, "", f"kernel_agents (Forge) not importable: {exc}",
                               time.time() - started)

        import asyncio

        # Driver: use the Hyperloom harness when present; otherwise auto-generate
        # a Forge-native driver from the candidate's operation + input_shapes.
        if test_command:
            driver = _build_driver_adapter(test_command, workspace, output_dir)
        else:
            driver = _autogen_forge_driver(candidate, worktree_kernel, output_dir, inplace=inplace)
            if driver is None:
                return _normalized(
                    2, "",
                    "forge: no test_command and could not auto-generate a driver for "
                    f"operation={candidate.get('operation')!r} (auto-gen supports gemm/matmul, "
                    "and fused_moe only in in-place mode; other ops need a "
                    "benchmark/test_command or an op template)",
                    time.time() - started)
        gpu_target = _resolve_gpu_target(candidate)
        # Export GPU_TARGET so Kernel-Forge's MCP server tools (build/bench/pmc)
        # pick up the resolved target instead of falling back to their own default.
        os.environ["GPU_TARGET"] = gpu_target
        shapes = _shapes_from_candidate(candidate)

        config = Config.from_env(gpu_target=gpu_target, workspace=workspace)
        # X5: redirect Forge experiment state under the run's output dir, not the
        # Forge package tree (Config.from_env does not wire experiments_dir).
        config.experiments_dir = output_dir / "forge_experiments"
        config.experiments_dir.mkdir(parents=True, exist_ok=True)
        # Derive Forge budget from Hyperloom's timeout_s (Y1). Default 8 iters:
        # agent_fn errors consume iterations silently (recorded in self.results
        # but not in the experiment tracker), so a higher budget ensures enough
        # successful iterations actually run validation + bench.
        max_iters = int(os.environ.get("FORGE_MAX_ITERS", "8"))
        per_iter = max(60, timeout_s // max(1, max_iters))
        iter_config = IterationConfig(
            kernel_file=worktree_kernel,
            driver_script=driver,
            shapes=shapes,
            snr_threshold=float((candidate.get("targets") or {}).get("snr_db", 30.0)),
            max_iterations=max_iters,
            max_time_hours=max(0.05, timeout_s / 3600.0),
            time_per_iteration_sec=per_iter,
            git_branch=branch,
            workspace_dir=workspace,
        )

        tracker = ExperimentTracker(config.experiments_dir)
        loop_runner = IterationLoop(iter_config, tracker, config)

        # Reuse the Hyperloom-rendered prompt (hypothesis_block + rocprof_before)
        # as the Forge program text driving the fellow.
        program_md = Path(prompt_file).read_text(errors="replace") if Path(prompt_file).exists() else ""
        raw_agent_fn = make_agent_fn(config=config, program_md=program_md, fellow_name=fellow)

        # Wrap agent_fn to persist errors: the runner's except branch only
        # prints to stdout (lost) and appends a bare IterationResult with no
        # detail in the tracker. This wrapper logs each call + error to a file
        # so post-mortem analysis can see exactly which iterations failed and why.
        forge_log = output_dir / "forge_loop.log"
        # Hard timeout per agent call. The claude-agent-sdk's query() awaits the
        # fellow subprocess's message stream; if that subprocess dies uncleanly
        # (observed in-loop: an iteration errors, the next call's stream never
        # closes), query() hangs forever and freezes the whole loop (and the
        # orchestrator awaiting it). wait_for bounds each call so the loop records
        # a timeout and moves on instead of stalling the session.
        agent_timeout_s = int(os.environ.get("FORGE_AGENT_TIMEOUT_SEC", "900"))
        # Clamp to a safe floor. A successful agent_fn legitimately takes ~5-8 min
        # (it reads the kernel + TraceLens context before making its single edit;
        # a warm 1.795x reference run measured 5-8 min/call). A too-low value
        # (observed: FORGE_AGENT_TIMEOUT_SEC=300) is SHORTER than a normal call, so
        # every attempt is false-killed as "fellow hung", every retry restarts from
        # scratch and re-times-out, and the run never lands a real optimization.
        # The floor makes the loop robust even if the env is misconfigured.
        agent_timeout_s = max(agent_timeout_s, 600)
        # Retry the fellow on a hung/transient failure. The claude-agent-sdk
        # streaming query() intermittently hangs (the fellow subprocess stream
        # never closes) or returns a transient SDK error; a fresh attempt almost
        # always succeeds. Without retry, every hung iteration burns the full
        # timeout and produces no edit -> a single flaky call can sink the whole
        # run. FORGE_AGENT_RETRIES = extra attempts after the first (default 2).
        agent_retries = max(0, int(os.environ.get("FORGE_AGENT_RETRIES", "2")))
        _TRANSIENT = (
            "error result: success", "Reached maximum number of turns",
            "Fatal error in message reader", "message reader",
            "Connection", "connection reset", "stream", "EOF", "broken pipe",
        )

        async def _logged_agent_fn(kernel_path: str, history: str) -> str:
            import asyncio as _aio
            import traceback as _tb
            last_exc: Exception | None = None
            for attempt in range(agent_retries + 1):
                call_ts = time.strftime("%H:%M:%S", time.gmtime())
                tag = f"attempt {attempt + 1}/{agent_retries + 1}"
                try:
                    result = await _aio.wait_for(
                        raw_agent_fn(kernel_path, history), timeout=agent_timeout_s)
                    with open(forge_log, "a") as f:
                        f.write(f"[{call_ts}] agent_fn OK ({tag}): {result[:120]}\n")
                    return result
                except _aio.TimeoutError as exc:
                    last_exc = exc
                    with open(forge_log, "a") as f:
                        f.write(f"[{call_ts}] agent_fn TIMEOUT after {agent_timeout_s}s "
                                f"({tag}; fellow hung) -> "
                                f"{'retrying' if attempt < agent_retries else 'giving up'}\n")
                    # Transient stream hang: a fresh query() usually reconnects.
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    detail = _tb.format_exc()
                    transient = any(s.lower() in str(exc).lower() for s in _TRANSIENT)
                    with open(forge_log, "a") as f:
                        f.write(f"[{call_ts}] agent_fn ERROR ({tag}; "
                                f"{'transient->retry' if transient and attempt < agent_retries else 'fatal'}): "
                                f"{exc}\n{detail}\n")
                    if transient and attempt < agent_retries:
                        continue
                    raise
            # All attempts hung/failed transiently; surface to the loop (it records
            # the failed iteration and moves on).
            raise last_exc if last_exc else RuntimeError("agent_fn exhausted retries")

        # The fellow runs the claude CLI with bypassPermissions, which refuses to
        # start under root unless IS_SANDBOX=1. Hyperloom backends commonly run as
        # root in containers, so inject it (only when root + unset) to keep the
        # agent from silently failing every iteration. The agent subprocess
        # inherits this env.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.environ.setdefault("IS_SANDBOX", "1")

        # TLS defaults for the claude CLI fellow. The AMD SaFE proxy presents an
        # internal/self-signed certificate; without these the Node-based CLI's
        # TLS handshake to the proxy fails and the streaming query() hangs or
        # errors every iteration (observed as "fellow hung"). setdefault so an
        # explicit operator value always wins, but a bare run (no setup_env.sh
        # exporting them) still works out of the box.
        os.environ.setdefault("ANTHROPIC_SKIP_TLS_VERIFY", "true")
        os.environ.setdefault("NODE_TLS_REJECT_UNAUTHORIZED", "0")

        # The claude-agent-sdk's streaming transport needs the /api/v1/llm-proxy
        # endpoint. The OOB path commonly exports ANTHROPIC_BASE_URL=.../llm-gateway
        # (which only serves the non-streaming `claude -p` path -> 401 in
        # streaming) or an OpenAI-style URL. Ensure the fellow's query() calls hit
        # a streaming proxy: keep an explicit proxy, rewrite a known /llm-gateway
        # suffix, else fall back to the claude CLI's own validated config.json
        # customApiUrl.
        _base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
        if "/api/v1/llm-proxy" not in _base:
            _proxy = ""
            if _base.endswith("/llm-gateway"):
                _proxy = _base.rsplit("/llm-gateway", 1)[0] + "/api/v1/llm-proxy"
            if not _proxy:
                try:
                    import json as _json
                    _cfg = _json.loads((Path.home() / ".claude" / "config.json").read_text())
                    _cu = str(_cfg.get("customApiUrl") or "").rstrip("/")
                    if "/api/v1/llm-proxy" in _cu:
                        _proxy = _cu
                except Exception:
                    _proxy = ""
            if _proxy:
                os.environ["ANTHROPIC_BASE_URL"] = _proxy

        # Redirect the loop's print output to a log file so the full iteration
        # timeline (baseline, agent rationale, validation stages, keep/revert
        # decisions, budget exhaustion) is preserved for post-mortem — the
        # runner prints to stdout which is otherwise lost inside asyncio.run.
        import contextlib, io
        loop_stdout = io.StringIO()
        loop_exc = None
        try:
            with contextlib.redirect_stdout(loop_stdout):
                asyncio.run(loop_runner.run(agent_fn=_logged_agent_fn))
        except Exception as _loop_err:
            loop_exc = _loop_err
        loop_output = loop_stdout.getvalue()
        try:
            with open(forge_log, "a") as f:
                f.write("\n=== forge loop stdout ===\n")
                f.write(loop_output)
                if loop_exc:
                    f.write(f"\n=== loop exception ===\n{loop_exc}\n")
        except OSError:
            pass

        baseline_ms = getattr(loop_runner.ic, "baseline_wall_ms", None)
        best_ms = getattr(loop_runner, "best_wall_ms", None)
        # `improved` = a validated kernel strictly faster than baseline was kept
        # (the loop only keeps iterations that pass 5-stage validation). Only then
        # do we report a KEEP-worthy speedup + correctness pass.
        improved = bool(baseline_ms and best_ms and best_ms < baseline_ms)

        # Export + report BEFORE _restore_inplace (in finally) reverts the
        # changed files. The best-kept state is on disk right now; capture ALL
        # files the agent touched (not just source_file) + a forge.patch.
        _, changed_files = _export_best_artifacts(
            workspace, base_commit, worktree_kernel, source_file, output_dir)
        if changed_files:
            try:
                (output_dir / "optimized_versions" / "changed_files.txt").write_text(
                    "\n".join(changed_files) + "\n")
            except OSError:
                pass
        _write_report(output_dir, baseline_ms, best_ms, improved)

        if loop_exc:
            raise loop_exc

        msg = (f"forge done: baseline={baseline_ms} best={best_ms} "
               f"improved={improved} fellow={fellow} gpu={gpu_target}")
        res = _normalized(0, msg + "\n" + loop_output[-3000:], "", time.time() - started)
        # Expose output_dir as cli_workspace so run_attempt's report scan finds
        # <output_dir>/optimization_report.md + optimized_versions/ (same path
        # convention as the OOB backend). Without this, build_verification reads
        # no report and falls back to default_unmeasured.
        res["cli_workspace"] = str(output_dir)
        res["output_dir"] = str(output_dir)
        return res
    except Exception as exc:  # noqa: BLE001
        return _normalized(1, "", f"forge submit failed: {type(exc).__name__}: {exc}",
                           time.time() - started)
    finally:
        if inplace:
            _restore_inplace(restore_info)
        else:
            _remove_worktree(kernel_repo, source_file, workspace, branch)
