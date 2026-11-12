# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Insert the windows_rocm_rocm bootstrap import at the top of vLLM's package __init__,
so the Windows/ROCm compatibility shims are installed before any vLLM submodule loads
torch.distributed. Idempotent.

Also applies the repo's vLLM source patches (patches/vllm/*.patch) to the checkout when
git is available, so a fresh clone reaches the documented behaviour without manual
`git -C vllm apply` steps. Patching is skipped (with a notice) when the checkout is not
a git work tree. Already-applied patches are detected from the tree itself, so re-runs are
safe and a partially reset clone gets exactly the missing patches back.

Usage: python tools/patch_vllm.py [path-to-vllm-checkout]   (default: ./vllm)
"""
import os
import shutil
import subprocess
import sys
import tempfile

MARK = "vllm_windows_rocm.bootstrap"
# The patches were generated against this commit (the v0.19.1 tag); see patches/README.md.
# A clone at any other commit may make them fail or apply with shifted context.
EXPECTED_VLLM_BASE = "b1388b1"
BLOCK = (
    "\n# --- vLLM-on-Windows-ROCm: install the single-process torch.distributed shim and\n"
    "# _C op fallbacks before any vllm submodule that imports torch.distributed is loaded.\n"
    "try:\n"
    "    import vllm_windows_rocm.bootstrap  # noqa: F401\n"
    "except Exception:\n"
    "    pass\n"
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_apply(vllm_root: str, patch_path: str, *extra: str) -> subprocess.CompletedProcess:
    """git apply with the flags that make Windows-checkout patches behave:
    --ignore-whitespace (CRLF vs LF mismatches between checkout and patch files)."""
    return subprocess.run(
        ["git", "-C", vllm_root, "apply", "--ignore-whitespace", "--whitespace=nowarn",
         *extra, patch_path],
        capture_output=True, text=True,
    )


def _patch_already_applied(vllm_root: str, patch_path: str, *extra: str) -> bool:
    """True when reverse-applying cleanly succeeds, i.e. the patch is already in the tree."""
    return _git_apply(vllm_root, patch_path, *extra, "--check", "-R").returncode == 0


def _touched_files(patch_path: str) -> list[str]:
    """Repo-relative paths a patch modifies, creates or deletes, in patch order."""
    files: list[str] = []
    with open(patch_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("--- a/", "+++ b/")):
                rel = line[6:].rstrip("\r\n").split("\t")[0]
                if rel not in files:
                    files.append(rel)
    return files


def _only(rel: str) -> str:
    """git apply argument limiting a patch to one file (glob characters escaped)."""
    return "--include=" + "".join("\\" + c if c in "*?[]\\" else c for c in rel)


def _applied_under_later(vllm_root: str, patch_dir: str, names: list[str], i: int,
                         rel: str) -> str | None:
    """The later patch that rewrote patch i's context in file `rel`, when patch i is applied
    underneath it there; None otherwise.

    Reverse detection cannot see patch i once a later patch changed lines inside its context
    (triton-attn-pc-scale over native-cache-ops). Verify it the way it was applied: copy the
    file to a scratch dir, reverse there the later patches applied to it (last first), then
    reverse-check patch i. `git apply` cannot chain several patches to one file in a single
    call, hence the copy; nothing in the clone is modified."""
    later = [n for n in names[i + 1:] if rel in _touched_files(os.path.join(patch_dir, n))]
    src = os.path.join(vllm_root, rel)
    if not later or not os.path.isfile(src):
        return None
    tmp = tempfile.mkdtemp(prefix="vw_patchcheck_")
    try:
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        # Outside a repository `git apply` patches files relative to cwd; the ceiling stops git
        # from discovering a repository above the scratch dir.
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=os.path.dirname(tmp))

        def reverse(name: str, *extra: str) -> bool:
            return subprocess.run(
                ["git", "apply", "--ignore-whitespace", "--whitespace=nowarn", "-R", _only(rel),
                 *extra, os.path.join(patch_dir, name)],
                cwd=tmp, env=env, capture_output=True,
            ).returncode == 0

        undone = [n for n in reversed(later) if reverse(n, "--check") and reverse(n)]
        return undone[-1] if undone and reverse(names[i], "--check") else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _apply_one(vllm_root: str, patch_dir: str, names: list[str], i: int) -> tuple[str, list[str]]:
    """Bring patch i into the tree. Returns (status line, error lines)."""
    name = names[i]
    p = os.path.join(patch_dir, name)
    if _patch_already_applied(vllm_root, p):
        return f"already applied: {name}", []
    r = _git_apply(vllm_root, p)
    if r.returncode == 0:
        return f"applied: {name}", []
    # Neither in the tree nor applicable as a whole. git apply is all-or-nothing per patch, so
    # a clone where only some of the patch's files were reset lands here, as does a patch whose
    # context a later patch rewrote. Decide file by file.
    applied, already, rewritten, errors = [], [], [], []
    for rel in _touched_files(p):
        if _patch_already_applied(vllm_root, p, _only(rel)):
            already.append(rel)
            continue
        rf = _git_apply(vllm_root, p, _only(rel))
        if rf.returncode == 0:
            applied.append(rel)
            continue
        over = _applied_under_later(vllm_root, patch_dir, names, i, rel)
        if over:
            rewritten.append(over)
            continue
        errors.append(rf.stderr.strip() or f"{rel}: patch does not apply")
    if errors:
        return f"FAILED to apply {name}", errors
    if applied:
        total = len(applied) + len(already) + len(rewritten)
        return (f"applied: {name} (to {len(applied)} of its {total} files; "
                "the others already had it)"), []
    if rewritten:
        return (f"already applied: {name} (part of its context was since rewritten by "
                f"{', '.join(sorted(set(rewritten)))})"), []
    return f"already applied: {name}", []


def apply_source_patches(vllm_root: str) -> int:
    patch_dir = os.path.join(_repo_root(), "patches", "vllm")
    if not os.path.isdir(patch_dir):
        return 0
    patches = sorted(f for f in os.listdir(patch_dir) if f.endswith(".patch"))
    if not patches:
        return 0
    probe = subprocess.run(
        ["git", "-C", vllm_root, "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
    )
    if probe.returncode != 0:
        print(f"notice: {vllm_root} is not a git work tree; skipping "
              f"{len(patches)} source patch(es), apply them manually per patches/README.md")
        return 0
    head = subprocess.run(
        ["git", "-C", vllm_root, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if head.returncode == 0 and not head.stdout.strip().startswith(EXPECTED_VLLM_BASE):
        print(f"WARNING: vllm checkout is at {head.stdout.strip()[:12]}, but the patches were "
              f"generated against {EXPECTED_VLLM_BASE} (tag v0.19.1). Continuing; patches that "
              "no longer apply cleanly will be reported below. See patches/README.md.")
    # Detection reads the tree, not a record of past runs: a record goes stale as soon as the
    # clone is reset (`git checkout .` keeps untracked files) and would then skip patches that
    # are no longer there.
    failures = 0
    for i, name in enumerate(patches):
        status, errors = _apply_one(vllm_root, patch_dir, patches, i)
        if not errors:
            print(status, flush=True)
            continue
        failures += 1
        print(status, file=sys.stderr, flush=True)
        for e in errors:
            print("  " + e.replace("\n", "\n  "), file=sys.stderr, flush=True)
        print(f"notice: {name} is neither in the tree nor applicable to it: the files above have "
              f"local edits, or the clone is not at {EXPECTED_VLLM_BASE}. See patches/README.md.",
              file=sys.stderr, flush=True)
    return 1 if failures else 0


def install_sitecustomize(venv_site: str) -> None:
    """Copy sitecustomize.py into site-packages so EVERY interpreter start (including
    vLLM's model-inspection subprocess `python -m vllm.model_executor.models.registry`,
    which does not load the platform plugin) applies the torch.distributed shim.
    Without it, `vllm serve` fails on un-cached architectures with
    ModuleNotFoundError: torch._C._distributed_c10d. Idempotent by content hash."""
    src = os.path.join(_repo_root(), "windows_rocm_plugin", "sitecustomize.py")
    if not os.path.isfile(src):
        print(f"notice: {src} not found; skipping sitecustomize install")
        return
    dst = os.path.join(venv_site, "sitecustomize.py")
    try:
        same = os.path.isfile(dst) and open(src, "rb").read() == open(dst, "rb").read()
    except OSError:
        same = False
    if same:
        print("sitecustomize: already installed")
        return
    try:
        import shutil

        shutil.copyfile(src, dst)
        print(f"sitecustomize: installed to {dst}")
    except OSError as e:
        print(f"notice: could not install sitecustomize ({e}); `vllm serve` may fail on "
              "model-architecture inspection. Copy windows_rocm_plugin/sitecustomize.py "
              "to your venv's site-packages manually.")


def _venv_site_packages() -> str | None:
    """site-packages of the python running this script (venv or not)."""
    import sysconfig

    p = sysconfig.get_paths().get("purelib")
    return p


def main(argv):
    vllm_root = argv[0] if argv else "vllm"
    init_py = os.path.join(vllm_root, "vllm", "__init__.py")
    if not os.path.isfile(init_py):
        print(f"not found: {init_py}", file=sys.stderr)
        return 1
    site = _venv_site_packages()
    if site:
        install_sitecustomize(site)
    src = open(init_py, encoding="utf-8").read()
    if MARK in src:
        print("already patched")
    else:
        anchor = "from .version import __version__"
        idx = src.find(anchor)
        if idx == -1:
            # fall back: prepend
            new = BLOCK + src
        else:
            eol = src.find("\n", idx)
            new = src[: eol + 1] + BLOCK + src[eol + 1 :]
        open(init_py, "w", encoding="utf-8", newline="\n").write(new)
        print("patched", init_py)
    return apply_source_patches(vllm_root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
