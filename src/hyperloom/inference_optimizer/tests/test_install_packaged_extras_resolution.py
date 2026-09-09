# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards that the packaged install resolves extras off the installed wheel.

The packaged branch of ensure_inference_optimizer() names the runtime deps by
EXTRA (`hyperloom-inference_optimizer[llm,forge]`) so pyproject.toml stays the
single source of truth. That only holds if pip can SEE the wheel it is naming.

The wheel lands via `pip install --target $REPO_ROOT`, and a --target dir is not
on the default sys.path. pip discovers installed distributions from sys.path and
does NOT scan the cwd, so with REPO_ROOT absent from PYTHONPATH the name misses
the local wheel and resolves from the package index instead -- installing the
last PUBLISHED release's metadata. An extra introduced in the version under
development is then invisible, so the pre-release gate can never exercise it.

The failure is silent in both directions: the neighbouring
`import hyperloom.inference_optimizer` guard runs as `python -` (cwd on
sys.path), so it passes from the leg root even when pip cannot see the wheel.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SH = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"

DIST = "hyperloom-inference_optimizer"


def _extract_func(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"could not locate {name}() in install.sh"
    return m.group(0)


# Records the PYTHONPATH each pip invocation actually carried; every import probe
# succeeds so the run reaches the pip call the way a healthy leg does.
_FAKE_PYTHON = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, sys

    args = sys.argv[1:]
    if args[:2] == ["-m", "pip"] and "install" in args:
        with open(os.environ["PIPLOG"], "a") as f:
            f.write(os.environ.get("PYTHONPATH", "") + "\\t" + " ".join(args) + "\\n")
    sys.exit(0)
    """
)


def _harness(target_root: Path) -> str:
    parts = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'log()  { echo "[log] $*"; }',
        'warn() { echo "[warn] $*" >&2; }',
        'die()  { echo "[die] $*" >&2; exit 1; }',
        "CHECK_ONLY=0",
        "DRY_RUN=0",
        "PIP_EXTRA=()",
        "HYPERLOOM_PACKAGED_INSTALL=1",
        f'REPO_ROOT="{target_root}"',
        'PYTHON="$FAKE_PYTHON"',
        _extract_func("_check_kernelforge_ready"),
        _extract_func("ensure_inference_optimizer"),
        "ensure_inference_optimizer",
    ]
    return "\n\n".join(parts) + "\n"


def _run(tmp_path: Path, *, pythonpath: str | None = None):
    """Run the packaged branch from the leg root, as the pre-release gate does."""
    target_root = tmp_path / "leg-root"
    target_root.mkdir()
    fake_py = tmp_path / "fakepy"
    fake_py.write_text(_FAKE_PYTHON, encoding="utf-8")
    fake_py.chmod(0o755)
    piplog = tmp_path / "pip.log"
    piplog.write_text("", encoding="utf-8")
    script = tmp_path / "harness.sh"
    script.write_text(_harness(target_root), encoding="utf-8")

    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "FAKE_PYTHON": str(fake_py),
        "PIPLOG": str(piplog),
    }
    # Default: no PYTHONPATH, which is the state after the .env scrub unsets it.
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=target_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    return proc, target_root, piplog.read_text(encoding="utf-8")


def _extras_call(piplog: str) -> tuple[str, str]:
    lines = [ln for ln in piplog.splitlines() if f"{DIST}[" in ln]
    assert len(lines) == 1, f"expected exactly one extras install, got:\n{piplog}"
    seen_pythonpath, argv = lines[0].split("\t", 1)
    return seen_pythonpath, argv


def test_extras_install_can_see_the_target_dir(tmp_path: Path) -> None:
    # Without REPO_ROOT on PYTHONPATH pip cannot see the --target wheel and
    # silently resolves the distribution from the index instead.
    proc, target_root, piplog = _run(tmp_path)
    assert proc.returncode == 0, proc.stdout
    seen_pythonpath, argv = _extras_call(piplog)
    assert str(target_root) in seen_pythonpath.split(":"), (
        f"the extras install ran with PYTHONPATH={seen_pythonpath!r}, which does not "
        f"contain the --target root {target_root}; pip will resolve {DIST} from the "
        f"index and read the last published release's extras.\nargv: {argv}"
    )


def test_a_preexisting_pythonpath_is_preserved(tmp_path: Path) -> None:
    # An isolated framework venv puts its own entries here; dropping them would
    # shadow that venv's torch, so REPO_ROOT is prepended rather than replacing.
    proc, target_root, piplog = _run(tmp_path, pythonpath="/opt/pre-existing")
    assert proc.returncode == 0, proc.stdout
    seen_pythonpath, _ = _extras_call(piplog)
    assert seen_pythonpath.split(":") == [str(target_root), "/opt/pre-existing"], (
        f"expected REPO_ROOT prepended to the inherited PYTHONPATH, got {seen_pythonpath!r}"
    )


def test_source_checkout_branch_still_installs_from_the_local_path() -> None:
    # The editable branch already names a path, so it resolves in-tree with no
    # PYTHONPATH help; it must stay a local install and never become a name.
    body = _extract_func("ensure_inference_optimizer")
    assert '-e "${REPO_ROOT}[test]"' in body, "the source-checkout branch must install the local project"
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert code.count(f'"{DIST}[') == 1, f"only the packaged branch may name {DIST} by distribution:\n{code}"
