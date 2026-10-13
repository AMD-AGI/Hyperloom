# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pinned Magpie source audit executed in the selected Magpie interpreter."""

MAGPIE_SOURCE_IDENTITY_CODE = r"""
import hashlib
import json
import re
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse


_MAGPIE_EXECUTION_SUFFIXES = (".py", ".sh", ".yaml", ".yaml.example", ".json")
# The pinned source commit contains Magpie/mcp/config.json, but its wheel does
# not publish that file (pyproject.toml only publishes top-level JSON).  It is
# unrelated to AgentX.  Hash the complete *published* execution tree so wheels
# and clean editable/source installs have one canonical identity.
_MAGPIE_UNPUBLISHED_PATHS = frozenset({"mcp/config.json"})
_AUDITED_MAGPIE_EXECUTION_TREES = {
    # Keep the release identity for native sessions resuming their accepted pin.
    "5d6aaa29f2ee78e53ec35c8bc2e082d8a10c47f0": {
        "file_count": 79,
        "tree_sha256": "afea9b95e992e0b715c955f5e72a7e92aca48d2c171ddff0e3d3e9082406e689",
    },
    # v0.3.0 plus the generic eval source-path hotfix, independently audited from its wheel.
    "a3339dc2776ee0c977fb3313fe89f56da7a91555": {
        "file_count": 79,
        "tree_sha256": "113f880b18ccd3ec26e6a432fcdf06a0c365520d51c3ed3c46069d33d7f07e93",
    },
    "d80eb4d3dad7fabe01ce81d049e2983adf2c86dd": {
        "file_count": 87,
        "tree_sha256": "4c827c3d4b6bb7f756964fdecaeb43b255470ee3285f5dc89d9d320c7e9164dc",
    },
    "cf805323f0a6db6c63a0aa4ffcce7acf4e830ea6": {
        "file_count": 87,
        "tree_sha256": "65b4acccbe70c05c033f75a5054344bd2fead13dc81878f343427c6c04142e4a",
    },
    "311d56429a3aadc3cfe1e1e0e2e533482d224cf0": {
        "file_count": 87,
        "tree_sha256": "f8be61cf4566f4daaf3c2463a600e49b11d97bade6632a7bb5b99050b4ce5656",
    },
    "938bfd7a9bc1f3970f3fbce8b65ad88a8e0958f5": {
        "file_count": 87,
        "tree_sha256": "af43bc3fcbe768b30b87cd5a23ff6f4d36e0e865a8db7375ca4e205970990768",
    },
    "63e3b19bee66700dbb012c538f121c9de0d86a2a": {
        "file_count": 87,
        "tree_sha256": "624a89b8e74adda9ab95e45d692cfb58a267b7e2d4de9a96076db24626d49e20",
    },
    "c5c80698fef1b89cc6882264b80d5b306d4e9328": {
        "file_count": 87,
        "tree_sha256": "1f475d413e7af20204da8b8abcd06a6a69916d3960f175c72d627663ce33d94f",
    },
    "658562345ad1a7e5a617e3631f3acfcec0eade4a": {
        "file_count": 86,
        "tree_sha256": "514ca3691163bb304cef31d392d9c10bbb5a3b3570b219f95ed12c6aeae29c69",
    },
    "d72965776df5416dad063c00237f6e389b841162": {
        "file_count": 81,
        "tree_sha256": "84dd7920ff992571c7dc6d485a4680ac3fc2afe7b6dde606dbd754d545a5729c",
    },
}


def _magpie_execution_tree_identity(package_root):
    '''Hash every published executable/config input below ``Magpie/``.'''
    package_root = Path(package_root).resolve(strict=True)
    files = {}
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts:
            continue
        relative_name = relative.as_posix()
        if path.is_symlink():
            raise RuntimeError(
                "installed Magpie execution tree must not contain symlinks: "
                + relative_name
            )
        if (
            not path.is_file()
            or relative_name in _MAGPIE_UNPUBLISHED_PATHS
            or not relative_name.endswith(_MAGPIE_EXECUTION_SUFFIXES)
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[relative_name] = digest.hexdigest()
    tree_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"file_count": len(files), "tree_sha256": tree_sha256}


def _validate_magpie_execution_tree(package_root, source_commit):
    expected = _AUDITED_MAGPIE_EXECUTION_TREES.get(source_commit)
    if expected is None:
        raise RuntimeError(
            "installed Magpie commit is not audited for native AgentX: "
            + (source_commit or "<unknown>")
        )
    actual = _magpie_execution_tree_identity(package_root)
    if actual != expected:
        raise RuntimeError(
            "installed Magpie published execution tree differs from the audited "
            f"commit: expected {expected}, got {actual}"
        )
    return actual


def _resolve_magpie_source_identity(package_root):
    '''Return the commit/url that own an imported Magpie package.

    A wheel normally lives below ``<some checkout>/.venv/site-packages``.  Git
    walks parent directories, so an unconditional ``git -C site-packages`` can
    accidentally report that enclosing checkout's HEAD as Magpie's commit.
    Prefer PEP 610 VCS provenance for wheels.  Only consult Git for an editable
    or direct source import whose package is the checkout's own top-level
    ``Magpie/`` directory.
    '''
    package_root = Path(package_root).resolve()
    dist = None
    try:
        dist = distribution("magpie-eval")
    except (PackageNotFoundError, OSError, TypeError, ValueError):
        pass
    try:
        direct_url = (
            json.loads(dist.read_text("direct_url.json") or "{}") if dist else {}
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        direct_url = {}
    if not isinstance(direct_url, dict):
        direct_url = {}

    source_url = str(direct_url.get("url") or "")
    dir_info = direct_url.get("dir_info")
    dir_info = dir_info if isinstance(dir_info, dict) else {}
    editable = dir_info.get("editable") is True
    vcs_info = direct_url.get("vcs_info")
    vcs_info = vcs_info if isinstance(vcs_info, dict) else {}
    vcs_commit = str(vcs_info.get("commit_id") or "").strip().lower()

    if dist is not None and not editable:
        try:
            distribution_root = Path(dist.locate_file("Magpie")).resolve(strict=True)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot bind the magpie-eval distribution to imported Magpie"
            ) from exc
        if distribution_root != package_root:
            raise RuntimeError(
                "magpie-eval distribution does not own the imported Magpie package: "
                f"distribution={distribution_root}, imported={package_root}"
            )

    if (
        dist is not None
        and not editable
        and str(vcs_info.get("vcs") or "").strip().lower() == "git"
        and re.fullmatch(r"[0-9a-f]{40}", vcs_commit)
    ):
        return vcs_commit, source_url

    if dist is not None and not editable:
        # A non-VCS wheel/local archive has no immutable source identity.  Do
        # not fall through to an unrelated repository above site-packages.
        return "", source_url

    source_root = package_root.parent
    if editable:
        parsed = urlparse(source_url)
        if parsed.scheme not in {"", "file"} or parsed.netloc not in {"", "localhost"}:
            return "", source_url
        source_root = Path(unquote(parsed.path)).resolve()

    top_proc = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if top_proc.returncode != 0:
        return "", source_url
    try:
        git_top = Path((top_proc.stdout or "").strip()).resolve(strict=True)
    except (OSError, RuntimeError):
        return "", source_url
    if editable and source_root != git_top:
        return "", source_url
    if package_root != (git_top / "Magpie").resolve():
        return "", source_url

    head_proc = subprocess.run(
        ["git", "-C", str(git_top), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = (head_proc.stdout or "").strip().lower() if head_proc.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return "", source_url
    status_proc = subprocess.run(
        [
            "git",
            "-C",
            str(git_top),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "Magpie",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        return "", source_url
    dirty = [
        line
        for line in (status_proc.stdout or "").splitlines()
        if line.strip()
        and not (line.startswith("?? ") and "__pycache__/" in line)
    ]
    if dirty:
        raise RuntimeError(
            "native AgentX requires a clean Magpie/ source tree: " + ", ".join(dirty)
        )
    return commit, source_url or git_top.as_uri()
"""
