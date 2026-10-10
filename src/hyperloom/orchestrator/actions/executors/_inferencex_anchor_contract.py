# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract between Hyperloom's patch anchors and the pinned InferenceX tree.

Shared by ``scripts/refresh_inferencex_anchor_contract.py``, which records the
contract, and the test that checks the recorded contract against the anchors; the
record itself is a test fixture
(``src/hyperloom/inference_optimizer/tests/fixtures/inferencex_anchor_contract.json``).
"""

from __future__ import annotations

import hashlib
import subprocess

from hyperloom.inference_optimizer.cli.preflight import _INFERENCEX_REPO_DEFAULT

from ._inferencex_patcher import (
    _ANCHOR_CONTRACT,
    EVAL_PROBE_TARGET_PARTS,
    count_anchor_hits,
)

REFRESH_CMD = "python scripts/refresh_inferencex_anchor_contract.py"
_FETCH_TIMEOUT_SEC = 30
_ENCODING = "utf-8"


PROBE_TARGET_PATH = "/".join(EVAL_PROBE_TARGET_PARTS)

# The second patcher aimed at the same file, and the one whose failure is louder.
MAGPIE_LIB_PATH = "benchmarks/benchmark_lib.sh"


def magpie_patch_applies(text: str) -> bool:
    """Whether the run_lm_eval argument-parser splice still finds its site."""
    from ._magpie_patcher import _patch_merged_case_parser

    return _patch_merged_case_parser(text) is not None


def _magpie_pattern_parts() -> list[str]:
    """The magpie-side patterns, for the fingerprint."""
    from . import _magpie_patcher as mp

    return [
        f"magpie_merged_catchall\x1f{MAGPIE_LIB_PATH}\x1f{mp._RUN_LM_EVAL_MERGED_CATCHALL_RE.pattern}",
        f"magpie_run_lm_eval_fn\x1f{MAGPIE_LIB_PATH}\x1f{mp._RUN_LM_EVAL_FN_MARKER}",
    ]


def anchors_by_file() -> dict[str, list[tuple[str, str]]]:
    """Group the patch anchors by the upstream file they are matched against."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, rel_parts, _sentinel, anchor in _ANCHOR_CONTRACT:
        grouped.setdefault("/".join(rel_parts), []).append((name, anchor))
    return grouped


def anchors_fingerprint() -> str:
    """Fingerprint the anchor definitions themselves."""
    parts = [f"{name}\x1f{'/'.join(rel_parts)}\x1f{anchor}" for name, rel_parts, _sentinel, anchor in _ANCHOR_CONTRACT]
    parts.append(f"probe_target\x1f{PROBE_TARGET_PATH}")
    parts.extend(_magpie_pattern_parts())
    return hashlib.sha256("\x1e".join(parts).encode(_ENCODING)).hexdigest()


def github_slug(clone_url: str) -> str:
    """Turn a clone URL into the ``owner/repo`` form the API expects."""
    return clone_url.rstrip("/").removesuffix(".git").split("github.com/", 1)[-1]


def fetch_pinned_file(rel_path: str, ref: str) -> str | None:
    """Fetch one upstream file at ``ref``, or ``None`` when unreachable."""
    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{github_slug(_INFERENCEX_REPO_DEFAULT)}/contents/{rel_path}?ref={ref}",
                "-H",
                "Accept: application/vnd.github.raw",
            ],
            capture_output=True,
            timeout=_FETCH_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode(_ENCODING, errors="replace")


def build_record(ref: str) -> dict:
    """Verify every anchor against upstream at ``ref`` and return the record."""
    files: dict[str, dict] = {}
    texts: dict[str, str] = {}
    for rel_path, anchors in anchors_by_file().items():
        text = fetch_pinned_file(rel_path, ref)
        if text is None:
            raise RuntimeError(f"cannot fetch {rel_path} at {ref}; is `gh auth status` clean?")
        texts[rel_path] = text
        hits = {name: count_anchor_hits(text, anchor) for name, anchor in anchors}
        broken = {name: n for name, n in hits.items() if n != 1}
        if broken:
            raise RuntimeError(
                f"{rel_path} at {ref}: expected each anchor to match exactly one site, got {broken}. "
                "Re-anchor these patches in _inferencex_patcher.py before recording the contract."
            )
        files[rel_path] = {
            "sha256": hashlib.sha256(text.encode(_ENCODING)).hexdigest(),
            "anchors": hits,
        }
    probe_target = fetch_pinned_file(PROBE_TARGET_PATH, ref)
    if probe_target is None:
        raise RuntimeError(
            f"cannot fetch {PROBE_TARGET_PATH} at {ref}. The probe and the request bounds are "
            "appended to this file; if upstream moved it, re-home them in _inferencex_patcher.py "
            "before recording the contract."
        )
    magpie_text = texts.get(MAGPIE_LIB_PATH) or fetch_pinned_file(MAGPIE_LIB_PATH, ref)
    if magpie_text is None:
        raise RuntimeError(f"cannot fetch {MAGPIE_LIB_PATH} at {ref}; is `gh auth status` clean?")
    if not magpie_patch_applies(magpie_text):
        raise RuntimeError(
            f"{MAGPIE_LIB_PATH} at {ref}: the run_lm_eval --concurrent-requests splice no longer "
            "finds its site. install.sh die()s when this patch cannot apply, so recording the "
            "contract now would ship a broken install. Re-anchor _magpie_patcher.py first."
        )
    return {
        "ref": ref,
        "anchors_fingerprint": anchors_fingerprint(),
        "refresh_with": REFRESH_CMD,
        "files": files,
        "probe_target": {
            "path": PROBE_TARGET_PATH,
            "sha256": hashlib.sha256(probe_target.encode(_ENCODING)).hexdigest(),
        },
        "magpie_patch": {"path": MAGPIE_LIB_PATH, "applies": True},
    }
