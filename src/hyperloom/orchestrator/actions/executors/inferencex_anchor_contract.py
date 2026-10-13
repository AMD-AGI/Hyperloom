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
import json
from pathlib import Path
import subprocess

from hyperloom.inference_optimizer.cli.preflight import INFERENCEX_REPO_DEFAULT

from ._inferencex_patcher import (
    _ANCHOR_CONTRACT,
    _ANCHOR_ALTERNATIVES,
    _BENCH_SERVING_REL_PARTS,
    EVAL_PROBE_TARGETS,
    EVAL_PROBE_TARGET_PARTS,
    count_anchor_hits,
)

_ENCODING = "utf-8"


REFRESH_CMD = "python scripts/refresh_inferencex_anchor_contract.py"
_FETCH_TIMEOUT_SEC = 30


PROBE_TARGET_PATH = "/".join(EVAL_PROBE_TARGET_PARTS)

# The second patcher aimed at the same file, and the one whose failure is louder.
MAGPIE_LIB_PATH = "benchmarks/benchmark_lib.sh"


def magpie_patch_applies(text: str) -> bool:
    """Whether the run_lm_eval argument-parser splice still finds its site."""
    from hyperloom.orchestrator.actions.executors._magpie_patcher import (
        _patch_merged_case_parser,
    )

    return _patch_merged_case_parser(text) is not None


def _magpie_pattern_parts() -> list[str]:
    """The magpie-side patterns, for the fingerprint."""
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    return [
        f"magpie_merged_catchall\x1f{MAGPIE_LIB_PATH}\x1f{mp._RUN_LM_EVAL_MERGED_CATCHALL_RE.pattern}",
        f"magpie_run_lm_eval_fn\x1f{MAGPIE_LIB_PATH}\x1f{mp._RUN_LM_EVAL_FN_MARKER}",
    ]


def anchors_fingerprint() -> str:
    """Fingerprint the anchor definitions themselves."""
    parts = [f"{name}\x1f{'/'.join(rel_parts)}\x1f{anchor}" for name, rel_parts, _sentinel, anchor in _ANCHOR_CONTRACT]
    parts.append(f"probe_target\x1f{PROBE_TARGET_PATH}")
    parts.extend(_magpie_pattern_parts())
    parts.append(json.dumps(_ANCHOR_ALTERNATIVES, sort_keys=True))
    parts.append(json.dumps([_BENCH_SERVING_REL_PARTS, EVAL_PROBE_TARGETS]))
    return hashlib.sha256("\x1e".join(parts).encode(_ENCODING)).hexdigest()


def github_slug(clone_url: str) -> str:
    """Turn a clone URL into the ``owner/repo`` form the API expects."""
    return clone_url.rstrip("/").removesuffix(".git").split("github.com/", 1)[-1]


def fetch_pinned_file(rel_path: str, ref: str, checkout: Path | None = None) -> str | None:
    """Fetch one upstream file at ``ref``, or ``None`` when unreachable."""
    command = (
        ["git", "-C", str(checkout), "show", f"{ref}:{rel_path}"]
        if checkout is not None
        else [
            "gh",
            "api",
            f"repos/{github_slug(INFERENCEX_REPO_DEFAULT)}/contents/{rel_path}?ref={ref}",
            "-H",
            "Accept: application/vnd.github.raw",
        ]
    )
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            timeout=_FETCH_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode(_ENCODING, errors="replace")


def upstream_candidates(paths: tuple[tuple[str, ...], ...]) -> list[str]:
    """Resolve project-relative targets in either repository layout."""
    return ["/".join((*prefix, *parts)) for prefix in (("inferencex-e2e",), ()) for parts in paths]


def _fetch_target(paths: tuple[tuple[str, ...], ...], ref: str, checkout: Path | None) -> tuple[str, str]:
    for path in upstream_candidates(paths):
        text = fetch_pinned_file(path, ref, checkout)
        if text is not None:
            return path, text
    raise RuntimeError(f"cannot read any supported path {upstream_candidates(paths)} at {ref}")


def build_record(ref: str, checkout: Path | None = None) -> dict:
    """Verify every active synthetic patch against immutable upstream objects."""
    files: dict[str, dict] = {}
    for name, parts, _sentinel, anchor in _ANCHOR_CONTRACT:
        paths = _BENCH_SERVING_REL_PARTS if name == "profile_extra_body" else (parts,)
        rel_path, text = _fetch_target(paths, ref, checkout)
        hits = count_anchor_hits(text, anchor)
        if hits != 1:
            raise RuntimeError(f"{rel_path} at {ref}: anchor {name} matched {hits} sites, expected exactly one")
        spec = files.setdefault(rel_path, {"sha256": hashlib.sha256(text.encode(_ENCODING)).hexdigest(), "anchors": {}})
        spec["anchors"][name] = hits
    probe_path, probe_text = _fetch_target(EVAL_PROBE_TARGETS, ref, checkout)
    magpie_path, magpie_text = _fetch_target((("benchmarks", "benchmark_lib.sh"),), ref, checkout)
    if not magpie_patch_applies(magpie_text):
        raise RuntimeError(f"{magpie_path} at {ref}: run_lm_eval --concurrent-requests splice no longer applies")
    return {
        "ref": ref,
        "anchors_fingerprint": anchors_fingerprint(),
        "refresh_with": REFRESH_CMD,
        "files": files,
        "probe_target": {"path": probe_path, "sha256": hashlib.sha256(probe_text.encode(_ENCODING)).hexdigest()},
        "magpie_patch": {"path": magpie_path, "applies": True},
    }


def anchors_by_file() -> dict[str, list[tuple[str, str]]]:
    """Group the patch anchors by the upstream file they are matched against."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, rel_parts, _sentinel, anchor in _ANCHOR_CONTRACT:
        grouped.setdefault("/".join(rel_parts), []).append((name, anchor))
    return grouped
