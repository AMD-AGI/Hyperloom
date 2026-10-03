# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract between Hyperloom's patch anchors and the pinned InferenceX tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.preflight import (
    _INFERENCEX_REF_DEFAULT,
    _INFERENCEX_REPO_DEFAULT,
)
from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
    _ANCHOR_CONTRACT,
    _ANCHOR_ALTERNATIVES,
    _BENCH_SERVING_REL_PARTS,
    EVAL_PROBE_TARGET_PARTS,
    EVAL_PROBE_TARGETS,
    count_anchor_hits,
)

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "inferencex_anchor_contract.json"
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
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


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
            f"repos/{github_slug(_INFERENCEX_REPO_DEFAULT)}/contents/{rel_path}?ref={ref}",
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
    return proc.stdout.decode("utf-8", errors="replace")


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
        spec = files.setdefault(rel_path, {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "anchors": {}})
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
        "probe_target": {"path": probe_path, "sha256": hashlib.sha256(probe_text.encode("utf-8")).hexdigest()},
        "magpie_patch": {"path": magpie_path, "applies": True},
    }


def load_record() -> dict:
    """Return the checked-in contract record."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


# --- hermetic ---------------------------------------------------------------


def test_recorded_ref_matches_the_pin_the_code_installs():
    """Bumping INFERENCEX_REF is exactly when an anchor silently rots, so the bump cannot be allowed to pass without someone re-verifying."""
    record = load_record()

    assert record["ref"] == _INFERENCEX_REF_DEFAULT, (
        f"INFERENCEX_REF moved to {_INFERENCEX_REF_DEFAULT} but the anchor contract was last "
        f"verified against {record['ref']}. Re-verify and refresh it: {REFRESH_CMD}"
    )


def test_recorded_fingerprint_matches_the_current_anchors():
    """The counts below describe what the anchors matched as they were written; editing one invalidates the record just
    as a pin bump does.
    """
    record = load_record()

    assert record["anchors_fingerprint"] == anchors_fingerprint(), (
        f"the patch anchors changed since the contract was recorded. Re-verify and refresh it: {REFRESH_CMD}"
    )


def test_record_covers_every_anchor_in_the_contract():
    """A newly added patch must be verified against upstream too, not just inherit the previous record's silence."""
    record = load_record()

    recorded = {name for spec in record["files"].values() for name in spec["anchors"]}
    assert recorded == {name for name, *_ in _ANCHOR_CONTRACT}, f"refresh the contract: {REFRESH_CMD}"


def test_every_recorded_anchor_matched_exactly_one_site():
    """One site is the whole contract: zero means the patch is inert, and more than one means the file drifted into a shape the patcher never handled."""
    record = load_record()

    hits = {name: n for spec in record["files"].values() for name, n in spec["anchors"].items()}
    assert all(n == 1 for n in hits.values()), hits


def test_record_covers_the_magpie_patch():
    """The louder of the two patchers aimed at benchmark_lib.sh."""
    record = load_record()

    assert record.get("magpie_patch", {}).get("applies") is True, (
        f"the contract predates the magpie patch check, or the splice no longer applies. {REFRESH_CMD}"
    )
    assert record["magpie_patch"]["path"] in upstream_candidates((("benchmarks", "benchmark_lib.sh"),))


# --- networked --------------------------------------------------------------


@pytest.mark.parametrize("rel_path", sorted(load_record()["files"]))
def test_pinned_upstream_still_matches_every_anchor(rel_path):
    """The layer that actually re-verifies."""
    record = load_record()
    text = fetch_pinned_file(rel_path, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    definitions = {name: anchor for name, _parts, _sentinel, anchor in _ANCHOR_CONTRACT}
    hits = {name: count_anchor_hits(text, definitions[name]) for name in record["files"][rel_path]["anchors"]}
    assert hits == record["files"][rel_path]["anchors"], (
        f"upstream {rel_path} no longer matches the recorded anchors. Re-anchor the affected "
        f"patches in _inferencex_patcher.py, then refresh: {REFRESH_CMD}"
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record["files"][rel_path]["sha256"], (
        f"the anchors still match, but {rel_path} is not the file the contract recorded. Refresh it: {REFRESH_CMD}"
    )


def test_recorded_probe_target_is_the_path_the_patcher_appends_to():
    """The probe has no anchor to rot, but it does need this file to exist: if upstream moves it the patch degrades to a warning and the eval runs unbounded again -- the exact failure the probe was written to stop."""
    record = load_record()

    assert record["probe_target"]["path"] in upstream_candidates(EVAL_PROBE_TARGETS), (
        f"the probe target moved to {PROBE_TARGET_PATH}. Re-verify and refresh: {REFRESH_CMD}"
    )


def test_pinned_upstream_still_carries_the_probe_target():
    """Networked counterpart: confirm the file is really there at the pin."""
    record = load_record()
    text = fetch_pinned_file(record["probe_target"]["path"], record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record["probe_target"]["sha256"], (
        f"{record['probe_target']['path']} changed upstream. The probe and the bounds are appended to it, so "
        f"re-read it before refreshing: {REFRESH_CMD}"
    )


def test_pinned_upstream_still_takes_the_magpie_splice():
    """Re-verify the splice against upstream, not just against the record."""
    record = load_record()
    text = fetch_pinned_file(record["magpie_patch"]["path"], record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    assert magpie_patch_applies(text), (
        f"{MAGPIE_LIB_PATH} at {record['ref'][:9]} no longer takes the run_lm_eval "
        f"--concurrent-requests splice; install.sh would die(). Re-anchor _magpie_patcher.py."
    )
