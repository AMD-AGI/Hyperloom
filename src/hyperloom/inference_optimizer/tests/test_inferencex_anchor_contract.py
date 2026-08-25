# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract between Hyperloom's patch anchors and the pinned InferenceX tree.

Hyperloom does not fork InferenceX; it rewrites a handful of upstream lines at
run time, each located by matching exact text. That makes a purely cosmetic
upstream edit indistinguishable from "nothing to patch", which is how the
eval-probe anchor once stayed broken across every checkout while the runs still
looked healthy.

The probe was since re-homed: it is appended to an upstream file instead of
anchored to a line, so it has no anchor to rot. It still has a dependency worth
pinning, though -- that file's path. Upstream moving it degrades the probe and
the request bounds back to a warning, and the eval runs unbounded again.

:func:`..._inferencex_patcher.verify_patch_anchors` catches that at launch, for
the user who is already affected. These tests catch it one step earlier -- when
*we* move ``INFERENCEX_REF`` or edit an anchor -- so the breakage never ships.

Two layers, because a unit test cannot count on network access:

* Hermetic, always runs: a checked-in record states the ref the anchors were
  last verified against, plus a fingerprint of the anchors themselves. Bumping
  the pin or editing an anchor without re-verifying fails immediately. The
  record is tiny; it deliberately does not vendor upstream's files.
* Networked, when reachable: fetch the pinned files and confirm every anchor
  still matches exactly one site. This is the layer that actually re-verifies,
  so the hermetic layer exists to make sure a human runs it.

Refresh the record with::

    python scripts/refresh_inferencex_anchor_contract.py
"""

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
    EVAL_PROBE_TARGET_PARTS,
    count_anchor_hits,
)

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "inferencex_anchor_contract.json"
REFRESH_CMD = "python scripts/refresh_inferencex_anchor_contract.py"
_FETCH_TIMEOUT_SEC = 30


PROBE_TARGET_PATH = "/".join(EVAL_PROBE_TARGET_PARTS)

# The second patcher aimed at the same file, and the one whose failure is
# louder. `_inferencex_patcher`'s anchors are covered above; `_magpie_patcher`
# splices a `--concurrent-requests` case into `run_lm_eval`'s argument parser,
# and when that splice does not apply install.sh `die`s outright (or, with
# MAGPIE_EVAL_FLAG_STRICT=0, every RUN_EVAL=true *synthetic* baseline aborts on
# "Unknown parameter"). It had no contract entry: the unit test that would have
# caught a break loads a fixture of the pinned benchmark_lib.sh that is not in
# the tree, so it self-skips and reports green.
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


def anchors_by_file() -> dict[str, list[tuple[str, str]]]:
    """Group the patch anchors by the upstream file they are matched against.

    Returns:
        Mapping of repo-relative path to its ``(anchor_name, anchor)`` pairs.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, rel_parts, _sentinel, anchor in _ANCHOR_CONTRACT:
        grouped.setdefault("/".join(rel_parts), []).append((name, anchor))
    return grouped


def anchors_fingerprint() -> str:
    """Fingerprint the anchor definitions themselves.

    Recorded counts describe what the anchors matched *as they were written at
    the time*. Editing an anchor therefore invalidates the record just as surely
    as bumping the pin does, and neither is detectable from the counts alone.

    Returns:
        A hex digest over every anchor's name, target path and pattern.
    """
    parts = [f"{name}\x1f{'/'.join(rel_parts)}\x1f{anchor}" for name, rel_parts, _sentinel, anchor in _ANCHOR_CONTRACT]
    parts.append(f"probe_target\x1f{PROBE_TARGET_PATH}")
    parts.extend(_magpie_pattern_parts())
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def github_slug(clone_url: str) -> str:
    """Turn a clone URL into the ``owner/repo`` form the API expects.

    Args:
        clone_url: An ``https://github.com/owner/repo.git`` style URL.

    Returns:
        The ``owner/repo`` slug.
    """
    return clone_url.rstrip("/").removesuffix(".git").split("github.com/", 1)[-1]


def fetch_pinned_file(rel_path: str, ref: str) -> str | None:
    """Fetch one upstream file at ``ref``, or ``None`` when unreachable.

    InferenceX is private, so this goes through ``gh`` rather than raw HTTP and
    every failure mode -- no ``gh``, no auth, no network, deleted path -- folds
    into ``None`` so the caller can degrade to the hermetic checks.

    Args:
        rel_path: Repo-relative path of the file to fetch.
        ref: The commit to fetch it at.

    Returns:
        The file's text, or ``None`` when it could not be retrieved.
    """
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
    return proc.stdout.decode("utf-8", errors="replace")


def build_record(ref: str) -> dict:
    """Verify every anchor against upstream at ``ref`` and return the record.

    Args:
        ref: The commit to verify against.

    Returns:
        The record to serialize into :data:`CONTRACT_PATH`.

    Raises:
        RuntimeError: When a file cannot be fetched, or an anchor does not match
            exactly one site -- recording a broken contract would defeat the
            point of having one.
    """
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
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
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
            "sha256": hashlib.sha256(probe_target.encode("utf-8")).hexdigest(),
        },
        "magpie_patch": {"path": MAGPIE_LIB_PATH, "applies": True},
    }


def load_record() -> dict:
    """Return the checked-in contract record.

    Returns:
        The parsed record.
    """
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


# --- hermetic ---------------------------------------------------------------


def test_recorded_ref_matches_the_pin_the_code_installs():
    """Bumping INFERENCEX_REF is exactly when an anchor silently rots, so the
    bump cannot be allowed to pass without someone re-verifying."""
    record = load_record()

    assert record["ref"] == _INFERENCEX_REF_DEFAULT, (
        f"INFERENCEX_REF moved to {_INFERENCEX_REF_DEFAULT} but the anchor contract was last "
        f"verified against {record['ref']}. Re-verify and refresh it: {REFRESH_CMD}"
    )


def test_recorded_fingerprint_matches_the_current_anchors():
    """The counts below describe what the anchors matched as they were written;
    editing one invalidates the record just as a pin bump does."""
    record = load_record()

    assert record["anchors_fingerprint"] == anchors_fingerprint(), (
        f"the patch anchors changed since the contract was recorded. Re-verify and refresh it: {REFRESH_CMD}"
    )


def test_record_covers_every_anchor_in_the_contract():
    """A newly added patch must be verified against upstream too, not just
    inherit the previous record's silence."""
    record = load_record()

    recorded = {name for spec in record["files"].values() for name in spec["anchors"]}
    assert recorded == {name for name, *_ in _ANCHOR_CONTRACT}, f"refresh the contract: {REFRESH_CMD}"


def test_every_recorded_anchor_matched_exactly_one_site():
    """One site is the whole contract: zero means the patch is inert, and more
    than one means the file drifted into a shape the patcher never handled."""
    record = load_record()

    hits = {name: n for spec in record["files"].values() for name, n in spec["anchors"].items()}
    assert all(n == 1 for n in hits.values()), hits


def test_record_covers_the_magpie_patch():
    """The louder of the two patchers aimed at benchmark_lib.sh.

    ``_inferencex_patcher``'s anchors degrade a probe; ``_magpie_patcher``'s
    splice failing makes install.sh ``die`` -- or, with strict mode off, aborts
    every RUN_EVAL=true synthetic baseline. It had no contract entry, and the
    unit test that would have caught it loads a fixture that is not in the tree
    and therefore self-skips.
    """
    record = load_record()

    assert record.get("magpie_patch", {}).get("applies") is True, (
        f"the contract predates the magpie patch check, or the splice no longer applies. {REFRESH_CMD}"
    )
    assert record["magpie_patch"]["path"] == MAGPIE_LIB_PATH


# --- networked --------------------------------------------------------------


@pytest.mark.parametrize("rel_path", sorted(anchors_by_file()))
def test_pinned_upstream_still_matches_every_anchor(rel_path):
    """The layer that actually re-verifies. Skipped without access to the
    private repo, which is why the hermetic checks above exist."""
    record = load_record()
    text = fetch_pinned_file(rel_path, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    hits = {name: count_anchor_hits(text, anchor) for name, anchor in anchors_by_file()[rel_path]}
    assert hits == record["files"][rel_path]["anchors"], (
        f"upstream {rel_path} no longer matches the recorded anchors. Re-anchor the affected "
        f"patches in _inferencex_patcher.py, then refresh: {REFRESH_CMD}"
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record["files"][rel_path]["sha256"], (
        f"the anchors still match, but {rel_path} is not the file the contract recorded. Refresh it: {REFRESH_CMD}"
    )


def test_recorded_probe_target_is_the_path_the_patcher_appends_to():
    """The probe has no anchor to rot, but it does need this file to exist: if
    upstream moves it the patch degrades to a warning and the eval runs unbounded
    again -- the exact failure the probe was written to stop."""
    record = load_record()

    assert record["probe_target"]["path"] == PROBE_TARGET_PATH, (
        f"the probe target moved to {PROBE_TARGET_PATH}. Re-verify and refresh: {REFRESH_CMD}"
    )


def test_pinned_upstream_still_carries_the_probe_target():
    """Networked counterpart: confirm the file is really there at the pin."""
    record = load_record()
    text = fetch_pinned_file(PROBE_TARGET_PATH, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record["probe_target"]["sha256"], (
        f"{PROBE_TARGET_PATH} changed upstream. The probe and the bounds are appended to it, so "
        f"re-read it before refreshing: {REFRESH_CMD}"
    )


def test_pinned_upstream_still_takes_the_magpie_splice():
    """Re-verify the splice against upstream, not just against the record.

    This is the check whose absence let a pin bump ship unverified on the
    patcher whose failure is a hard ``die``.
    """
    record = load_record()
    text = fetch_pinned_file(MAGPIE_LIB_PATH, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    assert magpie_patch_applies(text), (
        f"{MAGPIE_LIB_PATH} at {record['ref'][:9]} no longer takes the run_lm_eval "
        f"--concurrent-requests splice; install.sh would die(). Re-anchor _magpie_patcher.py."
    )
