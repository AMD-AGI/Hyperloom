# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Join hot-kernel rows to their resolved source location.

``hot_kernels_top15`` carries ``source_file`` but neither the line nor the
enclosing function. Both live in ``kernel_source_resolution.json``, the
versioned audit artifact written next to ``analysis.md``.

The pair matters more than it looks. A prompt that says a hot kernel sits at
``tuned_gemm.py:395 torch_gemm`` lets a patch answer name that location; one
that says only ``tuned_gemm.py`` lets it name a file. Since predicted patches
become free-form specialist mandates, the difference shows up as the
specialist's hit rate.

This is the one part of the request that reads from disk. The artifact sits in
the analysis run directory, which ``last_trace_analyze["analysis_md_path"]``
already points into.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.common.kernel_source_contract import (
    METHOD_REJECTED,
    METHOD_UNRESOLVED,
    SOURCE_RESOLUTION_FILENAME,
    SOURCE_RESOLUTION_SCHEMA_VERSION,
)

log = logging.getLogger(__name__)

#: Methods whose entries name no usable location. ``unresolved`` never found
#: one; ``rejected_non_path_sentinel`` found a placeholder and zeroed it. Both
#: still emit an entry, so forwarding their ``source_file`` would put a path the
#: producer already disbelieved in front of the model.
_UNUSABLE_METHODS = frozenset({METHOD_UNRESOLVED, METHOD_REJECTED})

_EXPECTED_MAJOR = SOURCE_RESOLUTION_SCHEMA_VERSION.split(".")[0]

#: Keys copied out of a resolution entry.
_SITE_KEYS = ("source_file", "source_line", "source_function")


def _site(entry: Any) -> dict[str, Any] | None:
    """Return the location fields of a usable entry, else ``None``."""
    if not isinstance(entry, dict):
        return None
    if str(entry.get("method") or METHOD_UNRESOLVED) in _UNUSABLE_METHODS:
        return None
    source_file = str(entry.get("source_file") or "").strip()
    if not source_file:
        return None
    line = entry.get("source_line")
    return {
        "source_file": source_file,
        "source_line": line if isinstance(line, int) else None,
        "source_function": str(entry.get("source_function") or "").strip() or None,
    }


def _artifact_path(analysis_md_path: str) -> Path | None:
    """Locate the resolution artifact relative to ``analysis.md``.

    The producer writes it into the kernel-agent *run* directory, while the
    upstream TraceLens report lands one level deeper in ``tracelens/``:

    .. code-block:: text

        <run_dir>/kernel_source_resolution.json
        <run_dir>/tracelens/analysis.md

    Both are checked because the bypass route puts the two side by side and the
    deterministic route does not. Looking only beside ``analysis.md`` silently
    finds nothing on the deterministic route, which is the common one.

    Args:
        analysis_md_path (str): Path to the report.

    Returns:
        Path | None: The first location that exists, else ``None``.
    """
    report = Path(analysis_md_path)
    for directory in (report.parent, report.parent.parent):
        candidate = directory / SOURCE_RESOLUTION_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_source_sites(analysis_md_path: Any) -> dict[str, dict[str, Any]]:
    """Read the resolution artifact written for this analysis run.

    The result is keyed by both ``kernel_id`` and kernel ``name`` so a caller
    can look up whichever it has. Ids are written last, so an id always wins
    over another row that happens to be *named* like that id.

    Args:
        analysis_md_path (Any): ``last_trace_analyze["analysis_md_path"]``. The
            artifact is looked up beside it and in the run directory above it.

    Returns:
        dict[str, dict[str, Any]]: Key to ``{source_file, source_line,
            source_function}``. Empty when the artifact is missing, unreadable,
            malformed, or written against a different schema major — source
            frames enrich the request, they never gate it.
    """
    raw = str(analysis_md_path or "").strip()
    if not raw:
        return {}
    path = _artifact_path(raw)
    if path is None:
        log.debug("predictor_source_sites: no artifact near %s", raw)
        return {}

    doc = read_json(path, default=None, require_dict=True)
    if not isinstance(doc, dict):
        log.debug("predictor_source_sites: %s is not a JSON object", path)
        return {}

    version = str(doc.get("schema_version") or "")
    if version.split(".")[0] != _EXPECTED_MAJOR:
        # The producer bumps the major on a field removal or meaning change, so
        # reading on is guessing at fields that may no longer mean what we want.
        log.warning(
            "predictor_source_sites: %s has schema_version %r, expected major %s; ignoring",
            path,
            version,
            _EXPECTED_MAJOR,
        )
        return {}

    entries = doc.get("entries")
    if not isinstance(entries, list):
        log.debug("predictor_source_sites: %s has no entries list", path)
        return {}

    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    skipped = 0
    for entry in entries:
        site = _site(entry)
        if site is None:
            skipped += 1
            continue
        name = str(entry.get("name") or "").strip()
        if name:
            by_name.setdefault(name, site)
        kernel_id = str(entry.get("kernel_id") or "").strip()
        if kernel_id:
            by_id[kernel_id] = site

    sites = {**by_name, **by_id}
    log.debug("predictor_source_sites: %d keys, %d entries skipped, from %s", len(sites), skipped, path)
    return sites


def attach_sites(
    kernels: list[dict[str, Any]] | None,
    sites: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return ``kernels`` with ``source_line`` / ``source_function`` filled in.

    Matching is by ``kernel_id`` first and ``name`` second. Rows are copied, not
    mutated: the input is ``last_trace_analyze["hot_kernels_top15"]``, which is
    live session state other readers share.

    An unmatched row keeps its own ``source_file`` and gets explicit ``None``
    for the frame. "File but no line" is the common case, and the renderer on
    the far side degrades to naming just the file.

    Args:
        kernels (list[dict[str, Any]] | None): Hot-kernel rows.
        sites (dict[str, dict[str, Any]] | None): Index from
            :func:`load_source_sites`.

    Returns:
        list[dict[str, Any]]: New rows, same order.
    """
    lookup = sites or {}
    out: list[dict[str, Any]] = []
    for kernel in kernels or []:
        row = dict(kernel)
        site = None
        for key in (row.get("kernel_id"), row.get("name")):
            candidate = str(key or "").strip()
            if candidate and candidate in lookup:
                site = lookup[candidate]
                break
        if site is not None:
            row.update({k: site[k] for k in _SITE_KEYS})
        else:
            for key in ("source_line", "source_function"):
                row.setdefault(key, None)
        out.append(row)
    return out
