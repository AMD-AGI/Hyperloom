"""Replay-bundle codec shared by Recipe writers and warm-start readers.

A throughput number is reusable only when it is bound to the exact state that
produced it: server argv, environment variables, workload, and every source
layer.  This module builds that atomic value and provides a GBrain-native
artifact transport.  Small diffs remain inline; larger diffs are written as
content-addressed GBrain pages and are hydrated through the same MCP credentials
as the recipe itself.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ...source_snapshot import MANIFEST_NAME

SCHEMA_VERSION = 1
REPLAY_BUNDLE_KEY = "replay_bundle"
ARTIFACT_PREFIX_ENV = "GBRAIN_REPLAY_ARTIFACT_PREFIX"
DEFAULT_ARTIFACT_PREFIX = "hyperloom-replay-artifacts"
INLINE_MAX_BYTES_ENV = "GBRAIN_REPLAY_INLINE_MAX_BYTES"
DEFAULT_INLINE_MAX_BYTES = 64 * 1024

_ARTIFACT_MARKER = '<!-- replay-artifact: patch file="bundle.diff" -->'
_FENCE_RE = re.compile(
    r'<!--\s*replay-artifact:\s*patch\s+file="bundle\.diff"\s*-->\s*'
    r"```diff\s*\n(?P<payload>.*?)\n```",
    re.DOTALL,
)


def canonical_server_argv(value: Any) -> list[str]:
    """Return the launch argv represented by ``value``.

    Legacy recipes store a shell-like string.  ``shlex.split`` is used exactly
    once to remove syntax quotes; the resulting tokens are the durable shape.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError:
        return []


def argv_to_env_string(argv: Any) -> str:
    """Encode argv for Magpie's unquoted ``$EXTRA_*_ARGS`` expansion.

    Quote characters stored inside an environment variable are data, not shell
    syntax.  Joining the already-parsed tokens without ``shlex.join`` therefore
    removes harmful outer quotes while retaining JSON's inner double quotes.
    Tokens containing whitespace cannot survive this transport and are refused.
    """
    tokens = canonical_server_argv(argv)
    if any(any(ch.isspace() for ch in token) for token in tokens):
        raise ValueError("server argv contains a whitespace-bearing token unsupported by Magpie env expansion")
    return " ".join(tokens)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bundle_digest(bundle: Mapping[str, Any]) -> str:
    """Hash replay semantics while ignoring artifact transport details."""
    digest_payload = copy.deepcopy(dict(bundle))
    digest_payload.pop("bundle_sha256", None)
    for artifact in digest_payload.get("source_artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact.pop("patch_content", None)
        artifact.pop("storage", None)
        artifact.pop("artifact_slug", None)
    return _sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _git_show(root: Path, revision: str, rel: str) -> tuple[bool, str]:
    if not revision:
        return False, ""
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{rel}"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        return False, ""
    try:
        return True, proc.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source artifact {rel!r} is binary and cannot be represented as a unified diff") from exc


def _diff_file(rel: str, old_exists: bool, old: str, new_exists: bool, new: str) -> str:
    if old_exists == new_exists and old == new:
        return ""
    lines = [f"diff --git a/{rel} b/{rel}\n"]
    if not old_exists:
        lines.append("new file mode 100644\n")
    elif not new_exists:
        lines.append("deleted file mode 100644\n")
    old_name = f"a/{rel}" if old_exists else "/dev/null"
    new_name = f"b/{rel}" if new_exists else "/dev/null"
    lines.extend(
        difflib.unified_diff(
            old.splitlines(keepends=True) if old_exists else [],
            new.splitlines(keepends=True) if new_exists else [],
            fromfile=old_name,
            tofile=new_name,
            lineterm="\n",
        )
    )
    rendered = "".join(lines)
    return rendered if rendered.endswith("\n") else rendered + "\n"


def _snapshot_groups(source_snapshots: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Squash ordered source snapshots into one final diff per repo/base SHA."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    repo_bases: dict[str, str] = {}
    for raw in source_snapshots:
        snapshot_dir = Path(str(raw.get("snapshot_dir") or ""))
        if not snapshot_dir.is_dir():
            return [], "source_snapshot_missing"
        try:
            manifest = json.loads((snapshot_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return [], "source_snapshot_manifest_invalid"
        root = Path(str(manifest.get("framework_root") or ""))
        base_sha = str(manifest.get("base_sha") or raw.get("base_sha") or "")
        if not root.is_dir() or not base_sha:
            return [], "source_snapshot_base_unavailable"
        repo = root.name.lower().replace("_meta", "")
        previous_base = repo_bases.setdefault(repo, base_sha)
        if previous_base != base_sha:
            return [], "source_snapshot_base_changed"
        key = (str(root), base_sha)
        group = groups.setdefault(
            key,
            {
                "repo": repo,
                "root": root,
                "base_sha": base_sha,
                "files": {},
                "provenance_layers": [],
            },
        )
        layer_id = str(raw.get("id") or manifest.get("provenance") or "")
        if layer_id:
            group["provenance_layers"].append(layer_id)
        for item in manifest.get("files") or []:
            if not isinstance(item, Mapping):
                continue
            rel = str(item.get("rel") or "").strip().lstrip("/")
            if not rel or ".." in Path(rel).parts:
                return [], "source_snapshot_path_unsafe"
            op = str(item.get("op") or "upsert")
            if op == "delete":
                group["files"][rel] = (False, "")
                continue
            try:
                content = (snapshot_dir / "files" / rel).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return [], "source_snapshot_binary"
            except OSError:
                return [], "source_snapshot_file_missing"
            group["files"][rel] = (True, content)

    artifacts: list[dict[str, Any]] = []
    for group in groups.values():
        patch_parts: list[str] = []
        changed_files: list[str] = []
        for rel, (new_exists, new) in sorted(group["files"].items()):
            try:
                old_exists, old = _git_show(group["root"], group["base_sha"], rel)
            except (OSError, subprocess.SubprocessError, ValueError):
                return [], "source_snapshot_base_read_failed"
            patch = _diff_file(rel, old_exists, old, new_exists, new)
            if patch:
                patch_parts.append(patch)
                changed_files.append(rel)
        patch_text = "".join(patch_parts)
        if not patch_text:
            continue
        artifacts.append(
            {
                "repo": group["repo"],
                "base_sha": group["base_sha"],
                "format": "unified-diff",
                "storage": "inline",
                "sha256": _sha256(patch_text),
                "bytes": len(patch_text.encode("utf-8")),
                "changed_files": changed_files,
                "provenance_layers": list(group["provenance_layers"]),
                "patch_content": patch_text,
            }
        )
    if source_snapshots and not artifacts:
        return [], "source_snapshot_empty"
    return artifacts, ""


def build_replay_bundle(
    *,
    env_spec: Mapping[str, Any],
    producer_session_id: str,
    baseline_throughput: float,
    optimized_throughput: float,
    workload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the atomic replay value for the measured champion."""
    config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
    raw_args = config.get("extra_server_args") or ""
    argv = canonical_server_argv(raw_args)
    envs = config.get("extra_envs") if isinstance(config.get("extra_envs"), Mapping) else {}
    snapshots = [
        item for item in (env_spec.get("source_snapshots") or []) if isinstance(item, Mapping)
    ]
    artifacts, reason = _snapshot_groups(snapshots)
    replayable = not reason and bool(argv or envs or artifacts)
    baseline = float(baseline_throughput or 0.0)
    optimized = float(optimized_throughput or 0.0)
    gain_pct = ((optimized - baseline) / baseline * 100.0) if baseline > 0.0 else 0.0
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "replayable": replayable,
        "reason": reason or ("" if replayable else "empty_replay_bundle"),
        "producer_session_id": str(producer_session_id or ""),
        "config": {
            "argv": argv,
            "extra_envs": {str(k): str(v) for k, v in envs.items()},
            "server_launch_flags": str(config.get("server_launch_flags") or ""),
        },
        "source_artifacts": artifacts,
        "workload": dict(workload or {}),
        "measurement": {
            "baseline_throughput": baseline,
            "optimized_throughput": optimized,
            "gain_pct": gain_pct,
            "measured_with_complete_bundle": replayable,
        },
    }
    bundle["bundle_sha256"] = _bundle_digest(bundle)
    return bundle


def _artifact_slug(sha256: str) -> str:
    prefix = os.environ.get(ARTIFACT_PREFIX_ENV, "").strip().strip("/") or DEFAULT_ARTIFACT_PREFIX
    return f"{prefix}/{sha256[:2]}/{sha256}"


def _artifact_page(artifact: Mapping[str, Any], canonical_id: str) -> str:
    patch = str(artifact.get("patch_content") or "")
    attrs = {
        "canonical_id": canonical_id,
        "sha256": str(artifact.get("sha256") or ""),
        "bytes": int(artifact.get("bytes") or len(patch.encode("utf-8"))),
        "format": "unified-diff",
    }
    return (
        "---\n"
        'type: "replay_artifact"\n'
        'title: "Hyperloom replay source artifact"\n'
        'tags: ["kind:replay_artifact"]\n'
        f"attrs: {json.dumps(attrs, ensure_ascii=False, sort_keys=True)}\n"
        "---\n\n"
        f"{_ARTIFACT_MARKER}\n```diff\n{patch.rstrip()}\n```\n"
    )


def externalize_large_artifacts(
    mcp: Any,
    *,
    canonical_id: str,
    bundle: Mapping[str, Any],
    inline_max_bytes: int | None = None,
) -> dict[str, Any]:
    """Write oversized inline artifacts as immutable GBrain pages."""
    out = copy.deepcopy(dict(bundle))
    try:
        threshold = int(
            inline_max_bytes
            if inline_max_bytes is not None
            else os.environ.get(INLINE_MAX_BYTES_ENV, "") or DEFAULT_INLINE_MAX_BYTES
        )
    except (TypeError, ValueError):
        threshold = DEFAULT_INLINE_MAX_BYTES
    for artifact in out.get("source_artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("storage") == "gbrain_page":
            # Readers hydrate the bytes for execution; writers keep only the
            # immutable reference so metadata-only updates do not inline it.
            artifact.pop("patch_content", None)
            continue
        if artifact.get("storage") != "inline":
            continue
        patch = str(artifact.get("patch_content") or "")
        if len(patch.encode("utf-8")) <= threshold:
            continue
        slug = _artifact_slug(str(artifact.get("sha256") or _sha256(patch)))
        try:
            mcp.call(
                "put_page",
                {"slug": slug, "content": _artifact_page(artifact, canonical_id)},
            )
        except Exception:
            # Preserve the Recipe write as reference material, but never claim
            # the champion can be replayed when its source bytes were not
            # durably published.
            out["replayable"] = False
            out["reason"] = "artifact_publish_failed"
            artifact.pop("patch_content", None)
            continue
        artifact["storage"] = "gbrain_page"
        artifact["artifact_slug"] = slug
        artifact.pop("patch_content", None)
    return out


def _extract_artifact_page(page: Any) -> str:
    if not isinstance(page, Mapping):
        return ""
    body = str(page.get("compiled_truth") or page.get("body") or page.get("content") or "")
    match = _FENCE_RE.search(body)
    return match.group("payload") + "\n" if match else ""


def hydrate_replay_bundle(mcp: Any, bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Hydrate artifact-page references and fail closed on integrity errors."""
    out = copy.deepcopy(dict(bundle))
    if int(out.get("schema_version") or 0) != SCHEMA_VERSION:
        out["replayable"] = False
        out["reason"] = "unsupported_replay_bundle_schema"
        return out
    for artifact in out.get("source_artifacts") or []:
        if not isinstance(artifact, dict):
            out["replayable"] = False
            out["reason"] = "artifact_manifest_invalid"
            return out
        if artifact.get("storage") == "gbrain_page":
            slug = str(artifact.get("artifact_slug") or "")
            patch = _extract_artifact_page(mcp.call("get_page", {"slug": slug})) if slug else ""
            if not patch:
                out["replayable"] = False
                out["reason"] = "artifact_missing"
                return out
            artifact["patch_content"] = patch
        patch = str(artifact.get("patch_content") or "")
        if not patch or _sha256(patch) != str(artifact.get("sha256") or ""):
            out["replayable"] = False
            out["reason"] = "artifact_sha_mismatch"
            return out
    expected_bundle_sha = str(out.get("bundle_sha256") or "")
    if expected_bundle_sha and _bundle_digest(out) != expected_bundle_sha:
        out["replayable"] = False
        out["reason"] = "bundle_sha_mismatch"
    return out


def replay_patches(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project a hydrated bundle into the baseline executor's patch contract."""
    if not bundle.get("replayable"):
        return []
    out: list[dict[str, Any]] = []
    for artifact in bundle.get("source_artifacts") or []:
        if not isinstance(artifact, Mapping):
            continue
        patch = str(artifact.get("patch_content") or "")
        if not patch:
            continue
        out.append(
            {
                "patch_file": f"{artifact.get('repo') or 'framework'}-{str(artifact.get('sha256') or '')[:12]}.diff",
                "patch_content": patch,
                "target_repo": str(artifact.get("repo") or ""),
                "base_sha": str(artifact.get("base_sha") or ""),
                "sha256": str(artifact.get("sha256") or ""),
            }
        )
    return out


__all__ = [
    "REPLAY_BUNDLE_KEY",
    "SCHEMA_VERSION",
    "argv_to_env_string",
    "build_replay_bundle",
    "canonical_server_argv",
    "externalize_large_artifacts",
    "hydrate_replay_bundle",
    "replay_patches",
]
