# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading a warm-start prior into the values a replay is built from.

A prior arrives as nested records: a recipe with its history, a kernel section
naming champion artifacts, and a donor's recommendation. Everything here turns
one of those records into a value PRELUDE can act on -- a launch overlay, a
kernel apply plan, a gain anchor, the checkouts an overlay was measured on --
and raises rather than guesses when two records disagree. Nothing here reads
or writes session state.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from ..knowledge.remote_recipe.sanitize import HOST_ORIGIN_KEY


# The donor's identity, carried flattened on the outcome and re-nested for the
# event. Listed once so the two directions cannot drift apart.
WARM_REPLAY_DONOR_FIELDS = (
    "donor_canonical_id",
    "donor_model",
    "donor_session_id",
    "donor_family_tags",
    "donor_gain_pct",
    "donor_breakdown_link",
)


def as_float(value: Any) -> float:
    """Read a recorded number, answering 0.0 for a record that carries something else."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def merge_named_current_recipe_configs(
    owners: list[tuple[str, Mapping[str, Any]]],
) -> tuple[str, dict[str, str]]:
    """Merge named config snapshots with exact duplicate conflict checks."""
    return _merge_named_server_args(owners), _merge_named_extra_envs(owners)


def _merge_named_server_args(owners: list[tuple[str, Mapping[str, Any]]]) -> str:
    """Concatenate the owners' server args, refusing two spellings of one flag."""
    pairs: dict[str, tuple[str, ...]] = {}
    order: list[str] = []
    for owner, config in owners:
        for key, rendered in _server_arg_flags(owner, str(config.get("extra_server_args") or "").strip()):
            prior = pairs.get(key)
            if prior is not None and prior != rendered:
                raise ValueError(f"current Recipe config conflict for {key}: {prior!r} != {rendered!r}")
            if prior is None:
                pairs[key] = rendered
                order.append(key)
    return " ".join(token for key in order for token in pairs[key])


def _server_arg_flags(owner: str, raw: str) -> list[tuple[str, tuple[str, ...]]]:
    """Split one owner's server args into flags and the values each carries.

    A bare word is a positional with no flag to compare against, so it is keyed
    by its own identity and never conflicts with anything.
    """
    from hyperloom.inference_optimizer.grid_server_args import (
        tokenize_server_args_preserving_json,
    )

    if not raw:
        return []
    parsed = tokenize_server_args_preserving_json(raw)
    if parsed is None:
        raise ValueError(f"{owner} extra_server_args cannot be tokenized safely")
    _normalized, tokens = parsed
    flags: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--"):
            flags.append((f"__positional_{len(flags)}_{owner}", (token,)))
            continue
        key, has_equals, inline = token.partition("=")
        values: list[str] = [inline] if has_equals else []
        while not has_equals and index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        flags.append((key, (key, *values)))
    return flags


def _merge_named_extra_envs(owners: list[tuple[str, Mapping[str, Any]]]) -> dict[str, str]:
    """Merge the owners' env bundles, refusing two values for one variable."""
    envs: dict[str, str] = {}
    for owner, config in owners:
        raw_envs = config.get("extra_envs")
        if not isinstance(raw_envs, Mapping):
            continue
        for raw_key, raw_value in raw_envs.items():
            key = str(raw_key)
            value = str(raw_value)
            if key in envs and envs[key] != value:
                raise ValueError(f"current Recipe env conflict for {key}: {envs[key]!r} != {value!r} ({owner})")
            envs[key] = value
    return envs


#: Row keys naming a file path, which a plan entry resolves locally instead of carrying.
_KERNEL_ROW_PATH_KEYS = frozenset({"source_file", "source_files", "target_file", "target_files", "target_path"})
#: Row keys a plan entry's ``meta`` leaves out: the paths, plus the refs it resolves itself.
_KERNEL_META_EXCLUDED = _KERNEL_ROW_PATH_KEYS | {"patch", "tuned_file", "files"}


def warm_kernel_plan_entry(kb: Any, column: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    """One champion kernel row as a local apply item, or None when its content did not download."""
    entry: dict[str, Any] = {
        "column": column,
        "meta": {key: value for key, value in row.items() if key not in _KERNEL_META_EXCLUDED},
        # Preserve the exact Recipe row so CLOSE can carry the validated kernel content and its refs forward
        # on the same inference page.
        "recipe_row": {key: value for key, value in row.items() if key not in _KERNEL_ROW_PATH_KEYS},
    }
    # The checkout this item was applied into.
    host_origin = row.get(HOST_ORIGIN_KEY)
    if isinstance(host_origin, Mapping):
        entry["apply_root"] = str(host_origin.get("apply_root") or "").strip()
    if patch_ref := str(row.get("patch") or "").strip():
        patch_local = kb.prior_file(patch_ref)
        if patch_local is not None:
            entry["patch_path"] = str(patch_local)
    tuned_ref = str(row.get("tuned_file") or "").strip()
    if column == "gemm" and tuned_ref:
        tuned_local = kb.prior_file(tuned_ref)
        if tuned_local is not None:
            entry["source_paths"] = [str(tuned_local)]
    if entry.get("patch_path") or (column == "gemm" and entry.get("source_paths")):
        return entry
    return None


def add_kernel_envs(envs: dict[str, str], source: Any) -> None:
    """Fold one declared env bundle into ``envs``, refusing two values for one variable."""
    if not isinstance(source, dict):
        return
    for raw_key, raw_value in source.items():
        key = str(raw_key).strip()
        if not key:
            continue
        value = str(raw_value)
        if key in envs and envs[key] != value:
            raise ValueError(f"current Recipe kernel env conflict for {key}: {envs[key]!r} != {value!r}")
        envs[key] = value


def point_tuner_envs_at_local_files(envs: dict[str, str], results: Any, local: list[str]) -> None:
    """Point each accepted tuner's variable at the copy of its table this run downloaded."""
    by_name = {Path(path).name: path for path in local}
    for tuner in (results.get("kept") or []) if isinstance(results, dict) else []:
        if not isinstance(tuner, dict):
            continue
        add_kernel_envs(envs, tuner.get("envs"))
        env_var = str(tuner.get("env_var") or "").strip()
        if not env_var:
            continue
        # Each accepted tuner owns its own table, so match by the recorded value's filename.
        recorded = str(tuner.get("env_value") or envs.get(env_var) or "").strip()
        local_copy = by_name.get(Path(recorded).name) if recorded else None
        if local_copy is None and len(local) == 1:
            local_copy = local[0]
        if local_copy:
            envs[env_var] = local_copy


def point_envs_at_local_files(envs: dict[str, str], local: list[str]) -> None:
    """Re-point any remaining variable that still names a file this run downloaded."""
    by_name = {Path(path).name: path for path in local}
    for name, value in list(envs.items()):
        if not value or "/" not in value:
            continue
        replacement = by_name.get(Path(value).name)
        if replacement and replacement != value:
            envs[name] = replacement


def warm_recipe_history_rows(warm: Any, key: str) -> list[dict[str, Any]]:
    """The ``what_worked``/``what_failed`` rows a warm-start prior recorded.

    The history may sit on the recipe, under its ``attrs``, or under the
    ``exact_history`` an identity match carries; whichever is present wins.
    """
    recipe = warm.get("recipe") if isinstance(warm, Mapping) else None
    attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, Mapping) else {}
    exact_history = attrs.get("exact_history")
    rows = (exact_history if isinstance(exact_history, Mapping) else attrs).get(key)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def warm_recipe_rejection(row: Mapping[str, Any], fingerprint: str, tier: str) -> dict[str, Any]:
    """One explore-ledger rejection standing for a config the prior already lost on."""
    return {
        "name": str(row.get("name") or "")[:120],
        "fingerprint": fingerprint,
        "reason": "warm_recipe_what_failed",
        "extra_server_args": str(row.get("extra_server_args") or "").strip(),
        "extra_envs": dict(row["extra_envs"]) if isinstance(row.get("extra_envs"), dict) else {},
        "source": "warm_start_recipe",
        "source_tier": tier,
        # Preserved for forensics; not used by the dedup gate.
        "gain_pct": row.get("gain_pct"),
        "error_class": row.get("error_class") or row.get("reason"),
    }


def recorded_apply_roots(provenance: Any) -> dict[str, str]:
    """Map each overlay ref to the checkout the record says it was applied into."""
    roots: dict[str, str] = {}
    for row in provenance or []:
        if not isinstance(row, Mapping):
            continue
        origin = row.get("host_origin")
        if not isinstance(origin, Mapping):
            continue
        for ref, root in (origin.get("apply_roots") or {}).items():
            name = str(ref or "").strip()
            value = str(root or "").strip()
            if name and value:
                roots.setdefault(name, value)
    return roots


def overlay_provenance_summary(sdk_replay: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise how the overlays this replay is about to apply were captured."""

    def _count(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    rows = [row for row in (sdk_replay.get("provenance") or []) if isinstance(row, Mapping)]
    overlays = len(sdk_replay.get("patches") or [])
    if not overlays and not rows:
        return {}
    return {
        "overlays": overlays,
        "realized": sum(1 for row in rows if row.get("realized")),
        "incomplete": sum(1 for row in rows if row.get("complete") is False),
        "artifacts_outside_root": sum(_count(row.get("artifacts_outside_root")) for row in rows),
    }


#: The two ways an overlay's recorded checkout can fail this host: never named, or gone.
ROOT_MISSING = "framework_apply_root_missing"
ROOT_ABSENT = "framework_apply_root_absent"

#: The donor tier a config carries once the confidence gate has dropped it.
SUPPRESSED_LOW_CONFIDENCE = "suppressed_low_confidence"


class WarmReplayConfig(NamedTuple):
    """The launch overlay a warm prior asks a replay to carry, and who it came from."""

    args: str
    envs: dict[str, Any]
    confidence: float
    source: str
    donor_tier: str
    donor_expected_gain: float
    suppressed: bool


class WarmKernelOverlay(NamedTuple):
    """The kernel half a combined replay carries, and the launch overlay it brings."""

    pending: list[dict[str, Any]]
    applied: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    args: str
    envs: dict[str, str]


class WarmLaunch(NamedTuple):
    """The server args and envs a replay launches with."""

    args: str
    envs: dict[str, Any]


class WarmPatchPlan(NamedTuple):
    """The code overlays a replay applies, and whether every one of them has to land."""

    patches: list[dict[str, Any]]
    required_timeline: bool


class WarmReplayRequest(NamedTuple):
    """What a warm prior asks a replay to carry, and the prior it was derived from."""

    config: WarmReplayConfig
    overlay: WarmKernelOverlay
    patches: list[dict[str, Any]]
    required_patch_timeline: bool
    tier: str
    confidence: float
    expected_gain: float
    donor: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        """How an outcome names the prior this replay came from."""
        return {"warm_recipe_tier": self.tier, "warm_recipe_conf": self.confidence, **self.donor}


class RecordedRoots(NamedTuple):
    """The checkouts a set of overlays was measured on, and those this host does not have."""

    recorded: list[str]
    absent: list[str]

    @property
    def complete(self) -> bool:
        """Every overlay named the checkout it was applied into."""
        return all(self.recorded)

    @property
    def named(self) -> list[str]:
        """The checkouts that were named at all."""
        return [root for root in self.recorded if root]


def recorded_framework_roots(patches: Sequence[Any]) -> RecordedRoots:
    """Which checkouts a set of code overlays says it was measured on."""
    recorded = [str((entry or {}).get("framework_root") or "").strip() for entry in patches]
    return RecordedRoots(recorded, [root for root in dict.fromkeys(recorded) if root and not Path(root).is_dir()])


def warm_replay_donor_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    """The donor identity as an outcome carries it, flattened under ``donor_``."""
    return {field: source[field] for field in WARM_REPLAY_DONOR_FIELDS if source.get(field) not in (None, "", [])}


def warm_replay_launch_config(
    recipe: Mapping[str, Any],
    sdk_replay: Mapping[str, Any],
    donor_replay: Mapping[str, Any],
    confidence: float,
) -> WarmReplayConfig:
    """Where this replay's launch overlay comes from: the current Recipe, a donor, or the prior's own best."""
    if sdk_replay:
        return WarmReplayConfig(
            args=str(sdk_replay.get("extra_server_args") or "").strip(),
            envs=dict(sdk_replay.get("extra_envs") or {}),
            confidence=confidence,
            source=str(recipe.get("canonical_id") or ""),
            donor_tier="self",
            donor_expected_gain=as_float(recipe.get("validated_gain_pct")),
            suppressed=False,
        )
    return _donor_launch_config(donor_replay, confidence) or _prior_best_launch_config(recipe, confidence)


def _donor_launch_config(donor_replay: Mapping[str, Any], confidence: float) -> WarmReplayConfig | None:
    """The config a donor recommended, or None when it recommended none."""
    args = str(donor_replay.get("extra_server_args") or "").strip()
    envs = donor_replay.get("extra_envs") if isinstance(donor_replay.get("extra_envs"), dict) else {}
    if not args and not envs:
        return None
    recorded_confidence = donor_replay.get("config_confidence")
    return WarmReplayConfig(
        args=args,
        envs=dict(envs),
        confidence=float(confidence if recorded_confidence is None else recorded_confidence),
        source=str(donor_replay.get("config_source") or ""),
        donor_tier=str(donor_replay.get("config_tier") or "self"),
        donor_expected_gain=as_float(donor_replay.get("expected_gain_pct")),
        suppressed=False,
    )


def _prior_best_launch_config(recipe: Mapping[str, Any], confidence: float) -> WarmReplayConfig:
    """The prior's own best config; ``best_config`` may be top-level or nested under attrs."""
    attrs = recipe.get("attrs") or recipe
    best = attrs.get("best_config") if isinstance(attrs.get("best_config"), dict) else {}
    envs = best.get("extra_envs") if isinstance(best.get("extra_envs"), dict) else {}
    return WarmReplayConfig(
        args=str(best.get("extra_server_args") or "").strip(),
        envs=dict(envs),
        confidence=confidence,
        source=str(recipe.get("canonical_id") or ""),
        donor_tier="self",
        donor_expected_gain=0.0,
        suppressed=False,
    )


def warm_replay_patch_plan(
    sdk_replay: Mapping[str, Any],
    donor_replay: Mapping[str, Any],
    recipe_attrs: Mapping[str, Any],
    config: WarmReplayConfig,
) -> WarmPatchPlan:
    """The code overlays this replay applies; a current Recipe derives fail-closed mode from its SDK timeline."""
    if config.suppressed:
        return WarmPatchPlan([], False)
    if sdk_replay:
        return WarmPatchPlan(list(sdk_replay.get("patches") or []), bool(sdk_replay.get("timeline")))
    return WarmPatchPlan(
        list(donor_replay.get("patches") or []),
        bool(recipe_attrs.get("required_patch_timeline")),
    )


def warm_expected_gain(recipe_attrs: Mapping[str, Any], donor_expected_gain: float) -> float:
    """The historical gain a replay is anchored to: the donor's, else the best session's, else the flat one."""
    if donor_expected_gain > 0:
        return donor_expected_gain
    sessions = recipe_attrs.get("sessions")
    rows = sessions if isinstance(sessions, list) else []
    gains = [as_float(row.get("gain_pct")) for row in rows if isinstance(row, dict)]
    anchor = max(gains) if gains else donor_expected_gain
    if anchor > 0:
        return anchor
    # Last-chance fallback for offline-ingested seed rows.
    flat = as_float(recipe_attrs.get("gain_pct"))
    return flat if flat > 0 else anchor


class StagedKernelPlan(NamedTuple):
    """The plan items that can contribute on this host, and the launch overlay they merge to."""

    item_envs: dict[int, dict[str, str]]
    item_targets: dict[int, list[str]]
    extra_server_args: str
    extra_envs: dict[str, str]


class WarmKernelApply(NamedTuple):
    """What landing a kernel plan left behind."""

    applied: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    deferred: int
    errors: int


def refuse_warm_kernel_entry(entry: dict[str, Any], exc: BaseException) -> bool:
    """Mark one kernel item as the error that stopped the sequence; always False."""
    entry["decision"] = "ERROR"
    entry["apply_result"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
    return False
