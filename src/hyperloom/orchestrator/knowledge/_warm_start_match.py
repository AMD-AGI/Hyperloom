# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which stored Recipe KB rows may serve a workload: compatibility, degradation tiers and the donor gate."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, NamedTuple

from packaging.version import InvalidVersion, Version

from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES
from hyperloom.orchestrator.knowledge.recipe_kb import cid_to_path_components


log = logging.getLogger(__name__)


def _text(value: Any) -> str:
    """*value* as trimmed text, with ``None`` and the other empty values collapsing to ``""``."""
    return str(value or "").strip()


#: SKU -> ISA family. Derived from the board identity table, so a new board becomes a
#: fallback donor for its own ISA in the commit that adds it.
_GPU_ISA_BY_SKU = {sku: arch for sku, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items()}
#: ``ep`` and the partition mode suffix at any node count, so they appear both inside a cluster suffix and on their
#: own. Kept as a named fragment rather than repeated, so the two forms cannot drift.
_SHAPE_SUFFIX = r"(?:_ep[1-9]\d*)?(?:_(?:dpx|qpx|cpx))?"
#: A single-node shape suffix has to parse here too, or ``_hardware_fallback_values`` stops offering the same-ISA
#: SKUs for exactly the rows these suffixes were added for. The alternation keeps ``_ws`` mandatory for the cluster
#: form, so an unrecognised suffix still fails to parse instead of being mistaken for a backend name.
_TOPOLOGY_SUFFIX_RE = re.compile(
    r"(?:_ws[1-9]\d*(?:_pd[1-9]\d*p[1-9]\d*d)?"
    rf"(?:_tp[1-9]\d*)?{_SHAPE_SUFFIX}(?:_[a-z0-9-]+)?"
    rf"|{_SHAPE_SUFFIX})$"
)


class _SearchIdentity(NamedTuple):
    """The seven-tuple dimensions, model aside, that a warm-start search is issued under."""

    hardware: str
    framework: str
    model_type: str
    arch_slug: str
    framework_version: str
    precision: str


class _TierQuery(NamedTuple):
    """One degradation tier: the labels it searches on and the dimensions it still pins."""

    tier: str
    confidence: float
    labels: dict[str, str]
    hardware_in: list[str] | None
    relax_framework_version: bool


class _DonorTarget(NamedTuple):
    """The workload a borrowed replay config has to be safe for."""

    arch_slug: str
    model_type: str
    conc: Any = None
    isl: Any = None
    osl: Any = None

    def for_identity(self, identity: "_SearchIdentity") -> "_DonorTarget":
        """The same request shape, re-pinned to the architecture and model type *identity* names."""
        return self._replace(arch_slug=identity.arch_slug, model_type=identity.model_type)


class _ConfigDonor(NamedTuple):
    """The row a replay config is borrowed from, and how far the search had to degrade to reach it."""

    row: Mapping[str, Any] | None
    tier: str
    confidence: float


def _parse_hardware_topology(hardware: str) -> tuple[str, str, str] | None:
    """Return ``(SKU, ISA family, topology suffix)`` for a known GPU."""
    value = _text(hardware).lower()
    for sku, family in _GPU_ISA_BY_SKU.items():
        if value == sku:
            return sku, family, ""
        if value.startswith(f"{sku}_"):
            suffix = value[len(sku) :]
            if _TOPOLOGY_SUFFIX_RE.fullmatch(suffix):
                return sku, family, suffix
    return None


def _hardware_fallback_values(hardware: str) -> list[str]:
    """List same-ISA SKUs carrying the target's exact topology suffix."""
    parsed = _parse_hardware_topology(hardware)
    if parsed is None:
        value = _text(hardware).lower()
        return [value] if value else []
    _sku, family, suffix = parsed
    return [f"{sku}{suffix}" for sku, sku_family in _GPU_ISA_BY_SKU.items() if sku_family == family]


def _hardware_is_compatible(target: str, candidate: str) -> bool:
    """Accept exact hardware or a known same-ISA SKU with exact topology."""
    target_value = _text(target).lower()
    candidate_value = _text(candidate).lower()
    if candidate_value == target_value:
        return bool(target_value)
    target_parsed = _parse_hardware_topology(target_value)
    candidate_parsed = _parse_hardware_topology(candidate_value)
    return bool(
        target_parsed
        and candidate_parsed
        and target_parsed[1] == candidate_parsed[1]
        and target_parsed[2] == candidate_parsed[2]
    )


def _framework_semver(version: str) -> Version | None:
    """Parse a framework's PEP 440 version."""
    value = _text(version)
    if not value:
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def _framework_version_is_compatible(target: str, candidate: str) -> bool:
    """Accept exact or nearest non-newer PEP 440 version in one release line."""
    target_value = _text(target).lower()
    candidate_value = _text(candidate).lower()
    if candidate_value == target_value:
        return bool(target_value)
    target_key = _framework_semver(target_value)
    candidate_key = _framework_semver(candidate_value)
    return bool(
        target_key
        and candidate_key
        and candidate_key.release[:2] == target_key.release[:2]
        and candidate_key <= target_key
    )


def _warm_search_tiers(identity: _SearchIdentity) -> tuple[_TierQuery, ...]:
    """Build the shared ordered seven-tuple degradation tiers."""
    common = {
        "framework_name": identity.framework,
        "model_type": identity.model_type,
        "architectures": identity.arch_slug,
    }
    hardware_values = _hardware_fallback_values(identity.hardware)
    return (
        _TierQuery(
            tier="same_arch_class",
            confidence=0.95,
            labels={
                **common,
                "hardware": identity.hardware,
                "framework_version": identity.framework_version,
                "precision": identity.precision,
            },
            hardware_in=None,
            relax_framework_version=False,
        ),
        _TierQuery(
            tier="same_gpu_isa",
            confidence=0.85,
            labels={
                **common,
                "framework_version": identity.framework_version,
                "precision": identity.precision,
            },
            hardware_in=hardware_values,
            relax_framework_version=False,
        ),
        _TierQuery(
            tier="compatible_framework_version",
            confidence=0.72,
            labels={**common, "precision": identity.precision},
            hardware_in=hardware_values,
            relax_framework_version=True,
        ),
    )


def _known_labels(labels: Mapping[str, str]) -> dict[str, str]:
    """Drop the labels a search cannot narrow on: the empty ones and the unknown placeholders."""
    return {
        key: value for key, value in labels.items() if value and value not in ("unknown_model_type", "unknown_arch")
    }


def _candidate_dimension(row: Mapping[str, Any], key: str) -> str:
    """Read one seven-tuple dimension from metadata or canonical id."""
    alias = "framework_name" if key == "framework" else key
    value = row.get(key)
    if value in (None, ""):
        value = row.get(alias)
    if value not in (None, ""):
        return str(value).strip().lower()
    try:
        dimensions = cid_to_path_components(str(row.get("canonical_id") or ""))
    except ValueError:
        return ""
    names = (
        "model",
        "hardware",
        "framework_name",
        "model_type",
        "architectures",
        "framework_version",
        "precision",
    )
    return str(dict(zip(names, dimensions)).get(alias) or "").strip().lower()


def _max_session_gain(row: Mapping[str, Any]) -> float:
    """Return the MAX ``gain_pct`` across a row's sessions (fallback flat gain_pct)."""
    best = 0.0
    sessions = row.get("sessions")
    if isinstance(sessions, list):
        for s in sessions:
            if not isinstance(s, Mapping):
                continue
            try:
                g = float(s.get("gain_pct") or 0.0)
            except (TypeError, ValueError):
                continue
            best = max(best, g)
    if best <= 0:
        try:
            best = max(best, float(row.get("validated_gain_pct") or row.get("gain_pct") or 0.0))
        except (TypeError, ValueError):
            pass
    return best


def _config_replay_args_envs(row: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Extract a replayable ``(args, envs)`` pair from a row's best_config."""
    best_config = row.get("best_config") if isinstance(row.get("best_config"), Mapping) else {}
    args = _text(best_config.get("extra_server_args"))
    envs = best_config.get("extra_envs") or {}
    if not isinstance(envs, Mapping):
        envs = {}
    return args, {str(k): str(v) for k, v in envs.items()}


def _has_replayable_config(row: Mapping[str, Any]) -> bool:
    """True when ``row`` carries a non-empty champion config (args OR envs)."""
    if not isinstance(row, Mapping):
        return False
    if isinstance(row.get("replay_material_available"), bool):
        # Remote candidate material is inspected by its owning AgentKB SDKs while isolated; T0 receives only this
        # capability bit.
        return bool(row.get("replay_material_available"))
    if isinstance(row.get("replay_config_available"), bool):
        return bool(row.get("replay_config_available"))
    args, envs = _config_replay_args_envs(row)
    return bool(args or envs)


def _recipe_is_actionable(row: Mapping[str, Any]) -> bool:
    """True when a warm recipe carries something worth replaying or priors."""
    if not isinstance(row, Mapping):
        return False
    if isinstance(row.get("replay_material_available"), bool):
        return bool(row.get("replayable") and row.get("replay_material_available"))
    if isinstance(row.get("replayable"), bool):
        return bool(row.get("replayable"))
    if _config_is_actionable(row.get("best_config")):
        return True
    try:
        if float(row.get("best_throughput") or 0.0) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    return any(row.get(key) for key in ("what_worked", "what_failed", "pitfalls", "lessons"))


def _config_is_actionable(best_config: Any) -> bool:
    """True when a stored champion config has something to replay; env-only or args-only still counts."""
    if not isinstance(best_config, Mapping) or not best_config:
        return False
    args = _text(best_config.get("extra_server_args"))
    envs = best_config.get("extra_envs") or best_config.get("envs") or {}
    return bool(args or (isinstance(envs, Mapping) and envs))


def _shape_matches(donor: Mapping[str, Any], target_value: Any, donor_key: str) -> bool:
    """True when the donor pins the same positive shape dimension, or the target pins none."""
    try:
        target_shape = int(target_value)
    except (TypeError, ValueError):
        return True
    if target_shape <= 0:
        return True
    try:
        donor_shape = int(donor.get(donor_key))
    except (TypeError, ValueError):
        return False
    return donor_shape > 0 and target_shape == donor_shape


def _donor_dimensions_match(donor: Mapping[str, Any], target: _DonorTarget) -> bool:
    """True when the donor names a concrete architecture and model type, both matching the target's."""
    from hyperloom.inference_optimizer.recipe_snapshot_constants import _architectures_slug

    donor_arch_value = donor.get("architectures")
    if donor_arch_value in (None, "", []):
        donor_arch_value = _candidate_dimension(donor, "architectures")
    donor_arch = _architectures_slug(donor_arch_value)
    donor_mt = _candidate_dimension(donor, "model_type")
    unknown = {"", "unknown", "unknown_arch", "unknown_model_type"}
    if donor_arch in unknown or donor_mt in unknown:
        return False
    if target.arch_slug and donor_arch and donor_arch != target.arch_slug:
        return False
    target_mt = _text(target.model_type).lower()
    return not (target_mt and donor_mt and donor_mt != target_mt)


def _donor_is_trustworthy(donor: Mapping[str, Any], target: _DonorTarget) -> bool:
    """Gate a BORROWED (cross-model) warm-replay config donor."""
    if not isinstance(donor, Mapping):
        return False
    if isinstance(donor.get("replayable"), bool) and not donor.get("replayable"):
        return False
    if not _has_replayable_config(donor):
        return False
    # Require evidence of a real positive gain.
    if _max_session_gain(donor) <= 0:
        return False
    if not _donor_dimensions_match(donor, target):
        return False
    return all(
        (
            _shape_matches(donor, target.conc, "conc"),
            _shape_matches(donor, target.isl, "isl"),
            _shape_matches(donor, target.osl, "osl"),
        )
    )


def _rank_warm_candidates(
    rows: list[Mapping[str, Any]],
    *,
    target_framework_version: str,
) -> list[Mapping[str, Any]]:
    """Rank framework proximity, validated gain, then recency."""
    target_version = str(target_framework_version or "")
    ranked = list(rows)
    ranked.sort(
        key=lambda row: str(row.get("updated_at") or ""),
        reverse=True,
    )
    ranked.sort(key=_max_session_gain, reverse=True)
    ranked.sort(
        key=lambda row: _framework_proximity(row, target_version),
        reverse=True,
    )
    return ranked


def _framework_proximity(row: Mapping[str, Any], target_version: str) -> Version:
    """A row's framework version when it is still compatible with the target's, else the lowest sort key."""
    candidate = _candidate_dimension(row, "framework_version")
    if not _framework_version_is_compatible(target_version, candidate):
        return Version("0.dev0")
    return _framework_semver(candidate) or Version("0.dev0")


def _search_warm_candidates(
    kb: Any,
    *,
    labels: dict[str, str],
    hardware_in: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Run exact identity search and fail soft on backend errors."""
    kwargs: dict[str, Any] = {"label_match": labels, "limit": 100}
    if hardware_in is not None and getattr(kb, "mode", "") == "remote":
        kwargs["hardware_in"] = hardware_in
    try:
        rows = kb.search(**kwargs)
    except Exception as exc:  # noqa: BLE001
        log.info("warm-start exact candidate search failed: %s", exc)
        return []
    return [row for row in (rows or []) if isinstance(row, Mapping)]


def _remote_candidate_matches(
    kb: Any,
    row: Mapping[str, Any],
    *,
    labels: Mapping[str, str],
) -> bool:
    """Validate exact server dimensions and current replay metadata."""
    if getattr(kb, "mode", "") != "remote":
        return True
    if (
        row.get("replayable") is not True
        or str(row.get("view_source") or "") != "current"
        or row.get("replay_material_available") is not True
    ):
        return False
    return all(_candidate_dimension(row, key) == str(value).strip().lower() for key, value in labels.items())


def _select_remote_candidate(kb: Any, row: Mapping[str, Any]) -> bool:
    """Ask remote adapters to materialize T0's accepted candidate."""
    select = getattr(kb, "select_candidate", None)
    if not callable(select):
        return True
    try:
        return bool(select(row))
    except Exception as exc:  # noqa: BLE001 — reject and continue cascade
        log.warning(
            "warm-start candidate materialization failed for %s: %s",
            row.get("canonical_id"),
            exc,
        )
        return False


def _take_unseen(rows: list[Mapping[str, Any]], seen: set[str]) -> list[Mapping[str, Any]]:
    """The rows whose canonical id *seen* has not offered yet, recording each one as offered."""
    fresh: list[Mapping[str, Any]] = []
    for row in rows:
        candidate_id = str(row.get("canonical_id") or "")
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            fresh.append(row)
    return fresh


def _candidate_matches_tier(
    candidate: Mapping[str, Any],
    *,
    identity: _SearchIdentity,
    query: _TierQuery,
) -> bool:
    """True when a candidate still carries every dimension its tier pins."""
    if query.hardware_in is not None and not _hardware_is_compatible(
        identity.hardware,
        _candidate_dimension(candidate, "hardware"),
    ):
        return False
    if _candidate_dimension(candidate, "precision") != identity.precision.strip().lower():
        return False
    return not query.relax_framework_version or _framework_version_is_compatible(
        identity.framework_version,
        _candidate_dimension(candidate, "framework_version"),
    )


def _trustworthy_candidates(
    rows: list[Mapping[str, Any]],
    *,
    identity: _SearchIdentity,
    query: _TierQuery,
    target: _DonorTarget,
) -> list[Mapping[str, Any]]:
    """The rows at one tier that match its pinned dimensions and clear the donor gate."""
    return [
        row
        for row in rows
        if _candidate_matches_tier(row, identity=identity, query=query) and _donor_is_trustworthy(row, target)
    ]


def _with_exact_history(
    candidate: Mapping[str, Any],
    exact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach exact-identity priors without changing donor replay metrics."""
    merged = dict(candidate)
    if not isinstance(exact, Mapping):
        return merged
    merged["exact_history"] = {
        "canonical_id": str(exact.get("canonical_id") or ""),
        "view": dict(exact.get("view") or {}),
        **{
            key: list(exact.get(key) or [])
            for key in (
                "what_worked",
                "what_failed",
                "remaining_gaps",
                "lessons",
                "pitfalls",
                "sessions",
            )
        },
        "validated_gain_pct": exact.get("validated_gain_pct"),
        "gain_pct": exact.get("gain_pct"),
    }
    return merged


def _find_config_donor(
    kb: Any,
    *,
    cid: str,
    identity: _SearchIdentity,
    target: _DonorTarget,
) -> _ConfigDonor:
    """Borrow a replayable config through the standard degradation tiers."""
    if not (identity.model_type or identity.arch_slug):
        return _ConfigDonor(None, "", 0.0)
    seen = {cid}
    for query in _warm_search_tiers(identity):
        rows = _search_warm_candidates(
            kb,
            labels=_known_labels(query.labels),
            hardware_in=query.hardware_in,
        )
        usable = _trustworthy_candidates(
            _take_unseen(rows, seen),
            identity=identity,
            query=query,
            target=target,
        )
        if rows and not usable:
            log.info(
                "warm-start config donor tier %s rejected all %d candidates",
                query.tier,
                len(rows),
            )
        ranked = _rank_warm_candidates(usable, target_framework_version=identity.framework_version)
        if ranked:
            return _ConfigDonor(ranked[0], query.tier, query.confidence)
    return _ConfigDonor(None, "", 0.0)
