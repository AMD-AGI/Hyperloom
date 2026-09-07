# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``replay_sufficiency``: the authoritative verdict over a projected recipe.

``status`` is ``"insufficient"`` whenever any reason stands and ``"sufficient"``
only when every rule holds; it is never inferred from how many optional keys
happen to be present. An absent decision reads as insufficient, so a section
produced by an older writer cannot be mistaken for one this contract judged.
"""

from __future__ import annotations

import shlex
from typing import Any, Iterable, Mapping, Sequence

from .steps import BUILD_KIND, PATCH_KIND, SETUP_KIND, command_digest, select_linked_build

SCHEMA_VERSION = 1

BLOCKS_REPLAY = "replay"
BLOCKS_ASSERTION = "assertion_validation"
BLOCKS_BOTH = "both"

#: The closed code set, each with what it blocks. A code outside this table is
#: itself insufficient.
REASON_BLOCKS: dict[str, str] = {
    "not_evaluated": BLOCKS_BOTH,
    "activation_incomplete": BLOCKS_BOTH,
    "launch_evidence_mismatch": BLOCKS_BOTH,
    "runtime_rebuild_required": BLOCKS_BOTH,
    "root_unidentified": BLOCKS_REPLAY,
    "root_unmappable": BLOCKS_REPLAY,
    "accepted_stack_not_launched": BLOCKS_BOTH,
    "source_snapshot_incomplete": BLOCKS_REPLAY,
    "source_snapshot_missing": BLOCKS_REPLAY,
    "artifact_not_self_contained": BLOCKS_REPLAY,
    "setup_occurrences_unknown": BLOCKS_BOTH,
    "setup_effect_outside_verified_launch": BLOCKS_BOTH,
    "setup_ledger_truncated": BLOCKS_BOTH,
    "setup_inputs_incomplete": BLOCKS_REPLAY,
    "build_attempt_unjoined": BLOCKS_REPLAY,
    "build_inputs_incomplete": BLOCKS_REPLAY,
    "environment_closure_absent": BLOCKS_REPLAY,
    "closure_scope_incomplete": BLOCKS_REPLAY,
    "assertions_not_at_keep": BLOCKS_ASSERTION,
    "credential_required": BLOCKS_REPLAY,
}

#: Requested launch settings the observed side is not expected to confirm: the
#: run-local names the extractor is entitled to strip.
_RUN_LOCAL_REQUESTED_FLAGS: frozenset[str] = frozenset(
    {
        "host",
        "port",
        "nccl_port",
        "dist_init_addr",
        "base_gpu_id",
        "gpu_id_step",
        "node_rank",
        "nnodes",
        "random_seed",
        "download_dir",
        "pid",
    }
)

#: The requested spellings of each parallelism axis ``observed_model_binding``
#: reports, so the requested side is compared against the binding rather than
#: against a launch line the extractor strips these from.
_PARALLELISM_AXES: tuple[tuple[str, frozenset[str]], ...] = (
    ("tp", frozenset({"tensor_parallel_size", "tp_size", "tp"})),
    ("dp", frozenset({"data_parallel_size", "dp_size"})),
    ("pp", frozenset({"pipeline_parallel_size", "pp_size"})),
)

#: Compared through ``observed_model_binding.model_digest`` instead.
_MODEL_IDENTITY_FLAGS: frozenset[str] = frozenset(
    {"model", "model_path", "tokenizer", "tokenizer_path", "served_model_name"}
)

#: Carried by ``observed_model_binding`` instead, and compared there.
_MODEL_AND_PARALLELISM_FLAGS: frozenset[str] = _MODEL_IDENTITY_FLAGS.union(
    *(flags for _axis, flags in _PARALLELISM_AXES)
)

#: Names every spawned build inherits and the ambient closure excludes on no
#: path, so a key list missing either was computed over less than the closure.
_AMBIENT_FLOOR: frozenset[str] = frozenset({"PATH", "HOME"})

_BUILTIN_REQUIRED_INPUTS: tuple[str, ...] = (
    "component",
    "repo_url",
    "ref",
    "resolved_sha",
    "gpu_arch",
    "max_jobs",
    "torch_constraint_mode",
    "env_digest",
    "ambient_digest",
)


def _reason(code: str, scope: str = "") -> dict[str, Any]:
    return {"code": code, "blocks": REASON_BLOCKS[code], "scope": scope}


def read_status(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the decision a consumer must read off ``section``.

    An absent key is the one absence that carries meaning: the producer predates
    this contract, so nothing was judged. An unrecognized code is likewise
    insufficient -- the vocabulary is closed.
    """
    decision = (section or {}).get("replay_sufficiency")
    if not isinstance(decision, Mapping):
        return {"schema_version": SCHEMA_VERSION, "status": "insufficient", "reasons": [_reason("not_evaluated")]}
    reasons = [r for r in decision.get("reasons") or [] if isinstance(r, Mapping)]
    if any(str(r.get("code")) not in REASON_BLOCKS for r in reasons):
        return {**dict(decision), "status": "insufficient"}
    return dict(decision)


def _normalize_flag(name: str) -> str:
    return str(name or "").lstrip("-").replace("-", "_")


def _requested_flags(argv: str) -> dict[str, str]:
    """Parse an argv string into ``{normalized_flag: value}``."""
    try:
        tokens = shlex.split(str(argv or ""))
    except ValueError:
        return {}
    flags: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            index += 1
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            flags[_normalize_flag(name)] = value
        else:
            value = ""
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                value = tokens[index + 1]
                index += 1
            flags[_normalize_flag(token)] = value
        index += 1
    return flags


def _activation_reasons(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Judge the graded launch: was this configuration actually launched?"""
    reasons: list[dict[str, Any]] = []
    accepted_config = section.get("accepted_config") or {}
    if str(section.get("accepted_config_source") or "") == "advanced_merge":
        reasons.append(_reason("activation_incomplete", "accepted_config_source"))
    if accepted_config and not accepted_config.get("config_path"):
        reasons.append(_reason("activation_incomplete", "config_path"))
    evidence = section.get("launch_evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        # The reason names the launch of a projected configuration; with none
        # projected there is no launch to have been observed.
        if accepted_config:
            reasons.append(_reason("activation_incomplete", "launch_evidence"))
        return reasons
    binding = evidence.get("observed_model_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    requested_digest = str(evidence.get("requested_model_digest") or "")
    if not str(binding.get("model_digest") or ""):
        reasons.append(_reason("activation_incomplete", "observed_model_binding"))
    elif requested_digest and requested_digest != str(binding.get("model_digest")):
        reasons.append(_reason("launch_evidence_mismatch", "model_digest"))
    reasons.extend(_parallelism_reasons(evidence, binding))
    reasons.extend(_requested_vs_observed_reasons(evidence))
    return reasons


def _parallelism_reasons(
    evidence: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Confirm each requested parallelism axis against the binding that reports it.

    The extractor strips these operands from the observed launch line, so the
    binding is the only observed side there is: without this comparison a server
    graded at a different width certifies the recipe that asked for one.
    """
    requested_flags = _requested_flags(str(evidence.get("requested_server_args") or ""))
    reasons: list[dict[str, Any]] = []
    for axis, spellings in _PARALLELISM_AXES:
        requested = next((v for f, v in requested_flags.items() if f in spellings and v), "")
        if not requested:
            continue
        observed = str(binding.get(axis) or "").strip()
        if not observed:
            reasons.append(_reason("activation_incomplete", f"requested:{axis}"))
        elif observed.lower() != requested.strip().lower():
            reasons.append(_reason("launch_evidence_mismatch", f"requested:{axis}"))
    return reasons


def _requested_vs_observed_reasons(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Confirm every replay-affecting requested setting against observed evidence.

    Judged against the *requested* side, not against the overlap: an
    intersection test silently passes a server that ignored a requested flag.
    """
    observed = _requested_flags(str(evidence.get("observed_server_launch_flags") or ""))
    identity = evidence.get("observed_server_identity")
    identity = {str(k): v for k, v in identity.items()} if isinstance(identity, Mapping) else {}
    reasons: list[dict[str, Any]] = []
    for flag, requested in _requested_flags(str(evidence.get("requested_server_args") or "")).items():
        if flag in _RUN_LOCAL_REQUESTED_FLAGS or flag in _MODEL_AND_PARALLELISM_FLAGS:
            continue
        if flag in observed:
            value = observed[flag]
        elif flag in identity:
            value = str(identity[flag])
        else:
            reasons.append(_reason("activation_incomplete", f"requested:{flag}"))
            continue
        if requested and str(value).strip().lower() != requested.strip().lower():
            reasons.append(_reason("launch_evidence_mismatch", f"requested:{flag}"))
    return reasons


def _acquisition_is_pinned(acquisition: Mapping[str, Any]) -> bool:
    """True when the acquisition names bytes rather than a moving source.

    A version-only wheel entry is not pinned: the install runs ``--upgrade``
    against the index, so re-running it reproduces whatever that index holds at
    replay time and the recorded version cannot detect the difference.
    """
    method = str(acquisition.get("acquisition_method") or "")
    if method == "editable_ref":
        return bool(acquisition.get("resolved_ref"))
    if method == "wheel":
        resolved = acquisition.get("resolved_packages")
        if not isinstance(resolved, Mapping) or not resolved:
            return False
        return all(isinstance(e, Mapping) and e.get("artifact_digest") for e in resolved.values())
    return False


def _runtime_reasons(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A runtime override the recipe cannot rebuild is not a replay path."""
    provenance = section.get("runtime_provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("override_keys"):
        return []
    if provenance.get("build_task_id"):
        return []
    acquisition = provenance.get("acquisition")
    if isinstance(acquisition, Mapping) and _acquisition_is_pinned(acquisition):
        return []
    return [_reason("runtime_rebuild_required", "runtime_provenance")]


def _root_reasons(section: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Judge that every replayed step and artifact names a reconstructable tree."""
    roots = {str(r.get("id")): r for r in (section.get("roots") or []) if isinstance(r, Mapping)}
    reasons: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if step.get("kind") != PATCH_KIND:
            continue
        record = roots.get(str(step.get("root_id") or ""))
        if record is None or not record.get("contributions"):
            reasons.append(_reason("root_unidentified", f"step[{index}]"))
    for index, artifact in enumerate(section.get("kept_artifacts") or []):
        if not isinstance(artifact, Mapping):
            continue
        record = roots.get(str(artifact.get("root_id") or ""))
        if record is None or not record.get("contributions"):
            reasons.append(_reason("root_unidentified", f"artifact[{index}]"))
    for root_id, record in roots.items():
        # A read that failed leaves is_git standing over no commit, which names
        # no tree a consumer can check out.
        if record.get("is_git") and not record.get("base_sha"):
            reasons.append(_reason("root_unidentified", root_id))
    anchors: dict[tuple[str, str], str] = {}
    for root_id, record in roots.items():
        target = record.get("replay_target") or {}
        anchor, rel = str(target.get("anchor") or "unmappable"), str(target.get("rel") or "")
        if anchor == "unmappable" or (anchor, rel) in anchors:
            reasons.append(_reason("root_unmappable", root_id))
            continue
        anchors[(anchor, rel)] = root_id
    return reasons


def _snapshot_reasons(
    section: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Judge the captured content against the stack it is supposed to contain."""
    snapshots = [s for s in (section.get("source_snapshots") or []) if isinstance(s, Mapping)]
    by_root = {str(s.get("root_id")): s for s in snapshots}
    reasons: list[dict[str, Any]] = []
    for record in section.get("roots") or []:
        # A capture that finds nothing returns no manifest at all, so a listed
        # root with no entry is the same failure with no record to test.
        if isinstance(record, Mapping) and str(record.get("id")) not in by_root:
            reasons.append(_reason("source_snapshot_missing", str(record.get("id"))))
    for snapshot in snapshots:
        if not snapshot.get("complete"):
            reasons.append(_reason("source_snapshot_incomplete", str(snapshot.get("root_id"))))
    reasons.extend(_expected_op_reasons(section, by_root, steps))
    return reasons


def _expected_op_reasons(
    section: Mapping[str, Any],
    by_root: Mapping[str, Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Match every expected target against the operation it was declared with.

    Presence is not coverage: a KEEP reached with its mutation inputs stripped
    holds the *base* file, so capturing it returns complete while the snapshot
    contains none of the accepted stack's changes.
    """
    expected = section.get("accepted_stack_targets")
    declared = any(step.get("kind") == PATCH_KIND for step in steps) or bool(section.get("kept_artifacts"))
    named = isinstance(expected, Mapping) and any(targets for targets in expected.values())
    # A round dispatched with its mutation inputs stripped declares every target
    # of the stack it replays and names none of them, which no per-target
    # comparison below can reach.
    if declared and not named:
        return [_reason("accepted_stack_not_launched", "accepted_stack_targets")]
    if not isinstance(expected, Mapping):
        return []
    reasons: list[dict[str, Any]] = []
    for root_id, targets in expected.items():
        snapshot = by_root.get(str(root_id))
        captured = {
            str(f.get("rel")): str(f.get("op")) for f in ((snapshot or {}).get("files") or []) if isinstance(f, Mapping)
        }
        if any(captured.get(str(rel)) != str(op) for rel, op in (targets or {}).items()):
            reasons.append(_reason("accepted_stack_not_launched", str(root_id)))
    return reasons


def _artifact_reasons(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An artifact with no captured payload cannot be restored by the recipe.

    Containment is judged against the artifact's own root: the same
    framework-relative layout repeats across a checkout and its installed copy,
    so a pooled path set would let one root's capture answer for another's.
    """
    captured: dict[str, set[str]] = {}
    for snapshot in section.get("source_snapshots") or []:
        if not isinstance(snapshot, Mapping):
            continue
        rels = captured.setdefault(str(snapshot.get("root_id") or ""), set())
        rels.update(
            str(f.get("rel"))
            for f in (snapshot.get("files") or [])
            if isinstance(f, Mapping) and f.get("rel") and str(f.get("op") or "") != "missing"
        )
    reasons: list[dict[str, Any]] = []
    for index, artifact in enumerate(section.get("kept_artifacts") or []):
        if not isinstance(artifact, Mapping):
            continue
        rel = str(artifact.get("rel_target") or "")
        if rel and rel not in captured.get(str(artifact.get("root_id") or ""), set()):
            reasons.append(_reason("artifact_not_self_contained", f"artifact[{index}]"))
    return reasons


def _setup_reasons(enablement: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Judge occurrence identity, mutation reach, input identity and scope."""
    commands = [str(c) for c in (enablement.get("setup_commands") or []) if str(c)]
    ledger = [row for row in (enablement.get("setup_executions") or []) if isinstance(row, Mapping)]
    if not ledger:
        return [_reason("setup_occurrences_unknown", "setup_executions")] if commands else []
    reasons: list[dict[str, Any]] = []
    for row in ledger:
        scope = f"seq[{row.get('seq')}]"
        outcome = str(row.get("outcome") or "")
        # A failed install leaves partial state no replay reproduces, so it takes
        # no succession exemption whatever a later occurrence did.
        stranded = not row.get("present_at_final_launch") and not row.get("replayed_at_final_launch")
        if outcome == "failed" or (outcome == "applied" and stranded):
            reasons.append(_reason("setup_effect_outside_verified_launch", scope))
        # Only a completed execution mutated the accepted environment; a failed
        # one is already named an effect outside the verified launch.
        if outcome == "applied" and str(row.get("installer") or "") not in ("", "pip"):
            reasons.append(_reason("closure_scope_incomplete", scope))
        if outcome == "applied" and row.get("unresolved_inputs"):
            reasons.append(_reason("setup_inputs_incomplete", scope))
    reasons.extend(_truncation_reasons(commands, ledger))
    return reasons


def _truncation_reasons(
    commands: Sequence[str],
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """A durable command the accepted round never replayed was capped out of it.

    Every applied command is replayed as a base into every later round, so the
    only way one is absent from the accepted round's ledger is the resolver's
    cap -- which distinguishes "capped out of the validated run" from "no round
    proposed it". Membership identifies that round rather than the presence
    flag: a round whose every reached command failed carries no present row.
    """
    accepted = [row for row in ledger if row.get("at_accepted_round")]
    if not accepted:
        return []
    # Every row of that round, not only its applied ones: a command it reached
    # and failed is a mutation, which has its own reason.
    reached = {str(row.get("cmd_digest") or "") for row in accepted}
    missing = [cmd for cmd in commands if command_digest(cmd) not in reached]
    return [_reason("setup_ledger_truncated", "setup_commands")] if missing else []


def _build_reasons(
    enablement: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Judge that an admitted build is joined to a row and reproducible."""
    if not any(step.get("kind") == BUILD_KIND for step in steps):
        return []
    sentinel, row = select_linked_build(enablement)
    task_id = str((sentinel or {}).get("task_id") or "")
    if row is None:
        return [_reason("build_attempt_unjoined", task_id)]
    inputs = row.get("build_inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        return [_reason("build_inputs_incomplete", task_id)]
    driver = str(row.get("build_driver") or "")
    missing = [key for key in _BUILTIN_REQUIRED_INPUTS if not inputs.get(key)]
    if not row.get("installed_versions"):
        missing.append("installed_versions")
    if not _AMBIENT_FLOOR.issubset({str(k) for k in inputs.get("ambient_keys") or []}):
        missing.append("ambient_keys")
    # A command the platform spawns unread cannot be shown to depend only on the
    # recorded inputs, so it is incomplete however complete the rest is.
    if missing or driver == "custom_command":
        return [_reason("build_inputs_incomplete", task_id)]
    return []


def _closure_reasons(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Judge the KEEP-time environment closure and assertion observation."""
    reasons: list[dict[str, Any]] = []
    closure = section.get("environment_closure")
    if not isinstance(closure, Mapping) or not (closure.get("distributions") or {}):
        reasons.append(_reason("environment_closure_absent", "environment_closure"))
    if not (section.get("installed_versions_at_keep") or {}):
        reasons.append(_reason("assertions_not_at_keep", "installed_versions_at_keep"))
    return reasons


def _credential_reasons(
    enablement: Mapping[str, Any],
    section: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Name every class of credential a replay operator must supply."""
    reasons: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if step.get("kind") == SETUP_KIND and step.get("credential_class"):
            reasons.append(_reason("credential_required", f"step[{index}]"))
        if step.get("kind") == BUILD_KIND:
            inputs = step.get("build_inputs") or {}
            command = inputs.get("build_command") or {}
            if (
                inputs.get("credential_class")
                or inputs.get("credential_channels")
                or (isinstance(command, Mapping) and command.get("credential_class"))
            ):
                reasons.append(_reason("credential_required", f"step[{index}]"))
    for row in enablement.get("setup_executions") or []:
        if isinstance(row, Mapping) and row.get("credential_channels"):
            reasons.append(_reason("credential_required", f"seq[{row.get('seq')}]"))
    provenance = section.get("runtime_provenance")
    acquisition = (provenance or {}).get("acquisition") if isinstance(provenance, Mapping) else None
    if isinstance(acquisition, Mapping) and (
        acquisition.get("credential_channels") or acquisition.get("credential_class")
    ):
        reasons.append(_reason("credential_required", "runtime_provenance"))
    return reasons


def _payload_references(
    section: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """Return ``(code, scope, path, sha256)`` for every byte a replay is handed.

    The recipe carries manifests, digests and class names; these name the bytes
    behind them, each with the refusal its absence earns and, where its recorder
    took one, the digest the delivered bytes must still hash to. A patch step is
    not among them: its own ``path`` names the authoring workspace no bundle
    ships, while the content it produced is its root's snapshot payload.
    """
    refs: list[tuple[str, str, str, str]] = []
    for snapshot in section.get("source_snapshots") or []:
        if not isinstance(snapshot, Mapping):
            continue
        ref = str(snapshot.get("snapshot_ref") or "").strip("/")
        root_id = str(snapshot.get("root_id"))
        rows = [row for row in (snapshot.get("files") or []) if isinstance(row, Mapping)]
        # A declared deletion has no payload to deliver; a snapshot with no
        # reference at all has nowhere for one to be.
        payloads = [str(row.get("rel") or "").strip("/") for row in rows if str(row.get("op") or "") != "delete"]
        if not ref:
            refs.append(("source_snapshot_missing", root_id, "", ""))
            continue
        refs.extend(("source_snapshot_missing", root_id, f"{ref}/files/{rel}", "") for rel in payloads if rel)
    accepted_config = section.get("accepted_config") or {}
    config_path = str(accepted_config.get("config_path") or "").strip("/")
    if config_path:
        digest = str(accepted_config.get("config_digest") or "")
        refs.append(("artifact_not_self_contained", "config_path", config_path, digest))
    for index, step in enumerate(steps):
        # A digest names the bytes an install consumed; only the delivery makes
        # them obtainable.
        for identity in step.get("input_identity") or ():
            if not isinstance(identity, Mapping):
                continue
            rel = str(identity.get("rel") or "").strip("/")
            if rel:
                refs.append(("artifact_not_self_contained", f"step[{index}]", rel, str(identity.get("sha256") or "")))
    return refs


def referenced_payloads(
    section: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    """Return the ``(path, sha256)`` payloads a delivery must be asked about.

    The digest is ``""`` where the recipe recorded none; where it recorded one,
    a delivery carrying different bytes is not carrying this payload. One path
    referenced under two digests yields two entries: two occurrences consumed
    different bytes there, and a delivery holding one is not holding the other.
    """
    return {(path, digest) for _code, _scope, path, digest in _payload_references(section, steps) if path}


def _delivery_reasons(
    section: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    delivered: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Refuse a bundle whose referenced bytes it does not actually carry.

    A reference the delivery omits is not a thinner recipe -- it is one an
    independent consumer cannot execute, so the export fails closed rather than
    reporting a self-contained artifact it is not. Judged per captured file, not
    per manifest: the manifest travels in this section, and the bytes it names
    do not. Judged per ``(path, digest)`` besides, so a delivery satisfying one
    occurrence's bytes does not answer for another occurrence's at that path.
    """
    packaged = {(str(path).strip("/"), str(digest or "")) for path, digest in delivered}
    reasons: list[dict[str, Any]] = []
    for code, scope, path, digest in _payload_references(section, steps):
        if (path, digest) not in packaged:
            reasons.append(_reason(code, scope))
    return reasons


def evaluate_replay_sufficiency(
    enablement: Mapping[str, Any],
    *,
    steps: Sequence[Mapping[str, Any]],
    section: Mapping[str, Any],
    delivered_payloads: Iterable[tuple[str, str]] | None = None,
    launch_argv_refused: bool = False,
) -> dict[str, Any]:
    """Decide whether the projected recipe can be replayed.

    Args:
        enablement: The durable enablement state, keyed by field name.
        steps: The projected ``recipe_steps`` array.
        section: The emitted enablement section this decision travels in.
        delivered_payloads: The ``(session-relative path, sha256)`` payloads an
            export actually delivers. When given, every payload the recipe
            references must be among them or the export fails closed; when
            ``None`` no delivery is being assembled and the recipe is judged on
            its content alone.
        launch_argv_refused: Whether the sanitizer refused a launch line it could
            not represent, so the evidence carries no observed argv at all.

    Returns:
        ``{schema_version, status, reasons}``, ``"sufficient"`` only when no
        reason stands.
    """
    reasons: list[dict[str, Any]] = []
    if launch_argv_refused:
        # A launch line the export cannot represent is not a thinner evidence
        # object; nothing observed remains to confirm the requested settings.
        reasons.append(_reason("activation_incomplete", "observed_server_launch_flags"))
    reasons.extend(_activation_reasons(section))
    reasons.extend(_runtime_reasons(section))
    reasons.extend(_root_reasons(section, steps))
    reasons.extend(_snapshot_reasons(section, steps))
    reasons.extend(_artifact_reasons(section))
    reasons.extend(_setup_reasons(enablement))
    reasons.extend(_build_reasons(enablement, steps))
    reasons.extend(_closure_reasons(section))
    reasons.extend(_credential_reasons(enablement, section, steps))
    if delivered_payloads is not None:
        reasons.extend(_delivery_reasons(section, steps, delivered_payloads))
    deduped: list[dict[str, Any]] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "insufficient" if deduped else "sufficient",
        "reasons": deduped,
    }
