"""Build the enablement section and the replay recipe from the lane's durable state.

The recorder only writes what it is handed: the lane computes the recipe here
and passes it to ``enablement_event.finish``, so the recorder never imports the
orchestrator.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.fs_utils import relative_posix
from hyperloom.inference_optimizer.breakdown.recorder.recorder_warnings import RECORDING_ERRORS, note_failure
from hyperloom.inference_optimizer.breakdown.session_package import deliverable

from hyperloom.common.coerce import to_mapping

from .attempts import build_attempt_summary
from .steps import PATCH_KIND
from .sufficiency import read_status

log = logging.getLogger(__name__)


#: Distinguishes "this state never recorded the field" from a recorded ``None``.
_ABSENT = object()


_RECIPE_STATE_FIELDS: tuple[str, ...] = (
    "active_runtime",
    "accepted_config",
    "accepted_config_source",
    "accepted_stack_targets",
    "base_sha",
    "build_extensions_not_carried",
    "build_manifest",
    "levers_without_readers",
    "environment_closure",
    "framework_root",
    "installed_versions_at_keep",
    "kept_artifacts",
    "kept_patches",
    # ``select_linked_build`` falls back from ``last_specialist_task_id`` to the
    # kept rounds, precisely because that marker is one-shot and is normally
    # already consumed by the time a build is linked. Omitted here, the fallback
    # exists at runtime and can never fire in the recorded recipe: a build
    # reachable only through a round's durable identity disappears from the
    # projection, which is the case the fallback was added for.
    "kept_rounds",
    "kept_stack_action",
    "last_specialist_task_id",
    "launch_argv_refused",
    "launch_evidence",
    "patch_roots",
    "patch_targets",
    "roots",
    "setup_commands",
    "setup_executions",
    "source_snapshots",
)


#: Reasons that specifically deny a closed dependency set. The status cannot read
#: "verified" while one stands: pinned components are not a pinned environment,
#: and a closure captured over the Python layer alone does not cover a build or
#: an installer outside it.
_CLOSURE_DENYING_CODES: frozenset[str] = frozenset(
    {
        "build_inputs_incomplete",
        "build_attempt_unjoined",
        "environment_closure_absent",
        "closure_scope_incomplete",
        "setup_occurrences_unknown",
        "setup_ledger_truncated",
    }
)


def _eg(state: dict, name: str, default: Any = None) -> Any:
    """Read an enablement round field from the nested ``enablement`` state dict."""
    return to_mapping(state.get("enablement")).get(name, default)


def _closure_status(decision: dict[str, Any], enablement: dict[str, Any]) -> str:
    """Return whether the recipe pins the dependency set, not merely components.

    Args:
        decision: The ``replay_sufficiency`` verdict this section carries.
        enablement: The projected round state, read for its execution ledger.
    """
    # The ledger is the only record of which installer families ran, and the
    # only durable one that records a failed execution, so an empty one leaves
    # the closure's scope unobserved rather than clean.
    if not (enablement.get("setup_executions") or []):
        return "unverified"
    denied = {str(r.get("code")) for r in decision.get("reasons") or []} & _CLOSURE_DENYING_CODES
    return "unverified" if denied else "verified"


def _recipe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Read the enablement fields the recipe contract is projected from."""
    return {name: _eg(state, name) for name in _RECIPE_STATE_FIELDS}


def _delivered_payloads(
    out: dict[str, Any],
    steps: list[dict[str, Any]],
    session_dir: Path,
) -> set[tuple[str, str]] | None:
    """What the session bundle actually hands a consumer of this recipe.

    The recipe names the bytes behind its manifests and digests, and the
    packager says which of them arrive as the recipe describes them; neither
    side restates the other's rules. ``None`` when the recipe references
    nothing, which is the one case with nothing to deliver.
    """
    from .sufficiency import referenced_payloads

    referenced = referenced_payloads(out, steps)
    if not referenced:
        return None
    return deliverable(session_dir, referenced)


def _recipe_accepted_config(enablement: dict[str, Any], state: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    """The accepted config, pointed at the archived file it was read from.

    Empty when nothing was accepted. The archived copy wins over the probe's
    own config: it is the one the session bundle ships.
    """
    from .projections import project_accepted_config

    accepted_config = project_accepted_config(enablement.get("accepted_config"))
    if not accepted_config:
        return {}
    archived = str(_eg(state, "accepted_config_path", "") or "")
    config_path = archived or str(_eg(state, "probe_config_path", "") or "")
    if config_path:
        accepted_config["config_path"] = relative_posix(Path(config_path), session_dir) or config_path
    return accepted_config


def _recipe_projections(enablement: dict[str, Any]) -> dict[str, Any]:
    """The portable projections of the accepted stack, empty values omitted."""
    from .projections import project_roots, project_runtime_provenance, project_source_snapshots

    candidates = (
        ("roots", project_roots(enablement.get("roots"))),
        ("source_snapshots", project_source_snapshots(enablement.get("source_snapshots"))),
        ("accepted_stack_targets", enablement.get("accepted_stack_targets") or {}),
        ("base_sha", str(enablement.get("base_sha") or "") or None),
        ("runtime_provenance", project_runtime_provenance(enablement)),
        # The interpreter is an absolute path on this host; the setup script reads it from state.
        (
            "environment_closure",
            {k: v for k, v in (enablement.get("environment_closure") or {}).items() if k != "interpreter"},
        ),
        ("installed_versions_at_keep", enablement.get("installed_versions_at_keep") or None),
    )
    return {key: value for key, value in candidates if value}


def _collect_recipe(
    out: dict[str, Any],
    state: dict[str, Any],
    *,
    session_dir: Path,
) -> None:
    """Emit the ordered replay contract and the verdict over it.

    ``recipe_steps`` is emitted only when non-empty, so a session that
    contributed nothing emits no key at all. ``replay_sufficiency`` is emitted
    unconditionally beside it: its own absence is the one absence that carries
    meaning, and a consumer must read it as insufficient.
    """
    from . import build_recipe_steps, evaluate_replay_sufficiency
    from .projections import project_launch_evidence

    enablement = _recipe_state(state)
    steps = build_recipe_steps(enablement, attempt_summary=build_attempt_summary)
    if steps:
        # Normalized on the way out, never on the way in: ``patch_targets`` and
        # ``patch_roots`` are keyed by the raw path, so rewriting the input makes
        # every lookup miss and the whole recipe reports ``patch_targets_unknown``
        # -- a portability fix that silently costs the steps their targets. The
        # published value is the only part a consumer reads.
        out["recipe_steps"] = [_portable_step(st, session_dir) for st in steps]
    accepted_config = _recipe_accepted_config(enablement, state, session_dir)
    if accepted_config:
        out["accepted_config"] = accepted_config
    evidence, argv_refused = project_launch_evidence(enablement.get("launch_evidence"))
    out["accepted_config_source"] = str(enablement.get("accepted_config_source") or "") or None
    out["launch_evidence"] = evidence
    out.update(_recipe_projections(enablement))
    # Tri-state observations: ``None`` (could not be read) and ``[]`` (clean)
    # mean opposite things, so they are copied through a sentinel rather than
    # dropped when falsy. A session that predates the field stays absent.
    for _tri in ("build_extensions_not_carried", "levers_without_readers"):
        _val = _eg(state, _tri, _ABSENT)
        if _val is not _ABSENT:
            out[_tri] = _val
    decision = evaluate_replay_sufficiency(
        enablement,
        steps=steps,
        section=out,
        delivered_payloads=_delivered_payloads(out, steps, session_dir),
        launch_argv_refused=argv_refused or bool(enablement.get("launch_argv_refused")),
    )
    out["replay_sufficiency"] = decision
    out["dependency_closure_status"] = _closure_status(decision, enablement)
    from ..setup_script import setup_script_record

    setup_script = setup_script_record(session_dir, sufficient=decision["status"] == "sufficient")
    if setup_script:
        out["setup_script"] = setup_script


def _lane_engaged(state: dict[str, Any]) -> bool:
    """Whether the section is emitted at all.

    The section exists when the lane did something or was explicitly turned off;
    with ``all`` the default, "armed but never needed" is the case that stays
    hidden. A round does not park its task id in the enablement state, so the
    evidence that one ran is the specialist it settled onto, the per-round
    records it kept, and the setup rows it ran.

    ``launch_observation_path`` is deliberately NOT read here. All three of its
    writers (``writeback._record_enablement_eval_trigger`` and the two boot
    failure paths) set it from the *trigger* observation -- the failed launch or
    failed eval that gives the lane something to author against -- and they run
    before any round is dispatched. Reading it would emit the section for a
    session whose lane never opened a round.
    """
    # eval_kind is NOT cleared on success, so it can identify an eval-origin
    # enablement even after the run succeeds and origin is reset to "".
    have_eval = str(_eg(state, "origin", "") or "") == "eval" or bool(_eg(state, "baseline_eval_kind"))
    # A state document carrying no mode is read as the SharedState default,
    # which is the value the lane actually ran under.
    mode = str(state.get("enablement_mode") or "all").strip().lower() or "all"
    engaged = have_eval or any(
        _eg(state, name)
        for name in (
            "last_specialist_task_id",
            "kept_rounds",
            "kept_patches",
            "setup_executions",
            "human_review_logged",
        )
    )
    provisioned = any(
        _eg(state, name) for name in ("active_runtime", "attempt_runtimes", "build_manifest", "last_build_failure")
    )
    return engaged or mode == "off" or provisioned


def _portable_patch_ref(raw: str, session_dir: Path) -> str:
    """A patch reference that names no directory on the authoring host."""
    rel = relative_posix(Path(raw), session_dir)
    if not rel:
        return Path(raw).name
    return Path(rel).name if Path(rel).is_absolute() else rel


def _portable_step(step: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    """A recipe step with its host paths reduced to what a consumer can act on.

    ``root`` is the tree a patch was resolved against, and it is absolute by
    construction. Nothing reads it -- the rules join on ``root_id``, and
    ``project_roots`` drops the same path for the same reason -- so shipping it
    only tells the consumer about a directory layout that is not theirs.
    """
    if step.get("kind") != PATCH_KIND:
        return step
    out = dict(step)
    if out.get("path"):
        out["path"] = _portable_patch_ref(str(out["path"]), session_dir)
    out.pop("root", None)
    return out


def _portable_kept_rounds(rounds: list[Any], session_dir: Path) -> list[dict[str, Any]]:
    """Reduce each kept round to the linkage a replay joins on.

    ``_push_kept_round`` stores authoring-workspace patch paths and raw artifact
    dicts carrying source and target; copied verbatim they would put absolute
    paths from this machine into a recipe meant to be replayed on another.
    A round's patches are sourced from the archive by name, so the name is the
    whole linkage and the directory it sat in on this machine is not part of it.
    Checked explicitly rather than through an ``or`` fallback:
    ``relative_posix`` falls back to ``str(path)``, so ``or`` never fires and
    the absolute path would travel exactly as if nothing had been done.

    An artifact travels as its ``rel_target`` where the producer recorded one
    and as the target's own name otherwise; ``target`` verbatim is an install
    path on this machine.
    """
    return [
        {
            "task_id": str(r.get("task_id") or ""),
            "patches": [_portable_patch_ref(str(p), session_dir) for p in (r.get("patches") or [])],
            "artifacts": [
                str(a.get("rel_target") or "") or Path(str(a.get("target") or "")).name
                for a in (r.get("artifacts") or [])
                if isinstance(a, dict) and (a.get("rel_target") or a.get("target"))
            ],
        }
        for r in rounds
        if isinstance(r, dict)
    ]


def _portable_kept_artifacts(
    artifacts: list[Any],
    *,
    root_ids: Mapping[str, str],
    framework_root: str,
) -> list[dict[str, Any]]:
    """Reduce each kept artifact to the portable pair the replay rules read.

    ``rel_target`` and ``root_id`` are the only two the rules consult.
    ``target`` is an install path on this machine: shipped verbatim it named a
    directory the consumer does not have, for a field nothing reads.
    """
    return [
        {
            "rel_target": str(a.get("rel_target") or ""),
            "kind": str(a.get("kind") or ""),
            "root_id": root_ids.get(str(a.get("root") or "") or framework_root) or None,
        }
        for a in artifacts
        if isinstance(a, dict) and a.get("target")
    ]


def _collect_landed_stack(out: dict[str, Any], state: dict[str, Any], *, session_dir: Path) -> None:
    """Emit the rounds and artifacts the lane landed."""
    from .steps import root_ids_by_path

    # Exported, not only fed to the projection: a build linked
    # through a round's durable identity is reachable only from here once the
    # one-shot ``last_specialist_task_id`` has been consumed, which is the normal
    # state by the time a build lands.
    kept_rounds_raw = _eg(state, "kept_rounds")
    if isinstance(kept_rounds_raw, list) and kept_rounds_raw:
        out["kept_rounds"] = _portable_kept_rounds(kept_rounds_raw, session_dir)
    kept_artifacts_raw = _eg(state, "kept_artifacts")
    framework_root = str(_eg(state, "framework_root", "") or "")
    if isinstance(kept_artifacts_raw, list) and kept_artifacts_raw:
        out["kept_artifacts"] = _portable_kept_artifacts(
            kept_artifacts_raw,
            root_ids=root_ids_by_path({"roots": _eg(state, "roots")}),
            framework_root=framework_root,
        )


def collect_enablement(session_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Collect the enablement replay contract from the durable round state.

    The lane's own account of what it did is recorded at author time; what is
    projected here is the part no author-time site can state -- the ordered
    ``recipe_steps`` a consumer would replay and the ``replay_sufficiency``
    verdict over them, judged against the evidence the session actually
    captured. The landed-stack keys are kept because the verdict is computed
    over the emitted section, not over raw state.
    """
    if not _lane_engaged(state):
        return {}
    out: dict[str, Any] = {}
    _collect_landed_stack(out, state, session_dir=session_dir)
    _collect_recipe(out, state, session_dir=session_dir)
    return out


#: The replay contract's own keys, as opposed to the working inputs
#: :func:`collect_enablement` judges them over. ``replay_sufficiency``
#: is the verdict a consumer outside this session reads to decide whether the
#: recipe can be replayed at all; the rest is the evidence that verdict was
#: reached over, kept beside it so the decision can be re-derived rather than
#: merely trusted.
_RECIPE_KEYS: tuple[str, ...] = (
    "recipe_steps",
    "replay_sufficiency",
    "dependency_closure_status",
    # Path and sha256 of the script that reproduces the setup, and whether it
    # does so on its own -- claimed only beside a sufficient verdict.
    "setup_script",
    "accepted_stack_targets",
    "source_snapshots",
    "roots",
    "base_sha",
    "runtime_provenance",
    "environment_closure",
    "installed_versions_at_keep",
    "accepted_config_source",
    "launch_evidence",
    # Tri-state, and inputs to the verdict above: ``None`` says the scan could
    # not be made, ``[]`` that it came back clean. The collector distinguishes
    # them with a sentinel and omits the key entirely when the state never
    # recorded one, so the copy below carries all three readings. Left out, a
    # consumer reading ``build_extensions_not_carried`` or ``levers_unverified``
    # in the reasons had no way to see what they were decided over -- which is
    # the one thing this block exists to keep beside the verdict.
    "build_extensions_not_carried",
    "levers_without_readers",
    # ``select_linked_build`` falls back to these when the one-shot specialist
    # marker has been consumed, which it normally has by the time a build lands.
    "kept_rounds",
)

#: Ceiling on the serialized recipe. A real host's environment closure runs to
#: roughly 12 KB, so this is generous; what it exists for is that nothing else
#: on this path bounds the block. Exceeding it does NOT truncate: a shortened
#: closure is indistinguishable from a narrow one, and the verdict was computed
#: over the full payload, so the pair would contradict each other. The recipe is
#: replaced by the explicit ``not_evaluated`` decision instead, which every
#: consumer already reads as insufficient.
_MAX_RECIPE_BYTES = 256 * 1024


def recipe_for(enablement: Any, *, session_dir: str, mode: str = "") -> dict[str, Any]:
    """Project the durable round state onto the replay contract and judge it.

    This is the one fact about an enablement lane that no author-time record can
    state: every ``enablement_event.record_*`` writes what one round did at the
    moment it did it, while the recipe is a statement about the *stack* -- what
    a consumer outside this session would have to replay, and whether the
    session captured enough for that to be possible. Called at the terminal,
    because that is the first moment the accepted stack is complete.

    Never raises, and never returns an empty verdict: an absent
    ``replay_sufficiency`` is read as insufficient by contract, so a projection
    that could not run records the explicit ``not_evaluated`` decision rather
    than leaving the key out and letting a consumer infer it.

    Args:
        enablement: The durable ``EnablementRound``.
        session_dir: Session root the snapshot refs are expressed against.
        mode: The lane's mode, needed only so the collector's own gate opens.

    Returns:
        The replay-contract subset of the collected section.
    """
    section: dict[str, Any] = {}
    try:
        state = asdict(enablement) if is_dataclass(enablement) else dict(to_mapping(enablement))
        collected = collect_enablement(Path(str(session_dir or ".")), {"enablement": state, "enablement_mode": mode})
        section = {key: collected[key] for key in _RECIPE_KEYS if key in collected}
        # The collector emits this unconditionally beside the steps; carrying its
        # own absence forward would hand a consumer a recipe with no verdict.
        if "kept_artifacts" in collected:
            section["kept_artifacts"] = collected["kept_artifacts"]
    except RECORDING_ERRORS as exc:
        note_failure(section="enablement_event", error=exc, detail="enablement event: recipe projection failed")
    if not isinstance(section.get("replay_sufficiency"), Mapping):
        section["replay_sufficiency"] = read_status({})
    try:
        oversize = len(json.dumps(section, default=str).encode("utf-8")) > _MAX_RECIPE_BYTES
    except (TypeError, ValueError):
        oversize = True
    if oversize:
        note_failure(
            section="enablement_event",
            error=ValueError("recipe exceeds the recorded ceiling"),
            detail="enablement event: recipe too large to record; reporting it as unjudged",
        )
        return {"replay_sufficiency": read_status({})}
    return section
