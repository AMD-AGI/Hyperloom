# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one reader of the baseline-prelude child's session: the main flow's launch info, ``state.json`` and reports.

``state.json`` is the main flow's internal persistence format, not an interface, so every key read from it is read
here and nowhere else, and a schema version other than the pinned one is refused rather than guessed at. What the
reader returns is folded into the root :class:`~hyperloom.contracts.recipe.Recipe` by :func:`root_recipe`; what a
recipe cannot express yet, a kept patch outside the framework's git checkout among it, is named in
:attr:`PreludeArtifacts.unexpressed`, which the root recipe carries as its ``unexpressed`` mark. The launch is rebuilt from
``enablement.accepted_config``, the source the main flow writes ``reports/enablement/enablement_setting.sh`` from;
that script is not parsed, only located (:attr:`PreludeArtifacts.setup_script`), so the setup the root recipe leaves
out can be traced to it.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.jsonio import read_json
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, PatchRef, Recipe, RecipeError, with_unexpressed
from hyperloom.inference_optimizer.breakdown.stop_reasons import outcome_status
from hyperloom.inference_optimizer.grid_server_args import strip_benchmark_harness_flags
from hyperloom_multibranch.scheduling.settings import PRIMARY_METRIC

#: ``LATEST_STATE_SCHEMA_VERSION`` of src/hyperloom/orchestrator/state/shared_state.py, which this package may not
#: import; a test compares the two.
MB_LATEST_STATE_SCHEMA_VERSION = 6

#: How long one read-only git command may take before the checkout counts as unreadable.
GIT_TIMEOUT_SECONDS = 60.0

MB_STATE_FILE = "state.json"
#: src/hyperloom/orchestrator/enablement/artifacts.py ``write_setting_script``, under the child session directory.
MB_SETUP_SCRIPT = Path("reports") / "enablement" / "enablement_setting.sh"
#: The key of the main flow's ``--launch-info-file`` (src/hyperloom/inference_optimizer/cli ``_emit_launch_info``).
MB_LAUNCH_SESSION_DIR_KEY = "session_dir"
#: The decode ceiling of a roofline snapshot (src/hyperloom/inference_optimizer/roofline_snapshot.py).
MB_ROOFLINE_PEAK_KEY = "theoretical_peak_tok_per_sec"
#: The ``outcome_status`` of a run that measured a baseline (src/hyperloom/inference_optimizer/breakdown/stop_reasons.py).
MB_OUTCOME_COMPLETED = "completed"
#: src/hyperloom/orchestrator/enablement/lane.py: an accepted config without ``args_mode`` appends its server args.
MB_DEFAULT_ARGS_MODE = "append"
#: What enablement kept that a recipe cannot express yet, by its ``state.json`` name under ``enablement``, which is
#: also its kind in :data:`~hyperloom.contracts.recipe.UNEXPRESSED_KINDS`.
MB_UNEXPRESSED_ENABLEMENT = ("setup_commands", "kept_artifacts", "active_runtime", "environment_closure")
#: Launch controls of ``enablement.accepted_config`` a recipe's launch config cannot express yet
#: (src/hyperloom/orchestrator/enablement/revalidation.py carries them with the accepted launch); each is also its kind.
MB_UNEXPRESSED_LAUNCH = ("remove_args", "unset_envs")
#: The unexpressed kind of a non-default ``args_mode`` and of a kept patch outside the framework checkout.
ARGS_MODE_KIND = "args_mode"
PATCH_KIND = "patch"
DEFAULT_REMOTE = "origin"


class PreludeReadError(RuntimeError):
    """The child session is missing, of another schema, or holds something the root recipe cannot even mark."""


@dataclass(frozen=True)
class KeptPatch:
    """A patch enablement kept, and the source tree it applies to."""

    root: Path
    patch_file: Path

    @property
    def unexpressed(self) -> str:
        """The root recipe's mark for this patch when no component of it can carry the patch."""
        return f"{PATCH_KIND}:{self.root / self.patch_file.name}"


@dataclass(frozen=True)
class UpstreamPin:
    """Where a component's source comes from: its git remote and the commit the patches apply on."""

    repo_url: str
    ref: str


@dataclass(frozen=True)
class PreludeArtifacts:
    """What the baseline prelude left behind, as the root recipe and the session profile need it.

    ``framework_root`` is the framework source tree enablement patched, or the checkout the operator named; it is
    ``None`` when the child named neither or it is not a git checkout (an installed package). ``kept_patches`` apply to
    that checkout; ``foreign_patches`` apply to any other tree, so no component can carry them. ``unexpressed`` names,
    as recipe ``unexpressed`` entries, the foreign patches and whatever else enablement kept that a recipe cannot
    express yet. ``roofline_ceiling`` is keyed by the recorded-value metric. ``setup_script`` is the child's
    ``enablement_setting.sh``, ``None`` when it wrote none.
    """

    session_dir: Path
    stop_reason: str
    baseline_tput: float
    launch: LaunchConfig
    kept_patches: tuple[KeptPatch, ...]
    foreign_patches: tuple[KeptPatch, ...]
    base_sha_by_root: Mapping[str, str]
    framework_root: Path | None
    framework_version: str
    roofline_ceiling: Mapping[str, float]
    unexpressed: tuple[str, ...]
    setup_script: Path | None

    @property
    def succeeded(self) -> bool:
        return outcome_status(self.stop_reason, self.baseline_tput) == MB_OUTCOME_COMPLETED


def read_child_session(launch_info: Path) -> Path | None:
    """The child's session directory from its launch info; ``None`` when the child never wrote one."""

    if not launch_info.exists():
        return None
    document = _json_object(launch_info)
    session_dir = document.get(MB_LAUNCH_SESSION_DIR_KEY)
    if not isinstance(session_dir, str) or not session_dir:
        raise PreludeReadError(f"{launch_info} names no {MB_LAUNCH_SESSION_DIR_KEY}")
    return Path(session_dir)


def read_prelude(session_dir: Path) -> PreludeArtifacts:
    """Read the child session in ``session_dir``; raises :class:`PreludeReadError` on anything it cannot trust.

    Blocks: telling the framework's patches from the others runs ``git``.
    """

    state = _json_object(session_dir / MB_STATE_FILE)
    version = state.get("schema_version")
    if version != MB_LATEST_STATE_SCHEMA_VERSION:
        raise PreludeReadError(
            f"{session_dir / MB_STATE_FILE} has schema version {version!r}; this reader knows "
            f"{MB_LATEST_STATE_SCHEMA_VERSION}"
        )
    enablement = _mapping(state.get("enablement"), "enablement")
    accepted = _mapping(enablement.get("accepted_config"), "enablement.accepted_config")
    framework_root = _framework_root(state, enablement)
    checkout = None if framework_root is None else _checkout(framework_root)
    patches = _kept_patches(session_dir, enablement, framework_root)
    kept = tuple(patch for patch in patches if checkout is not None and _checkout(patch.root) == checkout)
    foreign = tuple(patch for patch in patches if patch not in kept)
    return PreludeArtifacts(
        session_dir=session_dir,
        stop_reason=str(state.get("stop_reason") or ""),
        baseline_tput=_number(state.get("baseline_tput"), "baseline_tput"),
        launch=_launch_config(accepted),
        kept_patches=kept,
        foreign_patches=foreign,
        base_sha_by_root=_text_map(enablement.get("base_sha_by_root"), "enablement.base_sha_by_root"),
        framework_root=None if checkout is None else framework_root,
        framework_version=str(state.get("framework_version") or ""),
        roofline_ceiling=_roofline_ceiling(state.get("roofline_snapshots"), state.get("baseline_roofline_ceiling")),
        unexpressed=(*_unexpressed(enablement, accepted), *(patch.unexpressed for patch in foreign)),
        setup_script=session_dir / MB_SETUP_SCRIPT if (session_dir / MB_SETUP_SCRIPT).is_file() else None,
    )


def framework_pin(artifacts: PreludeArtifacts) -> UpstreamPin | None:
    """The framework checkout's remote at the commit enablement recorded before patching; ``None`` without a checkout.

    Enablement keys that commit by the root it patched, which may name any path inside the checkout, so a key whose
    git toplevel is the checkout's matches. Without a recorded commit the checkout's ``HEAD`` is the base only when
    enablement kept no patch: with kept patches ``HEAD`` may already hold them, so that is refused.
    """

    root = artifacts.framework_root
    if root is None:
        return None
    remotes = _git(root, "remote").split()
    if not remotes:
        raise PreludeReadError(f"{root} has no git remote to pin the framework to")
    remote = DEFAULT_REMOTE if DEFAULT_REMOTE in remotes else remotes[0]
    ref = _base_sha(root, artifacts.base_sha_by_root)
    if not ref and artifacts.kept_patches:
        raise PreludeReadError(f"enablement kept patches on {root} but recorded no commit they apply on")
    return UpstreamPin(repo_url=_git(root, "remote", "get-url", remote), ref=ref or _git(root, "rev-parse", "HEAD"))


def _base_sha(root: Path, base_sha_by_root: Mapping[str, str]) -> str:
    exact = base_sha_by_root.get(str(root))
    if exact:
        return exact
    toplevel = _toplevel(root)
    for key, sha in base_sha_by_root.items():
        try:
            if sha and _toplevel(Path(key)) == toplevel:
                return sha
        except PreludeReadError:
            continue
    return ""


def root_recipe(
    artifacts: PreludeArtifacts,
    framework: str,
    pin: UpstreamPin | None,
    patches: tuple[PatchRef, ...],
    base_image_digest: str,
) -> Recipe:
    """The baseline as a recipe on ``base_image_digest``: the accepted launch config and ``framework`` pinned with the
    stored ``patches``, marked with everything the prelude kept that it leaves out.

    ``patches`` are the framework's kept patches as the experiment repository stored them, in the same order.
    """

    if patches and pin is None:
        raise PreludeReadError(f"{len(patches)} kept patches but no framework checkout to pin them to")
    try:
        components = () if pin is None else (ComponentPatches(framework, pin.repo_url, pin.ref, patches),)
        recipe = Recipe(base_image_digest=base_image_digest, launch=artifacts.launch, components=components)
        return with_unexpressed(recipe, artifacts.unexpressed)
    except RecipeError as exc:
        raise PreludeReadError(f"the prelude's result is not a valid recipe: {exc}") from exc


def _json_object(path: Path) -> Mapping[str, object]:
    try:
        document = read_json(path, strict=True)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreludeReadError(f"cannot read {path}: {exc}") from exc
    return _mapping(document, str(path))


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PreludeReadError(f"{what} is not a JSON object: {value!r}")
    return value


def _text_map(value: object, what: str) -> Mapping[str, str]:
    return {str(key): str(item) for key, item in _mapping(value, what).items()}


def _number(value: object, what: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreludeReadError(f"{what} is not a number: {value!r}")
    return float(value)


def _strings(value: object, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise PreludeReadError(f"{what} is not a list: {value!r}")
    return tuple(str(item) for item in value)


def _framework_root(state: Mapping[str, object], enablement: Mapping[str, object]) -> Path | None:
    root = str(enablement.get("framework_root") or state.get("framework_repo_path") or "")
    return Path(root) if root else None


def _launch_config(accepted: Mapping[str, object]) -> LaunchConfig:
    """The accepted launch as a recipe holds it: without the benchmark harness flags the main flow's own export
    strips (src/hyperloom/orchestrator/enablement/artifacts.py ``write_setting_script``)."""

    envs = _text_map(accepted.get("extra_envs"), "accepted_config.extra_envs")
    try:
        return LaunchConfig(
            server_args=strip_benchmark_harness_flags(str(accepted.get("extra_server_args") or "").strip()),
            envs=envs,
        )
    except RecipeError as exc:
        raise PreludeReadError(f"the accepted launch config is not a recipe launch config: {exc}") from exc


def _kept_patches(
    session_dir: Path, enablement: Mapping[str, object], framework_root: Path | None
) -> tuple[KeptPatch, ...]:
    patch_roots = _mapping(enablement.get("patch_roots"), "enablement.patch_roots")
    kept = []
    for name in _strings(enablement.get("kept_patches"), "enablement.kept_patches"):
        root = str(patch_roots.get(name) or framework_root or "")
        if not root:
            raise PreludeReadError(f"kept patch {name} names no tree it applies to")
        patch_file = session_dir / name
        if not patch_file.is_file():
            raise PreludeReadError(f"kept patch {patch_file} does not exist")
        kept.append(KeptPatch(root=Path(root), patch_file=patch_file))
    return tuple(kept)


def _roofline_ceiling(snapshots: object, baseline_ceiling: object) -> Mapping[str, float]:
    """The latest snapshot's decode ceiling, else the baseline-arm ceiling.

    The main flow keeps the newest snapshot last, and records the baseline-arm ceiling as the backup for a run whose
    profiling wrote no snapshot (src/hyperloom/orchestrator/state/shared_state.py ``record_baseline_roofline_ceiling``).
    """

    for snapshot in (*reversed(_snapshot_list(snapshots)), _mapping(baseline_ceiling, "baseline_roofline_ceiling")):
        peak = snapshot.get(MB_ROOFLINE_PEAK_KEY) if isinstance(snapshot, Mapping) else None
        if isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak > 0:
            return {PRIMARY_METRIC: float(peak)}
    return {}


def _snapshot_list(snapshots: object) -> list[object]:
    if snapshots is None:
        return []
    if not isinstance(snapshots, list):
        raise PreludeReadError(f"roofline_snapshots is not a list: {snapshots!r}")
    return snapshots


def _unexpressed(enablement: Mapping[str, object], accepted: Mapping[str, object]) -> tuple[str, ...]:
    """What enablement kept that a recipe cannot express yet, as recipe ``unexpressed`` kinds."""

    kinds = [name for name in MB_UNEXPRESSED_ENABLEMENT if enablement.get(name)]
    kinds += [name for name in MB_UNEXPRESSED_LAUNCH if accepted.get(name)]
    if str(accepted.get("args_mode") or MB_DEFAULT_ARGS_MODE) != MB_DEFAULT_ARGS_MODE:
        kinds.append(ARGS_MODE_KIND)
    return tuple(kinds)


def _checkout(root: Path) -> Path | None:
    """The git checkout ``root`` lies in; ``None`` when it lies in none."""

    try:
        return _toplevel(root)
    except PreludeReadError:
        return None


def _toplevel(root: Path) -> Path:
    try:
        return Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    except PreludeReadError as exc:
        raise PreludeReadError(f"{root} is not a git checkout, so a patch on it has no component: {exc}") from exc


def _git(root: Path, *args: str) -> str:
    """The stripped standard output of ``git -C root args``. Blocks."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args), capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_SECONDS
        )
    except subprocess.CalledProcessError as exc:
        raise PreludeReadError(f"git {' '.join(args)} in {root} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreludeReadError(f"git {' '.join(args)} in {root} failed: {exc}") from exc
    return completed.stdout.strip()


__all__ = [
    "MB_LATEST_STATE_SCHEMA_VERSION",
    "KeptPatch",
    "PreludeArtifacts",
    "PreludeReadError",
    "UpstreamPin",
    "framework_pin",
    "read_child_session",
    "read_prelude",
    "root_recipe",
]
