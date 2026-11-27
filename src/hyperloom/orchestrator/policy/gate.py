# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PolicyGate — the single chokepoint every parsed Intent passes before any side effect.

The gate decides synchronously from the intent and static config, and makes no
database call. A rule that needs a resource -- the bring-up round holding the
machine, the pool a specialist leases its cards from -- is judged by the
admission step that holds that resource.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.framework_paths import resolved_within
from hyperloom.common.visible_devices import detect_gpu_count
from hyperloom.inference_optimizer.protocol.intent import EXTEND_LEASE_MAX_SEC, Intent, IntentType
from hyperloom.inference_optimizer.protocol.action_surfaces import COORDINATOR_OWNED_KERNEL_REQUEST_KINDS
from ..mcp_tool_names import qualified_context_tool_name
from ..specialists.domains import normalize_dispatch_tags
from .custom_specialist import deny_coordinator_dispatched_delegate
from .denial import PolicyDenied
from ..specialists.profile import SCOPE_FREEFORM as SPECIALIST_SCOPE_FREEFORM
from .validators import (
    log_specialist_dispatch_coherence,
    validate_freeform_task_description,
    validate_gemm_tuning_request,
    validate_prune_branch,
    validate_specialist_framework,
)


# Specialist dispatch action name.
SPECIALIST_ACTION_NAME: str = "specialist"


# Specialist / Explore parallelism caps — single source of truth across layers.
# Research-lane ceiling fallback used when the GPU count cannot be probed.
RESEARCH_LANE_CEILING_FALLBACK: int = 2


def research_lane_ceiling() -> int:
    """Dynamic ceiling on concurrent research-lane specialists (``2 × GPU``; falls back to :data:`RESEARCH_LANE_CEILING_FALLBACK`).

    Returns:
        int: twice the detected GPU count, or
            :data:`RESEARCH_LANE_CEILING_FALLBACK` when no GPUs can be probed.
    """
    gpus = detect_gpu_count()
    if gpus > 0:
        return 2 * gpus
    return RESEARCH_LANE_CEILING_FALLBACK


# Specialist task identity prefix: the dispatcher stamps ``specialist:<task_id>`` as the
# source of a specialist result (explore parses the task_id back out), and a send_message
# addressed to ``specialist:<task_id>`` is delivered to that specialist's inbox.
SPECIALIST_FROM_AGENT_PREFIX: str = "specialist:"


# R5 — external tool names a REQUEST may not carry as its kind.

#: PR Monitor *readonly* surfaces.
PR_MONITOR_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "mcp__pr_monitor__pr_repos_list",
        "mcp__pr_monitor__pr_repo_stats",
        "mcp__pr_monitor__pr_list",
        "mcp__pr_monitor__pr_get",
        "mcp__pr_monitor__pr_files",
        "mcp__pr_monitor__pr_file_patch",
        "mcp__pr_monitor__pr_patches",
        "mcp__pr_monitor__pr_blob",
        "mcp__pr_monitor__pr_commit_files",
        "mcp__pr_monitor__pr_commit_file",
        "mcp__pr_monitor__pr_pr_file_baseline",
        "mcp__pr_monitor__pr_search",
    }
)

#: Web tools. The orchestration agent holds them as tools, never as action names.
WEB_TOOL_NAMES: frozenset[str] = frozenset({"WebSearch", "WebFetch"})


# The agents a REQUEST may name as its ``target_agent``.
REQUEST_TARGETS: frozenset[str] = frozenset({"kernel_agent"})


# SESSION_DIR path containment: PATH_LIKE_FIELDS must point inside session_dir
# (checked recursively). SOURCE_LIKE_FIELDS are exempt -- they name framework
# source, which lives outside it by construction.
PATH_LIKE_FIELDS: frozenset[str] = frozenset(
    {
        "trace_input",
        "candidates_path",
        "patch_path",
        "target_file",
        "config_path",
        "output_dir",
        "workspace",
        "workspace_path",
        "trace_dir",
        "main_trace_path",
        "report_path",
        "json_path",
        "md_path",
        "session_dir",
        "manifest_path",
    }
)

# Exempt from path validation: where a patch may land is decided by the
# integration step that applies it.
SOURCE_LIKE_FIELDS: frozenset[str] = frozenset({"source_file", "framework_source_root"})


# Multi-node profile trace dirs live outside session_dir but must be referenceable by trace_dir / main_trace_path / trace_input (runtime-resolved).
def _trace_path_allowlist() -> tuple[str, ...]:
    """Multi-node profile trace path allowlist (runtime-resolved).

    Returns:
        tuple[str, ...]: a single-element tuple holding the multi-node profile
            trace root, normalized with a trailing ``/``. Boundary safety is
            enforced by :func:`resolved_within`, not by the trailing slash.
    """
    from hyperloom.inference_optimizer.session.paths import mn_profile_trace_root

    root = str(mn_profile_trace_root()).rstrip("/") + "/"
    return (root,)


# Subset of PATH_LIKE_FIELDS that also accept :func:`_trace_path_allowlist` (others stay strictly session-rooted).
TRACE_PATH_LIKE_FIELDS: frozenset[str] = frozenset(
    {
        "trace_dir",
        "main_trace_path",
        "trace_input",
    }
)


@dataclass
class PolicyGate:
    """Validate every intent emitted by an agent reactor.

    With a ``session_dir``, PATH_LIKE_FIELDS must resolve under it.
    """

    session_dir: Path | None = None

    # Public API
    def validate_intent(self, from_agent: str, intent: Intent) -> None:
        """Raise :class:`PolicyDenied` if the intent is not allowed (structural checks first, then paths).

        Args:
            from_agent (str): the identity of the emitting agent.
            intent (Intent): the parsed intent to validate.

        Raises:
            PolicyDenied: when the intent is not permitted; the ``rule``
                attribute identifies which guard fired.
        """
        payload = intent.payload or {}

        # Per-intent structural validators
        if intent.type == IntentType.DELEGATE:
            self._validate_delegate(payload)
        elif intent.type == IntentType.REQUEST:
            self._validate_request(from_agent, payload)
        elif intent.type == IntentType.EXTEND_LEASE:
            self._validate_extend_lease(payload)
        elif intent.type == IntentType.PRUNE_BRANCH:
            validate_prune_branch(payload)
        # ALERT carries no structural check.

        # Path-containment guard for PATH_LIKE_FIELDS in the payload.
        self.validate_payload_paths(from_agent, intent.type, payload)

    # Per-intent validators
    def _validate_delegate(self, payload: dict[str, Any]) -> None:
        """Validate a ``DELEGATE`` intent: it names an action, and a specialist dispatch keeps its contract.

        Args:
            payload: Delegate payload with ``action_name`` and optional ``params``.
        """
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("delegate intent missing action_name", rule="payload")
        # ``specialist`` bypasses the catalogue; ``_validate_specialist_dispatch`` owns its contract.
        if action_name == SPECIALIST_ACTION_NAME:
            params = payload.get("params")
            if isinstance(params, dict):
                deny_coordinator_dispatched_delegate(params)
            self._validate_specialist_dispatch(payload)

    def _validate_request(self, from_agent: str, payload: dict[str, Any]) -> None:
        """Validate a ``REQUEST`` intent.

        Checks that ``target_agent`` is in :data:`REQUEST_TARGETS`, that
        ``kind`` is present and is not a Coordinator-owned lane. GEMM-tuning
        ownership and external-tool collision guards are applied to the kind
        as well. Refused here, such a request is a denial; past the gate the
        router would answer it with a failed reply, which the failure ledger
        and the recipe's ``what_failed`` record.

        Args:
            from_agent (str): the emitting agent, named in the denial.
            payload (dict[str, Any]): the request payload, expected to
                carry ``target_agent`` and ``kind``.

        Returns:
            None: returns silently when the request is permitted.

        Raises:
            PolicyDenied: if the target is missing/disallowed, ``kind`` is
                missing, or one of the applied action guards fires.
        """
        target = str(payload.get("target_agent", "")).strip()
        if not target:
            raise PolicyDenied("request missing target_agent", rule="payload")
        if target not in REQUEST_TARGETS:
            raise PolicyDenied(
                f"role={from_agent!r} cannot request target_agent={target!r} (allowed: {sorted(REQUEST_TARGETS)!r})",
                rule="request_target",
            )
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied("request missing kind", rule="payload")
        if kind in COORDINATOR_OWNED_KERNEL_REQUEST_KINDS:
            raise PolicyDenied(
                f"request kind {kind!r} is a Coordinator-owned kernel lane and not LLM-requestable",
                rule="request_kind",
                hint=(
                    "the lane runs at KERNEL entry once its own gate passes, "
                    "targeted from the nomination and bounded by the lane budget; "
                    "its outcome arrives as <kind>_done. Wait for that event and "
                    "`integrate` the KEEPs it queues."
                ),
            )
        validate_gemm_tuning_request(kind)
        self._validate_tool_whitelist_collision(from_agent, kind)

    # R5 — tool_whitelist_role
    def _validate_tool_whitelist_collision(self, role_name: str, kind: str) -> None:
        """Reject a REQUEST kind that names an external tool.

        Args:
            role_name (str): the name of the emitting role.
            kind (str): the request kind being checked.

        Raises:
            PolicyDenied: when the kind is a PR Monitor or web tool.
        """
        if kind not in PR_MONITOR_TOOL_NAMES | WEB_TOOL_NAMES:
            return
        raise PolicyDenied(
            f"role={role_name!r} cannot invoke tool {kind!r}",
            rule="tool_whitelist_role",
            hint=(
                f"Tool {kind!r} is restricted to "
                f"specialist sub-agents as an action name. The "
                f"orchestration agent reaches KB / PR Monitor through the "
                f"Coordinator-mediated KnowledgePlane facade instead, and "
                f"additionally holds WebSearch / WebFetch "
                f"directly as tools."
            ),
        )

    def _validate_specialist_dispatch(self, payload: dict[str, Any]) -> None:
        """Enforce the specialist-delegate contract (Inv-11.2): an anchored dispatch names its gap, a freeform one its task.

        Args:
            payload (dict[str, Any]): the delegate intent payload carrying
                ``params`` (tags, scope, gap_canonical_id, task_description, ...).

        Raises:
            PolicyDenied: when params are malformed or name an unregistered
                framework, the gap id is missing, or a freeform dispatch has no
                task description. Tag / scope incoherence is logged rather than denied.
        """
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise PolicyDenied(
                "delegate{action='specialist'}: params must be a dict",
                rule="specialist_dispatch_source",
                hint="pass params={tags, gap_canonical_id, ...} per §3.5 §6",
            )
        validate_specialist_framework(params)

        scope_raw = str(params.get("scope") or "").strip().lower()
        # ``params.tags`` is canonical; ``params.domain`` is accepted as a single-tag alias.
        tags = normalize_dispatch_tags(params)
        # scope='freeform' has no domain anchor, and a bare dispatch (no scope, no
        # anchor) defaults to that cheap lane: both skip the tag / gap vocabulary
        # checks and still require a task_description.
        if scope_raw == SPECIALIST_SCOPE_FREEFORM or not (scope_raw or tags):
            validate_freeform_task_description(params)
            return

        # Observed, not enforced: resolve_specialist_profile re-infers the scope
        # and the runner synthesizes an empty result for an unresolvable anchor.
        log_specialist_dispatch_coherence(scope_raw, tags)

        if not str(params.get("gap_canonical_id") or "").strip():
            raise PolicyDenied(
                "delegate{action='specialist'}: params.gap_canonical_id required",
                rule="specialist_dispatch_source",
                hint=(
                    "Pass the canonical_id of the gap to work on: one the '=== Current gaps ===' block lists, "
                    "or a stable id of your own for a problem it does not list. Or dispatch scope='freeform' "
                    "with a task_description."
                ),
            )

    def _validate_extend_lease(self, payload: dict[str, Any]) -> None:
        """Validate an ``EXTEND_LEASE`` intent.

        The per-step bound is static config; ``TaskRegistry.extend_lease``
        decides whether the task still holds a lease, and applies the extension.

        Args:
            payload: The payload carrying ``task_id``, ``extra_sec`` and an
                optional ``reason``.

        Raises:
            PolicyDenied: When ``extra_sec`` exceeds :data:`EXTEND_LEASE_MAX_SEC`.
        """
        running_tasks = qualified_context_tool_name("get_running_tasks")
        extra_sec = int(payload["extra_sec"])
        if extra_sec > EXTEND_LEASE_MAX_SEC:
            raise PolicyDenied(
                f"extend_lease extra_sec={extra_sec} outside (0, {EXTEND_LEASE_MAX_SEC}]",
                rule="extend_lease_bounds",
                hint=(
                    f"Extend at most {EXTEND_LEASE_MAX_SEC}s per call; call again while the task runs to extend "
                    f"further. Re-check {running_tasks} between steps."
                ),
            )

    def _path_under_session(self, value: str) -> bool:
        """Return whether a path resolves inside the active session_dir.

        Args:
            value (str): the path string to test.

        Returns:
            bool: True when ``value`` resolves to or under the session
                directory; False if it escapes or cannot be resolved.
        """
        try:
            sd = self.session_dir.resolve()
            v = Path(str(value)).resolve()
        except (OSError, RuntimeError):
            return False
        return v == sd or v.is_relative_to(sd)

    def _path_in_trace_allowlist(self, value: str) -> bool:
        """Match a value against runtime-resolved trace path prefixes (multi-node shared profile dir outside session_dir).

        Args:
            value (str): the path string to test.

        Returns:
            bool: True when ``value`` resolves to or under any runtime-resolved
                trace path root, else False.
        """
        return any(resolved_within(value, p) for p in _trace_path_allowlist())

    def validate_payload_paths(self, from_agent: str, intent_type: IntentType, payload: dict[str, Any]) -> None:
        """Walk payload (recursively); reject path-like values escaping session_dir. No-op when session_dir is None.

        Run on every intent, and on the task params the Coordinator builds from
        what a specialist delivered.

        Args:
            from_agent (str): the author of the payload, used in error messages.
            intent_type (IntentType): the intent type, used in error messages.
            payload (dict[str, Any]): the payload to walk for path-like fields.

        Raises:
            PolicyDenied: when a path-like value escapes session_dir and the
                trace allowlist that a trace field may also resolve under.
        """
        if self.session_dir is None:
            return

        def visit(node: Any, path_keys: tuple[str, ...]) -> None:
            """Recursively scan a payload node for escaping path values.

            Args:
                node (Any): the current payload node (dict, list/tuple,
                    string, or scalar) being walked.
                path_keys (tuple[str, ...]): the chain of dict keys leading
                    to ``node``; its last element is the field name used to
                    decide which allowlist applies.

            Returns:
                None.

            Raises:
                PolicyDenied: when a path-like string escapes the session
                    directory and its applicable allowlists.
            """
            if isinstance(node, dict):
                for k, v in node.items():
                    visit(v, path_keys + (str(k),))
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    visit(item, path_keys)
                return
            if not isinstance(node, str) or not node.strip():
                return
            key = path_keys[-1] if path_keys else ""
            if key in SOURCE_LIKE_FIELDS:
                return
            if key not in PATH_LIKE_FIELDS:
                return
            self._check_payload_path(from_agent, intent_type, key, node)

        visit(payload, ())

    def _check_payload_path(self, from_agent: str, intent_type: IntentType, key: str, node: str) -> None:
        """Reject one path-like payload value escaping session_dir and every allowlist its field may use."""
        if self._path_under_session(node):
            return
        # Multi-node profile traces live outside session_dir; allow trace-input fields against the trace allowlist.
        if key in TRACE_PATH_LIKE_FIELDS and self._path_in_trace_allowlist(node):
            return
        raise PolicyDenied(
            f"role={from_agent!r} {intent_type.value} payload field "
            f"{key!r}={node!r} escapes session_dir={self.session_dir!s}",
            rule="path_outside_session_dir",
            hint=(
                "emit paths verbatim from SharedState (e.g. "
                "last_profile_trace) or under SESSION_DIR; "
                "multi-node trace fields may also resolve under "
                f"{list(_trace_path_allowlist())!r}"
            ),
        )


__all__ = [
    "PATH_LIKE_FIELDS",
    "PolicyDenied",
    "PolicyGate",
    "TRACE_PATH_LIKE_FIELDS",
    "SOURCE_LIKE_FIELDS",
]
