"""``fa agent`` subcommand entry point (P2 PR-D skeleton).

Registered into the top-level ``fa`` parser by
:mod:`framework_agent.runtime.cli` so that existing ``fa schema`` /
``fa candidates`` / ``fa explore`` / ``fa kb`` commands keep working
side-by-side.

Two stages, used by :class:`FrameworkAgentBackend` in
``inference_optimizer``:

* ``fa agent prepare-task --task task.json --output-bundle bundle.json``
  Stage A: read the task descriptor, run AST scan when enabled (PR-E
  fills the scanner), package the LLM bundle. P2 PR-D skeleton: returns
  an empty bundle so the subprocess contract is testable without
  libcst.
* ``fa agent commit-result --envelope envelope.json --task-id <id>``
  Stage B: validate the envelope against §4.6 jsonschema and persist it
  to ``runs/framework/<task_id>/envelope.json`` (path derived from
  ``--session-dir`` -- defaults to the cwd's ``runs/framework/<task_id>/``
  if not provided).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .envelope import EnvelopeValidationError, validate_envelope


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file. Raises ValueError with a precise message on parse failure."""
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"bad JSON in {path}: {exc}") from exc


def _emit_json(obj: dict[str, Any], path: Path | None) -> None:
    """Write ``obj`` to ``path`` (creating parents) and echo to stdout."""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Stage A: prepare-task
# ---------------------------------------------------------------------------
def cmd_prepare_task(args: argparse.Namespace) -> None:
    """Read a task descriptor and emit an LLM bundle.

    Task schema (PR-D minimum):

        {
            "task_id":            "fw-20260520-deadbeef",
            "kind":               "framework_optimize" | "framework_integrate",
            "session_dir":        "/workspace/hyperloom",
            "target_framework":   "vllm" | "sglang",
            "ast_scan_enabled":   true,
            "ast_frameworks":     ["sglang"],
            "kb_partition":       "framework_optimization"
        }

    PR-D bundle (skeleton; PR-E fills ast_findings; PR-G fills kb_priors):

        {
            "bundle_version":   "1",
            "task":             <verbatim task input>,
            "ast_findings":     null,
            "kb_priors":        null,
            "prompt":           "...",
            "prepared_at_ms":   <epoch ms>
        }
    """
    task_path = Path(args.task).expanduser().resolve()
    out_path = (
        Path(args.output_bundle).expanduser().resolve()
        if args.output_bundle else None
    )
    try:
        task = _load_json(task_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: prepare-task failed to load --task: {exc}",
              file=sys.stderr)
        sys.exit(2)
    kind = str(task.get("kind") or "").strip()
    if kind not in ("framework_optimize", "framework_integrate"):
        print(
            f"ERROR: prepare-task: task.kind={kind!r} not in "
            "{'framework_optimize','framework_integrate'}",
            file=sys.stderr,
        )
        sys.exit(2)

    bundle: dict[str, Any] = {
        "bundle_version": "1",
        "task": task,
        "ast_findings": None,    # PR-E fills
        "kb_priors": None,       # PR-G fills
        "prompt": f"<placeholder prompt for {kind}>",  # PR-G writes real prompt
        "prepared_at_ms": int(time.time() * 1000),
    }
    _emit_json(bundle, out_path)


# ---------------------------------------------------------------------------
# Stage B: commit-result
# ---------------------------------------------------------------------------
def cmd_commit_result(args: argparse.Namespace) -> None:
    """Validate and persist a RESPONSE envelope.

    Persists to ``<session_dir>/runs/framework/<task_id>/envelope.json``
    if ``--session-dir`` and ``--task-id`` are provided; otherwise the
    envelope is only echoed to stdout (useful for tests / dry-runs).
    """
    env_path = Path(args.envelope).expanduser().resolve()
    try:
        envelope = _load_json(env_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: commit-result failed to load --envelope: {exc}",
              file=sys.stderr)
        sys.exit(2)
    try:
        validate_envelope(envelope)
    except EnvelopeValidationError as exc:
        print(f"ERROR: commit-result envelope invalid: {exc}",
              file=sys.stderr)
        sys.exit(2)

    persist_path: Path | None = None
    if args.session_dir and args.task_id:
        persist_path = (
            Path(args.session_dir).expanduser().resolve()
            / "runs" / "framework" / args.task_id / "envelope.json"
        )
    _emit_json(envelope, persist_path)


# ---------------------------------------------------------------------------
# Subparser registration (called by runtime/cli.py)
# ---------------------------------------------------------------------------
def register_subparser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``agent`` subcommand to the top-level ``fa`` parser.

    Keeps the wiring out of :mod:`framework_agent.runtime.cli` so that
    PR-D adds only an import + one ``register_subparser`` call there.
    """
    agent_p = sub.add_parser(
        "agent",
        help="Sibling-skill subprocess used by inference_optimizer's "
             "FrameworkAgentBackend (P2+). prepare-task / commit-result.",
    )
    agent_sub = agent_p.add_subparsers(dest="agent_cmd", required=True)

    prep_p = agent_sub.add_parser(
        "prepare-task",
        help="Stage A: read task.json -> LLM bundle.",
    )
    prep_p.add_argument("--task", required=True,
                        help="Path to the task descriptor JSON.")
    prep_p.add_argument("--output-bundle", default="",
                        help="Path to write the LLM bundle. Empty = stdout only.")
    prep_p.set_defaults(func=cmd_prepare_task)

    commit_p = agent_sub.add_parser(
        "commit-result",
        help="Stage B: validate + persist a RESPONSE envelope.",
    )
    commit_p.add_argument("--envelope", required=True,
                          help="Path to the envelope JSON to validate + persist.")
    commit_p.add_argument("--task-id", default="",
                          help="Task id used to derive the persist path "
                               "<session_dir>/runs/framework/<task_id>/.")
    commit_p.add_argument("--session-dir", default="",
                          help="Session dir root for persisting. Empty = "
                               "validate only, echo to stdout, no disk write.")
    commit_p.set_defaults(func=cmd_commit_result)


__all__ = ["cmd_commit_result", "cmd_prepare_task", "register_subparser"]
