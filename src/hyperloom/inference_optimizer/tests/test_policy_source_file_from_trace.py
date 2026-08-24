# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PolicyGate accepts a ``source_file`` in the shape a profile trace records it.

TraceLens names a frame as ``<relative path>(<line>): <function>``. The path is
relative to the tree under optimization and the suffix is not part of it, so a
verbatim allowlist check resolves it against the process CWD and rejects every
one of them.

This is not cosmetic: once the roofline evidence actually reaches the
orchestration prompt, the model cites those frames when it dispatches a
specialist, and the gate cancels the task before it runs. A whole session
plateaued that way with 13 of 20 specialists cancelled and no work done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.roles.agent_role import default_role_registry


def _gate(tmp_path: Path) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        session_dir=tmp_path,
        strict_paths=True,
    )


def _framework_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the session at a checkout and create the frame's file inside it."""
    tree = tmp_path / "HY-WorldPlay-e2e"
    (tree / "hyvideo" / "models" / "transformers" / "modules").mkdir(parents=True)
    (tree / "hyvideo" / "models" / "transformers" / "modules" / "attention.py").write_text(
        "def sequence_parallel_attention_vision():\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tree))
    return tree


def _dispatch_intent(source_file: str) -> Intent:
    return Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "specialist",
            "params": {
                "domain": "framework_rewrite_specialist",
                "gap_canonical_id": "local_explore",
                "source_file": source_file,
            },
        },
    )


TRACE_FRAME = "hyvideo/models/transformers/modules/attention.py(206): sequence_parallel_attention_vision"


def test_trace_frame_with_line_and_function_is_accepted(tmp_path, monkeypatch):
    """The exact string a roofline hot-kernel row carries must pass."""
    _framework_tree(tmp_path, monkeypatch)
    _gate(tmp_path).validate_intent("orchestration", _dispatch_intent(TRACE_FRAME))


def test_bare_relative_path_into_the_session_tree_is_accepted(tmp_path, monkeypatch):
    """Same frame without the annotation suffix."""
    _framework_tree(tmp_path, monkeypatch)
    _gate(tmp_path).validate_intent(
        "orchestration",
        _dispatch_intent("hyvideo/models/transformers/modules/attention.py"),
    )


def test_absolute_path_into_the_session_tree_still_accepted(tmp_path, monkeypatch):
    """The pre-existing absolute form must keep working."""
    tree = _framework_tree(tmp_path, monkeypatch)
    _gate(tmp_path).validate_intent(
        "orchestration",
        _dispatch_intent(str(tree / "hyvideo" / "models" / "transformers" / "modules" / "attention.py")),
    )


def test_relative_traversal_out_of_the_tree_is_still_denied(tmp_path, monkeypatch):
    """Resolving relative paths must not become an escape hatch."""
    _framework_tree(tmp_path, monkeypatch)
    with pytest.raises(PolicyDenied) as exc:
        _gate(tmp_path).validate_intent(
            "orchestration",
            _dispatch_intent("../../../../etc/passwd"),
        )
    assert exc.value.rule == "source_file_outside_trusted_scope"


def test_relative_path_denied_when_no_session_tree_is_named(tmp_path, monkeypatch):
    """With no checkout named, a relative path has nothing to resolve against."""
    monkeypatch.delenv("FRAMEWORK_REPO_PATH", raising=False)
    for var in ("WORLDPLAY_REPO_PATH", "WORLDPLAY_DIR", "CUSTOM_REPO_PATH", "CUSTOM_DIR"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(PolicyDenied) as exc:
        _gate(tmp_path).validate_intent("orchestration", _dispatch_intent(TRACE_FRAME))
    assert exc.value.rule == "source_file_outside_trusted_scope"


def test_absolute_path_outside_every_scope_is_still_denied(tmp_path, monkeypatch):
    """The rule this test file relaxes must still reject a real escape."""
    _framework_tree(tmp_path, monkeypatch)
    with pytest.raises(PolicyDenied) as exc:
        _gate(tmp_path).validate_intent("orchestration", _dispatch_intent("/etc/passwd"))
    assert exc.value.rule == "source_file_outside_trusted_scope"


# Placeholder/vendor-label forms TraceLens can leave in source_file instead of
# an empty string (test_source_resolution_guards.py _SENTINELS covers the
# producer side; this covers the gate degrading them to an omitted field
# rather than denying the whole delegate as a bogus path).
_ABSENT_SENTINELS = (
    "Not found",
    "N/A",
    "none",
    "unknown",
    "TBD",
    "<unresolved>",
    "AITER (vendor)",
    "Triton (vendor)",
)


@pytest.mark.parametrize("sentinel", _ABSENT_SENTINELS)
def test_absent_value_sentinel_is_accepted_not_denied(tmp_path, monkeypatch, sentinel):
    """A known placeholder degrades the delegate gracefully instead of denying it."""
    _framework_tree(tmp_path, monkeypatch)
    _gate(tmp_path).validate_intent("orchestration", _dispatch_intent(sentinel))


def test_sentinel_match_is_case_insensitive(tmp_path, monkeypatch):
    _framework_tree(tmp_path, monkeypatch)
    _gate(tmp_path).validate_intent("orchestration", _dispatch_intent("NOT FOUND"))


def test_sentinel_pass_through_is_logged(tmp_path, monkeypatch, caplog):
    """A sentinel-driven accept must be visible in logs, not indistinguishable
    from a normal accept -- previously this branch returned silently.
    """
    _framework_tree(tmp_path, monkeypatch)
    with caplog.at_level("INFO", logger="hyperloom.orchestrator.policy.gate"):
        _gate(tmp_path).validate_intent("orchestration", _dispatch_intent("Not found"))
    assert any("absent-value sentinel" in record.message and "Not found" in record.message for record in caplog.records)
