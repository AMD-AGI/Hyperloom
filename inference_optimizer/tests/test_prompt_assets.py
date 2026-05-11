"""Verify the system_prompts/*.md assets and the cli loader helpers.

Pairs with ``test_p0_2_roles_policy.py::test_system_prompt_files_exist_and_nonempty``
which already covers existence + role-name presence for the four canonical
roles. This module pins the additional invariants introduced when
``_DEFAULT_ORCH_PROMPT`` / ``_DEFAULT_ORCH_PROMPT_NO_KERNEL`` /
``_DEFAULT_CRITIC_PROMPT`` were migrated out of ``cli.py``:

* The two orchestration variants and ``critic.md`` exist on disk.
* The cli loader helpers return content that contains the load-bearing
  markers each prompt is expected to expose (so accidental deletions
  surface here instead of silently degrading the orchestration loop).
* ``--no-kernel`` mode loads a different prompt that omits the
  kernel-opt pipeline.
"""

from __future__ import annotations

from inference_optimizer.cli import (
    _load_critic_prompt,
    _load_orchestration_prompt,
)
from inference_optimizer.paths import asset_system_prompts_dir


# ---------------------------------------------------------------------------
# .md asset files
# ---------------------------------------------------------------------------
def test_orchestration_md_exists_on_disk():
    p = asset_system_prompts_dir() / "orchestration.md"
    assert p.is_file(), f"missing orchestration prompt: {p}"
    assert "Orchestration" in p.read_text(encoding="utf-8")


def test_orchestration_no_kernel_md_exists_on_disk():
    p = asset_system_prompts_dir() / "orchestration.no_kernel.md"
    assert p.is_file(), f"missing orchestration.no_kernel prompt: {p}"
    text = p.read_text(encoding="utf-8")
    assert "Orchestration" in text
    assert "Kernel agent is DISABLED" in text


def test_critic_md_exists_on_disk():
    p = asset_system_prompts_dir() / "critic.md"
    assert p.is_file(), f"missing critic prompt: {p}"
    assert "Critic" in p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------
def test_orchestration_loader_full_pipeline_markers():
    """Default loader (no_kernel=False) yields the full kernel-opt playbook."""
    text = _load_orchestration_prompt(no_kernel=False)
    assert "DECISION FRAMEWORK" in text
    assert "KERNEL-OPT PIPELINE" in text
    assert "step K1" in text
    assert "step K2" in text
    assert "step K3" in text
    assert "SESSION_DIR" in text
    assert "HARD RULES" in text


def test_orchestration_loader_no_kernel_drops_pipeline():
    """``--no-kernel`` variant must not advertise kernel-opt steps."""
    text = _load_orchestration_prompt(no_kernel=True)
    assert "Kernel agent is DISABLED" in text
    assert "step K1" not in text
    assert "step K2" not in text
    assert "step K3" not in text
    assert "Do NOT emit REQUEST to any agent" in text


def test_orchestration_loader_two_variants_differ():
    full = _load_orchestration_prompt(no_kernel=False)
    bare = _load_orchestration_prompt(no_kernel=True)
    assert full != bare
    assert len(full) > len(bare)


def test_critic_loader_contains_payload_contract():
    text = _load_critic_prompt()
    assert "Critic" in text
    assert "review_verdict" in text
    assert "target_proposal_msg_id" in text
    assert "verdict" in text
    assert "reasoning" in text


def test_loaders_return_text_matching_files_on_disk():
    """Loader output is byte-identical to the source asset (no munging)."""
    full = _load_orchestration_prompt(no_kernel=False)
    bare = _load_orchestration_prompt(no_kernel=True)
    critic = _load_critic_prompt()
    assert full == (asset_system_prompts_dir() / "orchestration.md").read_text(
        encoding="utf-8"
    )
    assert bare == (
        asset_system_prompts_dir() / "orchestration.no_kernel.md"
    ).read_text(encoding="utf-8")
    assert critic == (asset_system_prompts_dir() / "critic.md").read_text(
        encoding="utf-8"
    )
