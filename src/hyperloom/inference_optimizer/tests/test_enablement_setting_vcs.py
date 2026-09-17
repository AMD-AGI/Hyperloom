# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The script's VCS kind must describe the root the script names."""

from __future__ import annotations

import subprocess
from pathlib import Path
from hyperloom.orchestrator.enablement import artifacts as art
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


def test_the_vcs_kind_follows_the_corrected_root(tmp_path: Path, monkeypatch):
    """Correcting the root while announcing the old root's VCS is unusable.

    A git checkout corrected to an installed package that still announces "git"
    emits ``git -C "$FRAMEWORK_ROOT" apply`` against a tree with no repository.
    """
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(site / "aiter")], check=True)

    session = tmp_path / "sess"
    round_dir = art.enablement_round_dir(session, "r1") / "patches"
    round_dir.mkdir(parents=True)
    (round_dir / "p.patch").write_text(
        "--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-x\n+y\n", encoding="utf-8"
    )

    seen: dict[str, object] = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return "#!/usr/bin/env bash\n"

    import hyperloom.inference_optimizer.reference_script as rs

    monkeypatch.setattr(rs, "render_reference_script", _capture)

    enablement = EnablementRound()
    enablement.framework_root = str(site / "aiter")
    enablement.kept_rounds = [{"task_id": "r1", "patches": ["p.patch"], "artifacts": []}]
    art.write_setting_script(session, enablement, "vllm")

    assert seen.get("framework_root") == str(site / "vllm"), "the script must name the patches' tree"
    assert seen.get("framework_root_vcs") != "git", "and describe that tree, not the one it replaced"
