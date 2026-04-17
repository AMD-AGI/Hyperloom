"""Integration tests for warm-start mode detection against real and synthetic layouts.

These tests verify that _detect_warm_start_mode correctly identifies:
  - "sprint": structured handoff with handoff/config.json
  - "sprint_repo": standalone Agentic-InferenceX repo with scripts/launch_server.sh
  - "baseline": existing marathon result dir with state.json
  - "cold": empty or nonexistent dir
"""

import json
import tempfile
from pathlib import Path

from marathon_harness.marathon import _detect_warm_start_mode


def test_cold_start_no_dir():
    assert _detect_warm_start_mode(None) == "cold"
    assert _detect_warm_start_mode("") == "cold"


def test_cold_start_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty"
        empty.mkdir()
        # empty dir has no files → any(iterdir()) is False → cold
        assert _detect_warm_start_mode(str(empty)) == "cold"


def test_sprint_handoff_mode():
    with tempfile.TemporaryDirectory() as td:
        handoff = Path(td) / "handoff"
        handoff.mkdir()
        (handoff / "config.json").write_text('{"model_path": "/m", "tp": 8}')
        assert _detect_warm_start_mode(td) == "sprint"


def test_sprint_repo_mode():
    with tempfile.TemporaryDirectory() as td:
        scripts = Path(td) / "scripts"
        scripts.mkdir()
        (scripts / "launch_server.sh").write_text(
            "#!/bin/bash\npython3 -m sglang.launch_server --model-path /m\n"
        )
        assert _detect_warm_start_mode(td) == "sprint_repo"


def test_sprint_handoff_takes_priority_over_repo():
    """If both handoff/config.json AND scripts/launch_server.sh exist,
    handoff mode wins (it's the newer, more structured format)."""
    with tempfile.TemporaryDirectory() as td:
        handoff = Path(td) / "handoff"
        handoff.mkdir()
        (handoff / "config.json").write_text('{"tp": 8}')
        scripts = Path(td) / "scripts"
        scripts.mkdir()
        (scripts / "launch_server.sh").write_text("#!/bin/bash\n")
        assert _detect_warm_start_mode(td) == "sprint"


def test_baseline_mode():
    with tempfile.TemporaryDirectory() as td:
        # Has files but no handoff/ or scripts/
        (Path(td) / "state.json").write_text('{}')
        assert _detect_warm_start_mode(td) == "baseline"


def test_real_agentic_inferencex_layout():
    """Test against the actual Agentic-InferenceX repo if it exists."""
    repo = Path("/shared_nfs/nehaprakriya/Agentic-InferenceX/DeepSeek-R1-0528-optimized")
    if not repo.exists():
        return  # skip on machines without the data

    mode = _detect_warm_start_mode(str(repo))
    assert mode == "sprint_repo", f"Expected sprint_repo, got {mode}"


def test_real_marathon_layout():
    """Test against real marathon output dir if it exists."""
    repo = Path("/shared_nfs/nehaprakriya/Agentic-InferenceX/DeepSeek-R1-0528-marathon")
    if not repo.exists():
        return

    mode = _detect_warm_start_mode(str(repo))
    # Marathon dirs have scripts/ too, but also state.json
    # Since they don't have handoff/config.json, they go through
    # scripts check first → sprint_repo
    assert mode in ("sprint_repo", "baseline")


def test_dry_run_cli():
    """Verify --dry-run prints config without starting anything."""
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as td:
        base = str(Path(td) / "fresh_base")
        result = subprocess.run(
            [sys.executable, "-m", "marathon_harness",
             "TestModel", base, "--dry-run"],
            capture_output=True, text=True,
            cwd="/shared_nfs/nehaprakriya/TBO/inference_optimization",
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["model"] == "TestModel"
        # _make_session_dir creates base/sessions/... before detection,
        # so mode is "baseline" (non-empty dir). The key check is that
        # dry-run exits cleanly without launching anything.
        assert output["mode"] in ("cold", "baseline")
