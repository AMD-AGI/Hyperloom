from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.orchestrator.knowledge.recipe_kb.replay_bundle import (
    argv_to_env_string,
    build_replay_bundle,
    canonical_server_argv,
    externalize_large_artifacts,
    hydrate_replay_bundle,
    replay_patches,
)
from hyperloom.orchestrator.source_snapshot import snapshot_source_layer


def test_server_argv_removes_shell_quotes_but_preserves_json() -> None:
    raw = """--compilation-config '{"cudagraph_mode":"FULL"}' --max-num-seqs 64"""
    argv = canonical_server_argv(raw)
    assert argv == [
        "--compilation-config",
        '{"cudagraph_mode":"FULL"}',
        "--max-num-seqs",
        "64",
    ]
    assert argv_to_env_string(argv) == (
        '--compilation-config {"cudagraph_mode":"FULL"} --max-num-seqs 64'
    )


def test_config_only_bundle_is_atomic_and_replayable() -> None:
    bundle = build_replay_bundle(
        env_spec={
            "config": {
                "extra_server_args": "--kv-cache-dtype fp8",
                "extra_envs": {"USE_AITER": "1"},
            },
            "source_snapshots": [],
        },
        producer_session_id="session-a",
        baseline_throughput=100.0,
        optimized_throughput=120.0,
        workload={"conc": 64},
    )
    assert bundle["replayable"] is True
    assert bundle["config"]["argv"] == ["--kv-cache-dtype", "fp8"]
    assert bundle["measurement"]["gain_pct"] == 20.0
    assert bundle["source_artifacts"] == []


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_source_snapshot_becomes_squashed_unified_diff(tmp_path: Path) -> None:
    repo = tmp_path / "aiter"
    repo.mkdir()
    _git(repo, "init")
    target = repo / "configs" / "qwen.csv"
    target.parent.mkdir()
    target.write_text("M,N,K\n32,1,2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base_sha = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    target.write_text("M,N,K\n32,1,2\n64,1,2\n", encoding="utf-8")
    snapshot = snapshot_source_layer(
        framework_root=repo,
        base_sha=base_sha,
        rel_paths=["configs/qwen.csv"],
        dest_dir=tmp_path / "snapshot",
        provenance="layer-a",
    )
    assert snapshot is not None

    bundle = build_replay_bundle(
        env_spec={
            "config": {"extra_server_args": "", "extra_envs": {}},
            "source_snapshots": [
                {
                    "id": "layer-a",
                    "snapshot_dir": snapshot["snapshot_dir"],
                    "base_sha": base_sha,
                }
            ],
        },
        producer_session_id="session-a",
        baseline_throughput=100.0,
        optimized_throughput=130.0,
    )
    assert bundle["replayable"] is True
    artifact = bundle["source_artifacts"][0]
    assert artifact["repo"] == "aiter"
    assert "+64,1,2" in artifact["patch_content"]
    assert replay_patches(bundle)[0]["target_repo"] == "aiter"
    _git(repo, "reset", "--hard", base_sha)
    patch_path = tmp_path / "bundle.diff"
    patch_path.write_text(artifact["patch_content"], encoding="utf-8")
    subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert "+64,1,2" not in target.read_text(encoding="utf-8")
    assert "64,1,2" in target.read_text(encoding="utf-8")


class _FakeMcp:
    def __init__(self) -> None:
        self.pages: dict[str, str] = {}

    def call(self, tool: str, args: dict):
        if tool == "put_page":
            self.pages[args["slug"]] = args["content"]
            return {"ok": True}
        if tool == "get_page":
            content = self.pages.get(args["slug"], "")
            body = content.split("---\n", 2)[-1].lstrip("\n") if content else ""
            return {"compiled_truth": body} if content else None
        raise AssertionError(tool)


def test_large_artifact_round_trips_through_gbrain_page() -> None:
    patch = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    import hashlib

    bundle = {
        "schema_version": 1,
        "replayable": True,
        "source_artifacts": [
            {
                "storage": "inline",
                "sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "bytes": len(patch),
                "patch_content": patch,
            }
        ],
    }
    mcp = _FakeMcp()
    external = externalize_large_artifacts(
        mcp,
        canonical_id="inference:test",
        bundle=bundle,
        inline_max_bytes=1,
    )
    artifact = external["source_artifacts"][0]
    assert artifact["storage"] == "gbrain_page"
    assert "patch_content" not in artifact
    hydrated = hydrate_replay_bundle(mcp, external)
    assert hydrated["replayable"] is True
    assert hydrated["source_artifacts"][0]["patch_content"] == patch


def test_hydration_fails_closed_when_artifact_is_missing() -> None:
    hydrated = hydrate_replay_bundle(
        _FakeMcp(),
        {
            "schema_version": 1,
            "replayable": True,
            "source_artifacts": [
                {
                    "storage": "gbrain_page",
                    "artifact_slug": "missing",
                    "sha256": "deadbeef",
                }
            ],
        },
    )
    assert hydrated["replayable"] is False
    assert hydrated["reason"] == "artifact_missing"
