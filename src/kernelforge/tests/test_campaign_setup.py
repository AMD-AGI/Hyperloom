"""Unit tests for the campaign initialization helpers."""

from __future__ import annotations

import subprocess

import pytest

from kernelforge.loop.campaign_setup import parse_list, resolve_campaign
from kernelforge.loop.campaign_config import CampaignConfigStore


class TestParseList:
    def test_empty_string_returns_empty(self):
        assert parse_list("") == []

    def test_comma_separated(self):
        assert parse_list("a,b,c") == ["a", "b", "c"]

    def test_newline_separated(self):
        assert parse_list("a\nb\nc") == ["a", "b", "c"]

    def test_mixed_separators(self):
        assert parse_list("a,b\nc") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert parse_list(" a , b ") == ["a", "b"]

    def test_skips_blank_entries(self):
        assert parse_list("a,,b") == ["a", "b"]


def _git_workspace(tmp_path, name="workspace"):
    workspace = tmp_path / name
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "-b", "feature/test-campaign"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tests"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("def k(): return 1\n")
    driver.write_text("pass\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return workspace, kernel, driver


def _base_args(workspace, kernel, driver):
    return dict(
        workspace_dir=str(workspace),
        resume=False,
        prepare_task=False,
        kernel=str(kernel),
        driver=str(driver),
        kernel_backend="triton",
        snr_threshold=2.0,
    )


class TestResolveCampaign:
    @pytest.fixture(autouse=True)
    def _gpu_target(self, monkeypatch):
        monkeypatch.setenv("GPU_TARGET", "gfx950")

    def test_fresh_campaign_creates_and_saves_config(self, tmp_path):
        workspace, kernel, driver = _git_workspace(tmp_path)
        r = resolve_campaign(**_base_args(workspace, kernel, driver))
        assert r.campaign is not None
        assert r.save_deferred is False
        assert CampaignConfigStore(str(workspace)).exists()

    def test_prepare_task_defers_save(self, tmp_path):
        workspace, kernel, driver = _git_workspace(tmp_path)
        args = _base_args(workspace, kernel, driver)
        args["prepare_task"] = True
        r = resolve_campaign(**args)
        assert r.save_deferred is True
        assert not CampaignConfigStore(str(workspace)).exists()

    def test_resume_with_existing_config_returns_it(self, tmp_path):
        workspace, kernel, driver = _git_workspace(tmp_path)
        r1 = resolve_campaign(**_base_args(workspace, kernel, driver))
        stored_sha = r1.campaign.driver_sha256

        r2 = resolve_campaign(
            workspace_dir=str(workspace),
            resume=True,
            prepare_task=False,
            kernel="",
            driver="",
            snr_threshold=2.0,
        )
        assert r2.campaign.driver_sha256 == stored_sha
        assert r2.save_deferred is False

    def test_resume_with_extra_inputs_raises(self, tmp_path):
        workspace, kernel, driver = _git_workspace(tmp_path)
        resolve_campaign(**_base_args(workspace, kernel, driver))
        with pytest.raises(ValueError, match="immutable configuration"):
            resolve_campaign(
                workspace_dir=str(workspace),
                resume=True,
                prepare_task=False,
                kernel=str(kernel),
                driver=str(driver),
                snr_threshold=2.0,
            )

    def test_missing_kernel_raises_for_fresh(self, tmp_path):
        workspace, _kernel, driver = _git_workspace(tmp_path)
        with pytest.raises(ValueError, match="fresh campaign requires"):
            resolve_campaign(
                workspace_dir=str(workspace),
                resume=False,
                prepare_task=False,
                kernel="",
                driver=str(driver),
                snr_threshold=2.0,
            )

    def test_missing_driver_raises_for_fresh(self, tmp_path):
        workspace, kernel, _driver = _git_workspace(tmp_path)
        with pytest.raises(ValueError, match="fresh campaign requires"):
            resolve_campaign(
                workspace_dir=str(workspace),
                resume=False,
                prepare_task=False,
                kernel=str(kernel),
                driver="",
                snr_threshold=2.0,
            )

    def test_pending_retry_mismatch_raises(self, tmp_path):
        workspace, kernel, driver = _git_workspace(tmp_path)
        resolve_campaign(**_base_args(workspace, kernel, driver))
        assert CampaignConfigStore(str(workspace)).exists()

        other_kernel = workspace / "other.py"
        other_kernel.write_text("def k(): return 99\n")
        with pytest.raises(ValueError, match="does not match"):
            resolve_campaign(
                workspace_dir=str(workspace),
                resume=False,
                prepare_task=False,
                kernel=str(other_kernel),
                driver=str(driver),
                kernel_backend="triton",
                snr_threshold=2.0,
            )
