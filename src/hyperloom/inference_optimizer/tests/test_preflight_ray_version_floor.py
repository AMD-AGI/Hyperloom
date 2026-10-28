# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for preflight treating the Ray pin as a floor, not an exact version."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import preflight as cli_preflight

RAY_RUNTIME = Path(cli_preflight.__file__).parents[2] / "orchestrator" / "actions" / "executors" / "_ray_runtime.py"
# The functions that build the `ray start` argv for the head preflight provisions.
_HEAD_ARGV_BUILDERS = (
    "_resources_start_args",
    "_isolated_head_port_args",
    "ensure_ray_cluster",
    "force_restart_local_cluster",
)


@pytest.fixture(autouse=True)
def ray_defaults(monkeypatch):
    monkeypatch.delenv("RAY_VERSION", raising=False)


# How Ray declares the `ray start` options Hyperloom passes: a type each, so the
# shim refuses a wrong value the way Ray's own parser does.
_RAY_START_OPTION_DECLS: dict[str, str] = {
    "--head": "is_flag=True, default=False",
    "--port": "type=int, default=None",
    "--dashboard-host": "type=str, default='127.0.0.1'",
    "--dashboard-port": "type=int, default=8265",
    "--ray-client-server-port": "type=int, default=None",
    "--num-gpus": "type=int, default=None",
    "--resources": "type=str, default='{}'",
}


def write_ray_cli_shim(ray_pkg: Path, *, ray_version: str, start_options: dict[str, str] | None = None) -> None:
    """Write a ray shim shaped like ray/scripts/scripts.py: the commands on a `cli` group, `main` wrapping it."""
    options = _RAY_START_OPTION_DECLS if start_options is None else start_options
    (ray_pkg / "__init__.py").write_text(f'__version__ = "{ray_version}"\n')
    decorators = "".join(f"@click.option({opt!r}, {decl})\n" for opt, decl in options.items())
    (ray_pkg / "scripts" / "scripts.py").write_text(
        "import click\n"
        "\n\n"
        "@click.group()\n"
        "def cli():\n"
        "    pass\n"
        "\n\n"
        "@cli.command()\n"
        f"{decorators}"
        "def start(**kwargs):\n"
        "    pass\n"
        "\n\n"
        "def main():\n"
        "    return cli()\n"
    )


def _fake_site(tmp_path: Path, *, ray_version: str, start_options: dict[str, str] | None = None) -> Path:
    """A site dir holding that shim and the dist-info of the matching release."""
    site = tmp_path / "site"
    ray_pkg = site / "ray"
    (ray_pkg / "scripts").mkdir(parents=True)
    (ray_pkg / "scripts" / "__init__.py").write_text("")
    write_ray_cli_shim(ray_pkg, ray_version=ray_version, start_options=start_options)
    dist = site / f"ray-{ray_version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ray\nVersion: {ray_version}\n")
    return site


def _run_smoke(site: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", cli_preflight._RAY_SMOKE],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"},
    )


def test_smoke_accepts_newer_ray(tmp_path: Path) -> None:
    # The kernel-agent installer picks 2.55.0 on interpreters without a 2.44.1
    # wheel; preflight must not reject it.
    site = _fake_site(tmp_path, ray_version="2.55.0")

    result = _run_smoke(site)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2.55.0"


def test_smoke_accepts_the_pinned_ray_whose_cli_works(tmp_path: Path) -> None:
    # The pin paired with the click the installer resolves: the CLI parses every
    # option Hyperloom passes, so no version comparison may refuse it.
    site = _fake_site(tmp_path, ray_version="2.44.1")

    result = _run_smoke(site)

    assert result.returncode == 0, result.stderr


def test_shim_mirrors_ray_s_wrapper_main(tmp_path: Path) -> None:
    """Ray's `main` is a plain wrapper: reading `.commands` off it finds nothing."""
    site = _fake_site(tmp_path, ray_version="2.44.1")

    result = subprocess.run(
        [sys.executable, "-c", "from ray.scripts.scripts import main;print(hasattr(main, 'commands'))"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_smoke_rejects_a_cli_that_lost_an_option(tmp_path: Path) -> None:
    options = {opt: decl for opt, decl in _RAY_START_OPTION_DECLS.items() if opt != "--resources"}
    site = _fake_site(tmp_path, ray_version="2.44.1", start_options=options)

    result = _run_smoke(site)

    assert result.returncode == 1
    assert "ray start does not accept" in result.stderr
    # click words this differently across releases; what has to reach the log is
    # which option the installed CLI no longer knows.
    assert re.search(r"No such option.*--resources", result.stderr), result.stderr


def test_smoke_rejects_a_cli_that_refuses_the_value_hyperloom_passes(tmp_path: Path) -> None:
    # The option is still there, but a release narrowed what it takes: enumerating
    # names would pass this CLI and `ray start` would then fail on the head.
    options = dict(_RAY_START_OPTION_DECLS, **{"--num-gpus": "type=click.IntRange(0, 4), default=None"})
    site = _fake_site(tmp_path, ray_version="2.44.1", start_options=options)

    result = _run_smoke(site)

    assert result.returncode == 1
    assert "ray start does not accept" in result.stderr
    assert "--num-gpus" in result.stderr


def test_smoke_rejects_a_cli_that_fails_to_import(tmp_path: Path) -> None:
    site = _fake_site(tmp_path, ray_version="2.44.1")
    (site / "ray" / "scripts" / "scripts.py").write_text("raise ImportError('boom')\n")

    result = _run_smoke(site)

    assert result.returncode == 1
    assert "ray CLI import failed" in result.stderr


def test_smoke_rejects_ray_below_the_floor(tmp_path: Path) -> None:
    site = _fake_site(tmp_path, ray_version="2.40.0")

    result = _run_smoke(site)

    assert result.returncode == 1
    assert "ray too old" in result.stderr


def test_certified_start_options_are_the_ones_the_head_is_started_with() -> None:
    """The options the smoke asks the CLI parser for are the ones the head gets."""
    source = RAY_RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bodies = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HEAD_ARGV_BUILDERS
    }

    certified = {token.split("=", 1)[0] for token in cli_preflight._RAY_START_ARGV if token.startswith("--")}
    assert set(bodies) == set(_HEAD_ARGV_BUILDERS), f"ray start argv builders moved: {sorted(bodies)}"
    for name, body in bodies.items():
        for option in re.findall(r"--[a-z][a-z-]+", body):
            assert option in certified, f"{name} passes {option}, which preflight never certifies"


def test_probe_env_reindexes_hip_devices_from_rocr(monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")

    assert cli_preflight._ray_probe_env()["HIP_VISIBLE_DEVICES"] == "0"

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,5,6")
    assert cli_preflight._ray_probe_env()["HIP_VISIBLE_DEVICES"] == "0,1,2"


def test_probe_env_keeps_an_existing_hip_value(monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")

    assert cli_preflight._ray_probe_env()["HIP_VISIBLE_DEVICES"] == "1"


def test_probe_env_leaves_hip_unset_without_rocr(monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    assert "HIP_VISIBLE_DEVICES" not in cli_preflight._ray_probe_env()


class _InstallRecorder:
    """Stands in for subprocess.run over the pip install attempts."""

    def __init__(self, *, pinned_install_ok: bool) -> None:
        self.pinned_install_ok = pinned_install_ok
        self.install_specs: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        specs = [arg for arg in argv if arg.startswith("ray[")]
        self.install_specs.append(specs)
        pinned = any(spec.startswith("ray[default]==") for spec in specs)
        code = 0 if (self.pinned_install_ok or not pinned) else 1
        return subprocess.CompletedProcess(argv, code, "", "No matching distribution found")


def _ensure_ray_with(monkeypatch, recorder: _InstallRecorder, *, version_after: str) -> dict:
    smoke_results = iter(
        [
            subprocess.CompletedProcess([], 1, "", "ray import failed"),
            subprocess.CompletedProcess([], 0, f"{version_after}\n", ""),
        ]
    )
    monkeypatch.setattr(cli_preflight, "_ray_smoke", lambda _exe, *_overrides: next(smoke_results))
    monkeypatch.setattr(cli_preflight.subprocess, "run", recorder)
    return cli_preflight._ensure_ray("/usr/bin/python3", [])


def test_ensure_ray_keeps_the_pin_when_it_resolves(monkeypatch) -> None:
    recorder = _InstallRecorder(pinned_install_ok=True)

    result = _ensure_ray_with(monkeypatch, recorder, version_after="2.44.1")

    assert recorder.install_specs == [["ray[default]==2.44.1"]]
    assert result["version_after"] == "2.44.1"


def test_ensure_ray_falls_back_to_the_floor_when_the_pin_cannot_resolve(monkeypatch) -> None:
    # 2.44.1 predates cp314 wheels: the pinned install cannot resolve, so the
    # repair falls back to the floor and keeps whatever Ray pip lands on.
    recorder = _InstallRecorder(pinned_install_ok=False)

    result = _ensure_ray_with(monkeypatch, recorder, version_after="2.55.0")

    assert recorder.install_specs == [
        ["ray[default]==2.44.1"],
        ["ray[default]>=2.44.1"],
    ]
    assert result["version_after"] == "2.55.0"
    assert result["spec"] == "ray[default]>=2.44.1"


def test_ensure_ray_reports_what_the_repair_really_did(monkeypatch) -> None:
    # The floor spec is satisfied by the Ray already installed, so pip can exit
    # zero having installed nothing: the error must not claim an install landed.
    recorder = _InstallRecorder(pinned_install_ok=False)
    smoke_results = iter(
        [
            subprocess.CompletedProcess([], 1, "", "ray start does not accept --resources"),
            subprocess.CompletedProcess([], 1, "", "ray start does not accept --resources"),
        ]
    )
    monkeypatch.setattr(cli_preflight, "_ray_smoke", lambda _exe, *_overrides: next(smoke_results))
    monkeypatch.setattr(cli_preflight.subprocess, "run", recorder)

    with pytest.raises(RuntimeError) as excinfo:
        cli_preflight._ensure_ray("/usr/bin/python3", [])

    message = str(excinfo.value)
    assert "pip install ray[default]>=2.44.1 succeeded" in message
    assert "ray[default]==2.44.1 failed to install" in message
    assert "ray start does not accept --resources" in message
