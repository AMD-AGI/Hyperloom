# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the enablement runtime adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.enablement.runtime import adapters as ad
from hyperloom.orchestrator.enablement.runtime.adapters import (
    AtomAdapter,
    NullAdapter,
    SglangAdapter,
    VllmRocmAdapter,
    XditAdapter,
    get_adapter,
)
from hyperloom.orchestrator.enablement.runtime.stack_actions import EnablementStackAction, RuntimeRequest

_PINS = "torch==2.11.0+rocm7.2\ntriton==3.7.0+amd.rocm7.2.0"


class _FakeRun:
    """Programmable fake subprocess shim: maps argv-substring -> (rc, out, err).

    The ROCm-pin probe answers with a pinned torch unless a rule says otherwise.
    """

    def __init__(self, rules: list[tuple[str, int, str, str]] | None = None, default_rc: int = 0):
        self.rules = [*(rules or []), ("named={'torch'", 0, _PINS, "")]
        self.default_rc = default_rc
        self.calls: list[list[str]] = []

    def __call__(self, argv, env, cwd):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for needle, rc, out, err in self.rules:
            if needle in joined:
                return subprocess.CompletedProcess(argv, rc, out, err)
        return subprocess.CompletedProcess(argv, self.default_rc, "", "")

    def pip_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[1:4] == ["-m", "pip", "install"]]


@pytest.fixture(autouse=True)
def _no_real_venv(monkeypatch):
    monkeypatch.setattr(ad, "create_venv", lambda venv_dir: None)


# the request a specialist emits


def test_a_request_without_a_ref_or_packages_asks_for_nothing():
    assert RuntimeRequest.from_payload(None) is None
    assert RuntimeRequest.from_payload({"reason": "newer sglang"}) is None
    assert RuntimeRequest.from_payload({"packages": ["", "  "]}) is None


def test_a_request_parses_its_fields():
    request = RuntimeRequest.from_payload(
        {"ref": " PR:123 ", "packages": "transformers==4.57.1", "capability": "glm5_arch", "reason": "r"}
    )
    assert request == RuntimeRequest(
        ref="PR:123", packages=("transformers==4.57.1",), capability="glm5_arch", reason="r"
    )


# registry


def test_frameworks_without_a_venv_runtime_acquire_nothing():
    request = RuntimeRequest(ref="v0.5.0", packages=("x==1",))
    for adapter in (AtomAdapter(), XditAdapter(), get_adapter("totally_unknown_fw")):
        assert adapter.build_stack_action(request) is None
    assert isinstance(get_adapter("totally_unknown_fw"), NullAdapter)


def test_get_adapter_case_insensitive():
    assert isinstance(get_adapter("VLLM"), VllmRocmAdapter)
    assert isinstance(get_adapter("SGLang"), SglangAdapter)


# build_stack_action: the request becomes the action, with no operator setting in the way


def test_sglang_pr_url_becomes_an_editable_checkout_of_that_repo():
    action = SglangAdapter().build_stack_action(
        RuntimeRequest(ref="https://github.com/fork/sglang/pull/77", capability="glm5_arch")
    )
    assert action is not None
    assert (action.acquisition_method, action.repo_url, action.ref) == (
        "editable_ref",
        "https://github.com/fork/sglang",
        "PR:77",
    )
    assert action.gap_id == "gap.enablement.glm5_arch"


def test_sglang_tag_defaults_to_the_upstream_repo_and_keeps_its_packages():
    action = SglangAdapter().build_stack_action(RuntimeRequest(ref="v0.5.21", packages=("transformers==4.57.1",)))
    assert action is not None
    assert action.repo_url == "https://github.com/sgl-project/sglang.git"
    assert action.ref == "v0.5.21"
    assert action.packages == ("transformers==4.57.1",)


def test_a_ref_naming_no_checkout_is_refused():
    assert SglangAdapter().build_stack_action(RuntimeRequest(ref="https://example.com/some/repo")) is None


@pytest.mark.parametrize("adapter", [SglangAdapter, VllmRocmAdapter])
def test_packages_alone_become_a_package_install(adapter):
    action = adapter().build_stack_action(
        RuntimeRequest(packages=("transformers==4.57.1",), index_url="https://idx.example/simple")
    )
    assert action is not None
    assert action.acquisition_method == "wheel"
    assert action.packages == ("transformers==4.57.1",)
    assert action.index_url == "https://idx.example/simple"


def test_vllm_refuses_a_checkout_because_it_compiles():
    assert VllmRocmAdapter().build_stack_action(RuntimeRequest(ref="v0.11.0")) is None


# ROCm pins


def test_every_install_is_held_to_the_rocm_pins(tmp_path):
    run = _FakeRun(rules=[("sysconfig", 0, f"{tmp_path}\n", "")])
    action = SglangAdapter().build_stack_action(RuntimeRequest(packages=("transformers==4.57.1",)))
    assert SglangAdapter(run=run).provision(action, tmp_path / "attempt").ok
    (pip,) = run.pip_calls()
    assert "--upgrade" not in pip
    constraints = Path(pip[pip.index("--constraint") + 1])
    assert constraints.read_text(encoding="utf-8").splitlines() == _PINS.splitlines()


def test_an_interpreter_without_torch_fails_before_pip(tmp_path):
    run = _FakeRun(rules=[("named={'torch'", 0, "", "")])
    action = SglangAdapter().build_stack_action(RuntimeRequest(packages=("x==1",)))
    result = SglangAdapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is False
    assert "torch version" in result.error
    assert run.pip_calls() == []


# provision + ROCm verification (mocked run)


def _wheel_action() -> EnablementStackAction:
    return EnablementStackAction(
        kind="runtime_candidate",
        framework="vllm",
        gap_id="gap.enablement.missing_model_arch",
        capability="deepseek_v4",
        acquisition_method="wheel",
        index_url="https://rocm.repo/whl",
        packages=("vllm",),
    )


_PURELIB = "/attempt/venv/lib/python3.12/site-packages"


def test_vllm_provision_ok(tmp_path):
    run = _FakeRun(rules=[("m.version(", 0, "0.21.0", ""), ("sysconfig", 0, f"{_PURELIB}\n", "")])
    result = VllmRocmAdapter(run=run).provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is True, result.error
    assert result.runtime.venv_root.endswith("venv")
    assert result.runtime.bin_path.endswith("bin")
    assert result.installed_versions.get("vllm") == "0.21.0"


def test_vllm_provision_pip_fail(tmp_path):
    run = _FakeRun(rules=[("-m pip install", 1, "", "no matching distribution")])
    result = VllmRocmAdapter(run=run).provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is False
    assert "pip install failed" in result.error


def test_vllm_provision_rejects_cuda_torch(tmp_path):
    run = _FakeRun(rules=[("getattr(torch.version,'hip'", 1, "", "")])
    result = VllmRocmAdapter(run=run).provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is False
    assert "not a ROCm build" in result.error


def test_vllm_provision_rejects_non_rocm_vllm(tmp_path):
    run = _FakeRun(rules=[("current_platform", 1, "", "not rocm")])
    result = VllmRocmAdapter(run=run).provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is False
    assert "ROCm platform" in result.error


# the tree an acquired runtime imports, which is the tree its patches apply to


@pytest.mark.parametrize("adapter, package", [(VllmRocmAdapter, "vllm"), (SglangAdapter, "sglang")])
def test_a_framework_installed_into_the_venv_is_the_tree_patches_land_in(tmp_path, adapter, package):
    (tmp_path / "purelib" / package).mkdir(parents=True)
    run = _FakeRun(rules=[("sysconfig", 0, f"{tmp_path / 'purelib'}\n", "")])
    action = adapter().build_stack_action(RuntimeRequest(packages=(f"{package}==9.9",)))
    result = adapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is True, result.error
    assert result.runtime.source_root == str(tmp_path / "purelib")


def test_a_dependency_only_runtime_leaves_patches_on_the_shared_tree(tmp_path):
    """The framework is still imported from the shared tree, so the venv's site-packages is not its source."""
    (tmp_path / "purelib").mkdir()
    run = _FakeRun(rules=[("sysconfig", 0, f"{tmp_path / 'purelib'}\n", "")])
    action = SglangAdapter().build_stack_action(RuntimeRequest(packages=("transformers==4.57.1",)))
    result = SglangAdapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is True, result.error
    assert result.runtime.source_root == ""


@pytest.mark.parametrize("adapter", [VllmRocmAdapter, SglangAdapter])
def test_a_wheel_runtime_with_no_site_packages_fails_to_provision(tmp_path, adapter):
    run = _FakeRun(rules=[("sysconfig", 1, "", "")])
    action = adapter().build_stack_action(RuntimeRequest(packages=("x==1",)))
    result = adapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is False
    assert "site-packages" in result.error


_TORCH_PROBE = ("m.version('torch')", 0, "2.11.0+rocm7.2", "")


def _sglang_checkout(attempt: Path) -> None:
    """What the fetch leaves behind: a checkout whose pyproject declares a ROCm extra."""
    python_dir = attempt / "src" / "python"
    python_dir.mkdir(parents=True)
    (python_dir / "pyproject.toml").write_text(
        "[project.optional-dependencies]\nsrt_hip = ['torch']\n", encoding="utf-8"
    )


def test_sglang_editable_provision_fetches_the_pr_head_and_records_the_clone_it_imports(tmp_path):
    """An editable install executes the clone, and its diffs are cut against the repo root."""
    _sglang_checkout(tmp_path / "attempt")
    run = _FakeRun(rules=[_TORCH_PROBE])
    action = SglangAdapter().build_stack_action(RuntimeRequest(ref="PR:123", packages=("transformers==4.57.1",)))
    result = SglangAdapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is True, result.error
    checkout = tmp_path / "attempt" / "src"
    assert [
        "git",
        "-C",
        str(checkout),
        "fetch",
        "-q",
        "--depth",
        "1",
        action.repo_url,
        "refs/pull/123/head",
    ] in run.calls
    assert result.runtime.source_root == str(checkout)
    assert result.runtime.pythonpath_prefix == str(checkout / "python")
    (pip,) = run.pip_calls()
    assert pip[-3:] == ["-e", f"{checkout / 'python'}[srt_hip]", "transformers==4.57.1"]


def test_a_failed_install_reports_pips_error_and_logs_it(tmp_path):
    """Stderr warnings come first; the error at the end is what the result and provision.log carry."""
    warnings = "DEPRECATION: Loading egg at /opt/venv/lib/python3.12/site-packages/x.egg is deprecated.\n" * 20
    error = "RuntimeError: cargo is required to discover the Rust extension modules"
    _sglang_checkout(tmp_path / "attempt")
    run = _FakeRun(
        rules=[("-m pip install", 1, "Obtaining file:///src/python\n", f"{warnings}{error}\n"), _TORCH_PROBE]
    )
    action = SglangAdapter().build_stack_action(RuntimeRequest(ref="v0.5.21"))
    result = SglangAdapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is False
    assert error in result.error
    assert error in Path(result.log_path).read_text(encoding="utf-8")


def test_sglang_editable_provision_reports_a_ref_it_cannot_fetch(tmp_path):
    run = _FakeRun(rules=[(" fetch ", 128, "", "couldn't find remote ref v9")])
    action = SglangAdapter().build_stack_action(RuntimeRequest(ref="v9"))
    result = SglangAdapter(run=run).provision(action, tmp_path / "attempt")
    assert result.ok is False
    assert "git checkout of v9 failed" in result.error
    assert run.pip_calls() == []


# verify helpers


def test_verify_torch_is_rocm_true_false():
    assert ad.verify_torch_is_rocm("/py", run=_FakeRun(default_rc=0)) is True
    assert ad.verify_torch_is_rocm("/py", run=_FakeRun(default_rc=1)) is False


def test_verify_vllm_rocm_true_false():
    assert ad.verify_vllm_rocm("/py", run=_FakeRun(default_rc=0)) is True
    assert ad.verify_vllm_rocm("/py", run=_FakeRun(default_rc=1)) is False


def _raiser(exc: BaseException):
    def raise_it(*_args, **_kwargs):
        raise exc

    return raise_it


_VERIFIERS = {"torch": ad.verify_torch_is_rocm, "vllm": ad.verify_vllm_rocm}
_ADAPTERS = {"vllm_provision": VllmRocmAdapter, "sglang_provision": SglangAdapter}


def _outcome(site, failure, tmp_path, monkeypatch):
    """Run ``site`` with ``failure`` raised underneath it."""
    if site in _VERIFIERS:
        return _VERIFIERS[site]("/py", run=_raiser(failure))
    monkeypatch.setattr(ad, "create_venv", _raiser(failure))
    result = _ADAPTERS[site](run=_FakeRun()).provision(_wheel_action(), tmp_path / "attempt")
    return result.ok, result.error


@pytest.mark.parametrize("site", [*_VERIFIERS, *_ADAPTERS])
@pytest.mark.parametrize(
    "failure",
    [
        OSError("no such interpreter"),
        subprocess.CalledProcessError(1, ["python", "-m", "venv"]),
        subprocess.TimeoutExpired(cmd="probe", timeout=1),
    ],
    ids=["oserror", "called-process", "timeout"],
)
def test_an_expected_probe_or_venv_failure_is_absorbed(tmp_path, monkeypatch, site, failure):
    expected = False if site in _VERIFIERS else (False, f"venv setup failed: {failure!r}")
    outcome = _outcome(site, failure, tmp_path, monkeypatch)
    assert outcome == expected


@pytest.mark.parametrize("site", [*_VERIFIERS, *_ADAPTERS])
def test_an_unexpected_probe_or_venv_failure_propagates(tmp_path, monkeypatch, site):
    with pytest.raises(TypeError, match="unexpected failure"):
        _outcome(site, TypeError("unexpected failure"), tmp_path, monkeypatch)
