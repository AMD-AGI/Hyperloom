# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aiter's touched modules built on CPU before the test (#125 stage 2b), through the whole runner.

The image is the stand-in aiter (:mod:`hyperloom.measure.runner.native.tests._aiter_standin`) with ``module_gemm`` built
from ``csrc/kernels/k.cu``. The sheet patches ``k.cu`` and names a CPU build of ``module_gemm``: a real one, planned by
:class:`~hyperloom.measure.runner.native.aiter_modules.AiterModules`, laid as the farm, compiled by the stand-in's aiter
JIT build in the build runner's child, signed. The fake server (``test_run_jit``'s grid) builds ``module_gemm`` when it
does not find it in the test's aiter JIT directory, and logs ``start build`` then; what the result says is checked.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from hyperloom.measure import signing
from hyperloom.measure.compile import EnvEntry, EnvRoot, LocalBlobStore
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.laying import NamedBuild, NativeLayError, SheetBuilds, manifest_sha256
from hyperloom.measure.jit.native.record import READY, BuildRecord, OutputFile
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import Signer, Trust, TrustedKey
from hyperloom.measure.jit.modules import read_config
from hyperloom.measure.runner.native import aiter_modules
from hyperloom.measure.runner.native.aiter_modules import AiterModules
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import IsolatedBuilds, Isolation
from hyperloom.measure.runner.native.planning import NativePlanner, NativeSite
from hyperloom.measure.runner.native.service import ensure_build
from hyperloom.measure.runner import run as run_module
from hyperloom.measure.runner.native.tests import _aiter_standin as standin
from hyperloom.measure.runner.tests.test_run import SHEET_IMAGE
from hyperloom.measure.runner.tests.test_run_jit import (  # noqa: F401 - fixtures
    Rig,
    _kind,
    _states,
    env,
    key,
    rig,
)

PATCHED = "// patched k.cu\n"
BUILDER = "sha256:" + "c" * 64
GFX = "gfx950"  # the rig's GPU


class Built:
    """A signed CPU build of ``module_gemm`` from the patched ``k.cu``, in a store the runner is pointed at."""

    def __init__(
        self,
        rig: Rig,
        tmp: Path,
        monkeypatch,
        *,
        environ: dict | None = None,
        gfx: str = GFX,
        factory: str = standin.FACTORY,
    ) -> None:
        signing_key = signing.generate_private_key()
        public = signing.public_key_pem(signing_key)
        self.trust_file = tmp / "trust.json"
        self.trust_file.write_text(json.dumps({"min_epoch": 1, "keys": [{"public_key_pem": public, "epoch": 1}]}))
        self.store_dir = tmp / "build-store"
        self.store = BuildStore(self.store_dir, Trust((TrustedKey(public, 1),), 1))
        blobs = LocalBlobStore(tmp / "build-blobs")
        self.blob = blobs.put(PATCHED.encode())
        monkeypatch.setattr(aiter_modules, "locate_package", lambda name, python, cwd: rig.bases / name)
        machine = BuildMachine(SHEET_IMAGE, BUILDER, gfx, sys.executable)
        planner = NativePlanner(NativeSite(machine, {}, tmp / "sources", tmp / "image-root"), blobs)
        plan = AiterModules(planner, environ=environ or {})
        config = read_config(rig.bases / "aiter" / "jit" / "optCompilerConfig.json")
        built = plan(self.roots(), config, {standin.MODULE})
        assert built is not None and built.modules == {standin.MODULE}
        runner = IsolatedBuilds(planner, blobs, Isolation((), None), {aiter_modules.AITER_COMPONENT: factory})
        self.record: BuildRecord = ensure_build(
            self.store, built.inputs, runner(built.inputs), Signer(signing_key)
        ).record

    def roots(self) -> tuple[EnvRoot, ...]:
        entry = EnvEntry(standin.KERNEL, "upsert", self.blob, "0644")
        return (
            EnvRoot("aiter", "aiter", "amd-aiter", "0", ()),
            EnvRoot("aiter.meta", "aiter_meta", "amd-aiter", "0", (entry,)),
        )

    def environ(self) -> dict:
        return {"HYPERLOOM_BUILD_STORE": str(self.store_dir), "HYPERLOOM_BUILD_TRUST": str(self.trust_file)}


@pytest.fixture
def image(rig: Rig, monkeypatch) -> Rig:
    """The rig over the stand-in aiter. The fake server imports nothing, so the farm's aiter (a package root of the
    sheet, as acceptance always makes one) is read as imported: whether it was is ``test_run_jit``'s to check."""
    standin.aiter_image(rig.bases)
    monkeypatch.setattr(run_module, "refine_imports", lambda document, farm: document)
    return rig


def _sheet(rig: Rig, built: Built | None, *, root_id: str = "aiter", envs: dict | None = None):
    def make(run: Path) -> dict:
        package = {**rig.root(run, "aiter", "ops/mha.py", b"X = 1\n"), "entries": []}
        meta = {**rig.root(run, "aiter_meta", standin.KERNEL, PATCHED.encode()), "id": "aiter.meta"}
        sheet = rig.sheet(run, roots=[package, meta], envs=envs)
        if built is not None:
            ref = {"root_id": root_id, "key": built.record.key, "manifest_sha256": manifest_sha256(built.record)}
            sheet["environment"]["builds"] = [ref]
        return sheet

    return make


def test_the_cpu_build_is_aiters_own_build_of_the_patched_source_for_the_named_gpu(
    image, tmp_path, monkeypatch
) -> None:
    built = Built(image, tmp_path, monkeypatch)

    assert built.record.status == READY and [o.path for o in built.record.outputs] == [f"{standin.MODULE}.so"]
    (output,) = built.record.outputs
    blob = built.store_dir / "blobs" / output.sha256[:2] / output.sha256
    assert blob.read_bytes() == standin.built(GFX, PATCHED), "the patched k.cu, for the named GPU, with no GPU asked"


def test_the_server_loads_the_module_its_cpu_build_made_and_builds_nothing(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch)

    manifest, status, run = image.run("seeded", _sheet(image, built), extra_env=built.environ())

    assert manifest.outcome.failure_class is None, "the touched module came from its build, so nothing stale ran"
    assert manifest.jit_cache.builds_seen == 0, "the server found module_gemm.so and logged no start build"
    log = run / "out" / "files" / "variant_0" / "server.log"
    assert "start build" not in log.read_text()
    assert "1 touched aiter modules come from their CPU build (module_gemm)" in status["message"]
    contributed = _kind(manifest, "aiter_jit").contribution
    assert contributed is not None and contributed.files == 1, "and the test gives it to the JIT store under its key"


def test_a_second_test_finds_the_module_in_the_jit_store_under_its_key(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch)
    first, _, first_run = image.run("first", _sheet(image, built), extra_env=built.environ())
    image.publish(first_run, first)

    second, _, _ = image.run("second", _sheet(image, built), extra_env=built.environ())

    assert _states(second)["aiter_jit"] == "hit" and second.jit_cache.builds_seen == 0
    assert second.outcome.failure_class is None


def test_without_a_cpu_build_the_server_builds_the_touched_module_on_its_card(image) -> None:
    manifest, status, _ = image.run("cold", _sheet(image, None))

    assert manifest.jit_cache.builds_seen == 1 and manifest.outcome.failure_class is None
    assert "CPU build" not in status["message"]


def test_a_build_made_under_another_variable_than_the_servers_is_not_used(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch)

    manifest, status, _ = image.run(
        "other-env", _sheet(image, built, envs={"ENABLE_CK": "0"}), extra_env=built.environ()
    )

    assert manifest.jit_cache.builds_seen == 1, "the server builds module_gemm itself, with its ENABLE_CK"
    assert "the CPU build of aiter's modules is not used: the server's ENABLE_CK differ" in status["message"]
    assert manifest.outcome.failure_class is None


def test_a_build_the_service_made_under_the_servers_variables_is_used(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch, environ={"ENABLE_CK": "0"})

    manifest, _, _ = image.run("same-env", _sheet(image, built, envs={"ENABLE_CK": "0"}), extra_env=built.environ())

    assert manifest.jit_cache.builds_seen == 0 and manifest.outcome.failure_class is None


#: What the real image has outside aiter's build path: its ``setup.py`` and the Triton kernels below ``aiter/ops`` read a
#: variable the build of a module never reads (``AITER_USE_SYSTEM_TRITON``, set to 1 on the GPU pod).
OUTSIDE_THE_BUILD = {
    "setup.py": 'import os\nSYSTEM = os.environ.get("AITER_USE_SYSTEM_TRITON", "0")\n',
    "aiter/ops/triton/gluon/__init__.py": 'import os\nUSE = os.getenv("AITER_USE_SYSTEM_TRITON")\n',
}
#: ``module_gemm``'s code generator script (its ``blob_gen_cmd``), which reads a variable of its own.
GENERATOR_READS = {"csrc/kernels/gen.py": 'import os\nFLAVOR = os.environ.get("AITER_GEN_FLAVOR", "a")\n'}


def _with_files(rig: Rig, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = rig.bases / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_a_variable_only_code_outside_the_build_reads_does_not_keep_the_cpu_build_from_the_server(
    image, tmp_path, monkeypatch
) -> None:
    _with_files(image, OUTSIDE_THE_BUILD)
    built = Built(image, tmp_path, monkeypatch)
    sheet = _sheet(image, built, envs={"AITER_USE_SYSTEM_TRITON": "1"})

    manifest, status, _ = image.run("unrelated-env", sheet, extra_env=built.environ())

    assert "not used" not in status["message"], status["message"]
    assert manifest.jit_cache.builds_seen == 0, "the server loads module_gemm from its CPU build"
    assert "1 touched aiter modules come from their CPU build (module_gemm)" in " ".join(manifest.notes)


@pytest.mark.parametrize(("service", "used"), [("b", True), ("a", False)], ids=["same", "other"])
def test_a_variable_the_modules_generator_reads_decides_whether_the_cpu_build_is_used(
    image, tmp_path, monkeypatch, service, used
) -> None:
    _with_files(image, GENERATOR_READS)
    built = Built(image, tmp_path, monkeypatch, environ={"AITER_GEN_FLAVOR": service})

    manifest, status, _ = image.run(
        "generator-env", _sheet(image, built, envs={"AITER_GEN_FLAVOR": "b"}), extra_env=built.environ()
    )

    assert manifest.jit_cache.builds_seen == (0 if used else 1)
    refused = "the CPU build of aiter's modules is not used: the server's AITER_GEN_FLAVOR differ from its build's"
    assert (refused in status["message"]) is not used
    assert (f"jit cache: {refused}" in manifest.notes) is not used, "the signed result carries the runner's note"


def test_a_build_by_another_interpreter_than_the_servers_is_not_used(image, tmp_path, monkeypatch) -> None:
    """Another interpreter can be another Python, torch or HIP: the module it would build is not the build's."""
    built = Built(image, tmp_path, monkeypatch)
    other = tmp_path / "python-other"
    other.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    other.chmod(0o755)

    manifest, status, _ = image.run(
        "other-python", _sheet(image, built, envs={"HYPERLOOM_FRAMEWORK_PYTHON": str(other)}), extra_env=built.environ()
    )

    assert manifest.jit_cache.builds_seen == 1
    assert "the server's HYPERLOOM_FRAMEWORK_PYTHON differ from its build's" in status["message"]


def test_a_build_for_another_gpu_is_not_used(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch, gfx="gfx942")

    manifest, status, _ = image.run("other-gpu", _sheet(image, built), extra_env=built.environ())

    assert manifest.jit_cache.builds_seen == 1
    assert "it was built for gfx942 and this GPU is gfx950" in status["message"]


def test_a_touched_module_neither_built_nor_given_still_ends_as_stale_code(image, tmp_path, monkeypatch) -> None:
    """The 2a guard holds beside a CPU build: when the build is not used and the image's copy of the touched module is
    in the test's directory anyway, the result is a failed measurement, never the patch's numbers."""
    from hyperloom.measure.jit import session as session_module

    built = Built(image, tmp_path, monkeypatch)
    real = session_module.link_prebuilt
    monkeypatch.setattr(
        session_module, "link_prebuilt", lambda prebuilt, content, skip: real(prebuilt, content, frozenset())
    )

    manifest, _, _ = image.run("stale", _sheet(image, built, envs={"ENABLE_CK": "0"}), extra_env=built.environ())

    assert manifest.outcome.failure_class == "infra"
    assert manifest.outcome.failure_reason.startswith("stale_aiter_module: module_gemm ran as the image built it")


def test_a_runner_given_no_build_store_refuses_a_sheet_that_names_a_build(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch)

    status = image.refused("nostore", _sheet(image, built))

    assert status["error_class"] == "spec_rejected"
    assert "HYPERLOOM_BUILD_STORE and HYPERLOOM_BUILD_TRUST" in status["message"]


def test_a_module_build_named_for_another_package_is_refused(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch)

    def sheet(run: Path) -> dict:
        document = _sheet(image, built)(run)
        document["environment"]["roots"].append(image.root(run, "sglang", "srt/a.py", b"x = 2\n"))
        document["environment"]["builds"][0]["root_id"] = "sglang"
        return document

    status = image.refused("foreign", sheet, extra_env=built.environ())

    assert status["error_class"] == "spec_rejected"
    assert "sglang: a package root whose modules no CPU build makes" in status["message"]


def test_a_compile_that_claims_a_variable_it_was_not_given_is_not_believed(image, tmp_path, monkeypatch) -> None:
    """The compile runs candidate code: what the record says the compile was given is the service's word, never the
    compile's, so a compile cannot make its modules acceptable to a server whose build would read other values."""
    built = Built(image, tmp_path, monkeypatch, factory=standin.FORGING_FACTORY)
    assert "env_ENABLE_CK" not in built.record.toolchain and built.record.toolchain["compiler"] == "stand-in"

    manifest, status, _ = image.run("forged", _sheet(image, built, envs={"ENABLE_CK": "0"}), extra_env=built.environ())

    assert manifest.jit_cache.builds_seen == 1
    assert "the server's ENABLE_CK differ from its build's" in status["message"]


def test_a_module_whose_tuned_table_the_server_changes_is_built_by_the_server(image, tmp_path, monkeypatch) -> None:
    built = Built(image, tmp_path, monkeypatch)
    table = tmp_path / "mine.csv"
    table.write_text("M,N,K\n7,8,9\n")

    manifest, status, _ = image.run(
        "table", _sheet(image, built, envs={standin.TABLE_VARIABLE: str(table)}), extra_env=built.environ()
    )

    assert manifest.jit_cache.builds_seen == 1, "only the server reads the table its variable names"
    assert "module_gemm not taken from their CPU build: the server's tuned tables differ" in status["message"]
    assert manifest.outcome.failure_class is None


def test_a_module_build_whose_outputs_are_not_module_files_is_refused(tmp_path) -> None:
    signing_key = signing.generate_private_key()
    store = BuildStore(tmp_path / "store", Trust((TrustedKey(signing.public_key_pem(signing_key), 1),), 1))
    data = b"\x7fELF module"
    sha = hashlib.sha256(data).hexdigest()
    (tmp_path / "staging" / "blobs").mkdir(parents=True)
    (tmp_path / "staging" / "blobs" / sha).write_bytes(data)
    inputs = BuildInputs(SHEET_IMAGE, BUILDER, "aiter", GFX, "0" * 64, "aiter_jit/1")
    output = OutputFile(f"aiter/jit/{standin.MODULE}.so", sha, len(data), "0644")
    record = BuildRecord(inputs.key, inputs, READY, (output,), {}, "2026-10-27T00:00:00Z", 1.0, 1.0)
    store.publish(record, tmp_path / "staging", "", Signer(signing_key))
    builds = SheetBuilds(store, {"aiter": NamedBuild("aiter", record.key, manifest_sha256(record))}, SHEET_IMAGE)

    with pytest.raises(NativeLayError, match="has outputs outside its place"):
        builds.module_record("aiter")
