# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache through the runner, the local backend and the host (design 4.21, H2-09).

Every run here is a whole runner: a signed sheet in, the real ``LocalMeasurementBackend`` with only ``run_grid`` played (by
a grid that compiles a module into the directory the environment names, the way aiter does when it finds none there), and
a signed result out. What is published to the store is what the host would publish from that result. So "the second test
hits the cache" is the second runner finding, through verification, the module the first one compiled; and a candidate that
differs only in server arguments or in a framework patch finds the same cache, while one that changes what the compilers
build from does not.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from hyperloom.measure import farm as farm_module
from hyperloom.measure import local, signing
from hyperloom.measure.jit.publish import STAGING_PATH, publish_contributions
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.jit.tests.triton_cache import find_kernel, get_group
from hyperloom.measure.manifest import JitCacheFacts, ResultManifest
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC
from hyperloom.measure.runner import jit as runner_jit
from hyperloom.measure.runner import own_jit
from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom.measure.runner.tests.test_run import (  # noqa: F401 - fixtures
    POINT,
    _blob,
    _go,
    _sheet,
    _snapshot,
    env,
    key,
    runner,
)
from hyperloom.measure.tests.test_local_backend import _Grid, _Server

HASH_DIR = "0123456789abcdef0123456789abcdef"
MODULE = b"\x7fELF module_gemm built from the tree"
BUILD_LOG = (
    "[aiter] [pid=11 pname=MainProcess] start build [module_gemm] under /j/build/module_gemm\n"
    "\x1b[32mfinish build [module_gemm], cost 12.5s \x1b[0m\n"
)
CORE = (
    'import os\nA = os.getenv("AITER_CONFIG_GEMM_A8W8", "x")\nB = os.environ.get("ENABLE_CK", "1")\n'
    'C = os.environ.get("AITER_META_DIR")\n'
)
TABLE_VARIABLE = "AITER_CONFIG_GEMM_A8W8"
#: What aiter's cpp_itfs does with the home directory: the default of the root of its cache.
HOME_READ = (
    'import os\nHOME_PATH = os.environ.get("HOME") or os.environ.get("USERPROFILE")\nROOT = f"{HOME_PATH}/.aiter"\n'
)
STATES = ("aiter_jit", "aiter_cpp", "triton")


class _CompilingGrid(_Grid):
    """A ``run_grid`` whose server compiles what it does not find: one aiter module, one cpp_itfs library, one Triton kernel."""

    #: A compiler that does not trust what it was given and writes every Triton file again, to other bytes.
    recompile_triton = False

    def __init__(self, tmp_path: Path, files: Path, *, fail: bool = False) -> None:
        super().__init__(tmp_path)
        self.files = files
        self.fail = fail
        self.triton_found: list[bool] = []
        self.triton_group: dict[str, str] | None = None

    async def __call__(self, **kwargs):
        log = self.files / "variant_0" / "server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        compiled = {"AITER_JIT_DIR": "module_gemm.so", "AITER_ROOT_DIR": "build/pa_x/lib.so"}
        for variable, relative in compiled.items():
            target = Path(os.environ.get(variable) or self.tmp_path / "default" / variable) / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(MODULE if variable == "AITER_JIT_DIR" else f"{variable} product".encode())
                if variable == "AITER_JIT_DIR":
                    with open(log, "a", encoding="utf-8") as out:
                        out.write(BUILD_LOG)
        triton = Path(os.environ.get("TRITON_CACHE_DIR") or self.tmp_path / "default" / "TRITON_CACHE_DIR")
        self.triton_found.append(find_kernel(triton, HASH_DIR))
        self.triton_group = get_group(triton, HASH_DIR)
        for path in triton.rglob("kernel.*") if self.recompile_triton else ():
            path.write_bytes(b"compiled again, differently")
        with open(log, "a", encoding="utf-8") as out:
            out.write("Model loading took 28.5 GiB and 41.25 seconds\n")
        results = await super().__call__(**kwargs)
        if self.fail:
            results[0] = self.result(results[0].name, status="failed", returncode=1)
        return results


class _OutHandle:
    """The sandbox as the host sees it: ``/out/...`` is the runner's output directory."""

    def __init__(self, out: Path) -> None:
        self.out = out

    def _path(self, path: str) -> Path:
        assert path.startswith("/out/")
        return self.out / path.removeprefix("/out/")

    def read_file(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    def copy_out(self, source: str, destination: Path) -> None:
        destination.write_bytes(self._path(source).read_bytes())


class Rig:
    """Runs whole runners against one cache, one bases tree and one key, each in a directory of its own."""

    def __init__(self, tmp_path: Path, environ: dict, signing_key, monkeypatch) -> None:
        self.tmp_path = tmp_path
        self.base_environ = environ
        self.signing_key = signing_key
        self.monkeypatch = monkeypatch
        self.store = DirJitStore(tmp_path / "store")
        self.bases = tmp_path / "bases"
        self._make_bases()
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
        monkeypatch.setattr(runner_jit, "locate_package", self._locate)
        monkeypatch.setattr(own_jit, "locate_package", self._locate)
        monkeypatch.setattr(farm_module, "locate_package", self._locate)
        _Server(monkeypatch)

    def _make_bases(self) -> None:
        files = {
            "aiter/__init__.py": "",
            "aiter/jit/core.py": CORE,
            "aiter/jit/optCompilerConfig.json": "{}",
            "aiter/ops/mha.py": 'import os\nX = os.environ.get("AITER_DISABLE_FMHA_OPUS")\n',
            "csrc/cpp_itfs/utils.py": HOME_READ,
            "sglang/__init__.py": "",
            "sglang/srt/a.py": "x = 1\n",
        }
        for relative, text in files.items():
            path = self.bases / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def _locate(self, import_name: str, python: str, cwd: Path) -> Path:
        return self.bases / import_name

    def sheet(
        self,
        run: Path,
        *,
        envs: dict | None = None,
        roots: list | None = None,
        benchmark: dict | None = None,
        runtime: dict | None = None,
        **changes,
    ) -> dict:
        point = {
            "benchmark": {
                **POINT["benchmark"],
                "envs": {**POINT["benchmark"]["envs"], "HYPERLOOM_FRAMEWORK_PYTHON": sys.executable, **(envs or {})},
                **(benchmark or {}),
            }
        }
        text = yaml.safe_dump(point)
        environment = {"roots": roots or [], "runtime": runtime, "blob_root": "/b"}
        return _sheet(
            config=[{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}],
            environment=environment,
            **changes,
        )

    def root(self, run: Path, import_name: str, path: str, data: bytes) -> dict:
        entry = {"path": path, "op": "upsert", "blob": _blob(run, data), "mode": "0644"}
        return {
            "id": import_name,
            "kind": "package",
            "import_name": import_name,
            "dist": {"name": import_name, "version": "0"},
            "entries": [entry],
        }

    def directory_attachment(self, run: Path) -> dict:
        """An attachment that is a directory: unlike a table file it can hold code the server runs."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            data = b"print('hello')\n"
            member = tarfile.TarInfo("helper.py")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        return {"id": "ext", "kind": "dir", "blob": _blob(run, buffer.getvalue()), "manifest_sha256": "1" * 64}

    def _start(self, name: str, *, cache: bool, fail: bool, extra_env: dict | None) -> tuple[Path, dict]:
        """The directory of a run and its environment, with the fake grid in place."""
        run = self.tmp_path / name
        run.mkdir()
        key_file = run / "key.pem"
        key_file.write_bytes(signing.private_key_pem(self.signing_key))
        environ = {**self.base_environ, "HYPERLOOM_SIGNING_KEY_FILE": str(key_file), **(extra_env or {})}
        if cache:
            environ[JIT_CACHE_ENV] = str(self.store.root)
        self.grid = _CompilingGrid(run, run / "out" / "files", fail=fail)
        self.monkeypatch.setattr(local, "run_grid", self.grid)
        return run, environ

    def run(self, name: str, make_sheet, *, cache: bool = True, fail: bool = False, extra_env: dict | None = None):
        """Run one whole test in ``tmp_path/name``; give its manifest, its status and the directory."""
        run, environ = self._start(name, cache=cache, fail=fail, extra_env=extra_env)
        assert _go(run, environ, make_sheet(run)) == 0
        manifest = ResultManifest.from_bytes((run / "out" / "manifest.json").read_bytes())
        return manifest, json.loads((run / "out" / "status.json").read_text()), run

    def refused(self, name: str, make_sheet, *, cache: bool = True) -> dict:
        """Run one whole test that the runner refuses before it starts a server; give its status."""
        run, environ = self._start(name, cache=cache, fail=False, extra_env=None)
        assert _go(run, environ, make_sheet(run)) != 0
        assert not (run / "out" / "manifest.json").exists()
        return json.loads((run / "out" / "status.json").read_text())

    def publish(self, run: Path, manifest: ResultManifest) -> int:
        """What the host does after it verified the signed result."""
        scratch = run / "host"
        scratch.mkdir()
        return publish_contributions(self.store, _OutHandle(run / "out"), manifest.jit_cache, scratch)


@pytest.fixture
def rig(tmp_path, env, key, monkeypatch) -> Rig:
    return Rig(tmp_path, env, key, monkeypatch)


def _states(manifest: ResultManifest) -> dict[str, str]:
    return {kind.kind: kind.state for kind in manifest.jit_cache.kinds}


def _kind(manifest: ResultManifest, name: str):
    return next(kind for kind in manifest.jit_cache.kinds if kind.kind == name)


def _keys(manifest: ResultManifest) -> dict[str, str | None]:
    return {kind.kind: kind.key for kind in manifest.jit_cache.kinds}


def _arm(rig: Rig, **kwargs):
    """The sheet of a candidate: ``rig.sheet`` with what that candidate changes."""
    return lambda run: rig.sheet(run, **{k: (v(run) if callable(v) else v) for k, v in kwargs.items()})


def test_the_second_test_hits_the_cache_the_first_filled_and_builds_nothing(rig) -> None:
    first, _, first_run = rig.run("first", _arm(rig))
    assert rig.publish(first_run, first) == 3
    second, status, _ = rig.run("second", _arm(rig))

    assert _states(first) == dict.fromkeys(STATES, "miss") and _states(second) == dict.fromkeys(STATES, "hit")
    assert (first.jit_cache.builds_seen, second.jit_cache.builds_seen) == (1, 0), "the hit counter"
    assert (first.timing.jit_compile_sec, second.timing.jit_compile_sec) == (12.5, 0.0), "the cold-start breakdown"
    assert first.jit_cache.kinds[0].contribution is not None and second.jit_cache.kinds[0].contribution is None
    assert second.jit_cache.kinds[0].seed_files == 5, "Triton's kernel: four products and the group that lists them"
    assert status["message"] == ""
    names = [phase.name for phase in second.timing.phases]
    assert "jit_cache" in names and names.index("jit_cache") < names.index("kernel_tables")


def test_a_sandbox_with_a_home_and_a_scratch_directory_of_its_own_finds_the_cache_another_one_filled(rig) -> None:
    """Every local sandbox has its own ``HOME`` and ``TMPDIR``, and aiter reads ``HOME``: they are where things are kept."""

    def places(name: str) -> dict:
        return {"HOME": str(rig.tmp_path / name / "home"), "TMPDIR": str(rig.tmp_path / name / "tmp")}

    first, _, first_run = rig.run("first", _arm(rig), extra_env=places("first"))
    rig.publish(first_run, first)
    second, status, _ = rig.run("second", _arm(rig), extra_env=places("second"))

    assert _keys(second) == _keys(first), "where the sandbox keeps its things is not an input of the build"
    assert _states(second) == dict.fromkeys(STATES, "hit") and second.jit_cache.builds_seen == 0
    assert status["message"] == ""
    other, _, _ = rig.run("other", _arm(rig, envs={"ENABLE_CK": "0"}), extra_env=places("other"))
    assert _keys(other)["aiter_jit"] != _keys(first)["aiter_jit"], (
        "a variable that selects what is built still splits it"
    )


def test_a_kernel_triton_compiled_in_a_sandbox_that_is_gone_is_found_by_the_next_one_and_nothing_is_rewritten(
    rig,
) -> None:
    first, _, first_run = rig.run("first", _arm(rig))
    rig.publish(first_run, first)
    shutil.rmtree(first_run / "scratch")  # the sandbox, and the directories the group was written with
    second, status, _ = rig.run("second", _arm(rig))

    assert rig.grid.triton_found == [True], "Triton found the kernel through the group: it compiled nothing"
    group = rig.grid.triton_group
    assert group and all(Path(target).is_file() and str(rig.tmp_path / "second") in target for target in group.values())
    assert status["message"] == "" and _kind(second, "triton").contribution is None, "no file was rewritten"


def test_the_status_message_says_what_failed_verification_before_the_files_a_compiler_changed(rig, monkeypatch) -> None:
    base, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, base)
    digest = hashlib.sha256(MODULE).hexdigest()
    blob = rig.store.root / "blobs" / digest[:2] / digest
    blob.chmod(0o644)
    blob.write_bytes(bytes(reversed(MODULE)))
    monkeypatch.setattr(_CompilingGrid, "recompile_triton", True)

    _, status, _ = rig.run("noisy", _arm(rig))

    message = status["message"]
    assert message.index("the bytes are not the 36 bytes whose hash is listed; ignored") < message.index(
        "changed by the test"
    )
    assert "jit cache: triton: 4 files changed by the test; the stored bytes stay (" in message


def test_a_succeeded_tests_status_message_is_made_of_its_notes_whole_from_the_first(rig, monkeypatch) -> None:
    joined = []
    monkeypatch.setattr(runner, "join_notes", lambda notes: joined.append(list(notes)) or "from join_notes")

    _, status, _ = rig.run("only", _arm(rig))

    assert status["message"] == "from join_notes" and len(joined) == 1


def test_the_result_states_the_cold_start_parts_with_or_without_a_cache(rig) -> None:
    manifest, _, _ = rig.run("plain", _arm(rig), cache=False)

    assert manifest.jit_cache is None, "no cache configured: the result has no cache block"
    assert (manifest.timing.jit_compile_sec, manifest.timing.model_load_sec) == (12.5, 41.25)


def test_a_candidate_that_differs_only_in_server_arguments_or_a_framework_patch_finds_the_same_cache(rig) -> None:
    base, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, base)

    sglang_patch = _arm(
        rig,
        benchmark={"server_args": "--chunked-prefill-size 4096"},
        envs={"SGLANG_MAX_BATCH": "64"},
        roots=lambda run: [rig.root(run, "sglang", "srt/a.py", b"def schedule(batch):\n    return sorted(batch)\n")],
    )
    other, _, _ = rig.run("other", sglang_patch)

    assert _keys(other) == _keys(base), "nothing the candidate changed is an input of the build"
    assert _states(other) == dict.fromkeys(STATES, "hit") and other.jit_cache.builds_seen == 0


def test_an_aiter_patch_misses_the_aiter_caches_and_still_hits_the_one_triton_keeps_content_addressed(rig) -> None:
    base, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, base)

    patched = _arm(rig, roots=lambda run: [rig.root(run, "aiter", "jit/optCompilerConfig.json", b'{"flags": ["-O0"]}')])
    candidate, _, _ = rig.run("patched", patched)

    assert _states(candidate) == {"aiter_jit": "miss", "aiter_cpp": "miss", "triton": "hit"}
    assert (
        _keys(candidate)["aiter_jit"] != _keys(base)["aiter_jit"]
        and _keys(candidate)["triton"] == _keys(base)["triton"]
    )
    assert candidate.jit_cache.builds_seen == 1, "it builds what the patch may have changed"


def _aiter_patch(rig: Rig):
    return lambda run: [rig.root(run, "aiter", "jit/optCompilerConfig.json", b'{"flags": ["-O0"]}')]


def test_a_patched_aiter_finds_the_cache_another_sandbox_filled_though_the_farm_names_its_meta_directory(rig) -> None:
    first, _, first_run = rig.run("first", _arm(rig, roots=_aiter_patch(rig)))
    rig.publish(first_run, first)
    second, _, _ = rig.run("second", _arm(rig, roots=_aiter_patch(rig)))

    assert _keys(second)["aiter_jit"] == _keys(first)["aiter_jit"], "the image's meta directory is not the sandbox's"
    assert _states(second)["aiter_jit"] == "hit"


def test_a_meta_directory_the_sheet_names_instead_of_the_images_is_an_input_of_the_aiter_build(rig) -> None:
    elsewhere = rig.tmp_path / "other-meta"
    (elsewhere / "csrc").mkdir(parents=True)

    image, _, _ = rig.run("image", _arm(rig, roots=_aiter_patch(rig)))
    named, _, _ = rig.run("named", _arm(rig, roots=_aiter_patch(rig), envs={"AITER_META_DIR": str(elsewhere)}))

    assert _keys(named)["aiter_jit"] != _keys(image)["aiter_jit"], "sources from another tree are another build"


def _pinned_meta_directory(rig: Rig) -> str | None:
    """The ``AITER_META_DIR`` the last run's server was given by its YAML."""
    return rig.grid.configs[0]["benchmark"]["envs"].get("AITER_META_DIR")


_FARM_META = "/farm/site/aiter_meta"


def test_a_patched_aiter_server_is_told_the_farms_mirrored_meta_directory_and_an_unpatched_one_nothing(rig) -> None:
    rig.run("base", _arm(rig))
    assert _pinned_meta_directory(rig) is None

    rig.run("patched", _arm(rig, roots=_aiter_patch(rig)))

    pinned = _pinned_meta_directory(rig)
    assert pinned is not None and pinned.endswith(_FARM_META), "the farm's mirror of the image's meta directory"


def test_a_meta_directory_the_server_inherits_is_kept_and_one_the_test_unsets_is_not(rig) -> None:
    inherited = {"AITER_META_DIR": "/inherited/meta"}

    rig.run("environment", _arm(rig, roots=_aiter_patch(rig)), extra_env=inherited)
    assert _pinned_meta_directory(rig) is None, "the runner's environment already names one"
    rig.run("process", _arm(rig, roots=_aiter_patch(rig), process_env=inherited))
    assert _pinned_meta_directory(rig) is None, "the test's own process variables name one"
    rig.run("unset", _arm(rig, roots=_aiter_patch(rig), unset_envs=["AITER_META_DIR"]), extra_env=inherited)
    assert str(_pinned_meta_directory(rig)).endswith(_FARM_META), "the test removes it, so the server would find none"


def test_a_runtime_snapshot_is_part_of_the_aiter_key_because_the_sources_outside_the_package_are_in_it(rig) -> None:
    base, _, _ = rig.run("base", _arm(rig))

    def with_snapshot(run: Path) -> dict:
        snapshot, manifest = _snapshot(run)
        return rig.sheet(run, runtime={"snapshot": snapshot, "manifest_sha256": manifest})

    snapped, _, _ = rig.run("snapped", with_snapshot)

    assert _keys(snapped)["aiter_jit"] != _keys(base)["aiter_jit"] and _keys(snapped)["triton"] == _keys(base)["triton"]


def test_a_tuned_table_is_keyed_by_its_content_not_by_where_it_is(rig) -> None:
    def with_table(name: str, content: str, where: str):
        def sheet(run: Path) -> dict:
            table = run / where / "table.csv"
            table.parent.mkdir(parents=True, exist_ok=True)
            table.write_text(content)
            return rig.sheet(run, envs={TABLE_VARIABLE: str(table)})

        return sheet

    one, _, one_run = rig.run("one", with_table("one", "M,N,K\n1,2,3\n", "here"))
    rig.publish(one_run, one)
    moved, _, _ = rig.run("moved", with_table("moved", "M,N,K\n1,2,3\n", "elsewhere"))
    edited, _, _ = rig.run("edited", with_table("edited", "M,N,K\n1,2,4\n", "here"))

    assert _keys(moved)["aiter_jit"] == _keys(one)["aiter_jit"] and _states(moved)["aiter_jit"] == "hit"
    assert _keys(edited)["aiter_jit"] != _keys(one)["aiter_jit"] and _states(edited)["aiter_jit"] == "miss"
    assert _states(edited)["triton"] == "hit", "a code generator's table is not an input of Triton"


def test_a_variable_the_build_reads_changes_the_key_and_one_it_never_reads_does_not(rig) -> None:
    base, _, _ = rig.run("base", _arm(rig))
    unrelated, _, _ = rig.run("unrelated", _arm(rig, envs={"SGLANG_X": "1", "AITER_DISABLE_FMHA_OPUS": "1"}))
    build, _, _ = rig.run("build", _arm(rig, envs={"ENABLE_CK": "0"}))

    assert _keys(unrelated) == _keys(base), "a call-time switch only chooses among modules"
    assert _keys(build)["aiter_jit"] != _keys(base)["aiter_jit"] and _keys(build)["triton"] == _keys(base)["triton"]


def test_a_framework_patch_that_can_change_the_build_from_inside_the_server_is_part_of_the_key(rig) -> None:
    base, _, _ = rig.run("base", _arm(rig))
    sneaky = b"import os\nos.environ['ENABLE_CK'] = '0'\n"
    patched, _, _ = rig.run("patched", _arm(rig, roots=lambda run: [rig.root(run, "sglang", "srt/a.py", sneaky)]))

    assert _keys(patched)["aiter_jit"] != _keys(base)["aiter_jit"] and _keys(patched)["triton"] == _keys(base)["triton"]


def test_a_test_that_sets_aiter_rebuild_does_not_use_the_aiter_caches_and_says_so(rig) -> None:
    base, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, base)

    manifest, status, _ = rig.run("rebuild", _arm(rig, envs={"AITER_REBUILD": "1"}))

    assert _states(manifest) == {"aiter_jit": "disabled", "aiter_cpp": "disabled", "triton": "hit"}
    assert _keys(manifest)["aiter_jit"] is None
    assert "jit cache: aiter_jit is not used: AITER_REBUILD is set" in status["message"]


def test_a_sheet_that_sets_a_cache_directory_itself_keeps_it_and_that_kind_is_not_used(rig) -> None:
    manifest, status, _ = rig.run("own", _arm(rig, envs={"TRITON_CACHE_DIR": "/somewhere/else"}))

    assert _states(manifest)["triton"] == "disabled" and _states(manifest)["aiter_jit"] == "miss"
    assert "jit cache: triton is not used: the sheet sets TRITON_CACHE_DIR itself" in status["message"]


def test_a_tampered_cache_is_ignored_the_test_passes_the_result_says_so_and_the_next_publication_repairs_it(
    rig,
) -> None:
    base, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, base)
    digest = hashlib.sha256(MODULE).hexdigest()
    blob = rig.store.root / "blobs" / digest[:2] / digest
    blob.chmod(0o644)
    blob.write_bytes(bytes(reversed(MODULE)))

    tampered, status, tampered_run = rig.run("tampered", _arm(rig))

    assert _states(tampered) == {"aiter_jit": "verify_failed", "aiter_cpp": "hit", "triton": "hit"}
    assert tampered.outcome.returncode == 0 and tampered.jit_cache.builds_seen == 1, "it built the module itself"
    assert "the bytes are not the 36 bytes whose hash is listed; ignored" in status["message"]
    assert rig.publish(tampered_run, tampered) == 1
    repaired, _, _ = rig.run("repaired", _arm(rig))
    assert _states(repaired) == dict.fromkeys(STATES, "hit") and blob.read_bytes() == MODULE


def test_a_test_that_failed_contributes_nothing(rig) -> None:
    manifest, _, run = rig.run("failed", _arm(rig), fail=True)

    assert manifest.outcome.returncode != 0
    assert all(kind.contribution is None for kind in manifest.jit_cache.kinds)
    assert not (run / "out" / "jit-cache").exists() or not any((run / "out" / "jit-cache").iterdir())


def test_the_contribution_is_staged_outside_the_files_the_result_lists_and_is_published_only_as_signed(rig) -> None:
    first, _, run = rig.run("first", _arm(rig))

    assert not any(f.path.startswith("jit-cache") for f in first.files), "a cache of gigabytes is not result data"
    staged = run / "out" / "jit-cache" / "aiter_jit"
    assert (staged / "manifest.json").is_file() and STAGING_PATH == "/out/jit-cache"
    contribution = _kind(first, "aiter_jit").contribution
    assert (
        contribution is not None
        and hashlib.sha256((staged / "manifest.json").read_bytes()).hexdigest() == contribution.sha256
    )
    (staged / "manifest.json").write_bytes((staged / "manifest.json").read_bytes() + b" ")  # swapped after the signing

    assert rig.publish(run, first) == 2, "the swapped list is not the signed one: the other two kinds are published"
    assert not (rig.store.root / "snapshots" / "aiter_jit").exists()


def test_the_manifest_block_round_trips_through_the_strict_reader(rig) -> None:
    first, _, _ = rig.run("first", _arm(rig))

    again = JitCacheFacts.parse(
        json.loads((rig.tmp_path / "first" / "out" / "manifest.json").read_text())["jit_cache"], "jit_cache"
    )

    assert again == first.jit_cache


def _contributions(manifest: ResultManifest) -> dict[str, bool]:
    return {kind.kind: kind.contribution is not None for kind in manifest.jit_cache.kinds}


def test_a_test_with_patched_code_the_key_does_not_cover_reads_the_cache_and_adds_nothing_to_it(rig) -> None:
    """Code in the server can write any file into the directory its compiler uses, and the key of a test that carries a
    framework patch outside the build is the key of every test that does not: what it wrote must not reach them."""
    patch = _arm(
        rig, roots=lambda run: [rig.root(run, "sglang", "srt/a.py", b"def schedule(batch):\n    return batch\n")]
    )

    patched, status, patched_run = rig.run("patched", patch)

    assert _states(patched) == dict.fromkeys(STATES, "miss") and patched.jit_cache.builds_seen == 1
    assert _contributions(patched) == dict.fromkeys(STATES, False), "it built them, and none of it is published"
    assert "jit cache: aiter_jit is read only in this test" in status["message"]
    assert rig.publish(patched_run, patched) == 0 and not (rig.store.root / "snapshots").exists()
    base, _, base_run = rig.run("base", _arm(rig))
    assert _contributions(base) == dict.fromkeys(STATES, True), "a test with no patch is the one that fills the cache"
    rig.publish(base_run, base)
    again, _, _ = rig.run("again", patch)
    assert _states(again) == dict.fromkeys(STATES, "hit"), "and the patched test is given it"


def test_an_aiter_patch_is_covered_by_the_aiter_key_so_those_caches_take_its_contribution_and_triton_does_not(
    rig,
) -> None:
    patched = _arm(rig, roots=lambda run: [rig.root(run, "aiter", "jit/optCompilerConfig.json", b'{"flags": ["-O0"]}')])

    manifest, _, _ = rig.run("patched", patched)

    assert _contributions(manifest) == {"aiter_jit": True, "aiter_cpp": True, "triton": False}


def test_a_directory_attachment_is_code_no_key_covers_so_every_kind_is_read_only(rig) -> None:
    manifest, status, _ = rig.run("attached", _arm(rig, attachments=lambda run: [rig.directory_attachment(run)]))

    assert _contributions(manifest) == dict.fromkeys(STATES, False)
    assert "jit cache: triton is read only in this test" in status["message"]


def test_a_run_whose_server_never_imported_the_patched_aiter_ends_as_an_inert_patch_not_a_measurement(rig) -> None:
    """The stand-in server is no Python process, so none notes importing the farm's aiter: the patch never ran."""
    base, _, _ = rig.run("base", _arm(rig))
    patched, _, run = rig.run("patched", _arm(rig, roots=_aiter_patch(rig)))

    assert base.outcome.failure_class is None
    assert (patched.outcome.failure_class, patched.outcome.returncode) == ("infra", HOSTED_INFRA_FAILURE_RC)
    assert str(patched.outcome.failure_reason).startswith("inert_patch: no process of the server noted importing aiter")
    assert (run / "scratch" / "test" / "farm" / "site" / "sitecustomize.py").is_file(), "the runner installed the watch"
