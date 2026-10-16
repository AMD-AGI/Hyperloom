# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's main flow: wait for the sheet, check it, rebuild the environment, run the test, sign the result (4.8.2)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
import time
from collections.abc import MutableMapping
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyperloom.measure import signing
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.failures import Disposition, classify_test_failure
from hyperloom.measure.local import LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.manifest import encode_manifest, RESULT_SCHEMA
from hyperloom.measure.runner import hardware
from hyperloom.measure.attachments import AttachmentError, place_attachments
from hyperloom.measure.farm import Farm, FarmError, build_farm, verify_farm
from hyperloom.measure.runner.inventory import OutputRejected, inventory, write_farm_inventory
from hyperloom.measure.runner.platform_env import align_platform_env, take_topology
from hyperloom.measure.runner.rewrite import Placement, rewrite_config
from hyperloom.measure.runner.rounds import write_rounds
from hyperloom.measure.runner.service import Progress, RunnerFailure, wait_for_file, write_status
from hyperloom.measure.runner.tools import ToolError, ensure_inferencex, ensure_magpie
from hyperloom.measure.socket import RoundPlan, SessionFacts, TestCall
from hyperloom.measure.spec import MeasurementSpec, fingerprint, spec_sha256
from hyperloom.orchestrator.actions.executors import harvest_leaked_artifacts

__all__ = ["run"]

_START_FAILED = "runner_start_failed"
_SPEC_REJECTED = "spec_rejected"
_FAILED = "failed"
_FILES_DIR = "files"
_KEY_ENV = "HYPERLOOM_SIGNING_KEY_FILE"
_TIMED_OUT_RC = 124


@dataclass
class _Context:
    in_dir: Path
    out_dir: Path
    work: Path
    cache: Path
    environ: MutableMapping[str, str]
    started: float

    @property
    def files(self) -> Path:
        return self.out_dir / _FILES_DIR

    def read_blob(self, address: str) -> bytes:
        name = address.removeprefix("sha256:")
        try:
            return (self.in_dir / "blobs" / name).read_bytes()
        except OSError as exc:
            raise FarmError(f"blob {address} is not in /in/blobs") from exc


def _version() -> str:
    explicit = os.environ.get("HYPERLOOM_CODE_REVISION")
    if explicit:
        return explicit
    try:
        return importlib.metadata.version("hyperloom-inference_optimizer")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _visible(environ: MutableMapping[str, str]) -> list[int]:
    return [int(p) for p in environ.get("ROCR_VISIBLE_DEVICES", "").split(",") if p.strip().isdigit()]


def _load_key(environ: MutableMapping[str, str]):
    path = environ.pop(_KEY_ENV, "")
    if not path:
        raise RunnerFailure(_START_FAILED, f"{_KEY_ENV} is not set")
    try:
        return signing.load_private_key(Path(path))
    except (signing.SigningError, OSError) as exc:
        raise RunnerFailure(_START_FAILED, f"signing key unusable: {exc}") from exc


def _parse(raw: bytes) -> MeasurementSpec:
    try:
        return MeasurementSpec.from_bytes(raw)
    except DocumentError as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc


def _check_model(yaml_text: str, hashes: dict[str, str]) -> None:
    model = ((yaml.safe_load(yaml_text) or {}).get("benchmark") or {}).get("model")
    for relative, wanted in hashes.items():
        try:
            actual = hashlib.sha256((Path(str(model)) / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise RunnerFailure(_SPEC_REJECTED, f"model file {relative} unreadable: {exc}") from exc
        if actual != wanted:
            raise RunnerFailure(_SPEC_REJECTED, f"model file {relative} differs from the recorded hash")


def _preflight(spec: MeasurementSpec, visible: list[int]):
    facts = hardware.read_gpu_facts(visible)
    if facts is not None:
        h = spec.hardware
        problems = hardware.check_hardware(
            facts,
            gpu_type=h.gpu_type,
            gpu_count=h.gpu_count,
            power_cap_w=h.power_cap_w,
            partition_mode=h.partition_mode,
        )
        if problems:
            raise RunnerFailure("preflight_mismatch", "; ".join(problems))
    return facts


def _prepare(ctx: _Context, spec: MeasurementSpec, visible: list[int]) -> tuple[list[Path], Path, Farm | None]:
    """Tools, attachments, farm and the rewritten YAML; returns the point path and the InferenceX checkout."""
    try:
        inferencex = ensure_inferencex(spec.tools.inferencex, ctx.cache)
        if spec.procedure.benchmark_backend == "magpie":
            ctx.environ["MAGPIE_PATH"] = str(ensure_magpie(spec.tools.magpie, ctx.cache))
    except ToolError as exc:
        raise RunnerFailure(_START_FAILED, str(exc)) from exc
    try:
        placed = place_attachments(spec.attachments, ctx.work / "attach", ctx.read_blob)
        envs = (yaml.safe_load(spec.config[0].yaml).get("benchmark") or {}).get("envs") or {}
        python = str(envs.get("HYPERLOOM_FRAMEWORK_PYTHON") or "python3")
        farm = build_farm(spec.environment.roots, ctx.work / "farm", python=python, read_blob=ctx.read_blob)
        placement = Placement(
            placed=placed,
            pid_dir=ctx.work / "pids",
            inferencex_path=inferencex,
            visible_devices=ctx.environ.get("ROCR_VISIBLE_DEVICES"),
            farm=farm,
        )
        points = [
            rewrite_config(point.yaml, ctx.work / f"point_{index}.yaml", placement)
            for index, point in enumerate(spec.config)
        ]
        if farm is not None:
            for point in points:
                rewritten = yaml.safe_load(point.read_text())["benchmark"]["envs"]
                verify_farm(farm, rewritten, python=python, cwd=ctx.work)
    except (AttachmentError, FarmError) as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc
    return points, inferencex, farm


def _call(ctx: _Context, spec: MeasurementSpec, points: list[Path], inferencex: Path, progress: Progress) -> TestCall:
    p = spec.procedure
    remaining = p.deadline_unix - time.time() if p.deadline_unix else p.timeout_sec
    try:
        rounds = RoundPlan(warmup_rounds=p.warmup_rounds, measure_rounds=p.measure_rounds)
    except ValueError as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc
    ctx.environ["HYPERLOOM_BENCHMARK_BACKEND"] = p.benchmark_backend
    ctx.files.mkdir(parents=True, exist_ok=True)
    return TestCall(
        points=tuple(points),
        output_dir=ctx.files,
        timeout_sec=int(max(1, min(p.timeout_sec, remaining))),
        cwd=str(ctx.work),
        magpie_python="",
        task_id=spec.request_id,
        kind=spec.kind,  # type: ignore[arg-type]
        prep=p.prep,  # type: ignore[arg-type]
        rounds=rounds,
        accuracy=p.accuracy,
        result_dir=str(ctx.files / spec.outputs.result_dir),
        silence_timeout_sec=p.silence_timeout_sec,
        on_output=progress.touch,
        env_overrides=dict(spec.process_env),
        unset_envs=spec.unset_envs,
        session_facts=SessionFacts((), str(inferencex), spec.tools.magpie),
    )


def _execute(call: TestCall):
    """Run the test; a timeout is an outcome, not an exception, from here on."""
    try:
        return LocalMeasurementBackend().run_test(call), False
    except subprocess.TimeoutExpired:
        return None, True
    except TestCallRefused as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc


def _outcome_document(outcome, timed_out: bool) -> dict:
    if timed_out:
        verdict = classify_test_failure(returncode=_TIMED_OUT_RC, timed_out=True)
        rc, out, err = _TIMED_OUT_RC, "", "the test exceeded its time limit"
    else:
        rc, out, err = outcome.returncode, outcome.stdout, outcome.stderr
        verdict = classify_test_failure(returncode=rc, logs=[err])
    failed = verdict.disposition is not Disposition.SUCCEEDED
    failure_class = verdict.failure_class.value if verdict.failure_class else None
    return {
        "returncode": rc,
        "timed_out": timed_out,
        "error_class": verdict.failure_reason if failed and failure_class is None else None,
        "failure_class": failure_class,
        "failure_reason": verdict.failure_reason if failure_class else None,
        "stdout_tail": out[-4000:],
        "stderr_tail": err[-4000:],
    }


def _check_node(before, after, visible: list[int], facts) -> None:
    if facts is not None and len(hardware.read_gpu_facts(visible) or facts) < len(facts):
        raise RunnerFailure("node_fault", "fewer GPUs are visible after the test")
    if before and after:
        for gpu, was in before.items():
            if after.get(gpu) and after[gpu].uncorrectable > was.uncorrectable:
                raise RunnerFailure("node_fault", f"GPU {gpu} reported new uncorrectable ECC errors")


@dataclass(frozen=True)
class _Signed:
    """The sheet being answered and the key that signs the answer."""

    spec: MeasurementSpec
    raw: bytes
    key: Any


@dataclass(frozen=True)
class _Findings:
    """What the run observed and the bundle records besides the outcome."""

    files: Any
    facts: Any
    changed: Any
    timing: dict
    visible: list[int]


def _write_bundle(ctx: _Context, signed: _Signed, outcome: dict, found: _Findings) -> None:
    spec, raw, key = signed.spec, signed.raw, signed.key
    files, facts, changed, timing, visible = found.files, found.facts, found.changed, found.timing, found.visible
    pem = signing.public_key_pem(key)
    first = facts[0] if facts else None
    document: dict = {"outcome": outcome}
    document.update(
        {
            "schema": RESULT_SCHEMA,
            "request_id": spec.request_id,
            "spec_fingerprint": fingerprint(spec, hyperloom_version=_version()),
            "spec_sha256": spec_sha256(raw),
            "surface_sha256": None,
            "runner": {
                "sandbox_id": ctx.environ.get("E2B_SANDBOX_ID", ""),
                "signing_key_fingerprint": signing.public_key_fingerprint(pem),
                "node_name": ctx.environ.get("NODE_NAME"),
                "image_digest": spec.image.digest,
                "gpu_model": first.model if first else None,
                "power_cap_w": first.power_cap_w if first else None,
                "partition_mode": first.partition_mode if first else None,
                "hyperloom_version": _version(),
                "magpie_commit": spec.tools.magpie,
                "inferencex_commit": spec.tools.inferencex,
                "gpu_ids": visible,
            },
            "platform_env_changed": list(changed),
            "timing": timing,
            "files": [f.to_document() for f in files],
        }
    )
    manifest = encode_manifest(document)
    (ctx.out_dir / "manifest.json").write_bytes(manifest)
    (ctx.out_dir / "manifest.sig").write_text(signing.sign(manifest, key), encoding="ascii")


def _run_sheet(ctx: _Context, spec_path: Path, key, progress: Progress) -> None:
    raw = spec_path.read_bytes()
    received = time.time()
    spec = _parse(raw)
    visible = _visible(ctx.environ)
    _check_model(spec.config[0].yaml, dict(spec.model_hashes))
    facts = _preflight(spec, visible)
    before = hardware.ecc_counts(visible)
    changed = align_platform_env(ctx.environ, spec.platform_env)
    progress.phase("preparing")
    points, inferencex, farm = _prepare(ctx, spec, visible)
    call = _call(ctx, spec, points, inferencex, progress)
    progress.phase("running")
    started = time.time()
    outcome, timed_out = _execute(call)
    progress.phase("collecting")
    harvest_leaked_artifacts(ctx.files, subprocess_started_unix=started, leak_root=ctx.work)
    _check_node(before, hardware.ecc_counts(visible), visible, facts)
    write_farm_inventory(ctx.files, farm)
    write_rounds(ctx.files, outcome)
    try:
        files = inventory(ctx.files, max_total_bytes=spec.outputs.max_total_bytes)
    except OutputRejected as exc:
        raise RunnerFailure(exc.error_class, str(exc)) from exc
    timing = {
        "runner_started_unix": ctx.started,
        "spec_received_unix": received,
        "server_ready_unix": None,
        "finished_unix": time.time(),
    }
    _write_bundle(
        ctx,
        _Signed(spec, raw, key),
        _outcome_document(outcome, timed_out),
        _Findings(files, facts, changed, timing, visible),
    )


def run(
    in_dir: Path,
    out_dir: Path,
    *,
    scratch: Path,
    environ: MutableMapping[str, str] | None = None,
    spec_wait_sec: float = 600.0,
) -> int:
    """Run one test from ``in_dir/spec.json`` and leave the signed result in ``out_dir``; returns the exit code."""
    environ = os.environ if environ is None else environ
    out_dir.mkdir(parents=True, exist_ok=True)
    work = scratch / "test"
    for sub in ("farm", "attach", "pids"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    ctx = _Context(in_dir, out_dir, work, scratch / "cache", environ, time.time())
    with Progress(out_dir) as progress:
        try:
            key = _load_key(environ)
            take_topology(environ)
            progress.phase("waiting_spec")
            if not wait_for_file(in_dir / "READY", spec_wait_sec):
                write_status(out_dir, state=_FAILED, error_class="no_spec", message="no instruction sheet arrived")
                return 0
            _run_sheet(ctx, in_dir / "spec.json", key, progress)
        except RunnerFailure as failure:
            write_status(out_dir, state=_FAILED, error_class=failure.error_class, message=str(failure))
            return failure.exit_code
        except Exception as exc:  # noqa: BLE001 - the application reads status.json, not a traceback
            write_status(out_dir, state=_FAILED, error_class="runner_error", message=f"{type(exc).__name__}: {exc}")
            return 1
        write_status(out_dir, state="succeeded")
    return 0
