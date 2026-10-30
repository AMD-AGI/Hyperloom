# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's main flow: wait for the sheet, check it, rebuild the environment, run the test, sign the result (4.8.2)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import os
import subprocess
import time
import traceback
from collections.abc import Mapping, MutableMapping
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.outside_facts import ROCR_VISIBLE_DEVICES_ENV
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.contracts.measurement import AttachmentRef, FailureClass
from hyperloom.measure import signing
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.failures import (
    DISK_FULL,
    FOREIGN_PROCESS_ON_CARD,
    Disposition,
    FailureClassification,
    classify_test_failure,
    is_disk_full,
)
from hyperloom.measure.local import LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.magpie_launchers import install_launchers
from hyperloom.measure.manifest import encode_manifest, manifest_notes, ManifestFile, RESULT_SCHEMA
from hyperloom.measure.phases import (
    BOOTSTRAP,
    ENV_REBUILD,
    EVAL_DEPS,
    PREFLIGHT,
    RESULT_SIGNING,
    RUNNER_START,
    SHEET_WAIT,
    PhaseRecorder,
    manifest_document,
)
from hyperloom.measure.pins import IMAGE_DIGEST_VAR
from hyperloom.measure.jit.native.config import store_from_environment
from hyperloom.measure.jit.native.laying import NamedBuild, NativeLayError, SheetBuilds, sheet_builds
from hyperloom.measure.jit.session import JitCacheSession
from hyperloom.measure.runner import hardware
from hyperloom.measure.runner.card_watch import BEFORE_TEST, CardWatch
from hyperloom.measure.runner.eval_deps import EVAL_DEPS_LOG, EvalDepsError, install_eval_deps
from hyperloom.measure.runner.aiperf_patch import ReplayClientPatchError, patch_replay_client
from hyperloom.measure.attachments import AttachmentError, place_attachments
from hyperloom.measure.farm import (
    Farm,
    FarmError,
    build_farm,
    built_roots,
    inherited_environment,
    patched_configs,
    server_environment,
    server_python,
    verify_farm,
)
from hyperloom.measure.runner.jit import STAGING_DIR, ColdStart, Surroundings, build_session, cold_start
from hyperloom.measure.runner.import_outcome import install_import_watch, refine_imports
from hyperloom.measure.runner.jit_outcome import refine_outcome
from hyperloom.measure.runner.own_jit import OWN_AITER_JIT_DIR, own_aiter_jit
from hyperloom.measure.runner.rebuild_watchdog import raise_rebuild_watchdog
from hyperloom.measure.runner.trace_fit import fit_traces
from hyperloom.measure.runner.own_tmp import own_tmpdir
from hyperloom.measure.runner.inventory import OutputRejected, drop_agentx_runtime, inventory, write_farm_inventory
from hyperloom.measure.runner.platform_env import align_platform_env, take_topology
from hyperloom.measure.runner.rewrite import Placement, rewrite_config
from hyperloom.measure.runner.rounds import write_rounds
from hyperloom.measure.runner.service import Progress, RunnerFailure, join_notes, wait_for_file, write_status
from hyperloom.measure.runner.snapshot import SnapshotError, check_snapshot
from hyperloom.measure.runner.tools import ToolError, ensure_inferencex, ensure_magpie
from hyperloom.measure.socket import RoundPlan, SessionFacts, TestCall
from hyperloom.measure.spec import MeasurementSpec, fingerprint, spec_sha256
from hyperloom.measure.surface import extract_surface
from hyperloom.orchestrator.actions.cancel_channel import CancelScope
from hyperloom.orchestrator.actions.executors import harvest_leaked_artifacts
from hyperloom.sandbox.card_occupancy import card_guard, own_sandbox_processes

__all__ = ["run"]

_START_FAILED = "runner_start_failed"
_SPEC_REJECTED = "spec_rejected"
_NODE_FAULT = "node_fault"
_FAILED = "failed"
_FILES_DIR = "files"
_KEY_ENV = "HYPERLOOM_SIGNING_KEY_FILE"
#: The unix times the start-up script began and ended, which the script sets (``measure.runner.bootstrap``).
_BOOTSTRAP_STARTED_ENV = "HYPERLOOM_BOOTSTRAP_STARTED_UNIX"
_BOOTSTRAP_ENDED_ENV = "HYPERLOOM_BOOTSTRAP_ENDED_UNIX"
_TIMED_OUT_RC = 124


@dataclass
class _Context:
    in_dir: Path
    out_dir: Path
    work: Path
    cache: Path
    environ: MutableMapping[str, str]
    started: float
    #: Watches the visible cards for a process that is not this sandbox's; the signed result states what it saw.
    cards: CardWatch
    #: Checks that could not be made on this machine; a successful run says so in ``status.json``.
    notes: list[str] = field(default_factory=list)
    #: The phases of this run on this machine's clock; the signed result states them.
    phases: PhaseRecorder = field(default_factory=PhaseRecorder)

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
    return [int(p) for p in environ.get(ROCR_VISIBLE_DEVICES_ENV, "").split(",") if p.strip().isdigit()]


def _load_key(environ: MutableMapping[str, str]):
    path = environ.pop(_KEY_ENV, "")
    if not path:
        raise RunnerFailure(_START_FAILED, f"{_KEY_ENV} is not set")
    try:
        key = signing.load_private_key(Path(path))
    except (signing.SigningError, OSError) as exc:
        raise RunnerFailure(_START_FAILED, f"signing key unusable: {exc}") from exc
    # The key lives in this process from here on. A file left on disk would be readable by everything the test runs
    # as this account (the inference server, a repro case), which could then sign a result of its own.
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        raise RunnerFailure(_START_FAILED, f"signing key file cannot be removed: {type(exc).__name__}") from exc
    return key


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


def _check_runtime(spec: MeasurementSpec) -> None:
    runtime = spec.environment.runtime
    if runtime is None:
        return
    try:
        check_snapshot(Path(runtime.snapshot), runtime.manifest_sha256)
    except SnapshotError as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc


def _check_cards_free(ctx: _Context, visible: list[int]) -> None:
    """No process holds a visible card yet, and the card's memory is free (design 4.8.1 step 3).

    The process list says who holds a card, so it goes first: a card that is held fails as ``foreign_process_on_card``,
    naming the process, and only a card whose memory is taken by something it does not list fails as a start failure.
    """
    if not visible:
        ctx.notes.append(f"{ROCR_VISIBLE_DEVICES_ENV} names no card; the cards were not checked for other processes")
    ctx.cards.check(BEFORE_TEST)
    ctx.cards.enforce_guard()
    _check_vram_free(ctx, visible)


def _check_vram_free(ctx: _Context, visible: list[int]) -> None:
    """A visible GPU that still holds memory once the last test's server had time to release it is not free for the
    test (design 4.8.1 step 3)."""
    busy = hardware.released_gpus(visible)
    if busy is None:
        ctx.notes.append("GPU memory use could not be read; the visible GPUs were not checked for free memory")
    elif busy:
        raise RunnerFailure(
            _START_FAILED, f"GPU {', '.join(map(str, busy))} already holds memory: the card is not free"
        )


def _image_problems(ctx: _Context, wanted: str) -> list[str]:
    """The image the sheet is for against the digest the machine is told it runs; no container can state its own."""
    actual = ctx.environ.get(IMAGE_DIGEST_VAR, "").strip()
    if not actual:
        ctx.notes.append(f"{IMAGE_DIGEST_VAR} is not set; the image digest was not compared with the sheet's")
        return []
    return [] if actual == wanted else [f"the machine runs image {actual}, the sheet is for {wanted}"]


def _preflight(ctx: _Context, spec: MeasurementSpec, visible: list[int]):
    facts = hardware.read_gpu_facts(visible)
    problems = _image_problems(ctx, spec.image.digest)
    if facts is None:
        ctx.notes.append("GPU model, power cap and partition mode could not be read; they were not compared")
    else:
        h = spec.hardware
        problems += hardware.check_hardware(
            facts,
            gpu_type=h.gpu_type,
            gpu_count=h.gpu_count,
            power_cap_w=h.power_cap_w,
            partition_mode=h.partition_mode,
        )
    if problems:
        raise RunnerFailure("preflight_mismatch", "; ".join(problems))
    return facts


def _surface_sha256(spec: MeasurementSpec, points: list[Path]) -> str:
    """The hash of the measurement surface of the YAML the test will execute (design 4.20 step 3).

    The points of a sheet differ only in what the concurrency level decides, so the first speaks for all of them. Under
    Magpie the benchmark script a test executes is Magpie's own copy of its launcher, which the backend puts in the
    InferenceX checkout before the first round; it is put there first so that the hash is of the file that runs.
    """
    if spec.procedure.benchmark_backend == "magpie":
        install_launchers(points)
    return extract_surface(yaml.safe_load(points[0].read_text()) or {}).sha256


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
        python = server_python(yaml.safe_load(spec.config[0].yaml) or {}, default="python3")
        farm = build_farm(
            built_roots(spec.environment.roots, _sheet_builds(spec, ctx.environ)),
            ctx.work / "farm",
            python=python,
            read_blob=ctx.read_blob,
            inherited=inherited_environment(ctx.environ, spec.process_env, spec.unset_envs),
        )
        if farm is not None:
            install_import_watch(farm)
        placement = Placement(
            placed=placed,
            pid_dir=ctx.work / "pids",
            inferencex_path=inferencex,
            visible_devices=ctx.environ.get(ROCR_VISIBLE_DEVICES_ENV),
            farm=farm,
            path=ctx.environ.get("PATH"),
        )
        points = [
            rewrite_config(point.yaml, ctx.work / f"point_{index}.yaml", placement)
            for index, point in enumerate(spec.config)
        ]
        if farm is not None:
            for point in points:
                verify_farm(farm, server_environment(yaml.safe_load(point.read_text())), python=python, cwd=ctx.work)
    except (AttachmentError, FarmError, NativeLayError, ValueError) as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc
    return points, inferencex, farm


def _sheet_builds(spec: MeasurementSpec, environ: Mapping[str, str]) -> SheetBuilds | None:
    """The native builds the sheet names, read from the store this runner's environment names."""
    named = tuple(NamedBuild(b.root_id, b.key, b.manifest_sha256) for b in spec.environment.builds)
    return sheet_builds(named, store_from_environment(environ) if named else None, spec.image.digest)


def _patch_replay_client(ctx: _Context, points: list[Path], inferencex: Path) -> None:
    """Make the client of an AgentX test keep the output token count of a reply it found nothing in.

    The patch only adds a field to the export line of a failed request, so a test it cannot be made for runs without it
    and says so: its no-content failures are then counted but not told empty from short.
    """
    if not is_magpie_agentx((yaml.safe_load(points[0].read_text()) or {}).get("benchmark") or {}):
        return
    try:
        patch_replay_client(inferencex)
    except ReplayClientPatchError as exc:
        ctx.notes.append(
            f"the replay client was not patched to export the output token count of a failed request: {exc}"
        )


def _install_eval_deps(ctx: _Context, spec: MeasurementSpec, points: list[Path], inferencex: Path) -> None:
    """Install the accuracy evaluator before the server boots, when the test evaluates through InferenceX's lm-eval.

    In the round it runs on an idle server whose silence the watchdog counts, and only the round's end shows its
    output (``measure.runner.eval_deps``). A test the evaluator cannot be installed for is not run: its accuracy round
    could only fail the same way, later.
    """
    if not spec.procedure.accuracy or spec.procedure.benchmark_backend != "magpie":
        return
    benchmark = (yaml.safe_load(points[0].read_text()) or {}).get("benchmark") or {}
    if is_magpie_agentx(benchmark):
        return
    ctx.phases.begin(EVAL_DEPS)
    try:  # the round's PATH is this sandbox's (``rewrite_config``), so its python3 is the one installed for
        note = install_eval_deps(inferencex, environ=ctx.environ, log_path=ctx.files / EVAL_DEPS_LOG)
    except EvalDepsError as exc:
        raise RunnerFailure(_START_FAILED, str(exc)) from exc
    if note:
        ctx.notes.append(note)


def _case_attachments(spec: MeasurementSpec) -> tuple[AttachmentRef, ...]:
    """The attachments a repro test's case is looked for in, as unpacked under ``attach/<id>/``; no other kind has any."""
    if spec.kind != "repro":
        return ()
    return tuple(AttachmentRef(a.id, a.kind, a.blob.removeprefix("sha256:")) for a in spec.attachments)  # type: ignore[arg-type]


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
        attachments=_case_attachments(spec),
        phases=ctx.phases,
        short_prompts=p.short_prompts,
        long_context=p.long_context,
    )


def _execute(call: TestCall):
    """Run the test; a timeout is an outcome, not an exception, from here on."""
    try:
        return LocalMeasurementBackend().run_test(call), False
    except subprocess.TimeoutExpired:
        return None, True
    except TestCallRefused as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from exc


def _execute_watched(ctx: _Context, call: TestCall):
    """Run the test while the card watch looks at its cards; enforcing, the watch ends it at the first look that finds a
    process outside it, through the cancellation the test is handed."""
    end = CancelScope()
    call = dataclasses.replace(call, cancel=end)
    with own_tmpdir(call, ctx.environ), ctx.cards.watching(lambda: end.cancel(reason=FOREIGN_PROCESS_ON_CARD)):
        return _execute(call)


def _verdict_of(outcome) -> FailureClassification:
    """The backend's verdict when it gave a failure class, else a classification of the outcome's stderr.

    The backend read the server log and the benchmark's stderr of every failed round, which hold the causes the outcome's
    own stderr lacks and which this process does not know the paths of.
    """
    if outcome.failure_class is not None and outcome.failure_reason:
        failure_class = FailureClass(getattr(outcome.failure_class, "value", outcome.failure_class))
        return FailureClassification(Disposition.FAILED, failure_class, outcome.failure_reason)
    return classify_test_failure(returncode=outcome.returncode, logs=[outcome.stderr])


def _outcome_document(outcome, timed_out: bool) -> dict:
    if timed_out:
        verdict = classify_test_failure(returncode=_TIMED_OUT_RC, timed_out=True)
        rc, out, err = _TIMED_OUT_RC, "", "the test exceeded its time limit"
    else:
        rc, out, err = outcome.returncode, outcome.stdout, outcome.stderr
        verdict = _verdict_of(outcome)
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


def _check_node(ctx: _Context, before, after, visible: list[int], facts) -> None:
    if facts is not None:
        now = hardware.read_gpu_facts(visible)
        if now is None:
            raise RunnerFailure(_NODE_FAULT, "the GPUs cannot be read after the test, as they could before it")
        if len(now) < len(facts):
            raise RunnerFailure(_NODE_FAULT, "fewer GPUs are visible after the test")
    if before is None or after is None:
        unread = " and ".join(when for when, counts in (("before", before), ("after", after)) if counts is None)
        ctx.notes.append(f"ECC counts could not be read {unread} the test; there is no ECC verdict")
        return
    for gpu, was in before.items():
        if after.get(gpu) and after[gpu].uncorrectable > was.uncorrectable:
            raise RunnerFailure(_NODE_FAULT, f"GPU {gpu} reported new uncorrectable ECC errors")


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
    surface_sha256: str
    jit_cache: dict | None = None


def _write_bundle(ctx: _Context, signed: _Signed, outcome: dict, found: _Findings) -> None:
    spec, raw, key = signed.spec, signed.raw, signed.key
    files, facts, changed, timing, visible = found.files, found.facts, found.changed, found.timing, found.visible
    pem = signing.public_key_pem(key)
    first = facts[0] if facts else None
    statement = ctx.cards.statement()
    document: dict = {"outcome": outcome}
    document.update(
        {
            "schema": RESULT_SCHEMA,
            "request_id": spec.request_id,
            "spec_fingerprint": fingerprint(spec, hyperloom_version=_version()),
            "spec_sha256": spec_sha256(raw),
            "surface_sha256": found.surface_sha256,
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
                "rule_pack": rule_pack_identity(),
                "card_occupancy": None if statement is None else statement.to_document(),
            },
            "platform_env_changed": list(changed),
            "timing": timing,
            "files": [f.to_document() for f in files],
        }
    )
    if found.jit_cache is not None:
        document["jit_cache"] = found.jit_cache
    if ctx.notes:
        document["notes"] = manifest_notes(ctx.notes)
    manifest = encode_manifest(document)
    (ctx.out_dir / "manifest.json").write_bytes(manifest)
    (ctx.out_dir / "manifest.sig").write_text(signing.sign(manifest, key), encoding="ascii")


def _jit_session(
    ctx: _Context, spec: MeasurementSpec, points: list[Path], farm: Farm | None, facts
) -> JitCacheSession | None:
    """The test's use of the shared compiled-code cache, or ``None`` when none is configured (``measure.runner.jit``)."""
    around = Surroundings(ctx.environ, ctx.work / "jit", ctx.out_dir / STAGING_DIR, ctx.read_blob, ctx.notes)
    return build_session(spec, points, farm, facts[0] if facts else None, around)


def _own_directories(ctx: _Context, call: TestCall, session: JitCacheSession | None, farm: Farm | None) -> TestCall:
    """``call`` with the test's compiled-code cache, the tables its farm patches and, when the cache gives aiter no JIT
    directory, one of the test's own, so nothing the server builds lands in the image. A point that rebuilds stale aiter
    modules at its first prefill gets a watchdog that waits for the builds (``rebuild_watchdog``)."""
    own = own_aiter_jit(call, farm, session, ctx.environ, ctx.work / OWN_AITER_JIT_DIR)
    raised = raise_rebuild_watchdog(call, farm, ctx.environ, ctx.work)
    if raised is not None:
        ctx.notes.append(raised)
    return dataclasses.replace(
        call, jit=session, tables_patch=patched_configs(farm), env_overrides={**call.env_overrides, **own}
    )


def _cold_start_timing(cold: ColdStart) -> dict[str, float]:
    """The two parts of a start the result states beside its phases, when the server's logs gave them."""
    timing = {}
    if cold.logs:
        timing["jit_compile_sec"] = round(cold.builds.compile_seconds, 3)
    if cold.model_load_sec is not None:
        timing["model_load_sec"] = cold.model_load_sec
    return timing


def _run_sheet(ctx: _Context, spec_path: Path, key, progress: Progress) -> None:
    raw = spec_path.read_bytes()
    received = time.time()
    ctx.phases.begin(PREFLIGHT, at=received)
    spec = _parse(raw)
    visible = _visible(ctx.environ)
    _check_model(spec.config[0].yaml, dict(spec.model_hashes))
    _check_runtime(spec)
    facts = _preflight(ctx, spec, visible)
    before = hardware.ecc_counts(visible)
    changed = align_platform_env(ctx.environ, spec.platform_env)
    progress.phase("preparing")
    ctx.phases.begin(ENV_REBUILD)
    points, inferencex, farm = _prepare(ctx, spec, visible)
    _patch_replay_client(ctx, points, inferencex)
    surface = _surface_sha256(spec, points)
    session = _jit_session(ctx, spec, points, farm, facts)
    call = _own_directories(ctx, _call(ctx, spec, points, inferencex, progress), session, farm)
    _install_eval_deps(ctx, spec, points, inferencex)
    ctx.cards.check(BEFORE_TEST)
    ctx.cards.enforce_guard()
    progress.phase("running")
    started = time.time()
    outcome, timed_out = _execute_watched(ctx, call)
    ctx.cards.enforce_guard()
    observed = ctx.cards.observation()
    if observed is not None:
        ctx.notes.append(observed)
    ctx.phases.begin(RESULT_SIGNING)
    progress.phase("collecting")
    harvest_leaked_artifacts(ctx.files, subprocess_started_unix=started, leak_root=ctx.work)
    drop_agentx_runtime(ctx.files)
    _check_node(ctx, before, hardware.ecc_counts(visible), visible, facts)
    write_farm_inventory(ctx.files, farm)
    write_rounds(ctx.files, outcome)
    files = _inventory(ctx, spec)
    cold = cold_start(ctx.files)
    report = None if session is None else session.report(cold.builds)
    if report is not None:
        ctx.notes.extend(report.notes)
    timing = _timing(ctx, received, cold)
    _write_bundle(
        ctx,
        _Signed(spec, raw, key),
        refine_imports(refine_outcome(_outcome_document(outcome, timed_out), session, cold.builds), farm),
        _Findings(files, facts, changed, timing, visible, surface, None if report is None else report.document()),
    )


def _inventory(ctx: _Context, spec: MeasurementSpec) -> tuple[ManifestFile, ...]:
    """The output's files, a profile's traces first fitted under the size cap as far as they can be (``trace_fit``)."""
    fitted = fit_traces(ctx.files, spec.outputs.max_total_bytes)
    if fitted is not None:
        ctx.notes.append(fitted)
    try:
        return inventory(ctx.files, max_total_bytes=spec.outputs.max_total_bytes)
    except OutputRejected as exc:
        raise RunnerFailure(exc.error_class, str(exc)) from exc


def _timing(ctx: _Context, received: float, cold: ColdStart) -> dict[str, object]:
    """The result's timing: when the run started, received its spec and finished, its phases, which end now, and the
    two parts of its cold start."""
    return {
        "runner_started_unix": ctx.started,
        "spec_received_unix": received,
        "server_ready_unix": None,
        "finished_unix": time.time(),
        "phases": manifest_document(ctx.phases.finish()),
        **_cold_start_timing(cold),
    }


def _unix_time(ctx: _Context, name: str) -> float | None:
    """The unix time the start-up script left in ``name``, when it is one this run could have started after."""
    try:
        said = float(ctx.environ.get(name, ""))
    except ValueError:
        return None
    return said if 0.0 < said <= ctx.started else None


def _begin_phases(ctx: _Context) -> None:
    """Start the run's phases: the start-up script (when it said when it began and ended), the interpreter's start to this
    point, then the wait for the sheet."""
    began, ended = _unix_time(ctx, _BOOTSTRAP_STARTED_ENV), _unix_time(ctx, _BOOTSTRAP_ENDED_ENV)
    if began is not None:
        ctx.phases.begin(BOOTSTRAP, at=began)
        if ended is not None and ended >= began:
            ctx.phases.begin(RUNNER_START, at=ended)
    ctx.phases.begin(SHEET_WAIT, at=ctx.started)


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
    try:
        guard = card_guard(environ)
    except ValueError as exc:
        failure = RunnerFailure(_START_FAILED, str(exc))
        write_status(out_dir, state=_FAILED, error_class=failure.error_class, message=str(failure))
        return failure.exit_code
    watch = CardWatch(_visible(environ), own_sandbox_processes(environ), guard)
    ctx = _Context(in_dir, out_dir, work, scratch / "cache", environ, time.time(), watch)
    _begin_phases(ctx)
    with Progress(out_dir, files_dir=ctx.files) as progress:
        try:
            key = _load_key(environ)
            take_topology(environ)
            _check_cards_free(ctx, _visible(environ))
            progress.phase("waiting_spec")
            if not wait_for_file(in_dir / "READY", spec_wait_sec):
                write_status(out_dir, state=_FAILED, error_class="no_spec", message="no instruction sheet arrived")
                return 0
            _run_sheet(ctx, in_dir / "spec.json", key, progress)
        except RunnerFailure as failure:
            write_status(out_dir, state=_FAILED, error_class=failure.error_class, message=str(failure))
            return failure.exit_code
        except Exception as exc:  # noqa: BLE001 - the application reads status.json, not a traceback
            error_class = DISK_FULL if is_disk_full(exc) else "runner_error"
            message = f"{type(exc).__name__}: {exc} (at {_raised_at(exc)})"
            write_status(out_dir, state=_FAILED, error_class=error_class, message=message)
            return 1
        write_status(out_dir, state="succeeded", message=join_notes(ctx.notes))
    return 0


#: How many of the innermost frames a runner error names: enough to place a crash the result bundle cannot carry,
#: since a runner that failed writes no bundle and its own log stays in the ended sandbox.
_RAISED_AT_FRAMES = 4


def _raised_at(exc: BaseException) -> str:
    """Where ``exc`` was raised: its innermost distinct frames, innermost first, as ``file:line function``."""
    frames: list[str] = []
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        where = f"{Path(frame.filename).name}:{frame.lineno} {frame.name}"
        if where not in frames:
            frames.append(where)
        if len(frames) == _RAISED_AT_FRAMES:
            break
    return " < ".join(frames) or "no frame"
