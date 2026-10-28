# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stand-in for the measurement runner, started as a local sandbox template's command.

It speaks the runner's file protocol (``/in/READY``, ``/in/spec.json``, ``/out/progress.json``, ``/out/files``,
``/out/manifest.json``, ``/out/manifest.sig``, ``/out/status.json``) and signs with the key the sandbox mounted, but it
measures nothing: what it writes is chosen by ``fake_mode`` and friends in the YAML of the sheet it receives.

``fake_mode``: ``ok`` (default), ``fail_test`` (non-zero return code with a known failure signature), ``fail_unknown``
(non-zero, no signature), ``timed_out``, ``spec_rejected`` and ``node_fault`` (``status.json`` failures), ``tamper``
(a listed file changes after signing), ``wrong_key`` (signed with another key), ``other_sandbox``, ``other_request``,
``other_spec`` (the manifest names another sandbox, request or sheet), ``extra_file`` (a file the manifest does not
list), ``no_manifest`` and ``hang`` (never finishes); ``fake_ready: true`` stamps the server ready first.
``fake_delay`` seconds pass before the work starts;
``fake_throughput`` and ``fake_completed`` set the report. ``fake_accuracy`` adds an accuracy round to the rounds file
with that score (``none``: a round that holds no score). ``fake_trace: true`` adds the torch-profiler trace a profile
test leaves (``profile/torch_trace/...``), ``capture_only`` only a CUDA-graph capture sidecar; ``fake_report: false`` leaves out the benchmark report, as a profiled round
that printed no readable measurement. ``fake_repro: true`` runs the repro case the sheet carries for real (the shipped
attachment placement and ``run_case``) against no server, and lists the verdict and log it leaves in the manifest.
``fake_attached_env: [NAME, ...]`` places the sheet's attachments as the runner does and lists, in
``attached_env.json``, the path each variable holds in the served YAML, the sha256 of the file there, and the sha256 of
every file in the directory it is in (``directory``; a directory attachment beside it is left out).
``fake_foreign_cards: [card, ...]`` fails as the runner does when a process outside the test holds one of those cards, if the
sandbox was given any of them (``status.json`` says ``foreign_process_on_card``); ``fake_card_occupancy: {state, reason,
checks}`` signs what the runner says of its cards into the manifest; ``fake_notes: [note, ...]`` signs those notes.
``fake_phases: [[name, seconds], ...]`` states those phases, back to back, in the manifest's timing (``fake_phases_at``
is the unix time the first begins, on this fake runner's own clock: 9000 by default); a runner that states none is
the default. ``fake_legacy_decode: true`` lists the ``decode/outputs.jsonl`` that a runner of the removed decode pass left, so a
bundle written then can be verified again.
``fake_short_prompts: good`` leaves the outputs of a short-prompt check whose one process, the test's own server, behaves
so, when the sheet asks for the check (``bad``: wrong for short prompts, ``drift``: one long prompt differs late, ``fail``:
did not answer).
``fake_long_context: {broken: [family, ...], served: N}`` leaves the outputs of a long-context check whose server loses those
task families (``niah``, ``tracking``, ``words``) and serves ``served`` tokens, when the sheet asks for the check.
``fake_export: [line, ...]`` lists the per-request export of an AgentX round (``aiperf_artifacts/profile_export.jsonl``
below the round's directory) with those lines.
``fake_eval_in_warmup: true`` lays the rounds out as a graded test with a warmup leaves them (one evaluation, in the
warmup round): a warmup round and the accuracy round on one directory, the measured round on another, and the lm-eval
results file of ``fake_accuracy`` below the warmup's.
``fake_log_line: <text>`` writes a server log whose last line that is, so the runner's progress reports it as the last
line of the live output.
``fake_jit: true`` stages a contribution to the shared compiled-code cache (one aiter module, below ``/out/jit-cache``) and
signs it into the manifest as a test that compiled it would; ``fake_jit_swap: true`` changes the staged list after the
signing, as something that can write ``/out`` could.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.measure import signing
from hyperloom.measure.attachments import place_attachments, substitute
from hyperloom.measure.repro import REPRO_DIR, find_case, run_case
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.jit.store import Snapshot, StoredFile
from hyperloom.measure.manifest import RESULT_SCHEMA, encode_manifest
from hyperloom.measure.runner.service import Progress
from hyperloom.measure.long_context.tests.support import outputs_files as long_context_outputs
from hyperloom.measure.short_prompts.tests.support import outputs_files
from hyperloom.measure.spec import MeasurementSpec, fingerprint, spec_sha256

REPORT_PATH = "benchmark_fake_20261014_000000/benchmark_report.json"
TRACE_PATH = "profile/torch_trace/rank0.pt.trace.json.gz"
#: What ``fake_trace: capture_only`` leaves instead: the CUDA-graph capture sidecar a profile whose profiler never started
#: holds alone (named as on a NEXTN head, where the run ended before the capture window).
CAPTURE_TRACE_PATH = "profile/torch_trace/graph_capture_profile/EAGLEDraftExtendCudaGraphRunner_bs_14_rank0.json.gz"
EXPORT_PATH = "benchmark_fake_20261014_000000/aiperf_artifacts/profile_export.jsonl"
#: What a runner of the removed greedy decode pass left in a bundle: a header line, then one line per prompt.
LEGACY_DECODE_PATH = "decode/outputs.jsonl"
LEGACY_DECODE_IDENTITY = "sha256:0123456789abcdef|temperature=0|top_k=1"
LEGACY_DECODE_OUTPUTS = (
    json.dumps({"decode_pass": {"identity": LEGACY_DECODE_IDENTITY, "prompts": 2, "isl": 8, "max_tokens": 4}})
    + '\n{"prompt": 0, "tokens": [1, 2]}\n{"prompt": 1, "tokens": [3]}\n'
).encode()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_bytes(data)
    os.replace(scratch, path)


def _visible_cards() -> list[int]:
    return [int(i) for i in os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",") if i.strip().isdigit()]


def _runner_facts(key: Any, sandbox_id: str, mode: str) -> dict[str, Any]:
    public_pem = signing.public_key_pem(key)
    return {
        "sandbox_id": "sbx-other" if mode == "other_sandbox" else sandbox_id,
        "signing_key_fingerprint": signing.public_key_fingerprint(public_pem),
        "node_name": os.environ.get("NODE_NAME"),
        "image_digest": "sha256:" + "11" * 32,
        "gpu_model": "FakeGPU",
        "power_cap_w": 500,
        "partition_mode": "SPX/NPS1",
        "hyperloom_version": _version(),
        "magpie_commit": "e" * 40,
        "inferencex_commit": "f" * 40,
        "gpu_ids": _visible_cards(),
        "ecc_delta": {"correctable": 0, "uncorrectable": 0},
        "rule_pack": rule_pack_identity(),
    }


def _version() -> str:
    from importlib import metadata

    try:
        return metadata.version("hyperloom")
    except metadata.PackageNotFoundError:
        return "0"


def _write_files(out: Path, config: dict[str, Any], mode: str) -> dict[str, bytes]:
    report = {
        "success": mode != "fail_test",
        "throughput": {
            "output_throughput": float(config.get("fake_throughput", 5000.0)),
            "completed_requests": int(config.get("fake_completed", 64)),
            "num_prompts": int(config.get("fake_requested", 64)),
        },
    }
    files = {"server.log": b"server up\nserver down\n"}
    if config.get("fake_report", True):
        files[REPORT_PATH] = json.dumps(report).encode()
    if config.get("fake_trace") == "capture_only":
        files[CAPTURE_TRACE_PATH] = b"fake capture trace bytes"
    elif config.get("fake_trace"):
        files[TRACE_PATH] = b"fake trace bytes"
    if "fake_export" in config:
        files[EXPORT_PATH] = "".join(f"{line}\n" for line in config["fake_export"]).encode()
    if config.get("fake_warmup") and REPORT_PATH in files:
        files["warmup_round/benchmark_fake_20261014_000000/benchmark_report.json"] = files[REPORT_PATH]
    in_warmup = bool(config.get("fake_eval_in_warmup")) and REPORT_PATH in files
    if in_warmup:
        report_bytes = files.pop(REPORT_PATH)
        for variant in ("warmup", "test"):
            files[f"variant_00_{variant}/{REPORT_PATH}"] = report_bytes
    if mode == "fail_test":
        files["server.log"] = b"ModuleNotFoundError: No module named 'sglang_kernel'\n"
    if mode == "fail_unknown":
        files["server.log"] = b"something odd happened\n"
    bench = REPORT_PATH.rsplit("/", 1)[0]
    rounds = [{"point": 0, "phase": "measure", "dir": bench, "returncode": 0}]
    if in_warmup:
        rounds = [
            {"point": 0, "phase": "warmup", "dir": f"variant_00_warmup/{bench}", "returncode": 0},
            {"point": 0, "phase": "measure", "dir": f"variant_00_test/{bench}", "returncode": 0},
        ]
    if config.get("fake_inventory"):
        files["farm_inventory.json"] = json.dumps(config["fake_inventory"]).encode()
    if "fake_accuracy" in config:
        score = config["fake_accuracy"]
        values = [] if score is None else [{"task": "gsm8k", "metric": "exact_match,flexible-extract", "score": score}]
        if "fake_accuracy_version" in config:
            values = [{**value, "version": config["fake_accuracy_version"]} for value in values]
        accuracy_round = {**rounds[0], "phase": "accuracy", "accuracy": values}
        if in_warmup:
            rounds.insert(1, accuracy_round)  # chronological: warmup bench, the evaluation, measured bench
            if score is not None:
                results = {"results": {"gsm8k": {"exact_match,flexible-extract": float(score)}}}
                files[f"{rounds[0]['dir']}/eval_output/results_2026-10-14T10-00-00.json"] = json.dumps(results).encode()
        else:
            rounds.append(accuracy_round)
    if "fake_warnings" in config:
        rounds[0]["nonfatal_warnings"] = list(config["fake_warnings"])
    if config.get("fake_rounds") or "fake_accuracy" in config or "fake_warnings" in config:
        files["rounds.json"] = json.dumps(rounds).encode()
    if config.get("fake_legacy_decode"):
        files[LEGACY_DECODE_PATH] = LEGACY_DECODE_OUTPUTS
    for relative, data in files.items():
        _atomic(out / "files" / relative, data)
    return files


def _place(spec: MeasurementSpec, root: Path) -> dict[str, Path]:
    """The sheet's attachments, written under ``<root>/work/attach`` from the blobs the sheet came with."""
    blobs = root / "in" / "blobs"
    attach = root / "work" / "attach"
    attach.mkdir(parents=True, exist_ok=True)
    return place_attachments(
        spec.attachments, attach, lambda address: (blobs / address.removeprefix("sha256:")).read_bytes()
    )


def _attached_env_files(spec: MeasurementSpec, root: Path, out: Path, names: list[str]) -> dict[str, bytes]:
    """The path ``names`` hold in the YAML once the attachments are placed, and the sha256 of the file at each."""
    served = yaml.safe_load(substitute(spec.config[0].yaml, _place(spec, root))) or {}
    held = {}
    for name in names:
        listed = str(served["benchmark"]["envs"][name]).split(os.pathsep)
        path = Path(listed[-1])
        held[name] = {
            "listed": listed[:-1],
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "directory": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.iterdir() if p.is_file()
            },
        }
    report = json.dumps(held).encode()
    _atomic(out / "files" / "attached_env.json", report)
    return {"attached_env.json": report}


def _repro_files(spec: MeasurementSpec, root: Path, out: Path) -> dict[str, bytes]:
    """Place the sheet's attachments, run its case and return what it left below ``<out>/files/repro``."""
    placed = _place(spec, root)
    script = find_case(root / "work" / "attach", list(placed))
    run_case(script, base_url="http://127.0.0.1:1", model="fake-model", timeout_sec=30, output_dir=out / "files")
    directory = out / "files" / REPRO_DIR
    return {f"{REPRO_DIR}/{p.name}": p.read_bytes() for p in sorted(directory.iterdir())}


JIT_KEY = "a" * 64
JIT_MODULE = b"\x7fELF the module a fake test compiled"


def _stage_jit(out: Path) -> dict[str, Any]:
    """Stage one aiter module as a runner does and give the ``jit_cache`` block that signs it."""
    staged = out / "jit-cache" / AITER_JIT.name
    sha = hashlib.sha256(JIT_MODULE).hexdigest()
    _atomic(staged / "blobs" / sha, JIT_MODULE)
    snapshot = Snapshot(AITER_JIT.name, JIT_KEY, None, {"module_fake.so": StoredFile(sha, len(JIT_MODULE))})
    _atomic(staged / "manifest.json", snapshot.to_bytes())
    contribution = {"sha256": snapshot.sha256(), "files": 1, "bytes": len(JIT_MODULE)}
    kind = {
        "kind": AITER_JIT.name,
        "key": JIT_KEY,
        "state": "miss",
        "seed_files": 0,
        "seed_bytes": 0,
        "seed_snapshot": None,
        "contribution": contribution,
    }
    return {"kinds": [kind], "builds_seen": 1, "jit_compile_sec": 3.0}


def _timing(config: dict[str, Any]) -> dict[str, Any]:
    timing: dict[str, Any] = {
        "runner_started_unix": 1.0,
        "spec_received_unix": 2.0,
        "server_ready_unix": 3.0,
        "finished_unix": 4.0,
    }
    at = float(config.get("fake_phases_at", 9000.0))
    phases = []
    for name, seconds in config.get("fake_phases") or []:
        phases.append({"name": name, "started_unix": at, "seconds": float(seconds), "point": None})
        at += float(seconds)
    if phases:
        timing["phases"] = phases
    return timing


def _manifest(
    spec: MeasurementSpec, raw: bytes, key: Any, files: dict[str, bytes], mode: str, config: dict[str, Any]
) -> dict[str, Any]:
    failing = mode in ("fail_test", "fail_unknown", "timed_out")
    document = {
        "schema": RESULT_SCHEMA,
        "request_id": "other-request" if mode == "other_request" else spec.request_id,
        "spec_fingerprint": ("0" * 16) if mode == "other_spec" else fingerprint(spec, hyperloom_version=_version()),
        "spec_sha256": spec_sha256(raw),
        "surface_sha256": None,
        "runner": _runner_facts(key, os.environ["E2B_SANDBOX_ID"], mode),
        "outcome": {
            "returncode": 1 if failing else 0,
            "timed_out": mode == "timed_out",
            "error_class": None,
            "failure_class": "unclear" if mode in ("fail_unknown", "timed_out") else None,
            "failure_reason": (
                "no_known_signature" if mode == "fail_unknown" else "timeout" if mode == "timed_out" else None
            ),
            "stdout_tail": "ran",
            "stderr_tail": "ModuleNotFoundError: x" if mode == "fail_test" else "",
        },
        "platform_env_changed": [],
        "timing": _timing(config),
        "files": [
            {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(files.items())
        ],
    }
    if mode == "fail_test":
        document["outcome"]["failure_class"] = "candidate"
        document["outcome"]["failure_reason"] = "import_error"
    if "fake_card_occupancy" in config:
        document["runner"]["card_occupancy"] = config["fake_card_occupancy"]
    if "fake_notes" in config:
        document["notes"] = config["fake_notes"]
    return document


def _short_prompt_files(out: Path, spec: MeasurementSpec, config: dict[str, Any]) -> dict[str, bytes]:
    """The outputs of the short-prompt check the sheet asks for, as the test's own server would leave them when it behaves
    as ``fake_short_prompts`` (``good``, ``bad``, ``drift`` or ``fail``) says."""
    plan = spec.procedure.short_prompts
    if plan is None or "fake_short_prompts" not in config:
        return {}
    files = outputs_files(plan, config["fake_short_prompts"])
    for relative, data in files.items():
        _atomic(out / "files" / relative, data)
    return files


def _long_context_files(out: Path, spec: MeasurementSpec, config: dict[str, Any]) -> dict[str, bytes]:
    """The outputs of the long-context check the sheet asks for, as ``fake_long_context`` says its server behaved."""
    plan = spec.procedure.long_context
    if plan is None or "fake_long_context" not in config:
        return {}
    behaviour = config["fake_long_context"]
    files = long_context_outputs(plan, behaviour.get("broken", ()), served=behaviour.get("served"))
    for relative, data in files.items():
        _atomic(out / "files" / relative, data)
    return files


def _publish(out: Path, spec: MeasurementSpec, raw: bytes, config: dict[str, Any], mode: str) -> None:
    key = signing.load_private_key(Path(os.environ["HYPERLOOM_SIGNING_KEY_FILE"]))
    files = _write_files(out, config, mode)
    files.update(_short_prompt_files(out, spec, config))
    files.update(_long_context_files(out, spec, config))
    if config.get("fake_repro"):
        files.update(_repro_files(spec, Path(os.environ["HYPERLOOM_SANDBOX_ROOT"]), out))
    if config.get("fake_attached_env"):
        sandbox_root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
        files.update(_attached_env_files(spec, sandbox_root, out, config["fake_attached_env"]))
    manifest = _manifest(spec, raw, key, files, mode, config)
    if config.get("fake_jit"):
        manifest["jit_cache"] = _stage_jit(out)
    manifest_raw = encode_manifest(manifest)
    signer = signing.generate_private_key() if mode == "wrong_key" else key
    if mode == "tamper":
        original = bytearray((out / "files" / "server.log").read_bytes())
        original[0] ^= 0x01
        (out / "files" / "server.log").write_bytes(bytes(original))
    if config.get("fake_jit_swap"):
        staged = out / "jit-cache" / AITER_JIT.name / "manifest.json"
        staged.write_bytes(staged.read_bytes() + b" ")
    if mode == "extra_file":
        _atomic(out / "files" / "extra.txt", b"not listed")
    if mode != "no_manifest":
        _atomic(out / "manifest.json", manifest_raw)
        _atomic(out / "manifest.sig", signing.sign(manifest_raw, signer).encode("ascii"))
    _atomic(out / "status.json", json.dumps({"state": "succeeded", "error_class": None, "message": ""}).encode())


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    inbox, out = root / "in", root / "out"
    with Progress(out, files_dir=out / "files") as progress:
        progress.phase("waiting_spec")
        while not (inbox / "READY").exists():
            time.sleep(0.02)
        raw = (inbox / "spec.json").read_bytes()
        spec = MeasurementSpec.from_bytes(raw)
        config = yaml.safe_load(spec.config[0].yaml) or {}
        mode = str(config.get("fake_mode", "ok"))
        if "fake_log_line" in config:
            _atomic(out / "files" / "server.log", f"boot\n{config['fake_log_line']}\n".encode())
        if config.get("fake_ready"):
            _atomic(out / "files" / "server_ready_at", f"{time.time()} 5\n".encode())
        progress.touch()
        progress.phase("running")
        time.sleep(float(config.get("fake_delay", 0)))
        if mode == "hang":
            time.sleep(3600)
        if set(config.get("fake_foreign_cards") or ()) & set(_visible_cards()):
            mode = "foreign_process_on_card"
        if mode in ("spec_rejected", "node_fault", "foreign_process_on_card"):
            state = {"state": "failed", "error_class": mode, "message": f"fake {mode}"}
            _atomic(out / "files" / "server.log", b"partial log\n")
            _atomic(out / "status.json", json.dumps(state).encode())
            return 0
        _publish(out, spec, raw, config, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
