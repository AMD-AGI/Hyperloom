#!/usr/bin/env python3
"""Run the deterministic KTH-before-performance Hyperloom demo."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name

from hyperloom.orchestrator.kernel.controller_patch_integration import (
    integrate_controller_patches,
)
from hyperloom.orchestrator.kernel.kth_qualification import KthQualificationProvider
from hyperloom.orchestrator.state.shared_state import SharedState

CPU_PLAN_ID = "fixture/cpu-rne-correction-v1"
CPU_KERNEL_PATH = "kernels/quantize.py"
GPU_PLAN_ID = "host/gfx942-rmsnorm-fused-add-v1"
GPU_KERNEL_PATH = "kernels/rmsnorm_fused_add.py"
CAMPAIGN_FROZEN_HOSTS = {"tw042", "tw045"}
DEFAULT_GPU_IMAGE = "vllm/vllm-openai-rocm:v0.27.1"
RULE = "=" * 72
THIN = "-" * 72
GIT_ENV = {
    "GIT_AUTHOR_NAME": "kth-hyperloom-demo",
    "GIT_AUTHOR_EMAIL": "demo@local",
    "GIT_COMMITTER_NAME": "kth-hyperloom-demo",
    "GIT_COMMITTER_EMAIL": "demo@local",
}


def _say(*lines: str) -> None:
    for line in lines:
        print(line, flush=True)


def _section(title: str, *body: str) -> None:
    _say("", RULE, title, THIN, *body)


def _briefing(*, gpu: bool, host: str, image: str, plan_id: str, kernel_path: str) -> None:
    if gpu:
        problem = (
            "The kernel under test is fused add + RMSNorm. It writes residual_out,",
            "the residual stream that every later transformer layer reads. Rounding",
            "that tensor one-sided (truncate) instead of round-to-nearest-even (RNE)",
            "is a silent failure:",
            "",
            "  • Ordinary allclose / 1-ULP checks still pass. Each step looks close.",
            "  • Cosine similarity can sit at 0.99999. The run looks healthy.",
            "  • Over dozens of layers the signed bias compounds. The residual",
            "    stream is then the wrong stream.",
            "",
            "If Hyperloom measured tokens/s first, it could KEEP that wrong kernel.",
            "Qualification has to happen after apply and before any benchmark.",
            "",
            "This live path uses AITER rmsnorm2d_fwd_with_add on a real MI300X",
            f"(gfx942) at {host}. It does not rerun serving-lifecycle issue #4888.",
        )
        candidates = (
            "Both publications set micro_validated=true. That is intentional.",
            "A local acceptance test already said they look fine. Qualification",
            "asks a different question: did the hard contracts hold?",
            "",
            "  1. CAST=truncate  — AITER fused-add with one-sided rounding.",
            "     Expect Blocked (typically REF / NUMERICAL_POLICY / COMPOSITION).",
            "     Hyperloom must skip the benchmark and revert.",
            "  2. CAST=rne       — independent torch RNE control on the same plan.",
            "     Expect Eligible. Only then may Hyperloom time residual_out.",
        )
    else:
        problem = (
            "The kernel under test is a tiny rounding fixture. Changing ROUND_MODE",
            "from ceil to rne is the same silent-failure story without a GPU:",
            "",
            "  • An aggregate tolerance check can still pass.",
            "  • The rounding is systematically one-sided.",
            "  • An optimizer that benchmarks first can KEEP a biased kernel.",
            "",
            "This CPU path proves the Hyperloom gate and the KTH subprocess",
            "contract. It does not claim hardware performance.",
        )
        candidates = (
            "Both publications set micro_validated=true. That is intentional.",
            "",
            "  1. ROUND_MODE=ceil  — expect Blocked; skip benchmark; revert.",
            "  2. ROUND_MODE=rne   — expect Eligible; then a fixture KEEP.",
        )
    _section(
        "What you are watching (no prior KTH context needed)",
        "Hyperloom is an optimization loop: apply a kernel patch, measure, KEEP",
        "the faster one. Measuring a silently wrong kernel wastes the node and",
        "can promote a correctness bug into the 'best' commit.",
        "",
        "KTH (Kernel Trust Harness) is the gate in front of that measurement.",
        "After Hyperloom applies the patch it may send only:",
        "  • the base Git commit",
        "  • the exact patch bytes",
        "  • a host-owned plan ID",
        "It cannot send a command, adapter, import, test file, or kernel-path",
        "override. KTH's host registry decides the oracles and the reference.",
        "",
        "KTH returns one of three completed outcomes (not 'pass/fail'):",
        "  Eligible      — hard contracts held; performance evaluation is allowed",
        "  Blocked       — a hard contract failed; revert; do not benchmark",
        "  Inconclusive  — evidence is incomplete; never treat this as Eligible",
        "",
        "Blocked and Inconclusive still write an attestation. Neither can KEEP.",
        "Eligible is necessary for KEEP, not sufficient — performance still decides.",
    )
    _section("The problem being shown", *problem)
    _section(
        "Trust boundary (what the agent is not allowed to do)",
        f"Plan ID:     {plan_id}",
        f"Kernel path: {kernel_path}",
        "  The plan owns that path. The publication cannot override it.",
        "Publication JSON contains only plan_id under kth_qualification.",
        "Subject digest = SHA-256(base commit + patch bytes + kernel path + plan).",
        "If any of those change, the attestation is for a different subject.",
    )
    _section("The two candidates, in order", *candidates)
    _section(
        "What to watch in the numbers",
        "When KTH blocks, look for:",
        "  REF              — disagreed with an independent reference on the same tensors",
        "  NUMERICAL_POLICY — one-sided / rounding contract, even if allclose-ish",
        "  COMPOSITION      — one step would accept; many residual steps would not",
        "  PROVENANCE       — could not attest which inner path ran (Inconclusive)",
        "A cosine near 1.0 is not a pass. It is often the trap.",
    )
    if gpu:
        _section(
            "Where it will run",
            f"Node:   {host}   architecture gfx942 (MI300X)",
            f"Image:  {image}",
            "Device: HIP_VISIBLE_DEVICES=0  (one GPU; the other seven stay untouched)",
            "Each qualification is a real docker/SSH call. Expect a quiet 10–20 s",
            "wait per candidate. That wait is the gate working: Hyperloom must not",
            "start a benchmark while KTH is still deciding.",
        )
    _say("", RULE, "Starting the Hyperloom patch loop", RULE)


def _fmt_num(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _explain_detector(detector: str) -> str:
    meanings = {
        "REF": (
            "REF: the candidate disagreed with an independent reference on the same "
            "tensors. KTH did not look up an issue number; it compared residual_out to "
            "a torch implementation that is a different type from the candidate."
        ),
        "NUMERICAL_POLICY": (
            "NUMERICAL_POLICY: the signed-bias / rounding contract failed. Ordinary "
            "allclose can still pass while rounding is systematically one-sided."
        ),
        "COMPOSITION": ("COMPOSITION: error grew across repeated residual feedback instead of staying unbiased."),
        "PROVENANCE": (
            "PROVENANCE: KTH could not attest which inner path ran. Missing evidence is Inconclusive, never Eligible."
        ),
        "None": "No primary detector: hard contracts held (Eligible), or the ranker had nothing to name.",
    }
    key = detector or "None"
    return meanings.get(
        key,
        f"Primary detector {key} is the first contract failure KTH ranked.",
    )


def _explain_status(status: str) -> tuple[str, ...]:
    if status == "kth_blocked":
        return (
            "Hyperloom will NOT call the performance validator.",
            "Next automatic action: git apply --reverse on this patch.",
            "The bad residual cannot leak into the next candidate or into a KEEP.",
            "Repair feedback is kept: fix this one mechanism and replay that case.",
        )
    if status == "eligible":
        return (
            "Hard contracts held on the declared plan, including an independent",
            "reference comparison. Eligible is permission to measure — not a KEEP.",
            "Hyperloom may now invoke the performance validator exactly once.",
        )
    if status == "kth_inconclusive":
        return (
            "KTH did not prove the kernel safe or unsafe. Missing evidence cannot be",
            "promoted to Eligible. Hyperloom skips the benchmark and reverts.",
        )
    if status == "needs_review":
        return (
            "The subprocess contract itself failed (timeout, missing attestation,",
            "digest mismatch). Fail-closed: skip benchmark and revert.",
        )
    if status == "kept":
        return (
            "Performance evaluation ran only after Eligible. Hyperloom committed KEEP.",
            "On GPU this timing is kernel-level residual_out latency, not a serving",
            "benchmark and not a claim about end-to-end tokens/s.",
        )
    return (f"Hyperloom recorded status {status}.",)


def _composition_table(metrics: dict[str, Any]) -> list[str]:
    steps = metrics.get("per_step") or []
    if not isinstance(steps, list) or not steps:
        return []
    interesting = {1, 2, 10}
    last = steps[-1].get("step") if isinstance(steps[-1], dict) else None
    if isinstance(last, int):
        interesting.add(last)
        interesting.add(max(1, last // 2))
    rows = [
        "COMPOSITION table — why one-step allclose is not enough:",
        "  step  max_abs     mean_signed",
        "  ----  ----------  -----------",
    ]
    seen: set[int] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        number = int(step.get("step") or index + 1)
        if number not in interesting or number in seen:
            continue
        seen.add(number)
        rows.append(f"  {number:4d}  {_fmt_num(step.get('max_abs')):<10}  {_fmt_num(step.get('mean_signed'))}")
    if metrics.get("one_step_would_accept"):
        rows.append(
            f"  One-step check would ACCEPT this kernel (one_step_max_abs={_fmt_num(metrics.get('one_step_max_abs'))})."
        )
        rows.append(
            f"  After {metrics.get('steps')} residual feedback steps, max_abs={_fmt_num(metrics.get('final_max_abs'))}."
        )
        rows.append("  That growth is the silent failure. A KEEP here would promote it.")
    return rows


def _attestation_story(artifacts_dir: str) -> tuple[str, ...]:
    path = Path(artifacts_dir) / "attestation.json"
    if not path.is_file():
        return ("No attestation.json was written. Fail-closed: do not benchmark.",)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return (f"Could not read attestation.json: {error}",)
    if not isinstance(data, dict):
        return ("Attestation is not a JSON object.",)

    identity = data.get("candidate_identity") or {}
    coverage = data.get("mandatory_oracle_coverage") or {}
    hardware = data.get("hardware_identity") or {}
    findings = data.get("findings") or []
    visible = (hardware.get("visible_devices") or {}) if isinstance(hardware, dict) else {}
    adapters = sorted(
        {
            str(finding.get("candidate_identity"))
            for finding in findings
            if isinstance(finding, dict) and finding.get("candidate_identity")
        }
    )
    lines = [
        "Decoded from the attestation (this is the evidence, not a log slogan):",
        f"  execution_mode:     {data.get('execution_mode')}",
        f"  duration_s:         {data.get('duration_s')}",
    ]
    hip = visible.get("HIP_VISIBLE_DEVICES") if isinstance(visible, dict) else None
    if hip not in (None, "", "(unset)"):
        lines.append(f"  HIP_VISIBLE_DEVICES:{hip}")
    lines.extend(
        [
            f"  patch_sha256:       {identity.get('patch_sha256')}",
            f"  operation_spec:     {identity.get('operation_spec')}",
            f"  inner path(s):      {', '.join(adapters) or '(not named)'}",
            "  KTH chose that inner path from the patch + host plan, not from the agent.",
            f"  coverage.complete:  {coverage.get('complete')}",
            f"  cases executed:     {len(coverage.get('executed_cases') or [])} / "
            f"{len(coverage.get('expected_cases') or [])}",
        ]
    )
    missing = coverage.get("missing_oracles") or []
    if missing:
        lines.append(f"  missing oracles:    {', '.join(str(item) for item in missing)}")
        lines.append("  Missing oracles cannot be treated as a pass.")
    exercised = coverage.get("exercised_oracles") or []
    if exercised:
        lines.append(f"  exercised oracles:  {', '.join(str(item) for item in exercised)}")

    by_detector: dict[str, dict[str, Any]] = {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        detector = str(finding.get("detector_id") or finding.get("check_id") or "")
        if finding.get("status") == "violation" and detector and detector not in by_detector:
            by_detector[detector] = finding
    if not by_detector:
        lines.append("  No oracle reported status=violation. Hard contracts held on the cases that ran.")
    for detector, finding in by_detector.items():
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        lines.append("")
        lines.append(f"  {detector}: {finding.get('detail') or finding.get('status')}")
        if detector == "REF":
            lines.append(
                f"    max_abs={_fmt_num(metrics.get('max_abs'))}  "
                f"mae={_fmt_num(metrics.get('mae'))}  "
                f"frac_out_of_tol={_fmt_num(metrics.get('frac_elements_out_of_tol'))}"
            )
            lines.append("    frac_out_of_tol=0 means a typical allclose still passes. REF still failed.")
        elif detector == "NUMERICAL_POLICY":
            lines.append(
                f"    cosine={_fmt_num(metrics.get('cosine'))}  "
                f"signed_bias={_fmt_num(metrics.get('signed_bias'))}  "
                f"max_abs={_fmt_num(metrics.get('max_abs'))}"
            )
            lines.append("    Cosine near 1.0 is the trap: the kernel looks right and is still biased.")
        elif detector == "COMPOSITION":
            lines.extend(_composition_table(metrics))
    return tuple(lines)


def _request_story(publication, *, gpu: bool, ordinal: int, kernel_text: str) -> tuple[str, ...]:
    patch_sha = hashlib.sha256(publication.patch_path.read_bytes()).hexdigest()
    patch_preview = publication.patch_path.read_text(encoding="utf-8", errors="replace").strip()
    preview_lines = patch_preview.splitlines()
    if len(preview_lines) > 16:
        patch_preview = "\n".join(preview_lines[:16] + ["  ... (truncated)"])
    wait = (
        "Calling kth-qualify now. On gfx942 this is typically 10–20 seconds of silence."
        if gpu
        else "Calling kth-qualify now (CPU fixture; usually a few seconds)."
    )
    prior = ()
    if ordinal > 1:
        prior = (
            "Previous candidate was reverted. Working tree is back at the baseline",
            "except for this newly applied patch. Blocked work cannot leak forward.",
            "",
        )
    return (
        f"Candidate {ordinal}: Hyperloom already applied the patch. Qualify before measure.",
        *prior,
        f"Operator:     {publication.operator_id}",
        f"Plan:         {publication.kth_plan_id}",
        f"Base commit:  {publication.base_commit}",
        f"Kernel path:  {publication.kernel_path}",
        f"Kernel now:   {kernel_text}",
        f"Patch SHA-256:{patch_sha}",
        "Request fields: commit + patch bytes + plan ID. No command, adapter, or import.",
        "",
        "Applied patch (this is the whole candidate):",
        patch_preview,
        "",
        wait,
        "Hyperloom is blocked here on purpose until Eligible / Blocked / Inconclusive.",
        "There is no LLM in this decision.",
    )


class _NarratingProvider:
    """Print the human story around each KTH subprocess call."""

    def __init__(self, inner: KthQualificationProvider, kernel: Path, *, gpu: bool):
        self._inner = inner
        self._kernel = kernel
        self._gpu = gpu
        self._ordinal = 0

    def qualify(self, publication, *, artifacts_root):
        self._ordinal += 1
        kernel_text = self._kernel.read_text(encoding="utf-8").strip()
        _section(
            f"Candidate {self._ordinal}: apply, then qualify — never benchmark first",
            *_request_story(publication, gpu=self._gpu, ordinal=self._ordinal, kernel_text=kernel_text),
        )
        result = self._inner.qualify(publication, artifacts_root=artifacts_root)
        feedback = result.repair_feedback or {}
        mechanism = feedback.get("primary_mechanism") or {}
        replay = feedback.get("smallest_replay") or {}
        _section(
            f"KTH result for candidate {self._ordinal}",
            f"Verdict:             {result.verdict or result.status}",
            f"Hyperloom status:    {result.status}",
            f"Primary detector:    {result.primary_detector or '(none)'}",
            f"Subject digest:      {result.subject_digest or '(none)'}",
            f"Request id:          {result.request_id or '(none)'}",
            f"Performance reached: {result.performance_reached}  (must still be false here)",
            "",
            _explain_detector(result.primary_detector),
            "",
            *(_explain_status(result.status)),
            "",
            f"Why, in one line: {result.reason or mechanism.get('meaning') or result.status}",
        )
        if replay:
            _say(
                f"Smallest replay:  case={replay.get('case_id')}  "
                f"recipe={replay.get('recipe_id')}  seed={replay.get('seed')}"
            )
        if result.artifacts_dir:
            _say("", *_attestation_story(result.artifacts_dir))
            _say(f"Attestation dir:  {result.artifacts_dir}")
        return result

    def mark_performance_reached(self, result):
        _say(
            "",
            THIN,
            "Eligible stands. Hyperloom is crossing into performance evaluation.",
            "If this were a Blocked or Inconclusive candidate, this line would never print.",
            "KEEP is still not decided — only that measurement is allowed.",
            THIN,
        )
        return self._inner.mark_performance_reached(result)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_patch(repo: Path, kernel_path: str, content: str) -> str:
    path = repo / kernel_path
    original = path.read_text(encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    patch = _git(repo, "diff", "--binary", "--", kernel_path)
    path.write_text(original, encoding="utf-8")
    return patch + "\n"


def _publish(
    patches_root: Path,
    repo: Path,
    base_commit: str,
    *,
    ordinal: str,
    content: str,
    evidence: str,
    plan_id: str,
    kernel_path: str,
    gpu: str,
    backend: str,
) -> None:
    kernel_name = f"{ordinal}-fused-add" if plan_id.startswith("host/") else f"{ordinal}-quant-rne"
    operator_id = f"kernel:forge-loop:{kernel_name}:standalone:demo:{backend}:{gpu}"
    patch_dir = patches_root / operator_directory_name(operator_id)
    patch_dir.mkdir(parents=True)
    (patch_dir / "change.patch").write_text(
        _make_patch(repo, kernel_path, content),
        encoding="utf-8",
    )
    (patch_dir / "report.md").write_text(
        "\n".join(
            (
                f"# Candidate {ordinal}",
                "",
                "- micro_validated: true",
                "- tolerance evidence: ordinary allclose / 1-ULP envelope",
                f"- claim: {evidence}",
                "",
            )
        ),
        encoding="utf-8",
    )
    publication = {
        "schema_version": 2,
        "operator_id": operator_id,
        "identity": {
            "producer": "forge-loop",
            "kernel_name": kernel_name,
            "framework": "standalone",
            "framework_version": "demo",
            "backend": backend,
            "gpu": gpu,
        },
        "base_commit": base_commit,
        "best_commit": "b" * 40,
        "repo_root": str(repo),
        "kernel_path": kernel_path,
        "operator_name": kernel_name,
        "micro_validated": True,
        "manifest": {"changed_files": [kernel_path]},
        "kth_qualification": {"plan_id": plan_id},
    }
    (patch_dir / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n",
        encoding="utf-8",
    )


def _cpu_wrapper(root: Path, output: Path) -> tuple[Path, str]:
    kth_sha = _git(root, "rev-parse", "HEAD")
    wrapper = output / "kth-qualify"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'PYTHONPATH="{root / "src"}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'exec "{sys.executable}" -m kth.provider_cli "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper, kth_sha


def _gpu_wrapper(
    root: Path,
    output: Path,
    *,
    host: str,
    image: str,
    device: str,
) -> tuple[Path, str]:
    kth_sha = _git(root, "rev-parse", "HEAD")
    wrapper = output / "kth-qualify"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import shutil, subprocess, sys, uuid
            from pathlib import Path
            args = sys.argv[1:]
            request = Path(args[args.index("--request") + 1]).resolve()
            out = Path(args[args.index("--out") + 1]).resolve()
            kth_root = Path({str(root)!r})
            work = kth_root / "artifacts" / "demo_gpu_live" / request.parent.name
            work.mkdir(parents=True, exist_ok=True)
            shutil.copy2(request, work / "request.json")
            name = "kth_hl_demo_" + uuid.uuid4().hex[:10]
            remote = (
                "docker run --rm --name " + name + " --entrypoint python "
                "--device=/dev/kfd --device=/dev/dri --group-add video --group-add render "
                "--ipc=host --cap-add SYS_PTRACE --security-opt seccomp=unconfined "
                "-e HIP_VISIBLE_DEVICES={device} -e PYTHONPATH=/kth/src "
                "-e KTH_SHA={kth_sha} "
                "-v " + str(kth_root) + ":/kth -v " + str(work) + ":/work {image} "
                "-m kth.provider_cli --request /work/request.json --out /work/attestation.json"
            )
            proc = subprocess.run(
                [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    "-o", "StrictHostKeyChecking=accept-new", {host!r}, remote,
                ],
                capture_output=True,
                text=True,
            )
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            attestation = work / "attestation.json"
            if attestation.is_file():
                shutil.copy2(attestation, out)
            raise SystemExit(proc.returncode)
            """
        ),
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper, kth_sha


def _timeline(
    summary: dict[str, Any],
    performance_calls: int,
    *,
    gpu: bool,
) -> list[dict[str, Any]]:
    drift = "fused-add truncation on gfx942" if gpu else "round-up drift"
    correction = "torch RNE residual_out on gfx942" if gpu else "round-to-nearest-even correction"
    keep = (
        "kernel-level gfx942 timing evaluated; KEEP committed"
        if gpu
        else "fixture performance evaluated; KEEP committed"
    )
    events: list[dict[str, Any]] = [
        {
            "step": 1,
            "event": "micro_validation",
            "outcome": "both candidates published with tolerance evidence",
        }
    ]
    for result in summary["results"]:
        if result["status"] == "kth_blocked":
            events.append(
                {
                    "step": 2,
                    "event": "kth_qualification",
                    "candidate": drift,
                    "outcome": "Blocked",
                    "detector": result["kth_primary_detector"],
                    "performance_reached": result["performance_reached"],
                    "repair_feedback": result["repair_feedback"],
                }
            )
            events.append(
                {
                    "step": 3,
                    "event": "hyperloom_action",
                    "outcome": "evaluation skipped; patch reverted",
                }
            )
        elif result["status"] == "kept":
            events.append(
                {
                    "step": 4,
                    "event": "kth_qualification",
                    "candidate": correction,
                    "outcome": result["kth_verdict"],
                    "performance_reached": result["performance_reached"],
                }
            )
            events.append(
                {
                    "step": 5,
                    "event": "hyperloom_action",
                    "outcome": keep,
                }
            )
    events.append(
        {
            "step": 6,
            "event": "invariant",
            "outcome": f"performance validator called {performance_calls} time",
        }
    )
    return events


def _walkthrough_text(
    summary: dict[str, Any],
    performance_calls: int,
    *,
    gpu: bool,
    output: Path,
) -> str:
    rows = [
        "KTH × Hyperloom demo recap",
        RULE,
        "A kernel that passes a narrow micro-check can still be unsafe to benchmark.",
        "KTH qualifies first. Hyperloom measures only after Eligible.",
        "Eligible is permission to measure, not a KEEP. Blocked never reaches KEEP.",
        "",
    ]
    for result in summary.get("results") or []:
        rows.append(
            f"- {result.get('operator_id')}: status={result.get('status')} "
            f"verdict={result.get('kth_verdict') or 'n/a'} "
            f"detector={result.get('kth_primary_detector') or 'n/a'} "
            f"performance_reached={result.get('performance_reached')}"
        )
        if result.get("reason"):
            rows.append(f"    {result['reason']}")
    rows.extend(
        [
            "",
            f"Performance validator calls: {performance_calls} (must be 1)",
            f"Hardware path: {'live gfx942 fused-add' if gpu else 'CPU fixture rounding'}",
            f"Artifacts: {output}",
            "",
            "Glossary",
            "  Eligible        — hard contracts held; benchmarking is allowed",
            "  Blocked         — a hard contract failed; revert; do not benchmark",
            "  Inconclusive    — not enough evidence; never treat as Eligible",
            "  KEEP            — Hyperloom retained the patch after Eligible + measurement",
            "  micro_validated — a local tolerance check already passed; not qualification",
            "  REF             — disagreed with an independent reference implementation",
            "  NUMERICAL_POLICY — rounding / signed-bias contract failed",
            "  COMPOSITION     — error grew under residual feedback; one-step can still pass",
            "  subject digest  — SHA-256 binding commit + patch + path + plan",
            "  plan ID         — the only thing the agent may choose; host owns the rest",
        ]
    )
    return "\n".join(rows) + "\n"


def _write_terminal_svg(path: Path, events: list[dict[str, Any]]) -> None:
    lines = [
        "$ hyperloom-kth-demo",
        *[f"{event['step']}. {event['event']}: {event['outcome']}" for event in events],
        "$ verdict: completed",
    ]
    text = "\n".join(
        f'<text x="28" y="{48 + index * 27}">{html.escape(line)}</text>' for index, line in enumerate(lines)
    )
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="300" '
            'viewBox="0 0 1280 300">'
            '<rect width="1280" height="300" rx="12" fill="#111827"/>'
            '<g fill="#e5e7eb" font-family="monospace" font-size="17">'
            f"{text}</g></svg>\n"
        ),
        encoding="utf-8",
    )


def _ssh(host: str, remote: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=accept-new",
            host,
            remote,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _preflight_gpu(host: str, image: str, *, allow_campaign_host: bool) -> None:
    if host in CAMPAIGN_FROZEN_HOSTS:
        _say(
            f"WARNING: {host} holds banked KTH GPU evidence (#4888 / #4788).",
            "Continuing because --host named this node"
            + (" and --allow-campaign-host was set." if allow_campaign_host else "."),
            "This demo will not rewrite casegen_4888_* or gpu_real_tw042 trees,",
            "and it will not rerun AITER #4888.",
        )
    probe = _ssh(host, "hostname; rocminfo | awk '/Name:/{print $2}' | grep gfx || true")
    if probe.returncode != 0:
        raise SystemExit(f"cannot reach {host}: {probe.stderr or probe.stdout}")
    if "gfx942" not in probe.stdout:
        raise SystemExit(f"{host} does not expose gfx942:\n{probe.stdout}")
    images = _ssh(host, f"docker image inspect {image} --format '{{{{.Id}}}}'")
    if images.returncode != 0:
        raise SystemExit(f"{image} is not present on {host}: {images.stderr}")
    print(f"GPU preflight: {host} exposes gfx942; image {image} is present.", flush=True)
    print("Not rerunning AITER #4888. This path is fused-add residual_out only.", flush=True)


def _gpu_kernel_timing(host: str, image: str, device: str, kth_root: Path) -> dict[str, Any]:
    script = kth_root / "artifacts" / "demo_gpu_live" / "time_fused_add.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        textwrap.dedent(
            """\
            import time
            import torch
            from kth.v2.rounding import fp32_to_bf16_rne
            x = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            for _ in range(3):
                fp32_to_bf16_rne(x.float() + residual.float())
            torch.cuda.synchronize()
            started = time.perf_counter()
            steps = 20
            for _ in range(steps):
                fp32_to_bf16_rne(x.float() + residual.float())
            torch.cuda.synchronize()
            props = torch.cuda.get_device_properties(0)
            print(
                round((time.perf_counter() - started) / steps * 1000, 3),
                (torch.cuda.get_device_name(0) or "MI300X").replace(" ", "_"),
                getattr(props, "gcnArchName", "") or "gfx942",
            )
            """
        ),
        encoding="utf-8",
    )
    name = "kth_hl_perf_" + uuid.uuid4().hex[:10]
    remote = (
        f"docker run --rm --name {name} --entrypoint python "
        "--device=/dev/kfd --device=/dev/dri --group-add video --group-add render "
        "--ipc=host "
        f"-e HIP_VISIBLE_DEVICES={device} -e PYTHONPATH=/kth/src "
        f"-v {kth_root}:/kth {image} /kth/artifacts/demo_gpu_live/time_fused_add.py"
    )
    proc = _ssh(host, remote, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"gfx942 timing failed: {proc.stderr or proc.stdout}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"gfx942 timing produced no output: {proc.stderr}")
    parts = lines[-1].split()
    if len(parts) < 3:
        raise RuntimeError(f"gfx942 timing parse failed: {lines[-1]!r} stderr={proc.stderr!r}")
    return {
        "mean_ms": float(parts[0]),
        "device_name": parts[1].replace("_", " "),
        "arch": parts[2],
        "label": "kernel-level gfx942 residual_out timing; not a serving benchmark",
    }


async def _run(
    kth_root: Path,
    output: Path,
    *,
    gpu: bool,
    host: str,
    image: str,
    device: str,
    allow_campaign_host: bool = False,
) -> dict[str, Any]:
    plan_id = GPU_PLAN_ID if gpu else CPU_PLAN_ID
    kernel_path = GPU_KERNEL_PATH if gpu else CPU_KERNEL_PATH
    gpu_id = "mi300x-gfx942" if gpu else "fixture-cpu"
    backend = "aiter-rocm" if gpu else "python-fixture"
    baseline = 'CAST = "baseline"\n' if gpu else 'ROUND_MODE = "baseline"\n'
    drift = 'CAST = "truncate"\n' if gpu else 'ROUND_MODE = "ceil"\n'
    corrected = 'CAST = "rne"\n' if gpu else 'ROUND_MODE = "rne"\n'
    if gpu and host in CAMPAIGN_FROZEN_HOSTS:
        _say(
            "",
            RULE,
            f"Note: {host} is a campaign-frozen evidence node.",
            "Proceeding because you passed --host. Fused-add residual_out only.",
            RULE,
        )
    _briefing(gpu=gpu, host=host, image=image, plan_id=plan_id, kernel_path=kernel_path)
    if gpu:
        os.environ.pop("KTH_ALLOW_FIXTURE_PLANS", None)
        _section(
            "GPU preflight (before any patch is applied)",
            f"Checking that {host} exposes gfx942 and that {image} is present.",
            "This is not qualification and not a benchmark.",
        )
        _preflight_gpu(host, image, allow_campaign_host=allow_campaign_host)
    else:
        os.environ["KTH_ALLOW_FIXTURE_PLANS"] = "1"
        _say("CPU fixture plans are enabled only inside this process.")

    repo = output / "candidate_repo"
    session = output / "session"
    patches = output / "publication_artifacts"
    repo.mkdir(parents=True)
    session.mkdir()
    _git(repo, "init")
    kernel = repo / kernel_path
    kernel.parent.mkdir()
    kernel.write_text(baseline, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "demo baseline")
    base_commit = _git(repo, "rev-parse", "HEAD")

    _publish(
        patches,
        repo,
        base_commit,
        ordinal="01-drift",
        content=drift,
        evidence="ordinary allclose passed, but residual_out rounding drifted"
        if gpu
        else "aggregate tolerance check passed, but rounding direction drifted",
        plan_id=plan_id,
        kernel_path=kernel_path,
        gpu=gpu_id,
        backend=backend,
    )
    _publish(
        patches,
        repo,
        base_commit,
        ordinal="02-corrected",
        content=corrected,
        evidence="round-to-nearest-even residual_out restored",
        plan_id=plan_id,
        kernel_path=kernel_path,
        gpu=gpu_id,
        backend=backend,
    )
    _section(
        "Publications are ready — Hyperloom has not measured anything yet",
        f"Isolated repo:     {repo}",
        f"Baseline commit:   {base_commit}",
        f"Patch bundles:     {patches}",
        "Each bundle is what an optimizer would publish: a patch, a report, and",
        "publication.json with micro_validated=true plus only a plan_id.",
        "The loop will: apply → kth-qualify → (only if Eligible) measure → KEEP.",
        "If Blocked or Inconclusive: skip measure, revert, keep repair feedback.",
    )

    if gpu:
        wrapper, kth_sha = _gpu_wrapper(kth_root, output, host=host, image=image, device=device)
    else:
        wrapper, kth_sha = _cpu_wrapper(kth_root, output)
    state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
        framework_repo_path=str(repo),
    )
    state.save(session)
    performance_calls = 0
    expected_kernel = corrected

    async def fixture_performance(_publication) -> dict[str, Any]:
        nonlocal performance_calls
        performance_calls += 1
        assert kernel.read_text(encoding="utf-8") == expected_kernel
        _section(
            "Performance evaluation (only because KTH said Eligible)",
            "Hyperloom is allowed to measure now. A Blocked kernel never reaches here.",
            "This is the first time anything resembling 'speed' is allowed to run.",
            *(
                (
                    "This is still not a serving benchmark: we time residual_out on one GPU",
                    "and label the evidence. KEEP here means 'the loop may retain this patch',",
                    "not 'this model got faster in production'.",
                )
                if gpu
                else (
                    "This CPU path uses a fixture-labelled throughput number so the KEEP",
                    "path can be shown without claiming hardware performance.",
                )
            ),
        )
        evidence: dict[str, Any] = {"performance_evidence": "fixture-labelled"}
        if gpu:
            evidence = _gpu_kernel_timing(host, image, device, kth_root)
            _say(
                f"Kernel-level mean step: {evidence['mean_ms']} ms   {evidence['device_name']} {evidence['arch']}",
                evidence["label"],
            )
        else:
            _say("Fixture-labelled throughput 107 (baseline 100). This is not hardware.")
        return {
            "decision": "KEEP",
            "new_tput": 107.0,
            "gain_pct": 7.0,
            **evidence,
        }

    integration = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session,
        shared_state=state,
        validator=fixture_performance,
        kth_provider=_NarratingProvider(  # type: ignore[arg-type]
            KthQualificationProvider(
                executable=str(wrapper),
                expected_kth_sha=kth_sha,
                timeout_s=1800.0 if gpu else 300.0,
            ),
            kernel,
            gpu=gpu,
        ),
    )
    summary = integration.to_dict()
    timeline = _timeline(summary, performance_calls, gpu=gpu)
    result = {
        "demo_schema_version": "1.0.0",
        "fixture_performance": not gpu,
        "live_gfx942": gpu,
        "gpu_host": host if gpu else None,
        "gpu_image": image if gpu else None,
        "issue_4888_rerun": False,
        "kth_sha": kth_sha,
        "plan_id": plan_id,
        "base_commit": base_commit,
        "final_commit": _git(repo, "rev-parse", "HEAD"),
        "final_kernel": kernel.read_text(encoding="utf-8").strip(),
        "performance_validator_calls": performance_calls,
        "integration": summary,
        "timeline": timeline,
    }
    (output / "timeline.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "terminal_timeline.txt").write_text(
        "\n".join(f"{event['step']}. {event['event']}: {event['outcome']}" for event in timeline) + "\n",
        encoding="utf-8",
    )
    _write_terminal_svg(output / "demo_terminal.svg", timeline)
    walkthrough = _walkthrough_text(summary, performance_calls, gpu=gpu, output=output)
    (output / "walkthrough.txt").write_text(walkthrough, encoding="utf-8")
    if performance_calls != 1 or summary["kept_count"] != 1:
        raise RuntimeError("demo invariants failed")
    _section(
        "What to take away",
        "1. Micro-validation (allclose / a unit test) is not qualification.",
        "2. KTH decides Eligible / Blocked / Inconclusive with no model in the loop.",
        "3. Hyperloom skips measurement and reverts on Blocked or Inconclusive.",
        "4. Eligible is necessary for KEEP, not sufficient — performance still decides.",
        "5. Cosine ~1.0 or frac_out_of_tol=0 can still be Blocked (see COMPOSITION).",
        "6. Every request, attestation, digest, and Git commit is under the artifacts dir.",
        "",
        f"Artifacts: {output}",
        f"Readable recap: {output / 'walkthrough.txt'}",
        f"Machine JSON:   {output / 'timeline.json'}",
        f"Working tree kernel is now: {kernel.read_text(encoding='utf-8').strip()}",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kth-root",
        type=Path,
        required=True,
        help="KTH checkout containing the integration/hyperloom-provider branch",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Qualify live on gfx942 via SSH/docker. Does not rerun AITER #4888.",
    )
    parser.add_argument("--host", default="tw051", help="Allocated gfx942 node")
    parser.add_argument(
        "--allow-campaign-host",
        action="store_true",
        help="Allow tw042/tw045. Those nodes hold banked GPU evidence; default is refuse.",
    )
    parser.add_argument("--image", default=DEFAULT_GPU_IMAGE)
    parser.add_argument("--device", default="0", help="HIP_VISIBLE_DEVICES inside the container")
    args = parser.parse_args()
    kth_root = args.kth_root.expanduser().resolve()
    if not (kth_root / "src" / "kth" / "provider_cli.py").is_file():
        parser.error(f"{kth_root} does not contain the KTH provider")
    output = args.out.expanduser().resolve() if args.out else Path(tempfile.mkdtemp(prefix="hyperloom-kth-demo-"))
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(
            _run(
                kth_root,
                output,
                gpu=args.gpu,
                host=args.host,
                image=args.image,
                device=args.device,
                allow_campaign_host=args.allow_campaign_host,
            )
        )
    except Exception:
        print(f"Demo failed; artifacts preserved at {output}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
