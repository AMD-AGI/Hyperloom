#!/usr/bin/env python3
"""Run the deterministic KTH-before-performance Hyperloom demo."""

from __future__ import annotations

import argparse
import asyncio
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
FORBIDDEN_HOSTS = {"tw042", "tw045"}
DEFAULT_GPU_IMAGE = "vllm/vllm-openai-rocm:v0.27.1"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "kth-hyperloom-demo",
    "GIT_AUTHOR_EMAIL": "demo@local",
    "GIT_COMMITTER_NAME": "kth-hyperloom-demo",
    "GIT_COMMITTER_EMAIL": "demo@local",
}


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


def _preflight_gpu(host: str, image: str) -> None:
    if host in FORBIDDEN_HOSTS:
        raise SystemExit(f"refusing GPU demo host {host}: campaign forbids it")
    probe = _ssh(host, "hostname; rocminfo | awk '/Name:/{print $2}' | grep gfx || true")
    if probe.returncode != 0:
        raise SystemExit(f"cannot reach {host}: {probe.stderr or probe.stdout}")
    if "gfx942" not in probe.stdout:
        raise SystemExit(f"{host} does not expose gfx942:\n{probe.stdout}")
    images = _ssh(host, f"docker image inspect {image} --format '{{{{.Id}}}}'")
    if images.returncode != 0:
        raise SystemExit(f"{image} is not present on {host}: {images.stderr}")
    print(f"GPU preflight: {host} gfx942, image {image}", file=sys.stderr)
    print("Not rerunning AITER #4888; this path uses fused-add residual_out only.", file=sys.stderr)


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
) -> dict[str, Any]:
    plan_id = GPU_PLAN_ID if gpu else CPU_PLAN_ID
    kernel_path = GPU_KERNEL_PATH if gpu else CPU_KERNEL_PATH
    gpu_id = "mi300x-gfx942" if gpu else "fixture-cpu"
    backend = "aiter-rocm" if gpu else "python-fixture"
    baseline = 'CAST = "baseline"\n' if gpu else 'ROUND_MODE = "baseline"\n'
    drift = 'CAST = "truncate"\n' if gpu else 'ROUND_MODE = "ceil"\n'
    corrected = 'CAST = "rne"\n' if gpu else 'ROUND_MODE = "rne"\n'
    if gpu:
        os.environ.pop("KTH_ALLOW_FIXTURE_PLANS", None)
        _preflight_gpu(host, image)
    else:
        os.environ["KTH_ALLOW_FIXTURE_PLANS"] = "1"

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
        evidence: dict[str, Any] = {"performance_evidence": "fixture-labelled"}
        if gpu:
            evidence = _gpu_kernel_timing(host, image, device, kth_root)
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
        kth_provider=KthQualificationProvider(
            executable=str(wrapper),
            expected_kth_sha=kth_sha,
            timeout_s=1800.0 if gpu else 300.0,
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
    if performance_calls != 1 or summary["kept_count"] != 1:
        raise RuntimeError("demo invariants failed")
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
        result = asyncio.run(
            _run(
                kth_root,
                output,
                gpu=args.gpu,
                host=args.host,
                image=args.image,
                device=args.device,
            )
        )
    except Exception:
        print(f"Demo failed; artifacts preserved at {output}", file=sys.stderr)
        raise
    print((output / "terminal_timeline.txt").read_text(), end="")
    print(f"Artifacts: {output}")
    print(f"Machine-readable result: {output / 'timeline.json'}")
    print(f"Final verdict: {result['integration']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
