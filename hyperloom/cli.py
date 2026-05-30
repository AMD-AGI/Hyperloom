"""Hyperloom CLI — entry point for optimization sessions.

Commands:
  hyperloom optimize       Full optimization loop (profile + kernel + config)
  hyperloom kernel-phase   Kernel-only optimization on a frozen server
  hyperloom status         Show session status
  hyperloom stop           Request graceful stop
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from .bench import resolve_benchmark_plugin, run_benchmark
from .accuracy import resolve_accuracy_plugin, run_accuracy_gate
from .capabilities import Capabilities, detect_capabilities
from .config import (
    AccuracyConfig,
    BenchmarkConfig,
    ExecutionMode,
    SessionConfig,
)
from .critic import Verdict, review_patch
from .model_profile import detect_model_info
from .state import init_session, load_session, save_session

log = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Hyperloom — modular LLM inference optimization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--model", required=True, help="Path to model weights")
@click.option("--benchmark", default="", help="Path to benchmark script")
@click.option("--accuracy-script", default="", help="Custom accuracy eval script (for non-LLM: diffusion, audio, etc.). Default: lm_eval/gsm8k for LLMs")
@click.option("--framework", default="", help="Framework name (sglang, vllm, inferencex)")
@click.option("--launch-script", default="", help="Script to launch the serving framework")
@click.option("--port", default=8000, type=int, help="Serving port")
@click.option("--target-gain", default=10.0, help="Target improvement percentage")
@click.option("--max-hours", default=4.0, help="Maximum runtime in hours")
@click.option("--mode", type=click.Choice(["local", "cluster", "auto"]), default="auto")
@click.option("--gpus", default="", help="GPU IDs (e.g., '0,1,2,3')")
@click.option("--tp", default=0, type=int, help="Tensor parallelism degree (0=auto)")
@click.option("--agent-model", default="claude-opus-4-7", help="Model for agent dispatch (must be available on configured API endpoint)")
@click.option("--session-dir", default="", help="Session output directory")
@click.option("--output-format", default="json", help="Benchmark output format (json, regex, last_line)")
@click.option("--throughput-key", default="throughput", help="JSON key for throughput in benchmark output")
@click.option("--accuracy-threshold", default=0.0, type=float, help="Minimum accuracy score (0=use default per task)")
@click.option("--resume", is_flag=True, default=False, help="Resume existing session (skip server launch + baseline)")
def optimize(
    model: str,
    benchmark: str,
    accuracy_script: str,
    framework: str,
    launch_script: str,
    port: int,
    target_gain: float,
    max_hours: float,
    mode: str,
    gpus: str,
    tp: int,
    agent_model: str,
    session_dir: str,
    output_format: str,
    throughput_key: str,
    accuracy_threshold: float,
    resume: bool,
):
    """Run full optimization loop."""
    config = SessionConfig(
        model_path=model,
        benchmark=BenchmarkConfig(
            script=benchmark,
            framework=framework,
            output_format=output_format,
            throughput_key=throughput_key,
        ),
        accuracy=AccuracyConfig(
            script=accuracy_script,
            threshold=accuracy_threshold,
        ),
        mode=ExecutionMode(mode),
        target_gain=target_gain,
        max_hours=max_hours,
        gpus=gpus,
        tp=tp,
        port=port,
        launch_script=launch_script,
        agent_model=agent_model,
        session_dir=session_dir,
    )

    errors = config.validate()
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    if not config.session_dir:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        config.session_dir = f"sessions/{ts}"

    caps = detect_capabilities()
    click.echo("Hyperloom Optimization Session")
    click.echo(f"  Model: {config.model_path}")
    click.echo(f"  Mode: {config.mode.value}")
    click.echo(f"  Target: +{config.target_gain}%")
    click.echo(f"  Max hours: {config.max_hours}")
    click.echo(f"\nCapabilities:\n{caps.summary()}")

    model_info = detect_model_info(config.model_path)
    click.echo(f"\nModel: {model_info.name} ({model_info.architecture})")
    if model_info.is_moe:
        click.echo(f"  MoE: {model_info.num_experts} experts")
    if model_info.is_mla:
        click.echo("  MLA: yes")

    bench_plugin = resolve_benchmark_plugin(config, caps)
    acc_plugin = resolve_accuracy_plugin(config)

    click.echo(f"\nSession: {config.session_dir}")
    click.echo(f"Benchmark plugin: {bench_plugin.name}")
    click.echo(f"Accuracy gate: {acc_plugin.name}")

    server = None

    if resume:
        # Resume: load existing session state, assume server is already running
        from .state import load_session
        state = load_session(config.session_dir)
        if state is None:
            click.echo("ERROR: --resume requires an existing session. Run without --resume first.", err=True)
            sys.exit(1)

        click.echo(f"\n--- Resuming session (baseline: {state.baseline_throughput:.1f} tok/s) ---")

        # Verify server is healthy
        import requests
        try:
            r = requests.get(f"http://localhost:{config.port}/health", timeout=5)
            if r.status_code == 200:
                click.echo(f"Server healthy on port {config.port}")
            else:
                click.echo(f"WARNING: Server returned {r.status_code}", err=True)
        except Exception:
            click.echo(f"WARNING: Server not responding on port {config.port} — orchestrator will manage it", err=True)

        from .plugins.base import BenchResult, AccuracyResult
        baseline = BenchResult(throughput=state.baseline_throughput)
        baseline_acc = AccuracyResult(
            score=state.extra.get("baseline_accuracy", 0.9431) if hasattr(state, "extra") and state.extra else 0.9431,
            passed=True,
            threshold=acc_plugin._threshold if hasattr(acc_plugin, "_threshold") else 0.85,
        )
    else:
        state = init_session(config.session_dir, config.model_path, config.target_gain)

        # Launch serving framework
        from .server import launch_server
        click.echo("\n--- Launching serving framework ---")
        try:
            server = launch_server(config)
            click.echo(f"Server running (PID {server.pid}, port {config.port})")
        except (RuntimeError, FileNotFoundError) as e:
            click.echo(f"ERROR: Server launch failed: {e}", err=True)
            sys.exit(1)

        click.echo("\n--- Running baseline benchmark ---")
        baseline = run_benchmark(bench_plugin, config)
        state.baseline_throughput = baseline.throughput
        state.best_throughput = baseline.throughput
        state.current_throughput = baseline.throughput
        state.status = "running"
        save_session(config.session_dir, state)

        click.echo(f"Baseline: {baseline.throughput:.1f} {baseline.throughput_unit}")

        if not baseline.success:
            click.echo("ERROR: Baseline benchmark failed. Check your script.", err=True)
            server.stop()
            sys.exit(1)

        click.echo(f"\nOptimization loop starting (target: +{target_gain}% = {baseline.throughput * (1 + target_gain/100):.1f} {baseline.throughput_unit})")
        click.echo(f"Results will be saved to: {config.session_dir}")

        # Run accuracy baseline
        click.echo("\n--- Running baseline accuracy ---")
        baseline_acc = run_accuracy_gate(acc_plugin, config)
        click.echo(f"Accuracy baseline: {baseline_acc.metric_name} = {baseline_acc.score:.4f} ({'PASS' if baseline_acc.passed else 'FAIL'})")
        if not baseline_acc.passed:
            click.echo("WARNING: Baseline accuracy below threshold — optimizations must not degrade further.", err=True)

    # Launch orchestrator (SDK-based with native dispatch tools)
    click.echo("\n--- Starting optimization orchestrator (SDK) ---")
    from .orchestrator import run_sdk_orchestrator
    from .prompt_builder import build_orchestrator_prompt

    system_prompt = build_orchestrator_prompt(config, model_info, caps, baseline, baseline_acc)

    user_prompt = (
        f"Begin the optimization loop.\n"
        f"Baseline throughput: {baseline.throughput:.1f} {baseline.throughput_unit}\n"
        f"Target: {baseline.throughput * (1 + target_gain/100):.1f} {baseline.throughput_unit} (+{target_gain}%)\n"
        f"Accuracy baseline: {baseline_acc.score:.4f} (threshold: {baseline_acc.threshold})\n"
        f"Server running on port {config.port}. Launch script: {config.launch_script}\n"
        f"Benchmark command: python3 {os.environ.get('VLLM_BENCH_SCRIPT', 'benchmark_serving.py')} "
        f"--backend vllm --base-url http://localhost:{config.port} "
        f"--model {config.model_path} --dataset-name random "
        f"--random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 "
        f"--num-prompts 640 --max-concurrency 64 --request-rate inf --ignore-eos\n"
        f"Max runtime: {max_hours} hours.\n"
        f"Use dispatch_agents to launch specialist sub-agents for optimization work. "
        f"Do NOT try to optimize directly — dispatch specialists and collect their results. "
        f"Re-benchmark after each patch. Reject any change that fails accuracy."
    )

    os.environ["SESSION_DIR"] = str(Path(config.session_dir).resolve())
    os.environ["AGENT_MODEL"] = agent_model

    try:
        run_sdk_orchestrator(
            session_dir=config.session_dir,
            system_prompt=system_prompt,
            model=agent_model,
            user_prompt=user_prompt,
        )
    except KeyboardInterrupt:
        click.echo("\n[Interrupted] Stopping...")
    finally:
        if server:
            server.stop()


@main.command("kernel-phase")
@click.option("--model", required=True, help="Path to model weights")
@click.option("--benchmark", required=True, help="Path to benchmark script")
@click.option("--accuracy-script", default="", help="Custom accuracy eval script (default: lm_eval/gsm8k)")
@click.option("--gpus", default="", help="GPU IDs for kernel work")
@click.option("--mode", type=click.Choice(["local", "cluster", "auto"]), default="auto")
@click.option("--session-dir", default="", help="Session output directory")
def kernel_phase(model: str, benchmark: str, accuracy_script: str, gpus: str, mode: str, session_dir: str):
    """Run kernel-only optimization (assumes server is already running)."""
    config = SessionConfig(
        model_path=model,
        benchmark=BenchmarkConfig(script=benchmark),
        accuracy=AccuracyConfig(script=accuracy_script),
        mode=ExecutionMode(mode),
        gpus=gpus,
        session_dir=session_dir,
    )

    caps = detect_capabilities()
    click.echo("Hyperloom Kernel Phase")
    click.echo(f"  Model: {config.model_path}")
    click.echo(f"  Benchmark: {config.benchmark.script}")
    click.echo(f"\nCapabilities:\n{caps.summary()}")

    if not caps.geak and not caps.oob:
        click.echo("\nWARNING: Neither GEAK nor OOB available. Kernel optimization limited.", err=True)

    click.echo("\nKernel phase ready for dispatch.")


@main.command()
@click.option("--session-dir", default="sessions", help="Session directory to check")
def status(session_dir: str):
    """Show current session status."""
    state = load_session(session_dir)
    if state is None:
        click.echo("No active session found.")
        return

    click.echo(f"Session: {state.session_id}")
    click.echo(f"Status: {state.status}")
    click.echo(f"Baseline: {state.baseline_throughput:.1f} tok/s")
    click.echo(f"Current: {state.current_throughput:.1f} tok/s")
    click.echo(f"Best: {state.best_throughput:.1f} tok/s")
    click.echo(f"Gain: {state.gain_pct:.1f}% (target: {state.target_gain_pct}%)")
    click.echo(f"Iterations: {state.iteration}")
    click.echo(f"Agents dispatched: {len(state.agents)}")


@main.command()
@click.option("--session-dir", default="sessions", help="Session directory")
def stop(session_dir: str):
    """Request graceful stop of a running session."""
    state = load_session(session_dir)
    if state is None:
        click.echo("No active session found.")
        return

    stop_file = Path(session_dir) / "STOP"
    stop_file.write_text("stop requested\n")
    state.status = "stopped"
    save_session(session_dir, state)
    click.echo(f"Stop requested for session {state.session_id}")

    from hyperloom.breakdown import write_breakdown_json
    try:
        path = write_breakdown_json(session_dir)
        click.echo(f"Session breakdown written: {path}")
    except Exception as e:
        click.echo(f"WARNING: breakdown export failed: {e}", err=True)


@main.command()
@click.option("--session-dir", required=True, type=click.Path(exists=True), help="Session directory")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output path (default: <session_dir>/session_breakdown.json)")
@click.option("--report", is_flag=True, help="Also generate markdown report")
@click.option("--report-output", default=None, type=click.Path(), help="Report output path (default: <session_dir>/session_report.md)")
def breakdown(session_dir: str, output: str | None, report: bool, report_output: str | None):
    """Export session breakdown JSON (and optionally markdown report)."""
    from hyperloom.breakdown import write_breakdown_json, build, render_session_report

    path = write_breakdown_json(session_dir, output_path=output)
    click.echo(f"Breakdown written: {path} ({path.stat().st_size:,} bytes)")

    if report:
        bd = build(session_dir)
        md = render_session_report(bd)
        report_path = Path(report_output) if report_output else Path(session_dir) / "session_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md)
        click.echo(f"Report written: {report_path}")


# ─── SDK Orchestrator (agentic loop) ──────────────────────────────────────────


@main.command("run-orchestrator")
@click.option("--session-dir", required=True, type=click.Path(), help="Session directory")
@click.option("--system-prompt", required=True, type=click.Path(exists=True), help="Orchestrator system prompt file")
@click.option("--model", default="claude-sonnet-4-6", help="Model for orchestrator")
@click.option("--max-turns", default=10000, type=int, help="Maximum conversation turns")
@click.option("--target-file", default=None, type=click.Path(exists=True), help="Competitor target benchmarks JSON")
def run_orchestrator(
    session_dir: str,
    system_prompt: str,
    model: str,
    max_turns: int,
    target_file: str | None,
):
    """Run the SDK-based orchestrator agentic loop.

    The orchestrator LLM gets tools to dynamically dispatch agents,
    check their status, and collect results. It controls the full
    optimization schedule autonomously.
    """
    from hyperloom.orchestrator import run_sdk_orchestrator

    session_dir = os.path.abspath(session_dir)
    os.environ["SESSION_DIR"] = session_dir

    prompt_text = Path(system_prompt).read_text()

    target_context = ""
    if target_file:
        from hyperloom.state import load_targets
        targets = load_targets(target_file)
        target_context = "\n\n## Competitor Targets\n" + "\n".join(
            t.full_gap_summary([]) for t in targets
        )

    run_sdk_orchestrator(
        session_dir=session_dir,
        system_prompt=prompt_text + target_context,
        model=model,
        max_turns=max_turns,
    )


if __name__ == "__main__":
    main()
