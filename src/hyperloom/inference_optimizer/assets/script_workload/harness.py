# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark an unchanged user script at its optimization-module boundary.

The user script calls ``<module>.<function>(model)`` once and uses what it returns.
A child process runs the script with ``<module>`` resolved to a candidate file and
that function wrapped:

* ``record`` (identity candidate): the script runs to completion while the calls of
  the returned object are recorded; held-out perturbed inputs and their outputs are
  then derived from the same unmodified model.
* ``measure`` (any candidate): when the function returns, the harness replays the
  recorded calls on fresh perturbed inputs, mixes the held-out inputs in at random
  timed positions, saves the outputs, and exits before the rest of the script runs.

The parent never imports the candidate. It owns the references and the accuracy
limits, compares the saved outputs, checks the reported time against its own clock,
and writes the ``inferencex_result.json`` Hyperloom reads.

Only the standard library and torch are imported: this file runs in the model's
environment, which need not have Hyperloom installed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import torch

CONFIG = "script_workload.json"
SCOPE = "optimization_scope.json"
PERTURBATION = 1e-2
INPUT_POOL_BYTES = 512 << 20
# The timed calls must account for most of the timed loop's wall time as the parent sees it.
MIN_TIMED_FRACTION = 0.8
IDENTITY = '"""Identity optimization: the unmodified model."""\n\n\ndef {function}(model):\n    return model\n'


# --- pytree helpers -------------------------------------------------------------------


def _map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        mapped = [_map_tensors(item, fn) for item in value]
        return type(value)(*mapped) if hasattr(value, "_fields") else type(value)(mapped)
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    return value


def _leaves(value) -> list[torch.Tensor]:
    leaves: list[torch.Tensor] = []
    _map_tensors(value, lambda tensor: leaves.append(tensor) or tensor)
    return leaves


def _snapshot(value):
    return _map_tensors(value, lambda tensor: tensor.detach().clone())


def _plain(value):
    """Plain containers and CPU tensors only, so the parent can load them with weights_only=True."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _perturb(value, generator: torch.Generator | None = None):
    """Same shapes, dtypes and devices; floating tensors get relative noise, others are copied."""

    def perturb(tensor):
        if not tensor.is_floating_point():
            return tensor.detach().clone()
        if generator is None:
            noise = torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32)
        else:
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32).to(tensor.device)
        scale = tensor.detach().float().std().nan_to_num(1.0).clamp_min(1e-6) * PERTURBATION
        return (tensor.detach().float() + noise * scale).to(tensor.dtype)

    return _map_tensors(value, perturb)


# --- child: calling the model the way the script did -----------------------------------


def _invoke(target, call, inputs=None, autocast=None):
    args, kwargs = inputs if inputs is not None else (call["args"], call["kwargs"])
    if call["inference_mode"]:
        grad = torch.inference_mode()
    else:
        grad = torch.enable_grad() if call["grad"] else torch.no_grad()
    dtype = autocast or call["autocast"]
    cast = torch.autocast("cuda", dtype=getattr(torch, dtype)) if dtype else contextlib.nullcontext()
    with grad, cast:
        return target(*args, **kwargs)


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _CandidateFinder(importlib.abc.MetaPathFinder):
    """Resolve the optimization module to the candidate file and wrap its entry function."""

    def __init__(self, name: str, path: Path, function: str, wrap):
        self.name, self.path, self.function, self.wrap = name, path, function, wrap

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        spec = importlib.util.spec_from_file_location(fullname, self.path)
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            setattr(module, self.function, self.wrap(getattr(module, self.function)))

        spec.loader.exec_module = exec_module
        return spec


class _Recorder:
    """Forward everything to the wrapped object and record its entry calls."""

    def __init__(self, target, method: str, limit: int):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_limit", limit)
        object.__setattr__(self, "calls", [])

    def _record(self, fn, args, kwargs):
        if len(self.calls) >= self._limit:
            return fn(*args, **kwargs)
        snapshot = _snapshot((args, kwargs))
        output = fn(*args, **kwargs)
        autocast = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
        self.calls.append(
            {
                "args": snapshot[0],
                "kwargs": snapshot[1],
                "grad": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "autocast": str(autocast).removeprefix("torch.") if autocast is not None else None,
                "output": _plain(output),
            }
        )
        return output

    def __call__(self, *args, **kwargs):
        if self._method == "__call__":
            return self._record(self._target, args, kwargs)
        return self._target(*args, **kwargs)

    def __getattr__(self, name):
        attribute = getattr(self._target, name)
        if name == self._method:
            return lambda *args, **kwargs: self._record(attribute, args, kwargs)
        return attribute

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


def _entry(target, method: str):
    return target if method == "__call__" else getattr(target, method)


def _finish_recording(recorder: _Recorder, config: dict, out: Path) -> None:
    calls = recorder.calls
    if not calls:
        raise SystemExit(f"The script never called the object returned by {config['function']}()")
    entry = _entry(recorder._target, config["method"])
    generator = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
    held_out = [[_perturb((c["args"], c["kwargs"]), generator) for _ in range(config["held_out"])] for c in calls]

    def held_out_outputs(autocast=None):
        return [
            [_plain(_invoke(entry, call, inputs, autocast)) for inputs in variants]
            for call, variants in zip(calls, held_out)
        ]

    references = {"recorded": [call.pop("output") for call in calls], "held_out": held_out_outputs()}
    (out / "inputs").mkdir()
    (out / "references").mkdir()
    torch.save(calls, out / "inputs" / "calls.pt")
    torch.save(held_out, out / "inputs" / "held_out.pt")
    torch.save(references, out / "references" / "outputs.pt")
    autocast = config["tolerance"].get("autocast")
    if autocast:
        calibration = {
            "recorded": [_plain(_invoke(entry, call, autocast=autocast)) for call in calls],
            "held_out": held_out_outputs(autocast),
        }
        torch.save(calibration, out / "references" / "calibration.pt")


def _schedule(calls: int, held_out: int, passes: int, rng: random.Random) -> dict[tuple[int, int], int]:
    """A random timed pass for every held-out variant of every call: (pass, call) -> variant."""
    positions = {}
    for call in range(calls):
        for variant, pass_index in enumerate(rng.sample(range(passes), held_out)):
            positions[(pass_index, call)] = variant
    return positions


def _send_mark(text: str) -> None:
    fd = os.environ.get("HL_SCRIPT_WORKLOAD_MARK_FD")
    if fd:
        os.write(int(fd), f"{text}\n".encode())


def _warm_up(entry, calls, config):
    deadline = time.monotonic() + config["warmup_s"]
    passes = 0
    while passes < config["warmup_passes"] or time.monotonic() < deadline:
        for call in calls:
            _invoke(entry, call)
        passes += 1
    _synchronize()


def _pass_inputs(calls, held_out, schedule, pass_index):
    inputs = []
    for index, call in enumerate(calls):
        variant = schedule.get((pass_index, index))
        inputs.append(held_out[index][variant] if variant is not None else _perturb((call["args"], call["kwargs"])))
    return inputs


def _chunk_passes(calls, passes: int) -> int:
    """Passes whose fresh inputs fit the pre-generation budget at once."""
    per_pass = sum(t.numel() * t.element_size() for c in calls for t in _leaves((c["args"], c["kwargs"])))
    return max(1, min(passes, INPUT_POOL_BYTES // max(1, per_pass)))


def _timed_passes(entry, calls, held_out, config):
    """Time every pass on fresh inputs, generated per chunk outside the parent-visible timed sections."""
    rng = random.Random(os.urandom(16))
    torch.manual_seed(rng.getrandbits(63))
    passes = config["timed_passes"]
    schedule = _schedule(len(calls), len(held_out[0]), passes, rng)
    chunk = _chunk_passes(calls, passes)
    held_out_outputs = [[None] * len(variants) for variants in held_out]
    latencies = []
    for first in range(0, passes, chunk):
        pool = [_pass_inputs(calls, held_out, schedule, p) for p in range(first, min(first + chunk, passes))]
        kept = []
        _synchronize()
        _send_mark("start")
        for pass_index, inputs in enumerate(pool, start=first):
            elapsed = 0.0
            for index, call in enumerate(calls):
                _synchronize()
                start = time.perf_counter()
                output = _invoke(entry, call, inputs[index])
                _synchronize()
                elapsed += time.perf_counter() - start
                if (pass_index, index) in schedule:
                    kept.append((index, schedule[(pass_index, index)], output))
            latencies.append(elapsed * 1000)
        _send_mark("stop")
        for index, variant, output in kept:
            held_out_outputs[index][variant] = _plain(output)
    return latencies, held_out_outputs


def _output_survives_next_call(entry, calls) -> bool:
    """An output must not change when the next call runs; graph replay can reuse its buffers."""
    held = _invoke(entry, calls[0])
    copy = _snapshot(held)
    _invoke(entry, calls[0], _perturb((calls[0]["args"], calls[0]["kwargs"])))
    _synchronize()
    return all(torch.equal(a, b) for a, b in zip(_leaves(held), _leaves(copy)))


def _profile(entry, calls) -> str | None:
    trace_dir = os.environ.get("HYPERLOOM_PROFILE_TRACE_DIR")
    if os.environ.get("PROFILE") != "1" or not trace_dir:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        for call in calls:
            _invoke(entry, call)
        _synchronize()
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    trace = Path(trace_dir) / "model.pt.trace.json.gz"
    profiler.export_chrome_trace(str(trace))
    return str(trace)


def _measure(optimized, config: dict, inputs: Path, out: Path) -> None:
    entry = _entry(optimized, config["method"])
    calls = torch.load(inputs / "calls.pt", weights_only=False)
    held_out = torch.load(inputs / "held_out.pt", weights_only=False)
    _warm_up(entry, calls, config)
    latencies, held_out_outputs = _timed_passes(entry, calls, held_out, config)
    recorded_outputs = [_plain(_invoke(entry, call)) for call in calls]
    survives = _output_survives_next_call(entry, calls)
    trace = _profile(entry, calls)
    torch.save({"recorded": recorded_outputs, "held_out": held_out_outputs}, out / "outputs.pt")
    summary = {"pass_latencies_ms": latencies, "held_output": survives, "profiler_trace": trace}
    (out / "measurement.json").write_text(json.dumps(summary) + "\n")


def _wrap(mode: str, config: dict, inputs: Path, out: Path, recorders: list):
    def wrap(function):
        def recording(*args, **kwargs):
            recorder = _Recorder(function(*args, **kwargs), config["method"], config["record_calls"])
            recorders.append(recorder)
            return recorder

        def measuring(*args, **kwargs):
            _measure(function(*args, **kwargs), config, inputs, out)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        return recording if mode == "record" else measuring

    return wrap


def _run_script(config: dict) -> None:
    script = Path(config["script"])
    sys.argv = [str(script), *config["arguments"]]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


def child(mode: str, config: dict, candidate: Path, inputs: Path, out: Path) -> None:
    recorders: list[_Recorder] = []
    wrap = _wrap(mode, config, inputs, out, recorders)
    sys.meta_path.insert(0, _CandidateFinder(config["module"], candidate, config["function"], wrap))
    _run_script(config)
    if mode == "measure":
        raise SystemExit(f"The script exited without calling {config['module']}.{config['function']}()")
    if len(recorders) != 1:
        raise SystemExit(f"Expected one call of {config['function']}(), saw {len(recorders)}")
    _finish_recording(recorders[0], config, out)


# --- parent: references, limits, comparison and the report -----------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structure(value):
    return _map_tensors(value, lambda tensor: ("tensor", tuple(tensor.shape)))


def rel_l2(candidate, reference) -> float:
    """Largest relative L2 error over the output's tensors; inf on a structure or finiteness mismatch."""
    if candidate is None or _structure(candidate) != _structure(reference):
        return math.inf
    worst = 0.0
    for out, ref in zip(_leaves(candidate), _leaves(reference)):
        out, ref = out.double(), ref.double()
        if not torch.isfinite(out).all():
            return math.inf
        norm = torch.linalg.vector_norm(ref)
        diff = torch.linalg.vector_norm(out - ref)
        worst = max(worst, float(diff / norm) if norm > 0 else float(diff))
    return worst


def _errors(outputs: dict, references: dict) -> dict[str, list[float]]:
    """Per recorded call: its own error and the worst error over its held-out variants."""
    return {
        "recorded": [rel_l2(o, r) for o, r in zip(outputs["recorded"], references["recorded"])],
        "held_out": [
            max(rel_l2(o, r) for o, r in zip(outs, refs))
            for outs, refs in zip(outputs["held_out"], references["held_out"])
        ],
    }


def _limits(tolerance: dict, references: Path) -> list[float]:
    outputs = torch.load(references / "outputs.pt", weights_only=True)
    if "rel_l2" in tolerance:
        return [float(tolerance["rel_l2"])] * len(outputs["recorded"])
    errors = _errors(torch.load(references / "calibration.pt", weights_only=True), outputs)
    return [
        max(max(recorded, held_out) * tolerance["headroom"], tolerance["floor"])
        for recorded, held_out in zip(errors["recorded"], errors["held_out"])
    ]


def _child_env(runtime: Path, mark_fd: int) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "cache" / "inductor"),
        "TRITON_CACHE_DIR": str(runtime / "cache" / "triton"),
        "HL_SCRIPT_WORKLOAD_MARK_FD": str(mark_fd),
    }


def _read_marks(fd: int, marks: list[tuple[str, float]]) -> None:
    with os.fdopen(fd, "rb") as stream:
        for line in stream:
            marks.append((line.decode().strip(), time.monotonic()))


def _timed_wall_ms(marks: list[tuple[str, float]]) -> float:
    """Parent-clock duration of the child's timed sections; inf if the marks do not pair up."""
    labels = [label for label, _ in marks]
    if not marks or labels != ["start", "stop"] * (len(marks) // 2):
        return math.inf
    return sum(stop - start for (_, start), (_, stop) in zip(marks[::2], marks[1::2])) * 1000


def _run_child(
    mode: str, config_path: Path, candidate: Path, inputs: Path, runtime: Path, logs: Path
) -> list[tuple[str, float]]:
    """Run one child writing into ``runtime/out``; return the parent-clock times of the marks it sent."""
    config = json.loads(config_path.read_text())
    (runtime / "out").mkdir()
    read_fd, write_fd = os.pipe()
    marks: list[tuple[str, float]] = []
    reader = threading.Thread(target=_read_marks, args=(read_fd, marks), daemon=True)
    reader.start()
    command = [config["python"], "-B", __file__, "_child", mode, "--config", str(config_path)]
    command += ["--candidate", str(candidate), "--inputs", str(inputs), "--out", str(runtime / "out")]
    with (logs / f"{mode}_stdout.log").open("w") as stdout, (logs / f"{mode}_stderr.log").open("w") as stderr:
        process = subprocess.run(
            command,
            cwd=config["cwd"],
            env=_child_env(runtime, write_fd),
            stdout=stdout,
            stderr=stderr,
            pass_fds=(write_fd,),
        )
    os.close(write_fd)
    reader.join(timeout=5)
    if process.returncode:
        raise RuntimeError(f"The script exited {process.returncode} in {mode} mode; see {logs}/{mode}_stderr.log")
    return marks


def record(directory: Path) -> dict:
    """Record the references and accuracy limits once, with the identity module."""
    config_path = directory / CONFIG
    config = json.loads(config_path.read_text())
    store = directory / "reference"
    if store.exists():
        raise ValueError(f"{store} exists; references are recorded once and never replaced")
    with tempfile.TemporaryDirectory(prefix="script-workload-") as tmp:
        runtime = Path(tmp)
        identity = runtime / "identity.py"
        identity.write_text(IDENTITY.format(function=config["function"]))
        _run_child("record", config_path, identity, runtime, runtime, directory)
        shutil.copytree(runtime / "out", store)
    limits = _limits(config["tolerance"], store / "references")
    (store / "references" / "limits.json").write_text(json.dumps(limits) + "\n")
    return {"calls": len(limits), "rel_l2_limits": limits}


def _gate(errors: dict, limits: list[float], measurement: dict, timed_fraction: float) -> dict:
    cases = []
    for kind, values in errors.items():
        for index, (error, limit) in enumerate(zip(values, limits)):
            finite = math.isfinite(error)
            cases.append(
                {
                    "case": f"{kind}[{index}]",
                    "rel_l2": error if finite else None,
                    "rel_l2_max": limit,
                    "passed": finite and error <= limit,
                }
            )
    cases.append({"case": "held_output", "passed": measurement["held_output"]})
    cases.append(
        {"case": "timing_consistent", "timed_fraction": timed_fraction, "passed": timed_fraction >= MIN_TIMED_FRACTION}
    )
    return {"passed": all(case["passed"] for case in cases), "cases": cases}


def _candidate_file(directory: Path) -> Path:
    candidate = Path(json.loads((directory / SCOPE).read_text())["file"])
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("The optimization module must be a regular file")
    return candidate


def measure(directory: Path, result_dir: Path) -> dict:
    """Measure the current optimization module and write Hyperloom's result file."""
    config_path = directory / CONFIG
    config = json.loads(config_path.read_text())
    if _sha256(Path(config["script"])) != config["script_sha256"]:
        raise ValueError(f"{config['script']} changed since its references were recorded")
    store = directory / "reference"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "inferencex_result.json").unlink(missing_ok=True)
    content = _candidate_file(directory).read_bytes()
    with tempfile.TemporaryDirectory(prefix="script-workload-") as tmp:
        runtime = Path(tmp)
        candidate = runtime / f"{config['module']}.py"
        candidate.write_bytes(content)
        marks = _run_child("measure", config_path, candidate, store / "inputs", runtime, result_dir)
        measurement = json.loads((runtime / "out" / "measurement.json").read_text())
        outputs = torch.load(runtime / "out" / "outputs.pt", weights_only=True)
    references = torch.load(store / "references" / "outputs.pt", weights_only=True)
    limits = json.loads((store / "references" / "limits.json").read_text())
    latencies = measurement["pass_latencies_ms"]
    gate = _gate(_errors(outputs, references), limits, measurement, sum(latencies) / _timed_wall_ms(marks))
    latency = statistics.median(latencies)
    payload = {
        "framework": "custom",
        "workload_kind": "scriptable",
        "throughput_unit": "invocations/s",
        "output_throughput": 1000 / latency,
        "latency_s": latency / 1000,
        "quality_gate": gate,
        "bench_summary": {
            "latency_ms": latency,
            "calls_per_invocation": len(limits),
            "optimization_sha256": hashlib.sha256(content).hexdigest(),
            "profiler_trace": measurement["profiler_trace"],
        },
    }
    (result_dir / "inferencex_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("record")
    commands.add_parser("measure")
    internal = commands.add_parser("_child")
    internal.add_argument("mode", choices=("record", "measure"))
    for flag in ("--config", "--candidate", "--inputs", "--out"):
        internal.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    if args.command == "_child":
        config = json.loads(args.config.read_text())
        child(args.mode, config, args.candidate, args.inputs, args.out)
    elif args.command == "record":
        print(json.dumps(record(directory)))
    else:
        print(json.dumps(measure(directory, Path(os.environ["RESULT_DIR"]))))


if __name__ == "__main__":
    main()
