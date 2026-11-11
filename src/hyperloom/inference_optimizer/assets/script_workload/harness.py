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
import traceback

# Imported only once the user script has started (child) or the command needs it (parent):
# the child must observe the script's own import order, not the harness's.
torch = None

CONFIG = "script_workload.json"
SCOPE = "optimization_scope.json"
SCORER = "scorer.py"
DEVICE = "cuda"
MS_PER_S = 1000
# The child subcommand, its two modes, and the directory layout they exchange.
CHILD = "_child"
RECORD = "record"
MEASURE = "measure"
INPUTS_DIR = "inputs"
REFERENCES_DIR = "references"
OUT_DIR = "out"
OUTPUTS_FILE = "outputs.pt"
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


_SCORER = None


def _use_scorer(directory: Path) -> None:
    """Load the operator's optional scorer from the frozen benchmark directory.

    It lives beside this file, which the candidate cannot write to, so it is
    operator code throughout: a candidate cannot soften its own grading.
    """
    global _SCORER
    path = Path(directory) / SCORER
    if not path.is_file():
        return
    spec = importlib.util.spec_from_file_location("hl_script_scorer", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SCORER = module


def _capture(value, where: str = "output"):
    """One model output, reduced to data the parent can load and compare.

    A scorer's ``capture`` runs first when it defines one, for an output whose shape
    follows its input and cannot be compared as a fixed structure. ``_plain`` still
    runs after, so the result must be tensors and plain containers.
    """
    if _SCORER is not None and hasattr(_SCORER, "capture"):
        value = _SCORER.capture(value)
    return _plain(value, where)


def _plain(value, where: str = "output"):
    """Plain containers and CPU tensors only, so the parent can load them with weights_only=True.

    Any other leaf is refused rather than stringified. A ``repr`` is invisible to
    ``_leaves``, so an output made only of them has nothing to compare and scores a
    perfect zero; numpy also elides arrays over 1000 elements, rendering two
    different masks identically. Recording runs the unmodified model, so an
    unsupported output is refused before a campaign spends anything.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {str(key): _plain(item, f"{where}[{key!r}]") for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{where} is a {type(value).__name__}, which the accuracy gate cannot compare. "
        "Return tensors, or plain containers and scalars built from them."
    )


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
    cast = torch.autocast(DEVICE, dtype=getattr(torch, dtype)) if dtype else contextlib.nullcontext()
    with grad, cast:
        return target(*args, **kwargs)


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_torch():
    global torch
    import torch as module

    torch = module


def _import_site() -> str:
    """The innermost non-importlib, non-harness frame: the statement that triggered this import."""
    for frame in reversed(traceback.extract_stack()):
        if not frame.filename.startswith("<frozen") and frame.filename != __file__:
            return f"{frame.filename}:{frame.lineno}"
    return "unknown"


class _ImportOrder(importlib.abc.MetaPathFinder):
    """Note where torch was first imported; never resolves anything itself."""

    def __init__(self):
        self.torch_site = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" and self.torch_site is None:
            self.torch_site = _import_site()

    def observe(self) -> dict:
        """What the optimization module's settings can still reach, as of its import."""
        loaded = sys.modules.get("torch")
        return {
            "module_site": _import_site(),
            "torch_site": self.torch_site,
            "torch_imported": loaded is not None,
            "gpu_initialized": loaded is not None and loaded.cuda.is_initialized(),
            "compiler_imported": [m for m in ("torch._inductor", "torch._dynamo") if m in sys.modules],
        }


class _CandidateFinder(importlib.abc.MetaPathFinder):
    """Resolve the optimization module to the candidate file and wrap its entry function."""

    def __init__(self, name: str, path: Path, function: str, wrap, on_import):
        self.name, self.path, self.function, self.wrap, self.on_import = name, path, function, wrap, on_import

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        self.on_import()
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
        autocast = torch.get_autocast_dtype(DEVICE) if torch.is_autocast_enabled(DEVICE) else None
        self.calls.append(
            {
                "args": snapshot[0],
                "kwargs": snapshot[1],
                "grad": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "autocast": str(autocast).removeprefix("torch.") if autocast is not None else None,
                "output": _capture(output),
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
            [_capture(_invoke(entry, call, inputs, autocast)) for inputs in variants]
            for call, variants in zip(calls, held_out)
        ]

    references = {"recorded": [call.pop("output") for call in calls], "held_out": held_out_outputs()}
    (out / INPUTS_DIR).mkdir()
    (out / REFERENCES_DIR).mkdir()
    torch.save(calls, out / INPUTS_DIR / "calls.pt")
    torch.save(held_out, out / INPUTS_DIR / "held_out.pt")
    torch.save(references, out / REFERENCES_DIR / OUTPUTS_FILE)
    autocast = config["tolerance"].get("autocast")
    if autocast:
        calibration = {
            "recorded": [_capture(_invoke(entry, call, autocast=autocast)) for call in calls],
            "held_out": held_out_outputs(autocast),
        }
        torch.save(calibration, out / REFERENCES_DIR / "calibration.pt")


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
                    # Cloned here, moved to host after the chunk: a module that recycles its
                    # output buffer would otherwise collapse the chunk's samples onto the last.
                    kept.append((index, schedule[(pass_index, index)], _snapshot(output)))
            latencies.append(elapsed * MS_PER_S)
        _send_mark("stop")
        for index, variant, output in kept:
            held_out_outputs[index][variant] = _capture(output)
    return latencies, held_out_outputs


def _reuses_output_buffer(entry, calls) -> bool:
    """Whether the next call overwrites the output the previous one returned.

    A module that hands back storage it will write again is not a drop-in
    replacement: a caller that keeps an output across calls silently gets the
    newer value. Graph replay and AOTInductor both return static buffers, so a
    module using either must clone before it returns.
    """
    held = _invoke(entry, calls[0])
    copy = _snapshot(held)
    _invoke(entry, calls[0], _perturb((calls[0]["args"], calls[0]["kwargs"])))
    _synchronize()
    return not all(torch.equal(a, b) for a, b in zip(_leaves(held), _leaves(copy)))


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
    recorded_outputs = [_capture(_invoke(entry, call)) for call in calls]
    reused = _reuses_output_buffer(entry, calls)
    trace = _profile(entry, calls)
    torch.save({"recorded": recorded_outputs, "held_out": held_out_outputs}, out / OUTPUTS_FILE)
    summary = {"pass_latencies_ms": latencies, "output_buffer_reused": reused, "profiler_trace": trace}
    (out / "measurement.json").write_text(json.dumps(summary) + "\n")


def _wrap(mode: str, config: dict, inputs: Path, out: Path, recorders: list):
    def wrap(function):
        def recording(*args, **kwargs):
            _load_torch()
            recorder = _Recorder(function(*args, **kwargs), config["method"], config["record_calls"])
            recorders.append(recorder)
            return recorder

        def measuring(*args, **kwargs):
            _load_torch()
            _measure(function(*args, **kwargs), config, inputs, out)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        return recording if mode == RECORD else measuring

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
    order = _ImportOrder()

    def on_import():
        (out / "import_order.json").write_text(json.dumps(order.observe()) + "\n")

    sys.meta_path[:0] = [order, _CandidateFinder(config["module"], candidate, config["function"], wrap, on_import)]
    _run_script(config)
    if mode == MEASURE:
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
    if not _leaves(reference):
        # Nothing numeric to compare, so there is no evidence either way; a zero here
        # would read as a perfect score.
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


def _score(candidate, reference) -> float:
    """The error between one captured output and its reference; lower is better.

    ``rel_l2`` unless the operator supplied a scorer. A scorer that raises or
    returns a non-number stops the run: every candidate would otherwise be graded
    by a broken instrument, which reads as the model being wrong.
    """
    if _SCORER is None or not hasattr(_SCORER, "score"):
        return rel_l2(candidate, reference)
    value = _SCORER.score(candidate, reference)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{SCORER}.score returned {value!r}; it must return a number, lower being better")
    return float(value)


def _errors(outputs: dict, references: dict) -> dict[str, list[float]]:
    """Per recorded call: its own error and the worst error over its held-out variants."""
    return {
        "recorded": [_score(o, r) for o, r in zip(outputs["recorded"], references["recorded"])],
        "held_out": [
            max(_score(o, r) for o, r in zip(outs, refs))
            for outs, refs in zip(outputs["held_out"], references["held_out"])
        ],
    }


def _limits(tolerance: dict, references: Path) -> list[float]:
    outputs = torch.load(references / OUTPUTS_FILE, weights_only=True)
    if "limit" in tolerance:
        return [float(tolerance["limit"])] * len(outputs["recorded"])
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
    return sum(stop - start for (_, start), (_, stop) in zip(marks[::2], marks[1::2])) * MS_PER_S


def _run_child(
    mode: str, config_path: Path, candidate: Path, inputs: Path, runtime: Path, logs: Path
) -> list[tuple[str, float]]:
    """Run one child writing into ``runtime/out``; return the parent-clock times of the marks it sent."""
    config = json.loads(config_path.read_text())
    (runtime / OUT_DIR).mkdir()
    read_fd, write_fd = os.pipe()
    marks: list[tuple[str, float]] = []
    reader = threading.Thread(target=_read_marks, args=(read_fd, marks), daemon=True)
    reader.start()
    command = [config["python"], "-B", __file__, CHILD, mode, "--config", str(config_path)]
    command += ["--candidate", str(candidate), "--inputs", str(inputs), "--out", str(runtime / OUT_DIR)]
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
    _verify_scorer(directory, config)
    _use_scorer(directory)
    store = directory / "reference"
    if store.exists():
        raise ValueError(f"{store} exists; references are recorded once and never replaced")
    with tempfile.TemporaryDirectory(prefix="script-workload-") as tmp:
        runtime = Path(tmp)
        identity = runtime / "identity.py"
        identity.write_text(IDENTITY.format(function=config["function"]))
        _run_child(RECORD, config_path, identity, runtime, runtime, directory)
        shutil.copytree(runtime / OUT_DIR, store)
    limits = _limits(config["tolerance"], store / REFERENCES_DIR)
    (store / REFERENCES_DIR / "limits.json").write_text(json.dumps(limits) + "\n")
    import_order = json.loads((store / "import_order.json").read_text())
    return {"calls": len(limits), "rel_l2_limits": limits, "import_order": import_order}


def _gate(errors: dict, limits: list[float], measurement: dict, timed_fraction: float) -> dict:
    cases = []
    for kind, values in errors.items():
        for index, (error, limit) in enumerate(zip(values, limits)):
            finite = math.isfinite(error)
            cases.append(
                {
                    "case": f"{kind}[{index}]",
                    # Not named for relative L2: with a scorer these are its units.
                    "error": error if finite else None,
                    "error_max": limit,
                    "passed": finite and error <= limit,
                }
            )
    cases.append({"case": "output_buffer_reuse", "passed": not measurement["output_buffer_reused"]})
    cases.append(
        {"case": "timing_consistent", "timed_fraction": timed_fraction, "passed": timed_fraction >= MIN_TIMED_FRACTION}
    )
    return {"passed": all(case["passed"] for case in cases), "cases": cases}


def _verify_scorer(directory: Path, config: dict) -> None:
    """The scorer is pinned like the script: both decide whether a candidate passes."""
    recorded = config.get("scorer_sha256")
    path = Path(directory) / SCORER
    actual = _sha256(path) if path.is_file() else None
    if recorded != actual:
        raise ValueError(f"{path} does not match the scorer the references were recorded with")


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
    _verify_scorer(directory, config)
    _use_scorer(directory)
    store = directory / "reference"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "inferencex_result.json").unlink(missing_ok=True)
    content = _candidate_file(directory).read_bytes()
    with tempfile.TemporaryDirectory(prefix="script-workload-") as tmp:
        runtime = Path(tmp)
        candidate = runtime / f"{config['module']}.py"
        candidate.write_bytes(content)
        marks = _run_child(MEASURE, config_path, candidate, store / INPUTS_DIR, runtime, result_dir)
        measurement = json.loads((runtime / OUT_DIR / "measurement.json").read_text())
        outputs = torch.load(runtime / OUT_DIR / OUTPUTS_FILE, weights_only=True)
    references = torch.load(store / REFERENCES_DIR / OUTPUTS_FILE, weights_only=True)
    limits = json.loads((store / REFERENCES_DIR / "limits.json").read_text())
    latencies = measurement["pass_latencies_ms"]
    gate = _gate(_errors(outputs, references), limits, measurement, sum(latencies) / _timed_wall_ms(marks))
    latency = statistics.median(latencies)
    payload = {
        "framework": "custom",
        "workload_kind": "scriptable",
        "throughput_unit": "invocations/s",
        "output_throughput": MS_PER_S / latency,
        "latency_s": latency / MS_PER_S,
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
    commands.add_parser(RECORD)
    commands.add_parser(MEASURE)
    internal = commands.add_parser(CHILD)
    internal.add_argument("mode", choices=(RECORD, MEASURE))
    for flag in ("--config", "--candidate", "--inputs", "--out"):
        internal.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    if args.command == CHILD:
        config = json.loads(args.config.read_text())
        # The child captures outputs, so it needs the scorer too; the parent already
        # checked the pin before spawning it.
        _use_scorer(directory)
        child(args.mode, config, args.candidate, args.inputs, args.out)
        return
    _load_torch()
    if args.command == RECORD:
        print(json.dumps(record(directory)))
    else:
        print(json.dumps(measure(directory, Path(os.environ["RESULT_DIR"]))))


if __name__ == "__main__":
    main()
