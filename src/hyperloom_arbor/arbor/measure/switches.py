"""A kernel plan's stage S4, "in tree", checked by the harness at every KEEP.

S4 lands a new kernel at its call site behind a switch that defaults off, and
the kernel plan (the dossier skill's kernel-plan.md, the kernels skill) asks
five things of it: the switch defaults off, dispatch on every rank, zero JIT
compiles while serving, both switch positions tested, and the accuracy gate.
Until v0.9.1 the agent asserted them as flags in its STAGE line, which only
parallel mode's lanes judge. A Hyperloom MB branch kept such a step
(``HL_OSK_DECODE``, a split-K decode GEMM JIT-compiled at first call) whose
per-GEMM check passed at three batch sizes; served with the switch on, GSM8K
accuracy collapsed.

So the harness checks every switch a KEEP lands, lane or not:

- **Which:** the agent's ``SWITCH: NAME=VALUE`` lines (VALUE turns it on, ``1``
  when omitted), and every environment variable the change's Python starts to
  read that the tree did not name before and that neither the arm nor the
  campaign's launch sets. An undeclared one refuses the KEEP: the harness
  cannot know how to turn it on.
- **Off by default:** neither the arm's settings nor the campaign's launch set
  it to anything but off (``OFF_VALUES``), and the added code's own default,
  where a read names one (``os.getenv("X", "1")``), is off too. A new read
  whose default is on is not a switch: the arm's own run served that path.
- **Both positions:** the arm's own run served it off. The harness then serves
  the tree again with every switch on and runs the accuracy check there: the
  full GSM8K against the golden score less the tolerance on a GSM8K workload
  (arbor/measure/accuracy.py), the pack's correctness check on any other.
- **Dispatch on every rank:** that server's log names the switch in a line
  from each rank (``TP<n>`` or ``rank <n>`` tags), or in as many lines as
  ranks when its lines carry none. The kernel logs the line when it
  dispatches; vLLM's unknown-variable warning does not count.
- **Zero JIT while serving:** no file appears in a JIT cache (Triton, AITER,
  torch extensions and inductor, vLLM, FlyDSL) between that server's health
  and the end of its accuracy check. The arm's own run has warmed every cache
  the tree already used, so what appears now is the switch's path compiling
  while it serves.

A failure turns the KEEP into a REVERT, with the reason. A check that could not
run (no time left before the deadline, an eval with no score) refuses nothing,
as the short-prompt check refuses nothing (arbor/measure/short_prompts.py): the
switch stays off, so the KEEP serves as it was measured. Either way the switch
is recorded in ``<session>/switches.json`` with its verdict, and a later change
that turns it on is held to the accuracy gate in every mode (``turned_on``,
``accuracy.trigger``). The KEEP's snapshot carries the record for its host.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

FILE = "switches.json"          # <session>/: every switch a KEEP landed
ITER_FILE = "switches.json"     # iter_N/: this KEEP's check
OFF_VALUES = ("", "0", "false", "off", "no", "none", "disabled")
DEFAULT_ON = "1"

# Upper case only, as STAGE: "Switch:" is an ordinary word in a DONE line.
_SWITCH_LINE = re.compile(r"^[*_`>#\-\s]*SWITCH[*_`]*\s*:[\s*_`]*([A-Z][A-Z0-9_]+)(?:\s*=\s*([^\s`*,;]+))?")
_NAME = r"([A-Z][A-Z0-9_]{2,})"
_READS = re.compile(
    rf"""(?:os\.environ\.get|os\.getenv|environ\.get|getenv)\(\s*["']{_NAME}["']"""
    rf"""|environ\[\s*["']{_NAME}["']\s*\]"""
    rf"""|["']{_NAME}["']\s+(?:not\s+)?in\s+os\.environ"""
    rf"""|\benvs\.{_NAME}\b""")
# A read that names its default: a string, or a bare number or constant.
_DEFAULT = re.compile(
    rf"""(?:os\.environ\.get|os\.getenv|environ\.get|getenv)\(\s*["']{_NAME}["']\s*,\s*"""
    r"""(?:["']([^"']*)["']|(True|False|None|-?\d+))""")
# Paths whose Python is not the served tree: the harness's scratch, tests, benchmarks.
_NOT_SERVED = re.compile(r"(^|/)(arbor|tests?|benchmarks?|eval)/|(^|/)test_[^/]*\.py$|(^|/)conftest\.py$")
_RANK = re.compile(r"(?<![A-Za-z])TP[ _-]?(\d+)\b|(?<![A-Za-z])rank[ _=:-]*(\d+)\b", re.IGNORECASE)
_UNKNOWN_ENV = re.compile(r"unknown .*environment variable", re.IGNORECASE)

# Where a framework's kernels compile to: (the variable that moves it, its default).
JIT_CACHES = (
    ("TRITON_CACHE_DIR", "$HOME/.triton/cache"),
    ("AITER_JIT_DIR", "$HOME/.aiter/jit"),
    ("TORCH_EXTENSIONS_DIR", "$HOME/.cache/torch_extensions"),
    ("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_$USER"),
    ("VLLM_CACHE_ROOT", "$HOME/.cache/vllm"),
    ("FLYDSL_RUNTIME_CACHE_DIR", "$HOME/.flydsl/cache"),
)
JIT_LISTED = 20


def is_off(value) -> bool:
    return value is None or str(value).strip().lower() in OFF_VALUES


def declared(output: str) -> dict[str, str]:
    """``SWITCH: NAME=VALUE`` lines, decorated or not: ``{NAME: on value}``, the last per name."""
    out: dict[str, str] = {}
    for line in (output or "").splitlines():
        m = _SWITCH_LINE.match(line)
        if m:
            out[m.group(1)] = (m.group(2) or DEFAULT_ON).strip("\"'")
    return out


def _served_lines(git_diff: str):
    """The diff's added lines in the served tree's Python."""
    path = ""
    for line in (git_diff or "").splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            continue
        if line.startswith("+") and path.endswith(".py") and not _NOT_SERVED.search(path):
            yield line[1:]


def env_reads(git_diff: str) -> set[str]:
    """The environment variables the diff's added lines read, in the served tree's Python."""
    return {g for line in _served_lines(git_diff) for m in _READS.finditer(line) for g in m.groups() if g}


def code_defaults(git_diff: str) -> dict[str, str]:
    """The value the added code gives each variable it reads when it is unset
    (``os.getenv("X", "1")``), where a read names one: one that turns it on wins."""
    out: dict[str, str] = {}
    for line in _served_lines(git_diff):
        for m in _DEFAULT.finditer(line):
            name, value = m.group(1), m.group(2) if m.group(2) is not None else m.group(3)
            if name not in out or (is_off(out[name]) and not is_off(value)):
                out[name] = value
    return out


def landed(output: str, git_diff: str, settings: dict, launch_env: dict,
           existed: Callable[[str], bool]) -> tuple[dict[str, str], list[str]]:
    """The switches this change lands, ``{NAME: on value}``, and why the KEEP cannot stand as it
    is: a new variable it reads that it leaves unset and does not declare, and whose code
    default is off. ``settings`` are the arm's (knobs.settings, after its commit),
    ``launch_env`` the campaign's ``server_env``, and ``existed(name)`` says whether the tree
    named ``name`` before the change. A read whose default is on is served by the arm's own run."""
    switches = declared(output)
    defaults = code_defaults(git_diff)
    refusals = []
    for name in sorted(env_reads(git_diff) - set(switches)):
        if name in settings or name in (launch_env or {}) or not is_off(defaults.get(name)):
            continue   # a setting the arm or the launch gives, or a path the arm served
        if existed(name):
            continue   # one the tree already read
        refusals.append(
            f"the change starts to read {name}, which the tree did not name and the arm leaves "
            f"unset: a switch that defaults off. Declare it with `SWITCH: {name}=<on value>` so "
            f"the harness serves it on, or set it in the arm")
    return switches, refusals


def default_on(switches: dict[str, str], settings: dict, launch_env: dict,
               defaults: dict[str, str] | None = None) -> list[str]:
    """Why a switch is not off by default: the arm or the campaign's launch sets it on, or the
    code turns it on when it is unset (``code_defaults``)."""
    why = []
    for name in sorted(switches):
        for where, values in (("the arm", settings), ("the campaign's launch", launch_env or {})):
            if not is_off(values.get(name)):
                why.append(f"{name} is not off by default: {where} sets {name}={values.get(name)}")
                break
        else:
            if not is_off((defaults or {}).get(name)):
                why.append(f"{name} is not off by default: the code reads it as "
                           f"{defaults[name]!r} when it is unset")
    return why


def turned_on(session_dir, before: dict, after: dict) -> list[str]:
    """The switches an earlier KEEP landed (``<session>/switches.json``) that the arm turns on."""
    known = load(session_dir)
    return sorted(n for n in known if is_off(before.get(n)) and not is_off(after.get(n)))


def jit_roots(env: dict) -> list[str]:
    """The JIT caches the server writes, as the shell names them: each moved by its variable
    when the server's env sets it."""
    return [str(env.get(var)) if env.get(var) else default for var, default in JIT_CACHES]


def jit_command(roots: list[str]) -> str:
    """A shell command that prints every file under ``roots``. Two listings are compared, not
    mtimes against a clock: a cache on NFS is stamped by the server's clock, not this host's."""
    quoted = " ".join(f'"{r}"' for r in roots)   # "$HOME/..." expands; a root absent is skipped
    return f'for d in {quoted}; do [ -d "$d" ] && find "$d" -type f 2>/dev/null; done; true'


def dispatch(log: str, name: str, tp: int) -> tuple[bool, str]:
    """Does the log show ``name``'s path taken on every rank: a line naming it from each of the
    ``tp`` ranks, or ``tp`` lines naming it when no line carries a rank tag."""
    lines = [ln for ln in (log or "").splitlines() if name in ln and not _UNKNOWN_ENV.search(ln)]
    tp = max(1, int(tp or 1))
    if not lines:
        return False, f"no server log line names {name}: log it where the kernel dispatches"
    ranks = {next(g for g in m.groups() if g) for ln in lines for m in [_RANK.search(ln)] if m}
    if ranks:
        if len(ranks) >= tp:
            return True, f"{name} logged on ranks {', '.join(sorted(ranks, key=int))}"
        return False, f"{name} logged on rank(s) {', '.join(sorted(ranks, key=int))} of {tp}"
    if len(lines) >= tp:
        return True, f"{len(lines)} line(s) name {name}"
    return False, f"{len(lines)} line(s) name {name}, fewer than the {tp} ranks"


def check(config: dict, iter_dir, switches: dict[str, str], *, gsm8k: bool, tolerance: float,
          start: Callable[[dict, dict], int], healthy: Callable[[dict, int], bool],
          stop: Callable[[dict], None], run: Callable, jit_files: Callable[[dict], set[str]],
          read_log: Callable[[dict], str], clock: Callable[[], float] = time.time) -> dict:
    """Serve the tree with every switch on, run the accuracy check, compare the JIT caches before
    and after it, read the log, stop that server, and return the record: ``{verdict, reasons,
    switches, ...}`` with a verdict of pass, fail or not_run. Each phase keeps component mode's
    cap and deadline. ``start(config, extra_env)`` launches with the switches' values set."""
    from arbor import component
    out = Path(iter_dir) / "switches"
    rec: dict = {"switches": dict(switches), "verdict": "not_run", "reasons": [], "ts": _iso(clock())}
    began = clock()
    try:
        stop(config)
        boot = component.phase_seconds(config, "server")
        if boot is not None and boot <= 0:
            rec["reasons"].append("no time is left to serve the switches on")
            return rec
        component.phase("server")
        on = ", ".join(f"{k}={v}" for k, v in sorted(switches.items()))
        if not start(config, dict(switches)) or not healthy(config, int(boot if boot is not None else 600)):
            rec.update(verdict="fail", served=False)
            rec["reasons"].append(f"the tree does not serve with {on}: the server did not come up")
            return rec
        rec["served"] = True
        warm = jit_files(config)
        cap = component.phase_seconds(config, "correctness")
        if cap is not None and cap <= 0:
            rec["reasons"].append("no time is left for the accuracy check")
            return rec
        component.phase("correctness")
        failed, unread = _evaluate(config, out, switches, gsm8k, tolerance, run, cap, rec)
        written = sorted(jit_files(config) - warm)
        rec["jit"] = written[:JIT_LISTED]
        if written:
            failed.append(f"{len(written)} file(s) compiled into a JIT cache while serving with {on}, "
                          f"e.g. {written[0]}: warm the kernel before the server is healthy")
        log = read_log(config)
        tp = int(config.get("tp") or 1)
        rec["dispatch"] = {}
        for name in sorted(switches):
            ok, why = dispatch(log, name, tp)
            rec["dispatch"][name] = why
            if not ok:
                failed.append(f"dispatch: {why}")
        rec["reasons"] += failed + unread
        rec["verdict"] = "fail" if failed else ("not_run" if unread else "pass")
        return rec
    finally:
        try:
            stop(config)
        finally:
            rec["seconds"] = round(clock() - began, 1)
            save(iter_dir, rec)


def _evaluate(config: dict, out: Path, switches: dict, gsm8k: bool, tolerance: float, run, cap,
              rec: dict) -> tuple[list[str], list[str]]:
    """The accuracy check on the switch-on server: (what failed, what could not be read)."""
    from arbor.measure import accuracy
    on = ", ".join(sorted(switches))
    if gsm8k:
        acc = accuracy.measure(config, out, [f"switch {n} on" for n in sorted(switches)], tolerance,
                               out_dir=out / "accuracy", run=run, cap=cap)
        rec["accuracy"] = accuracy.brief(acc)
        verdict, why = accuracy.reading(acc)
        if verdict == "fail":
            return [f"accuracy with {on} on: {why}, full GSM8K"], []
        return [], ([f"accuracy with {on} on: {why}"] if verdict == "not_run" else [])
    result = run(config, out / "correctness", cap=cap)
    rec["correctness"] = {k: result.get(k) for k in ("passed", "score", "error") if k in result}
    if result.get("capped"):
        return [], [f"the correctness check with {on} on timed out after {result['capped']} s"]
    if not result.get("passed"):
        return [f"correctness with {on} on: {str(result.get('error') or 'failed')[:300]}"], []
    return [], []


def save(iter_dir, rec: dict) -> None:
    """This KEEP's check, ``iter_N/switches/switches.json``, which its snapshot carries."""
    _write(Path(iter_dir) / "switches" / ITER_FILE, rec)


def load(session_dir) -> dict[str, dict]:
    """``<session>/switches.json``: every switch a KEEP landed, by name."""
    try:
        doc = json.loads((Path(session_dir) / FILE).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def remember(session_dir, iteration: int, rec: dict) -> None:
    """Add a KEEP's switches to ``<session>/switches.json``."""
    doc = load(session_dir)
    for name, value in (rec.get("switches") or {}).items():
        doc[name] = {"on": value, "iteration": iteration, "verdict": rec.get("verdict"),
                     "reasons": rec.get("reasons") or []}
    _write(Path(session_dir) / FILE, doc)


def brief(rec: dict | None) -> dict | None:
    """The row's and the KEEP snapshot's record."""
    if not rec:
        return None
    return {k: rec[k] for k in ("switches", "verdict", "reasons", "accuracy", "correctness", "jit", "dispatch")
            if k in rec}


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
