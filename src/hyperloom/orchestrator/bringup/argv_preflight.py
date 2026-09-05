# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Judge a server argv with the installed framework's own argument parser.

Only the parser that will reject an argument knows which spellings it accepts
today, so the parser is what answers: the adapter supplies the source that
builds it, the probe runs that source in the interpreter that will serve.
Nothing here is a table of flags.

Three outcomes, never two: :data:`OK` and :data:`INVALID` are verdicts,
:data:`UNAVAILABLE` is the answer whenever the check could not be made and never
stops a round. Nothing is cached -- a round exists to change the installed
framework, so a probe that reached no parser this round may reach one next.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # nosec B404 - the probe is an interpreter this module resolves
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.bringup import BootObservation, Excerpt, LadderStage, redact

log = logging.getLogger(__name__)

#: Names this module's observations in downstream artifacts.
PRODUCER = "preflight.argv"

#: The marker an argv-rejection observation carries, and the name of the
#: terminal it produces.
ARGV_INVALID = "server_argv_invalid"

#: The stream name recorded on the excerpt, so a reader can tell parser output
#: from a server log.
STREAM = "argv_preflight"

OK = "ok"
INVALID = "invalid"
UNAVAILABLE = "unavailable"

#: The YAML env that pins the interpreter the server is launched with; when set
#: it wins over anything on ``PATH``.
SERVING_PYTHON_ENV = "HYPERLOOM_FRAMEWORK_PYTHON"

#: Reasons, in the vocabulary the report and the terminal use.
PARSED = "parsed"
PARSED_AFTER_DROP = "parsed_after_drop"
VALUE_REJECTED = "value_rejected"
#: The argv's one drop-only repair is not available -- already used, or there
#: is no digest to record a new one under.
REPAIR_SPENT = "repair_spent"
REPAIR_FAILED = "repair_failed"
NO_PARSER = "no_parser"
PROBE_FAILED = "probe_failed"
INTERPRETER_UNPROVEN = "interpreter_unproven"
INTERPRETER_MISMATCH = "interpreter_mismatch"

#: Identity is four ``sys``/``importlib`` lookups; the parse probe imports the
#: framework's entry point, so it is given longer.
_IDENTITY_TIMEOUT_SEC = 10.0
_PARSE_TIMEOUT_SEC = 30.0

#: How much probe output is carried into an observation.
_DETAIL_CHARS = 600

#: Runs one probe: ``(argv, env, timeout_sec) -> CompletedProcess``.
ProbeFn = Callable[[list[str], Mapping[str, str], float], "subprocess.CompletedProcess[str]"]

_IDENTITY_PROGRAM = (
    "import importlib.util as u, json, os, sys\n"
    "name = sys.argv[1]\n"
    "try:\n"
    "    spec = u.find_spec(name)\n"
    "    origin = getattr(spec, 'origin', '') or ''\n"
    "except BaseException:\n"
    "    origin = ''\n"
    "try:\n"
    "    import importlib.metadata as meta\n"
    "    dist = meta.version(name)\n"
    "except BaseException:\n"
    "    dist = ''\n"
    "print(json.dumps({\n"
    "    'executable': os.path.realpath(sys.executable),\n"
    "    'python': '.'.join(str(part) for part in sys.version_info[:3]),\n"
    "    'origin': os.path.realpath(origin) if origin else '',\n"
    "    'dist': dist,\n"
    "}))\n"
)

# ``parse_known_args`` so an unrecognised flag comes back as a token this module
# can name and drop. Required-ness is cleared first: the string under test is the
# tail the harness appends, not the whole invocation.
_PARSE_PROGRAM = (
    "import json, sys\n"
    "\n"
    "\n"
    "def _emit(status, message='', unknown=()):\n"
    "    print(json.dumps({'status': status, 'message': str(message)[-4000:], 'unknown': list(unknown)}))\n"
    "    raise SystemExit(0)\n"
    "\n"
    "\n"
    "class _Rejected(Exception):\n"
    "    pass\n"
    "\n"
    "\n"
    "def _reject(message, *_a, **_k):\n"
    "    raise _Rejected(str(message))\n"
    "\n"
    "\n"
    "argv = json.loads(sys.argv[1])\n"
    "try:\n"
    "    parser = _build_parser()\n"
    "except BaseException as exc:\n"
    "    _emit('unavailable', '%s: %s' % (type(exc).__name__, exc))\n"
    "for action in getattr(parser, '_actions', ()):\n"
    "    action.required = False\n"
    "for group in getattr(parser, '_mutually_exclusive_groups', ()):\n"
    "    group.required = False\n"
    "parser.error = _reject\n"
    "parser.exit = _reject\n"
    "try:\n"
    "    _ns, unknown = parser.parse_known_args(argv)\n"
    "except _Rejected as exc:\n"
    "    _emit('invalid', exc)\n"
    "except BaseException as exc:\n"
    "    _emit('unavailable', '%s: %s' % (type(exc).__name__, exc))\n"
    "if unknown:\n"
    "    _emit('invalid', 'unrecognized arguments: ' + ' '.join(unknown), unknown)\n"
    "_emit('ok')\n"
)


@dataclass(frozen=True)
class ArgvVerdict:
    """What the installed parser said about one server argv.

    Attributes:
        status: :data:`OK`, :data:`INVALID` or :data:`UNAVAILABLE`.
        reason: Which of this module's named outcomes was reached.
        detail: Parser or probe output, clipped.
        argv: The argv that should now launch -- the input, or the repaired one
            when a drop repair applied.
        text: ``argv`` as the argument string to write back.
        dropped: Flags removed by the one allowed repair.
        repaired_digest: The original argv's digest when a repair applied, so
            the caller can record that this argv has spent its one repair.
        probe_interpreter: The interpreter the parser ran in.
        serving_interpreter: The interpreter the server will run in.
    """

    status: str
    reason: str
    detail: str = ""
    argv: tuple[str, ...] = ()
    text: str = ""
    dropped: tuple[str, ...] = ()
    repaired_digest: str = ""
    probe_interpreter: str = ""
    serving_interpreter: str = ""

    @property
    def terminal(self) -> bool:
        """bool: True when this verdict must stop the round rather than launch."""
        return self.status == INVALID


@dataclass
class Probe:
    """One decoded probe result: the JSON the program printed, or why not.

    Attributes:
        payload: The decoded object, or ``None`` when nothing decoded.
        detail: Probe output or the reason there is no payload.
        ok: Whether a JSON object was decoded.
    """

    payload: Mapping[str, object] | None = None
    detail: str = ""
    ok: bool = False


def _default_probe(argv: list[str], env: Mapping[str, str], timeout_sec: float) -> "subprocess.CompletedProcess[str]":
    """Run one probe subprocess, captured and time-bounded."""
    return subprocess.run(  # nosec B603 - argv[0] is an interpreter this module resolved
        argv,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def _clip(text: str) -> str:
    """Return the informative tail of probe output, bounded for the record."""
    return text.strip()[-_DETAIL_CHARS:]


def run_probe_json(
    interpreter: str,
    program: str,
    args: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_sec: float,
    probe: ProbeFn | None = None,
) -> Probe:
    """Run a probe program and decode the single JSON object it prints.

    Args:
        interpreter: Interpreter to run the program with.
        program: The program source.
        args: Positional arguments appended after ``-c <program>``.
        env: Environment the probe runs in -- the launch env, so it resolves the
            same packages the server will.
        timeout_sec: Hard bound on the probe.
        probe: The subprocess shim; the real one when omitted.

    Returns:
        Probe: The decoded payload, or the reason there is none.
    """
    probe = probe if probe is not None else _default_probe
    if not interpreter:
        return Probe(detail="no interpreter to probe with")
    try:
        completed = probe([interpreter, "-c", program, *args], env, timeout_sec)
    except subprocess.TimeoutExpired:
        return Probe(detail=f"probe timed out after {timeout_sec:g}s")
    except (OSError, ValueError) as exc:
        return Probe(detail=f"{type(exc).__name__}: {exc}")
    stdout = completed.stdout
    stderr = completed.stderr
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except ValueError:
            break
        if isinstance(decoded, dict):
            return Probe(payload=decoded, detail=_clip(stderr), ok=True)
        break
    return Probe(detail=_clip(stderr or stdout) or f"exit={completed.returncode}")


def resolve_serving_interpreter(framework: str, launch_env: Mapping[str, str]) -> str:
    """Return the interpreter that will run the server, from the launch env.

    Read from the environment the launch is handed rather than the ambient
    process env, because that is what the server's own resolution reads.

    Args:
        framework: Framework the config serves.
        launch_env: The environment the benchmark subprocess is launched with.

    Returns:
        str: The interpreter path, or ``""`` when none could be resolved.
    """
    pinned = str(launch_env.get(SERVING_PYTHON_ENV, "")).strip()
    if pinned:
        return pinned
    path = str(launch_env.get("PATH", "")).strip() or None
    if framework.strip().lower() == "vllm":
        # With no pin, vLLM is launched through its console script, so the
        # interpreter that serves is the one that script's venv holds.
        console = shutil.which("vllm", path=path)
        if console:
            sibling = os.path.join(os.path.dirname(console), "python")
            if os.access(sibling, os.X_OK):
                return sibling
    return shutil.which("python3", path=path) or ""


def _resolve_probe_interpreter(framework: str) -> str:
    """Return the interpreter the capability probes use for ``framework``."""
    # Deferred: the resolution ladder lives beside the grid runner, whose
    # package imports the executors this module's callers live in.
    from hyperloom.orchestrator.actions.executors._grid_runner import _resolve_probe_python

    return _resolve_probe_python(framework)


def _identity(
    interpreter: str,
    framework: str,
    *,
    env: Mapping[str, str],
    probe: ProbeFn,
) -> tuple[Mapping[str, object] | None, str]:
    """Return ``interpreter``'s identity for ``framework``, or why it is unknown."""
    result = run_probe_json(
        interpreter,
        _IDENTITY_PROGRAM,
        [framework],
        env=env,
        timeout_sec=_IDENTITY_TIMEOUT_SEC,
        probe=probe,
    )
    if not result.ok or result.payload is None:
        return None, result.detail or "identity probe produced no answer"
    if not str(result.payload.get("origin", "")):
        return None, f"{framework} is not importable there"
    return result.payload, ""


def _identity_field(identity: Mapping[str, object], key: str) -> str:
    """Return one identity field, reading an absent key and a JSON null alike."""
    value = identity.get(key)
    return "" if value is None else str(value)


def _same_install(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """True when two identities name the same interpreter and framework build."""
    return all(
        _identity_field(left, key) == _identity_field(right, key) for key in ("executable", "python", "origin", "dist")
    )


def _offending_flags(unknown: Sequence[object]) -> tuple[str, ...]:
    """Return the flag names among a parser's leftover tokens, de-duplicated.

    A leftover that is not a flag is the value of one that is, and dropping a
    flag takes its value with it.
    """
    out: list[str] = []
    for token in unknown:
        text = str(token).strip()
        if not text.startswith("-"):
            continue
        name = text.split("=", 1)[0]
        if name not in out:
            out.append(name)
    return tuple(out)


def _parse(
    framework: str,
    parser_source: str,
    argv: Sequence[str],
    *,
    interpreter: str,
    env: Mapping[str, str],
    probe: ProbeFn,
) -> tuple[str, str, tuple[str, ...]]:
    """Run the framework's parser over ``argv``.

    Returns:
        tuple: ``(status, detail, offending_flags)`` where status is one of
        :data:`OK`, :data:`INVALID`, :data:`UNAVAILABLE`.
    """
    result = run_probe_json(
        interpreter,
        parser_source + "\n" + _PARSE_PROGRAM,
        [json.dumps(list(argv))],
        env=env,
        timeout_sec=_PARSE_TIMEOUT_SEC,
        probe=probe,
    )
    if not result.ok or result.payload is None:
        return UNAVAILABLE, result.detail or "parser probe produced no answer", ()
    payload = result.payload
    status = str(payload.get("status", ""))
    detail = _clip(str(payload.get("message", "")))
    unknown = payload.get("unknown")
    flags = _offending_flags(unknown if isinstance(unknown, list) else ())
    if status not in (OK, INVALID, UNAVAILABLE):
        return UNAVAILABLE, detail or f"parser probe reported {status!r}", ()
    return status, detail, flags


def check_server_argv(
    *,
    framework: str,
    argv: Sequence[str],
    text: str,
    launch_env: Mapping[str, str],
    repaired: Container[str] = (),
    digest: str = "",
    probe: ProbeFn | None = None,
) -> ArgvVerdict:
    """Decide whether the installed framework will accept ``argv``.

    The probe and serving interpreters are proven to be one -- same interpreter,
    same Python, same framework origin, same distribution version -- before any
    verdict is claimed; anything short of that is :data:`UNAVAILABLE`.

    One repair is allowed per distinct argv and it only ever removes: an
    unrecognised flag is dropped and the parser asked again. A value the parser
    rejected is never rewritten.

    Args:
        framework: Framework the config serves.
        argv: The sealed argv the server would receive.
        text: That argv as the argument string, for the drop repair.
        launch_env: The environment the benchmark subprocess is launched with.
        repaired: Digests of argvs that have already used their one repair.
        digest: This argv's digest, the key recorded against ``repaired``.
        probe: Subprocess shim, injected by tests.

    Returns:
        ArgvVerdict: The verdict, and the argv that should now launch.
    """
    from hyperloom.orchestrator.framework.adapters import get_adapter

    probe = probe if probe is not None else _default_probe
    name = framework.strip().lower()
    parser_source = get_adapter(name).argv_parser_source()
    if not parser_source:
        return ArgvVerdict(
            status=UNAVAILABLE,
            reason=NO_PARSER,
            detail=f"no adapter reaches {name or 'this framework'}'s parser",
            argv=tuple(argv),
            text=text,
        )

    serving = resolve_serving_interpreter(name, launch_env)
    probe_interpreter = _resolve_probe_interpreter(name)

    def _unavailable(reason: str, detail: str) -> ArgvVerdict:
        return ArgvVerdict(
            status=UNAVAILABLE,
            reason=reason,
            detail=detail,
            argv=tuple(argv),
            text=text,
            probe_interpreter=probe_interpreter,
            serving_interpreter=serving,
        )

    if not serving or not probe_interpreter:
        return _unavailable(INTERPRETER_UNPROVEN, "could not resolve both the probe and the serving interpreter")
    serving_id, serving_why = _identity(serving, name, env=launch_env, probe=probe)
    if serving_id is None:
        return _unavailable(INTERPRETER_UNPROVEN, f"serving interpreter: {serving_why}")
    probe_id, probe_why = _identity(probe_interpreter, name, env=launch_env, probe=probe)
    if probe_id is None:
        return _unavailable(INTERPRETER_UNPROVEN, f"probe interpreter: {probe_why}")
    if not _same_install(probe_id, serving_id):
        return _unavailable(
            INTERPRETER_MISMATCH,
            f"probe {json.dumps(dict(probe_id), sort_keys=True)} != serving {json.dumps(dict(serving_id), sort_keys=True)}",
        )

    status, detail, flags = _parse(
        name,
        parser_source,
        argv,
        interpreter=probe_interpreter,
        env=launch_env,
        probe=probe,
    )
    if status == UNAVAILABLE:
        return _unavailable(PROBE_FAILED, detail)
    if status == OK:
        return ArgvVerdict(
            status=OK,
            reason=PARSED,
            argv=tuple(argv),
            text=text,
            probe_interpreter=probe_interpreter,
            serving_interpreter=serving,
        )

    def _invalid(reason: str, why: str, *, dropped: tuple[str, ...] = ()) -> ArgvVerdict:
        return ArgvVerdict(
            status=INVALID,
            reason=reason,
            detail=why,
            argv=tuple(argv),
            text=text,
            dropped=dropped,
            probe_interpreter=probe_interpreter,
            serving_interpreter=serving,
        )

    if not flags:
        # The parser rejected a value, not a name. Choosing a replacement would
        # be the harness guessing what the framework meant.
        return _invalid(VALUE_REJECTED, detail)
    if not digest or digest in repaired:
        # No repair without a key to record it under: an unrecordable repair is
        # one the composer can trigger again every round, which is a loop, not
        # a repair.
        return _invalid(REPAIR_SPENT, detail, dropped=flags)

    from hyperloom.orchestrator.actions.executors._grid_server_args import (
        remove_server_args,
        tokenize_server_args_preserving_json,
    )

    split = tokenize_server_args_preserving_json(remove_server_args(text, list(flags)))
    if split is None:
        return _invalid(REPAIR_FAILED, f"{detail}\ndropping {' '.join(flags)} left an argv that cannot be tokenised")
    repaired_text, tokens = split
    retry_status, retry_detail, _retry_flags = _parse(
        name,
        parser_source,
        tokens,
        interpreter=probe_interpreter,
        env=launch_env,
        probe=probe,
    )
    if retry_status == UNAVAILABLE:
        return _unavailable(PROBE_FAILED, retry_detail)
    if retry_status != OK:
        return _invalid(REPAIR_FAILED, f"{detail}\nafter dropping {' '.join(flags)}: {retry_detail}", dropped=flags)
    log.warning(
        "argv preflight: %s rejected %s; dropped and re-validated once",
        name or "the framework",
        " ".join(flags),
    )
    return ArgvVerdict(
        status=OK,
        reason=PARSED_AFTER_DROP,
        detail=detail,
        argv=tuple(tokens),
        text=repaired_text,
        dropped=flags,
        repaired_digest=digest,
        probe_interpreter=probe_interpreter,
        serving_interpreter=serving,
    )


def argv_invalid_observation(verdict: ArgvVerdict, *, session_dir: Path | None = None) -> BootObservation:
    """Return the boot observation for an argv the installed parser refused.

    Recorded at :data:`~hyperloom.common.bringup.LadderStage.ARGV_PARSE` under
    this module's producer: no process was started, so there is no server log.

    Args:
        verdict: The refusing verdict.
        session_dir: Session root, redacted out of the recorded text.

    Returns:
        BootObservation: The observation to persist.
    """
    roots = (str(session_dir),) if session_dir else ()
    body = "\n".join(part for part in (f"server argv rejected ({verdict.reason})", verdict.detail) if part)
    excerpt = Excerpt(
        text=redact(body, roots=roots),
        stream=STREAM,
        byte_start=0,
        byte_end=len(body.encode("utf-8", "replace")),
    )
    return BootObservation(
        producer=PRODUCER,
        stage_reached=LadderStage.ARGV_PARSE,
        stage_failed=LadderStage.ARGV_PARSE,
        matched_marker=ARGV_INVALID,
        excerpt=excerpt,
        evidence_ref=STREAM,
    )


def is_argv_invalid(observation: BootObservation | None) -> bool:
    """True when ``observation`` is this module's argv refusal.

    An argv the framework never accepted is not a defect in framework source,
    which is what the enablement path repairs.

    Args:
        observation: A loaded observation, or ``None``.

    Returns:
        bool: Whether the observation names a refused argv.
    """
    return observation is not None and observation.producer == PRODUCER and observation.matched_marker == ARGV_INVALID


__all__ = [
    "ARGV_INVALID",
    "INVALID",
    "OK",
    "PRODUCER",
    "SERVING_PYTHON_ENV",
    "UNAVAILABLE",
    "ArgvVerdict",
    "Probe",
    "argv_invalid_observation",
    "check_server_argv",
    "is_argv_invalid",
    "resolve_serving_interpreter",
    "run_probe_json",
]
