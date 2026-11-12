"""The host's base launch, checked at Arbor's start (component mode).

A host such as Hyperloom's MB resolves one base launch per session and freezes it: the record
``hyperloom-base-launch/1`` holds the server's entrypoint, flags and environment, the golden
acceptance length and the gate's client. The host's
gate measures on that base, and Arbor must start from the same one: a hand-made launch that
differed from the recipe in one flag, or in one environment variable, made an Arbor's KEEPs fail
at the gate, or measured them on another base without a trace.

``component.base_launch`` names the record (``{"path", "digest"}``). Before Arbor starts any
server, ``check`` builds the base part of the launch it would run from the config alone, and
compares it with the record:

- **the entrypoint**, token for token (``component.entrypoint`` sets Arbor's, see
  arbor/frameworks/base.py);
- **the flags**, as flags and their values: ``--a=b`` reads as ``--a b``, a JSON value compares as
  JSON, vLLM's spellings of the tensor-parallel size are one flag, and the flags the host or
  Arbor owns (host, port, model path, served name, the record's ``runner_owned``) are left out;
- **the environment**, every variable but the GPU masks, the pack's eval settings (``GSM8K_MIN``,
  ``GSM8K_LIMIT``, which a host puts in ``server_env`` for the correctness run) and the record's
  ``runner_owned``: the golden acceptance variables are base environment like any other. A value
  the record redacts (``{"sha256": <hex>}``) compares by the SHA-256 of Arbor's value.

**The golden layer**: when the record's ``golden_al`` is set, its own ``args`` and ``env``
hold the keys of the golden acceptance (``SGLANG_SIMULATE_ACC_*``, the speculative flags), and the
top-level fields do not. The record's base is then ``args + golden_al.args`` and ``env`` with
``golden_al.env`` over it; Arbor's config holds them all in ``server_flags`` and ``server_env``.

Any difference refuses the start (exit 2, arbor/component.py ``refusals``), naming it, so no GPU
time is spent on a wrong base. ``host_launch.json`` in the session records both sides either way,
secrets masked, for the host to read: the host keeps the parser-aware digest, which needs the
framework's own argument parser, and checks the record against it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

SCHEMA = "hyperloom-base-launch/1"
HOST_LAUNCH_FILE = "host_launch.json"
HOST_LAUNCH_SCHEMA = "arbor-host-launch/1"

# Flags every framework adapter sets from the config itself (arbor/frameworks/): where the
# server listens, which model it serves and under which name. Never part of the base.
OWN_FLAGS = frozenset({"--host", "--port", "--model-path", "--model", "--served-model-name"})
# One flag, several spellings: vLLM's adapter writes --tensor-parallel-size, a recipe may not.
ALIASES = {"-tp": "--tensor-parallel-size", "--tp": "--tensor-parallel-size",
           "--tp-size": "--tensor-parallel-size"}
GPU_MASKS = frozenset({"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"})
# A variable whose name ends in one of these words, or names a secret or an API or access key
# anywhere, holds a secret: its value is never written. Whole segments at the end, so knobs keep
# their values: VLLM_MAX_NUM_BATCHED_TOKENS and SGLANG_SIMULATE_ACC_TOKEN_MODE are knobs, HF_TOKEN
# a secret. Custom headers can carry a gateway key.
_SECRET = re.compile(r"(?:^|_)(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?|HEADERS)$"
                     r"|(?:^|_)(?:SECRET|API_KEY|ACCESS_KEY)(?:_|$)", re.I)


def block(config) -> dict | None:
    """``component.base_launch``, or None when the host gives none."""
    from arbor import component

    value = component.block(config).get("base_launch")
    return value if isinstance(value, dict) else None


def errors(value) -> list[str]:
    """Why ``component.base_launch`` is malformed; [] when it is well formed."""
    if not isinstance(value, dict):
        return ["component.base_launch must be an object: {\"path\": <the record>, \"digest\": <its digest>}"]
    out = []
    if not (isinstance(value.get("path"), str) and value["path"]):
        out.append("component.base_launch.path must name the host's base launch record")
    if "digest" in value and not isinstance(value["digest"], str):
        out.append("component.base_launch.digest must be a string")
    return out


def entrypoint_errors(value) -> list[str]:
    """Why ``component.entrypoint`` is malformed; [] when it is well formed."""
    if isinstance(value, list) and value and all(isinstance(t, str) and t for t in value):
        return []
    return ["component.entrypoint must be a non-empty list of strings: the server's command, "
            "such as [\"python3\", \"-m\", \"sglang.launch_server\"]"]


def load(path) -> dict:
    """The record at ``path``; ValueError when it is not one."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"the base launch record {path} does not read: {exc}") from None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} record (schema {doc.get('schema') if isinstance(doc, dict) else None!r})")
    return doc


def _value(text: str) -> str:
    """A flag's value as compared: a JSON object or array in canonical form, else the text."""
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            return json.dumps(json.loads(stripped), sort_keys=True, separators=(",", ":"))
        except ValueError:
            pass
    return text


def flags(tokens: list[str], skip: frozenset[str] = frozenset()) -> list[tuple[str, tuple[str, ...]]]:
    """``tokens`` as (flag, values) pairs, sorted: ``--a=b`` reads as ``--a b``, the values are the
    tokens up to the next flag, an alias takes its flag's name, and a flag in ``skip`` is left
    out with its values. A value before any flag is a positional, under the flag "". A flag given
    twice with the same values counts once: the server reads the same thing (MB names vLLM's TP in
    the flags, and Arbor's vLLM command gives it too)."""
    out: list[tuple[str, list[str]]] = []
    for token in tokens:
        if token.startswith("-") and len(token) > 1 and not _is_number(token):
            name, eq, value = token.partition("=")
            out.append((ALIASES.get(name, name), [value] if eq else []))
        elif out:
            out[-1][1].append(token)
        else:
            out.append(("", [token]))
    return sorted({(name, tuple(_value(v) for v in values)) for name, values in out if name not in skip})


def split_flags(text: str) -> list[str]:
    """Server flags as words: split on whitespace outside quotes, a JSON value one word with its
    inner quotes, and one pair of outer shell quotes taken off a word (or off the value of
    ``--a='b'``). So ``--x {"a":1}`` and ``--x '{"a": 1}'`` give the same JSON value."""
    try:
        words = shlex.split(text, posix=False)
    except ValueError:
        words = text.split()
    out = []
    for word in words:
        name, eq, value = word.partition("=") if word.startswith("-") else ("", "", word)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out.append(f"{name}={value}" if eq else (value if not name else word))
    return out


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def secret(name: str) -> bool:
    return bool(_SECRET.search(name))


def masked(env: dict) -> dict:
    """``env`` with every secret's value replaced by its digest's first 12 hex digits."""
    return {k: (f"<masked sha256:{hashlib.sha256(str(v).encode()).hexdigest()[:12]}>" if secret(k) else str(v))
            for k, v in sorted(env.items())}


def ours(config) -> dict:
    """The base part of the launch Arbor would run from ``config``: the entrypoint, the flags
    after it, and the environment, before any change of its own (the static config, no KEEP's
    overlay)."""
    from arbor import frameworks

    config = getattr(config, "config", config) or {}
    adapter = frameworks.get(config)
    server_flags = str(config.get("server_flags") or "")
    entry = shlex.split(adapter.entrypoint(config))
    # The framework's part of the command, then the flags it appends last, split as a host writes
    # them: MB's JSON values are compact and unquoted, which a POSIX split would strip of quotes.
    argv = shlex.split(adapter.serve_cmd(config, "")) + split_flags(server_flags)
    rest = argv[len(entry):] if argv[:len(entry)] == entry else argv
    model_path = str(config.get("model_path") or "")
    if rest[:1] == [model_path] and model_path:        # vLLM takes the model as its first positional
        rest = rest[1:]
    env = {str(k): str(v) for k, v in (config.get("server_env") or {}).items()}
    env.update(adapter.extra_env(server_flags))
    return {"entrypoint": entry, "args": rest, "env": env, "model": model_path}


def _base(record: dict) -> tuple[list[str], dict]:
    """The record's base flags and environment: the top level with the golden layer's."""
    args = [str(t) for t in record.get("args") or []]
    env = dict(record.get("env") or {}) if isinstance(record.get("env"), dict) else {}
    golden = record.get("golden_al")
    if isinstance(golden, dict):
        args += [str(t) for t in golden.get("args") or []]
        if isinstance(golden.get("env"), dict):
            env.update(golden["env"])
    return args, env


def golden_keys(record: dict) -> tuple[frozenset[str], frozenset[str]]:
    """The flags and the environment variables the record's golden layer owns (``golden_al``); empty sets when
    there is none."""
    golden = record.get("golden_al") if isinstance(record, dict) else None
    if not isinstance(golden, dict):
        return frozenset(), frozenset()
    names = frozenset(name for name, _ in flags([str(t) for t in golden.get("args") or []]) if name)
    env = frozenset(str(k) for k in golden["env"]) if isinstance(golden.get("env"), dict) else frozenset()
    return names, env


def golden_changes(record: dict, before: dict, after: dict) -> list[str]:
    """Each flag or variable of the record's golden layer that ``after`` (a config: its ``server_flags`` and
    ``server_env``) sets otherwise than ``before``, one line each; [] when the golden acceptance is untouched.

    A host owns those keys: MB refuses a candidate that touches them on import, so a change to them is never Arbor's
    to keep."""
    names, env_names = golden_keys(record)
    if not names and not env_names:
        return []

    def settings(config: dict) -> dict:
        out: dict = {}
        for name, values in flags(split_flags(str(config.get("server_flags") or ""))):
            if name in names:
                out.setdefault(name, set()).add(values)
        return {name: tuple(sorted(v)) for name, v in out.items()}

    def variables(config: dict) -> dict:
        env = config.get("server_env") if isinstance(config.get("server_env"), dict) else {}
        return {k: str(env[k]) for k in env_names if k in env}

    def shown(values) -> str:
        if values is None:
            return "(not set)"
        if isinstance(values, str):
            return values
        return " | ".join(" ".join(v) or "(on)" for v in values)

    out = []
    old, new = settings(before), settings(after)
    for name in sorted(set(old) | set(new)):
        if old.get(name) != new.get(name):
            out.append(f"the golden flag {name}: {shown(old.get(name))} -> {shown(new.get(name))}")
    old_env, new_env = variables(before), variables(after)
    for name in sorted(set(old_env) | set(new_env)):
        if old_env.get(name) != new_env.get(name):
            out.append(f"the golden variable {name}: {shown(old_env.get(name))} -> {shown(new_env.get(name))}")
    return out


def _redacted(value) -> str | None:
    """The hex digest a record gives for a redacted value (``{"sha256": ...}``); None for a plain value."""
    if isinstance(value, dict) and isinstance(value.get("sha256"), str):
        return value["sha256"].lower().removeprefix("sha256:")
    return None


def _same_value(mine: str, theirs) -> bool:
    digest = _redacted(theirs)
    if digest is not None:
        return hashlib.sha256(mine.encode()).hexdigest() == digest
    return mine == str(theirs)


def _shown_value(name: str, value) -> str:
    digest = _redacted(value)
    if digest is not None:
        return f"<redacted sha256:{digest[:12]}>"
    return masked({name: value})[name]


def compare(launch: dict, record: dict) -> list[str]:
    """Every difference between Arbor's base ``launch`` (``ours``) and the host's ``record``,
    one line each; [] when they are the same launch."""
    from arbor.measure.accuracy import EVAL_KEYS

    owned = record.get("runner_owned") or {}
    skip_flags = OWN_FLAGS | {ALIASES.get(f, f) for f in owned.get("args") or [] if isinstance(f, str)}
    # The pack's eval size and floor ride in server_env (a host sets GSM8K_MIN there), but they are
    # the correctness run's settings, not the server's launch.
    skip_env = GPU_MASKS | set(EVAL_KEYS) | {e for e in owned.get("env") or [] if isinstance(e, str)}
    out = []
    want_entry = record.get("entrypoint")
    if isinstance(want_entry, list) and want_entry != launch["entrypoint"]:
        out.append(f"the entrypoint is {shlex.join(launch['entrypoint'])!r}, the host's is "
                   f"{shlex.join(str(t) for t in want_entry)!r} (component.entrypoint sets Arbor's)")
    # The model, given as a positional (vLLM's `vllm serve <model>`), is the host's to place.
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    models = {str(m) for m in (launch.get("model"), model.get("path"), model.get("id")) if m}
    mine = [f for f in flags(launch["args"], skip_flags) if not (f[0] == "" and set(f[1]) <= models)]
    record_args, record_env = _base(record)
    theirs = [f for f in flags(record_args, skip_flags) if not (f[0] == "" and set(f[1]) <= models)]
    for name, values in _only(mine, theirs):
        out.append(f"the flag {_shown(name, values)} is in Arbor's launch, not in the host's")
    for name, values in _only(theirs, mine):
        out.append(f"the flag {_shown(name, values)} is in the host's launch, not in Arbor's")
    mine_env = {k: v for k, v in launch["env"].items() if k not in skip_env}
    theirs_env = {str(k): v for k, v in record_env.items() if k not in skip_env}
    for k in sorted(set(mine_env) | set(theirs_env)):
        if k not in theirs_env:
            out.append(f"the variable {k}={masked({k: mine_env[k]})[k]} is in Arbor's environment, not in the host's")
        elif k not in mine_env:
            out.append(f"the variable {k}={_shown_value(k, theirs_env[k])} is in the host's environment, not in Arbor's")
        elif not _same_value(mine_env[k], theirs_env[k]):
            out.append(f"the variable {k} is {masked({k: mine_env[k]})[k]} in Arbor's environment and "
                       f"{_shown_value(k, theirs_env[k])} in the host's")
    return out


def _only(a: list, b: list) -> list:
    """The items of ``a`` that ``b`` does not have."""
    rest = list(b)
    out = []
    for item in a:
        if item in rest:
            rest.remove(item)
        else:
            out.append(item)
    return out


def _shown(name: str, values: tuple[str, ...]) -> str:
    return " ".join(([name] if name else []) + list(values)) or name


def check(config) -> tuple[dict | None, list[str]]:
    """``host_launch.json``'s document and the refusals: (None, []) when the config names no
    record. A record that does not read, or whose digest is not the one the host gave, refuses."""
    given = block(config)
    if given is None:
        return None, []
    launch = ours(config)
    doc = {"schema": HOST_LAUNCH_SCHEMA, "base_launch": dict(given),
           "entrypoint": launch["entrypoint"], "args": launch["args"], "env": masked(launch["env"]),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        record = load(given.get("path", ""))
    except ValueError as exc:
        doc.update(same=False, differences=[str(exc)])
        return doc, [str(exc)]
    out = []
    if given.get("digest") and record.get("digest") != given["digest"]:
        out.append(f"the base launch record {given['path']} has digest {record.get('digest')!r}, not "
                   f"{given['digest']!r}, the one the host gave: the record changed after the host froze it")
    out += compare(launch, record)
    doc.update(same=not out, differences=out, record_digest=record.get("digest"))
    return doc, [f"the launch differs from the host's base launch: {d}" for d in out]


def write(config, session_dir) -> dict | None:
    """``host_launch.json`` in ``session_dir`` (``check``'s document), or None when the config
    names no record. The host reads it whether the start was refused or not."""
    doc, _ = check(config)
    if doc is None:
        return None
    path = Path(session_dir) / HOST_LAUNCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc
