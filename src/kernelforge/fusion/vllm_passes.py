# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read which vLLM compile-fusion passes the TARGET install actually has switched on.

forge-fuse used to read "vLLM ships a compile pass for this chain" as "vLLM
already fuses it" and dropped the candidate as a no-op. That inference is wrong:
most ``PassConfig`` fusion flags are only resolved at runtime, so a pass can EXIST
while being disabled -- the fusion never runs and nobody turns it on. Enabling
vLLM's own QK-norm+RoPE pass on dense Qwen3 measured several percent of decode
throughput that was being left on the table.

Nothing about on/off is hardcoded here: the state is version-, platform- and
optimization-level dependent, so it is read out of the target vLLM (see
``_PROBE_SRC`` for the precedence, which mirrors how a real run resolves a flag).
Reading it in a subprocess is deliberate -- importing vLLM is heavy and its
platform init can abort, and the framework under test may live under a different
interpreter than the one running forge-fuse.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

log = logging.getLogger("kernelforge.fusion.vllm_passes")


def _within(path: str, root: str) -> bool:
    """Whether ``path`` resolves inside ``root`` (both may be symlinked)."""
    if not path or not root:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


_VLLM_PASS_PROBE_MARKER = "FORGE_VLLM_PASS_PROBE "

# Resolves each flag the way a real run does, which is NOT the annotation default:
# vLLM's optimization level (default -O2) owns most fusion flags, so a bare
# PassConfig would report every level-owned flag as off and invent opportunities
# that the runtime already takes. Precedence, all read out of the target vLLM:
#   1. the default optimization level's pass_config, when it pins a literal bool;
#   2. UNKNOWN when that entry is a predicate -- it is resolved from the full
#      VllmConfig (AITER on? model quantized? hidden size?), which cannot be
#      answered here, and guessing would mean proposing no-op work;
#   3. otherwise the resolved PassConfig attribute (flags no level owns, which is
#      where enable_qk_norm_rope_fusion lives).
# Every requested flag shares one import: that import is the whole cost, so probing
# per flag would re-pay it for every matched pattern.
_PROBE_SRC = """
import inspect, json, sys

flags = sys.argv[1:]
out = {"config_file": "", "package_root": "", "error": "", "level": "",
       "level_api": "", "flags": {}}
try:
    import vllm
    from vllm.config.compilation import PassConfig
    out["config_file"] = inspect.getsourcefile(PassConfig) or ""
    pkg = getattr(vllm, "__file__", "") or ""
    out["package_root"] = __import__("os").path.dirname(__import__("os").path.dirname(pkg))
    cfg = PassConfig()
    level_pass = {}
    try:
        from vllm.config.vllm import OPTIMIZATION_LEVEL_TO_CONFIG, VllmConfig
    except ImportError as exc:
        # CONFIRMED absent (other vLLM version): the PassConfig default is then the
        # whole story, so falling back to it is sound -- not an unknown.
        out["level_api"] = "absent: %s" % exc
    else:
        try:
            import dataclasses
            level = None
            for field in dataclasses.fields(VllmConfig):
                if field.name == "optimization_level":
                    level = field.default
                    break
            if level is None:
                level = getattr(VllmConfig, "optimization_level", None)
            entry = OPTIMIZATION_LEVEL_TO_CONFIG.get(level) or {}
            level_pass = (entry.get("compilation_config") or {}).get("pass_config") or {}
            out["level"] = str(level)
            out["level_api"] = "ok"
        except Exception as exc:
            # The API exists but did not read as expected (shape change, init
            # failure): NOT a benign fallback -- report so every verdict is unknown.
            out["error"] = "optimization level unreadable: %s: %s" % (type(exc).__name__, exc)
    for flag in flags:
        if flag in level_pass:
            value = level_pass[flag]
            if isinstance(value, bool):
                item = {"present": True, "enabled": value, "source": "level"}
            else:
                item = {"present": True, "enabled": None, "source": "level-dynamic"}
        elif hasattr(cfg, flag):
            item = {"present": True, "enabled": bool(getattr(cfg, flag)), "source": "default"}
        else:
            item = {"present": False, "enabled": None, "source": "absent"}
        out["flags"][flag] = item
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print("MARKER" + json.dumps(out))
""".replace("MARKER", _VLLM_PASS_PROBE_MARKER)

# One vLLM import, not an unbounded wait: a hung import must not stall the run.
DEFAULT_PROBE_TIMEOUT_S = 120


@dataclass(frozen=True)
class PassState:
    """Whether a vLLM compile-fusion pass exists in the target install and is on.

    ``enabled is None`` means the state could not be determined (no importable
    vLLM, probe failure, or a level-resolved predicate); it is NOT a guess of
    "off". ``present is False`` means the target has no such flag at all, which is
    different again: there is no framework implementation to claim.
    """

    flag: str
    present: bool = False
    enabled: Optional[bool] = None
    config_file: str = ""
    error: str = ""
    # Where the verdict came from: "level" (optimization level pins a literal),
    # "level-dynamic" (level resolves it from the full VllmConfig -> unknown),
    # "default" (no level owns it, so the PassConfig default stands), "absent".
    source: str = ""
    package_root: str = ""

    @property
    def missed(self) -> bool:
        """Framework implements this fusion but ships it switched OFF.

        An error voids the verdict: a probe that failed halfway can report a
        ``None`` attribute as ``False``, and acting on that would claim a pass
        that is really enabled.
        """
        return self.present and self.enabled is False and bool(self.config_file) and not self.error

    @property
    def claimable(self) -> bool:
        """Missed AND actually flippable by editing the ``PassConfig`` default.

        Only flags no optimization level owns qualify. A level that pins the flag
        (``source="level"``) overrides the class default at runtime, so flipping
        that default changes nothing -- it would export a patch with no behavioural
        effect. Those are left alone rather than fought: upstream pins
        ``fuse_attn_quant`` off through ``IS_QUANTIZED = False`` deliberately (see
        vllm-project/vllm#25689), and forcing it on would drive a path upstream
        has disabled on purpose.
        """
        return self.missed and self.source == "default"

    @property
    def undecidable(self) -> bool:
        """Present in the target but its state could not be established."""
        return self.present and self.enabled is None


@dataclass(frozen=True)
class TargetRuntime:
    """The ONE vLLM install a run probes, edits and serves.

    These three used to be resolved independently -- the probe imported vLLM under
    ``sys.executable``, serving booted whatever ``vllm`` was first on ``PATH``, and
    ``--framework-root`` only steered model-source lookup -- so a run could read
    state from install A, edit A, and then validate with launcher C. The
    interpreter is therefore derived FROM the serving launcher, which makes probe
    and serving the same install by construction instead of by coincidence, and
    ``require_root`` makes an explicitly requested framework root a hard
    precondition rather than a hint.
    """

    framework: str = ""
    python: str = ""
    launcher_exe: str = ""
    require_root: str = ""
    error: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        """Cache identity: two different installs must never share probe state."""
        return (self.python, self.launcher_exe, self.require_root)


def _launcher_interpreter(launcher_exe: str) -> str:
    """Interpreter a console-script launcher runs under, from its shebang."""
    if not launcher_exe:
        return ""
    try:
        with open(launcher_exe, "rb") as fh:
            first = fh.readline(512).decode("utf-8", "replace").strip()
    except OSError:
        return ""
    if not first.startswith("#!"):
        return ""  # binary/compiled launcher: cannot attribute an interpreter
    parts = first[2:].strip().split()
    if not parts:
        return ""
    # "#!/usr/bin/env python3" -> the interpreter is the argument.
    exe = parts[1] if parts[0].endswith("env") and len(parts) > 1 else parts[0]
    return exe if Path(exe).exists() else ""


def resolve_target_runtime(framework: str, *, framework_root: str = "", launcher_exe: str = "") -> TargetRuntime:
    """Pin the install that will be probed, edited and served.

    ``error`` is set (and the caller must not edit anything) when the launcher
    cannot be located or attributed to an interpreter -- guessing would risk
    editing an install other than the one under test.
    """
    exe = launcher_exe or shutil.which("vllm") or ""
    if not exe:
        return TargetRuntime(framework=framework, require_root=framework_root, error="no vllm launcher on PATH")
    python = _launcher_interpreter(exe)
    if not python:
        return TargetRuntime(
            framework=framework,
            launcher_exe=exe,
            require_root=framework_root,
            error=f"cannot determine the interpreter behind {exe}",
        )
    return TargetRuntime(framework=framework, python=python, launcher_exe=exe, require_root=framework_root)


def _marker_payload(stdout: str) -> Optional[dict]:
    """Last marker-tagged JSON object in ``stdout`` (vLLM prints banners around it)."""
    for line in reversed((stdout or "").splitlines()):
        idx = line.find(_VLLM_PASS_PROBE_MARKER)
        if idx < 0:
            continue
        try:
            payload = json.loads(line[idx + len(_VLLM_PASS_PROBE_MARKER) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


@lru_cache(maxsize=8)
def probe_pass_states(
    flags: tuple[str, ...],
    *,
    python: str = "",
    require_root: str = "",
    timeout_s: int = DEFAULT_PROBE_TIMEOUT_S,
) -> Mapping[str, PassState]:
    """Resolved state of every requested ``PassConfig`` flag, in ONE subprocess.

    Batched on purpose: the cost here is importing vLLM, so all flags share a
    single import (and a single timeout) instead of paying it per flag. Never
    raises -- any failure yields ``enabled=None`` (unknown) for every flag so
    callers stay conservative instead of acting on a guessed default.

    ``python`` and ``require_root`` are part of the cache key: probe state from one
    install must never be reused for another. When ``require_root`` is set, an
    install whose config file lives outside it is a hard failure (all-unknown) --
    the run was told which framework to target, so silently probing and editing a
    different one is worse than stopping.
    """
    wanted = tuple(dict.fromkeys(f for f in flags if f))
    if not wanted:
        return MappingProxyType({})

    def _all(**kw) -> Mapping[str, PassState]:
        return MappingProxyType({f: PassState(flag=f, **kw) for f in wanted})

    try:
        proc = subprocess.run(
            [python or sys.executable, "-c", _PROBE_SRC, *wanted],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _all(error=f"{type(exc).__name__}: {exc}")
    payload = _marker_payload(proc.stdout)
    if payload is None:
        why = (proc.stderr or proc.stdout or "no probe output").strip()[-300:]
        log.debug("vLLM pass probe produced no verdict for %s: %s", ", ".join(wanted), why)
        return _all(error=why)
    config_file = str(payload.get("config_file") or "")
    package_root = str(payload.get("package_root") or "")
    error = str(payload.get("error") or "")
    if require_root and not _within(config_file, require_root):
        return _all(
            error=(
                f"probed vLLM config {config_file or '<unknown>'} is outside the "
                f"requested framework root {require_root}"
            ),
            config_file=config_file,
            package_root=package_root,
        )
    values = payload.get("flags")
    values = values if isinstance(values, dict) else {}
    states = {}
    for flag in wanted:
        item = values.get(flag)
        item = item if isinstance(item, dict) else {}
        enabled = item.get("enabled")
        # A flag absent from the target install is "not present", NOT "disabled":
        # there is nothing to enable and nothing to claim.
        states[flag] = PassState(
            flag=flag,
            present=bool(item.get("present")),
            enabled=enabled if isinstance(enabled, bool) else None,
            config_file=config_file,
            error=error,
            source=str(item.get("source") or ""),
            package_root=package_root,
        )
        log.debug(
            "vLLM pass %s: present=%s enabled=%s source=%s file=%s error=%s",
            flag,
            states[flag].present,
            states[flag].enabled,
            states[flag].source,
            config_file,
            error,
        )
    return MappingProxyType(states)


def probe_pass_state(
    flag: str,
    *,
    python: str = "",
    require_root: str = "",
    timeout_s: int = DEFAULT_PROBE_TIMEOUT_S,
) -> PassState:
    """Resolved state of a single ``PassConfig`` flag (see :func:`probe_pass_states`)."""
    if not flag:
        return PassState(flag=flag, error="no config flag")
    got = probe_pass_states((flag,), python=python, require_root=require_root, timeout_s=timeout_s).get(flag)
    return got if got is not None else PassState(flag=flag, error="no probe result")


def verify_pass_enabled(
    flag: str,
    *,
    python: str = "",
    require_root: str = "",
    timeout_s: int = DEFAULT_PROBE_TIMEOUT_S,
) -> PassState:
    """Re-read a flag AFTER editing, bypassing the cache.

    Editing the ``PassConfig`` default only takes effect for flags nothing else
    overrides, so the edit must be confirmed against the target rather than
    assumed: an unconfirmed flip would export a patch with no behavioural effect.
    """
    probe_pass_states.cache_clear()
    return probe_pass_state(flag, python=python, require_root=require_root, timeout_s=timeout_s)


def _disabled_default_re(flag: str) -> re.Pattern[str]:
    return re.compile(
        rf"^(?P<pre>[ \t]*{re.escape(flag)}[ \t]*:[ \t]*bool[ \t]*=[ \t]*)(?P<val>None|False)(?P<post>.*)$",
        re.MULTILINE,
    )


def enable_pass_in_source(config_file: str, flag: str) -> bool:
    """Flip ``flag``'s disabled default to ``True`` in vLLM's ``PassConfig`` source.

    Deterministic (no LLM) and idempotent: returns False when there is no disabled
    default to flip -- already ``True``, flag absent, or file unreadable -- so a
    re-run never rewrites the file. Only the requested field's line is touched.
    """
    if not config_file or not flag:
        return False
    path = Path(config_file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("cannot read vLLM pass config %s: %s", config_file, exc)
        return False
    new_text, count = _disabled_default_re(flag).subn(lambda m: f"{m.group('pre')}True{m.group('post')}", text, count=1)
    if not count:
        return False
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        log.warning("cannot enable %s in %s: %s", flag, config_file, exc)
        return False
    log.info("enabled vLLM compile pass flag %s in %s", flag, config_file)
    return True
