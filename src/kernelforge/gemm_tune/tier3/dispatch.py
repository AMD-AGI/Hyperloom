# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turning a proposed candidate into something the referee can time."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..evidence import MOE_KEY_FIELDS, MOE_TABLE

log = logging.getLogger(__name__)

# Back-to-back invocations inside one captured graph.
GRAPH_INNER = 20

# Fresh-input correctness repeats, worst result counted.
CORRECTNESS_TRIALS = 8

# Bound the aggregate referee error metric, not an element-wise relative ratio.
MAX_RELATIVE_ERROR = 5e-2

#: MoE uses mean error because atomic reduction makes peak error noisy. On
#: MI355X, self-noise measured 0.00021 and real failures at least 0.1445.
MAX_MOE_MEAN_ERROR = 1e-2

#: Aiter logs a row before checking this prefix; require it to ensure the tuned
#: path ran instead of timing the heuristic fallback.
FLYDSL_KERNEL_PREFIX = "flydsl_"

#: MoE trials cost two calls and vary only activations because its weights are deterministic.
MOE_CORRECTNESS_TRIALS = 4

# Tables this module knows how to exercise. Everything else is honestly absent
# rather than approximated.
SUPPORTED_TABLES = ("bf16_tuned_gemm.csv", MOE_TABLE)

#: The single source for both mandate vocabulary and dispatch validation.
#: Keeping them together prevents advertised and accepted backends from drifting.
DENSE_BF16_BACKENDS: dict[str, str] = {
    "torch": "the unmodified path, `torch.matmul(a, b.t())`. No config. Propose it only as a control.",
    "hipblaslt": (
        "`solidx=<int>` (required). Must be an index hipBLASLt itself offers for these exact "
        "operands -- `aiter.hipb_findallsols` after `aiter.hipb_create_extension` is the "
        "authoritative list. An invented index is refused, because running one kills the process."
    ),
    "aiter_asm": "`kernelName=<str>` (required), `splitK=<int>` (default 0).",
    "aiter_opus": "`kernelId=<int>` (required), `splitK=<int>` (default 1).",
    "aiter_flydsl": (
        "`tile_m`, `tile_n`, `tile_k`, `split_k` (default 1), `block_m_warps`/`block_n_warps`/"
        "`block_k_warps` (default 1), `stages` (default 4), `async_copy`/`b_to_lds` (default True)."
    ),
}


#: What ``_FusedMoeAdapter._build`` will accept. One backend, because there is
#: one way in: aiter resolves a fused-MoE kernel pair by looking the dispatch
#: key up in a CSV, so a candidate *is* a row of that CSV and nothing else.
FUSED_MOE_BACKENDS: dict[str, str] = {
    "aiter_fmoe": (
        "`kernelName1=<str>` and `kernelName2=<str>` (both required) -- the stage-1 and stage-2 "
        "kernels, spelled exactly as aiter spells them. Optional: `block_m=<int>` (default 32), "
        "`ksplit=<int>` (default 0), `run_1stage`/`xbf16`/`flat` (default 0). Do not invent a "
        "name: enumerate them from the installed library, which keeps the authoritative "
        "registry in `aiter.ops.flydsl.moe_kernels` -- `get_flydsl_stage1_kernels(a, b, out)` "
        "and `get_flydsl_stage2_kernels(a, b, out)` each return `{name: params}` for every "
        "config it can compile. Filter that list against your own shape before proposing "
        "anything: a name encodes its tile as `t<M>x<N>x<K>`, and aiter silently downgrades a "
        "tile that does not divide the dimension it walks -- stage-1 `N` and stage-2 `K` are "
        "both `inter_dim` -- so a name whose tile does not divide runs a different kernel than "
        "the one you asked for. `block_m` is not a free knob either: it is the granularity the "
        "tokens are sorted into before either kernel indexes them, and it has to equal the "
        "`M` tile of *both* names. Measured on gfx950, every mismatch of the three ran without "
        "faulting and returned garbage -- a mean error of 1.1 to 1.4 against the default path -- "
        "and the mismatched pairs were the fastest thing in the search, so a tuner that pins "
        "`block_m` and trusts its clock will rank nonsense first. Both of those are refused here "
        "before any GPU work, so a search that ignores this paragraph will spend its budget "
        "collecting refusals rather than wrong answers. A pair that aiter does not "
        "resolve to is refused rather than "
        "timed: the adapter reads back which kernels aiter actually chose, and a row it ignored "
        "would otherwise be timed as the default path and scored as a tie."
    ),
}


def describe_correctness_rule(table: str) -> dict[str, Any] | None:
    """Return the adapter's referee rule, or ``None`` for the dense default.

    Fused-MoE compares with the untuned path because only quantized,
    pre-shuffled weights exist. The mandate quotes this same rule.
    """
    if table != MOE_TABLE:
        return None
    return {
        "trials": MOE_CORRECTNESS_TRIALS,
        "limit": MAX_MOE_MEAN_ERROR,
        "definition": (
            "mean|got - ref| / mean|ref| over the whole output tensor, where ref is this same "
            "call with an empty tuned-config CSV -- the path production serves today"
        ),
        "note": (
            "There is no fp32 reference to build here, so do not look for one: aiter hands over\n"
            "weights that are already quantized and pre-shuffled, and no unquantized copy exists.\n"
            "The reference is the unmodified path -- run the same key with an empty tuned-config\n"
            "CSV, keep the output, then point `AITER_CONFIG_FMOE` at your candidate and compare.\n"
            "That is the right question for a promotion anyway: would swapping this in change\n"
            "what we serve?\n"
            "\n"
            "The limit is tight on purpose and it is the one the harness applies, verbatim. A\n"
            "candidate above it is discarded whatever it timed. Measured on this hardware, the\n"
            "stage-1 kernels whose names carry `_kw2_fp4` were among the fastest in the registry\n"
            "and scored 0.1441-0.1446 by this metric -- fast and wrong, which is what the screen\n"
            "exists to catch. Vary the activations between trials; a kernel that is wrong\n"
            "intermittently passes a single check roughly at random."
        ),
    }


def describe_candidate_protocol(table: str) -> str:
    """Describe the adapter's candidate format, or return "" without one."""
    if table == MOE_TABLE:
        fields = ", ".join(f"`{f}`" for f in MOE_KEY_FIELDS)
        backends = "\n".join(f"- `{name}` -- {detail}" for name, detail in FUSED_MOE_BACKENDS.items())
        return (
            "Each candidate is an object with a `backend` and a `config`, and the shape it belongs\n"
            "to is the key it is filed under. A fused-MoE shape is not `MxNxK`: it is the whole\n"
            "dispatch key, written as `field=value` pairs joined by `|`, in this order:\n"
            f"{fields}.\n"
            "Values are copied verbatim from the same row of the CSV -- including the spelling of\n"
            "`act_type` (`ActivationType.Swiglu`), the dtypes (`torch.float4_e2m1fn_x2`) and\n"
            "`q_type` (`QuantType.per_1x32`). For example:\n"
            '`"token=512|model_dim=6144|inter_dim=384|expert=128|topk=4|'
            "act_type=ActivationType.Swiglu|dtype=torch.bfloat16|"
            "q_dtype_a=torch.float4_e2m1fn_x2|q_dtype_w=torch.float4_e2m1fn_x2|"
            'q_type=QuantType.per_1x32|use_g1u1=1|doweight_stage1=0"`.\n'
            "`config` is `key=value` pairs joined by `;` (never a comma).\n"
            "\n" + backends + "\n"
            "\n"
            "At least one of `kernelName1` and `kernelName2` must be a FlyDSL kernel (its name\n"
            f"starts with `{FLYDSL_KERNEL_PREFIX}`). aiter only consults a tuned row when one of\n"
            "the two is; otherwise it reads the row, ignores it, and runs its heuristic choice --\n"
            "so a pair without one is refused here rather than timed as a phantom win. The\n"
            "partner name may be a CK, CKTile or Opus kernel.\n"
            "\n"
            "Only the mxfp4 SwiGLU path is re-timable here (`q_type=QuantType.per_1x32`,\n"
            "`q_dtype_w=torch.float4_e2m1fn_x2`, `act_type=ActivationType.Swiglu`); a candidate\n"
            "for any other combination is recorded as not dispatchable, because building the\n"
            "operands for it in the layout its kernels expect is not something this harness knows\n"
            "how to do yet. Report what you find for those anyway -- it just cannot be promoted."
        )
    if table != "bf16_tuned_gemm.csv":
        return ""
    backends = "\n".join(f"- `{name}` -- {detail}" for name, detail in DENSE_BF16_BACKENDS.items())
    return (
        "Each candidate is an object with a `backend` and a `config`, and the shape it belongs\n"
        'to is the key it is filed under, written `MxNxK` -- `"8192x3456x1152"`, matching the\n'
        "M, N and K columns of the same row in the CSV. `config` is `key=value` pairs joined by\n"
        "`;` (never a comma); values that look like integers or `True`/`False` are read as such.\n"
        "\n"
        "These are the backends that can be re-dispatched, and the keys each one reads.\n"
        "A candidate naming anything else, or omitting a required key, is recorded as not\n"
        "dispatchable and cannot win -- so a well-measured candidate described in some other\n"
        "vocabulary is worth exactly nothing here.\n"
        "\n" + backends + "\n"
        "\n"
        "Anything the harness cannot re-dispatch is still worth reporting in the CSV; it just\n"
        "cannot be promoted. If the only axis you find is unreachable through these backends,\n"
        "say so plainly -- that is a real finding about this hardware, not a failure."
    )


def adapters_for(table: str) -> Any | None:
    """Return the dispatch adapter for a table, or None if we have none."""
    if table == "bf16_tuned_gemm.csv":
        return _Bf16DenseAdapter()
    if table == MOE_TABLE:
        return _FusedMoeAdapter()
    log.info(
        "tier3: no dispatch adapter for %s, so a generated tuner for it could not be re-timed; supported today: %s",
        table,
        ", ".join(SUPPORTED_TABLES),
    )
    return None


def parse_config(cfg: Any) -> dict[str, Any]:
    """``a=1;b=True;c=x`` into a dict, recovering ints and bools."""
    out: dict[str, Any] = {}
    for part in str(cfg).split(";"):
        key, sep, value = part.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if value in ("True", "False"):
            out[key] = value == "True"
            continue
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def shape_key(shape: str) -> tuple[int, int, int]:
    """``\"16x1536x7168\"`` into ``(16, 1536, 7168)``."""
    parts = str(shape).split("x")
    if len(parts) != 3:
        raise ValueError(f"tier3 shape must be MxNxK, got {shape!r}")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"tier3 shape must be MxNxK of integers, got {shape!r}") from exc
    return (m, n, k)


def moe_shape_key(shape: str) -> dict[str, str]:
    """Parse a named fused-MoE shape key, raising for missing fields."""
    parts = [p for p in str(shape).split("|") if p.strip()]
    got: dict[str, str] = {}
    for part in parts:
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"tier3 fused-MoE shape must be field=value pairs, got {part!r} in {shape!r}")
        got[name.strip()] = value.strip()
    missing = [f for f in MOE_KEY_FIELDS if f not in got]
    if missing:
        raise ValueError(f"tier3 fused-MoE shape {shape!r} is missing {', '.join(missing)}")
    return {f: got[f] for f in MOE_KEY_FIELDS}


class _FusedMoeAdapter:
    """Dispatch and validate aiter fused mxfp4 SwiGLU MoE candidates.

    Cache resets, observed kernel read-back, compatible operand layouts, and
    pair checks prevent silently timing a substituted/default kernel. Scope is
    the ``per_1x32`` fp4 path selected through ``AITER_CONFIG_FMOE``.
    """

    #: Exact aiter fused-MoE CSV schema: dispatch keys, kernels, and flags.
    _CSV_COLUMNS = (
        "gfx",
        "cu_num",
        *MOE_KEY_FIELDS,
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "xbf16",
        "flat",
        "tflops",
        "bw",
    )

    def __init__(self) -> None:
        self._operands: dict[str, Any] = {}
        self._in_play: dict[str, dict[str, Any]] = {}
        self._workdir = Path(tempfile.mkdtemp(prefix="forge_tier3_fmoe_"))
        self._baseline_csv = self._workdir / "baseline.csv"
        self._baseline_csv.write_text(",".join(self._CSV_COLUMNS) + "\n", encoding="utf-8")
        self._restore = os.environ.get("AITER_CONFIG_FMOE")

    # -- torch and aiter are imported lazily so this stays importable off-GPU --
    @staticmethod
    def _torch():
        import torch

        return torch

    @staticmethod
    def _fused_moe():
        import aiter.fused_moe as fm

        return fm

    def _point_at(self, path: Path) -> None:
        """Point the next call at *path*, clearing all three dispatch caches."""
        from aiter.jit.core import AITER_CONFIGS

        fm = self._fused_moe()
        os.environ["AITER_CONFIG_FMOE"] = str(path)
        AITER_CONFIGS.get_config_file.cache_clear()
        fm.cfg_2stages = None
        fm.get_2stage_cfgs.cache_clear()

    # ------------------------------------------------------------- operands --
    def _ops(self, shape: str) -> dict[str, Any] | None:
        """Build operands with aiter's tuner, or return None without its sources."""
        if shape in self._operands:
            return self._operands[shape]
        key = moe_shape_key(shape)
        torch = self._torch()
        try:
            import aiter
            from aiter import ActivationType, QuantType

            if key["q_type"] != "QuantType.per_1x32" or key["act_type"] != "ActivationType.Swiglu":
                log.info("tier3: %s is not the mxfp4 SwiGLU path; no fused-MoE operands for it", shape)
                self._operands[shape] = None
                return None
            codegen = Path(aiter.__file__).resolve().parent.parent / "csrc" / "ck_gemm_moe_2stages_codegen"
            if not (codegen / "gemm_moe_tune.py").is_file():
                log.info("tier3: aiter sources are not installed at %s; cannot build fused-MoE operands", codegen)
                self._operands[shape] = None
                return None
            import sys

            if str(codegen) not in sys.path:
                sys.path.insert(0, str(codegen))
            import gemm_moe_tune as moe_tune

            dtype = getattr(torch, key["dtype"].removeprefix("torch."))
            q_dtype_w = getattr(torch, key["q_dtype_w"].removeprefix("torch."))
            bundle = moe_tune.FmoeTuner.generate_data_2stages(
                int(key["token"]),
                int(key["model_dim"]),
                int(key["inter_dim"]),
                int(key["expert"]),
                int(key["topk"]),
                ActivationType.Swiglu,
                dtype,
                q_dtype_w,
                q_dtype_w,
                QuantType.per_1x32,
                key["use_g1u1"] not in ("0", "False", "false", ""),
                key["doweight_stage1"] not in ("0", "False", "false", ""),
                # blockM only sizes the pre-sorted buffers the low-level stage
                # entry points take. Nothing passed to fused_moe depends on it,
                # so the candidate's own block_m does not force a rebuild.
                32,
                1,
            )
        except Exception as exc:  # noqa: BLE001 - "cannot build operands" is data
            log.info("tier3: cannot build fused-MoE operands for %s: %r", shape, exc)
            self._operands[shape] = None
            return None
        self._operands[shape] = bundle
        return bundle

    def _hidden(self, shape: str, seed: int):
        torch = self._torch()
        key = moe_shape_key(shape)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(int(key["token"]), int(key["model_dim"]), device="cuda", dtype=torch.bfloat16, generator=gen)

    def _call(self, shape: str, hidden) -> Callable[[], Any] | None:
        bundle = self._ops(shape)
        if bundle is None:
            return None
        from aiter import ActivationType, QuantType

        fm = self._fused_moe()
        torch = self._torch()

        def run():
            return fm.fused_moe(
                hidden,
                bundle["w1_qt_shffle_flydsl"],
                bundle["w2_qt_shffle_flydsl"],
                bundle["topk_weights"],
                bundle["topk_ids"],
                activation=ActivationType.Swiglu,
                quant_type=QuantType.per_1x32,
                w1_scale=bundle["w1_scale_flydsl"],
                w2_scale=bundle["w2_scale_flydsl"],
                dtype=torch.bfloat16,
            )

        return run

    # ------------------------------------------------------- the three hooks --
    def as_graph(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        return _Bf16DenseAdapter.as_graph(self, fn)  # type: ignore[arg-type]

    def make_baseline(self, shape: str) -> Callable[[], Any]:
        """Build an untuned baseline using an empty table, not installed rows."""
        self._point_at(self._baseline_csv)
        run = self._call(shape, self._hidden(shape, 0))
        if run is None:
            # The runner only reaches here for a table this adapter claimed, so
            # a shape it cannot build is a real failure of the attempt rather
            # than something to paper over with a no-op.
            raise ValueError(f"tier3: no fused-MoE baseline could be built for {shape}")
        return self.as_graph(run)

    def make_dispatch(self, shape: str) -> Callable[[dict[str, Any]], Callable[[], Any] | None]:
        def dispatch(cand: dict[str, Any]) -> Callable[[], Any] | None:
            self._in_play[shape] = cand
            run = self._build(shape, cand)
            return self.as_graph(run) if run is not None else None

        return dispatch

    def make_correctness(self, shape: str) -> Callable[[Callable[[], Any]], bool]:
        def check(_dispatched: Callable[[], Any]) -> bool:
            cand = self._in_play.get(shape)
            if cand is None:
                return True
            return self._is_correct(shape, cand)

        return check

    def sync(self) -> Callable[[], Any]:
        return self._torch().cuda.synchronize

    # ------------------------------------------------------------ internals --
    def _candidate_csv(self, shape: str, cand: dict[str, Any]) -> tuple[Path, str, str] | None:
        """Write the candidate as the one row of a tuned-config file."""
        if str(cand.get("backend", "")) not in FUSED_MOE_BACKENDS:
            log.info("tier3: unknown backend %r in a fused-MoE candidate", cand.get("backend"))
            return None
        cfg = parse_config(cand.get("config", ""))
        kn1, kn2 = str(cfg.get("kernelName1", "")), str(cfg.get("kernelName2", ""))
        if not kn1 or not kn2:
            log.info("tier3: a fused-MoE candidate named no kernel pair: %r", cand.get("config"))
            return None
        if not aiter_honours_kernel_pair(kn1, kn2):
            log.info(
                "tier3: aiter would read the row for (%s, %s) and then ignore it -- one of the two "
                "names has to be FlyDSL's for the tuned branch to be taken at all; refusing",
                kn1,
                kn2,
            )
            return None
        key = moe_shape_key(shape)
        block_m = int(cfg.get("block_m", 32))
        wrong = flydsl_pair_misconfigured(kn1, kn2, block_m, int(key["inter_dim"]))
        if wrong:
            log.info("tier3: refusing (%s, %s) -- %s", kn1, kn2, wrong)
            return None
        row = {
            "gfx": self._gfx(),
            "cu_num": self._cu_num(),
            **key,
            "block_m": block_m,
            "ksplit": cfg.get("ksplit", 0),
            "us1": 0,
            "kernelName1": kn1,
            "err1": "0.0%",
            "us2": 0,
            "kernelName2": kn2,
            "err2": "0.0%",
            "us": 0,
            "run_1stage": int(bool(cfg.get("run_1stage", 0))),
            "xbf16": int(bool(cfg.get("xbf16", 0))),
            "flat": int(bool(cfg.get("flat", 0))),
            "tflops": 0,
            "bw": 0,
        }
        path = self._workdir / "candidate.csv"
        path.write_text(
            ",".join(self._CSV_COLUMNS) + "\n" + ",".join(str(row[c]) for c in self._CSV_COLUMNS) + "\n",
            encoding="utf-8",
        )
        return path, kn1, kn2

    def _gfx(self) -> str:
        from aiter.jit.core import get_gfx

        return get_gfx()

    def _cu_num(self) -> int:
        return int(self._torch().cuda.get_device_properties(0).multi_processor_count)

    def _build(self, shape: str, cand: dict[str, Any]) -> Callable[[], Any] | None:
        written = self._candidate_csv(shape, cand)
        if written is None:
            return None
        path, kn1, kn2 = written
        self._point_at(path)
        run = self._call(shape, self._hidden(shape, 0))
        if run is None:
            return None
        try:
            resolved = self._resolved_kernels(run)
        except Exception as exc:  # noqa: BLE001 - a candidate that raises is not dispatchable
            log.info("tier3: fused-MoE candidate %s/%s raised: %r", kn1, kn2, exc)
            self._point_at(self._baseline_csv)
            return None
        if resolved != (kn1, kn2):
            # A mismatched key runs the default path, which noise can falsely score as a win.
            log.info(
                "tier3: aiter did not take the candidate row for %s -- asked for (%s, %s), served %s; not timing it",
                shape,
                kn1,
                kn2,
                resolved,
            )
            self._point_at(self._baseline_csv)
            return None
        return run

    def _resolved_kernels(self, run: Callable[[], Any]) -> tuple[str, str] | None:
        """Read the kernel pair aiter actually chose from its own log."""
        import re

        records: list[str] = []

        class _Catch(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        aiter_log = logging.getLogger("aiter")
        handler = _Catch()
        previous = aiter_log.level
        aiter_log.addHandler(handler)
        aiter_log.setLevel(logging.INFO)
        try:
            run()
            self._torch().cuda.synchronize()
        finally:
            aiter_log.removeHandler(handler)
            aiter_log.setLevel(previous)
        pattern = re.compile(r"kernelName1='([^']*)', kernelName2='([^']*)'")
        for message in reversed(records):
            found = pattern.search(message)
            if found:
                return (found.group(1), found.group(2))
        return None

    def _is_correct(self, shape: str, cand: dict[str, Any]) -> bool:
        """Compare with the untuned path using fresh inputs.

        This tests promotion equivalence, not independent arithmetic correctness,
        because only quantized and pre-shuffled weights are available.
        """
        torch = self._torch()
        written = self._candidate_csv(shape, cand)
        if written is None:
            return False
        hiddens = [self._hidden(shape, 1 + i) for i in range(MOE_CORRECTNESS_TRIALS)]
        try:
            self._point_at(self._baseline_csv)
            refs = []
            for hidden in hiddens:
                run = self._call(shape, hidden)
                if run is None:
                    return False
                refs.append(run().float())
            torch.cuda.synchronize()

            self._point_at(written[0])
            worst = 0.0
            for hidden, ref in zip(hiddens, refs, strict=True):
                run = self._call(shape, hidden)
                if run is None:
                    return False
                got = run()
                torch.cuda.synchronize()
                worst = max(worst, mean_error(got, ref))
        except Exception as exc:  # noqa: BLE001 - a kernel that raises is wrong
            log.warning("tier3: fused-MoE correctness check raised, rejecting: %r", exc)
            return False

        if worst > MAX_MOE_MEAN_ERROR:
            log.error(
                "tier3: rejecting %s -- worst mean error over %d activation sets was %.4g, above the %.3g limit",
                str(cand.get("config"))[:80],
                MOE_CORRECTNESS_TRIALS,
                worst,
                MAX_MOE_MEAN_ERROR,
            )
            return False
        return True


class _Bf16DenseAdapter:
    """Dispatch, baseline and correctness for row-major bf16 A[M,K] x B[N,K]^T."""

    def __init__(self) -> None:
        self._operands: dict[tuple[int, ...], tuple[Any, Any]] = {}
        self._in_play: dict[str, dict[str, Any]] = {}
        self._hipb_ready = False
        self._hipb_sols: dict[tuple[int, ...], set[int]] = {}

    # -- torch is imported lazily so this module stays importable off-GPU --
    @staticmethod
    def _torch():
        import torch

        return torch

    def _ops(self, key: tuple[int, int, int]):
        if key not in self._operands:
            torch = self._torch()
            m, n, k = key
            torch.manual_seed(0)
            self._operands[key] = (
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
            )
        return self._operands[key]

    def as_graph(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """Replay many invocations per call, so dispatch cost is amortised."""
        torch = self._torch()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(GRAPH_INNER):
                    fn()
            return graph.replay
        except Exception as exc:  # noqa: BLE001 - capture is an optimisation
            log.debug("tier3: graph capture failed, timing raw: %r", exc)
            return fn

    def make_baseline(self, shape: str) -> Callable[[], Any]:
        torch = self._torch()
        key = shape_key(shape)
        a, b = self._ops(key)
        return self.as_graph(lambda: torch.matmul(a, b.t()))

    def make_dispatch(self, shape: str) -> Callable[[dict[str, Any]], Callable[[], Any] | None]:
        key = shape_key(shape)

        def dispatch(cand: dict[str, Any]) -> Callable[[], Any] | None:
            # The correctness check runs straight after this and needs to know which candidate is in play, because it
            # has to rebuild against fresh inputs rather than reuse this callable's fixed operands.
            self._in_play[shape] = cand
            call = self._build(key, cand)
            return self.as_graph(call) if call is not None else None

        return dispatch

    def make_correctness(self, shape: str) -> Callable[[Callable[[], Any]], bool]:
        key = shape_key(shape)

        def check(_dispatched: Callable[[], Any]) -> bool:
            cand = self._in_play.get(shape)
            if cand is None:
                return True
            return self._is_correct(key, cand)

        return check

    def sync(self) -> Callable[[], Any]:
        return self._torch().cuda.synchronize

    # ------------------------------------------------------------ internals --
    def _hipb_solutions(self, key: tuple[int, int, int], a, bt) -> set[int]:
        """Return hipBLASLt solution indices valid for these operands.

        Invalid handles or indices terminate the process in C++, so initialize
        the extension and reject values outside ``findallsols`` before timing.
        """
        import aiter

        torch = self._torch()
        if not self._hipb_ready:
            aiter.hipb_create_extension()
            self._hipb_ready = True
        if key not in self._hipb_sols:
            sols = aiter.hipb_findallsols(a, bt, None, torch.bfloat16, None, None, None, False, False)
            self._hipb_sols[key] = {int(s) for s in (sols or [])}
        return self._hipb_sols[key]

    def _build(self, key: tuple[int, int, int], cand: dict[str, Any]) -> Callable[[], Any] | None:
        """One candidate as a callable, or None when we cannot dispatch it.

        None is recorded by the referee as "not dispatchable", which is a result
        worth having; approximating what the candidate meant is not.
        """
        backend = str(cand.get("backend", ""))
        if backend not in DENSE_BF16_BACKENDS:
            # Validate against the mandate's source before importing aiter or allocating operands.
            log.info("tier3: unknown backend %r in a candidate", backend)
            return None

        import aiter

        torch = self._torch()
        cfg = parse_config(cand.get("config", ""))
        m, n, _k = key
        a, b = self._ops(key)

        try:
            if backend == "torch":
                return lambda: torch.matmul(a, b.t())

            if backend == "hipblaslt":
                sol = cfg.get("solidx")
                if sol is None:
                    return None
                bt = b.t()
                sol = int(sol)
                if sol not in self._hipb_solutions(key, a, bt):
                    log.info("hipblaslt solidx %s is not a solution for %s; not dispatching", sol, key)
                    return None
                return lambda: aiter.hipb_mm(a, bt, sol, None, torch.bfloat16, None, None, None, False, False)

            if backend == "aiter_asm":
                name = cfg.get("kernelName")
                if not name:
                    return None
                split_k = cfg.get("splitK", 0)
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                return lambda: aiter.gemm_a16w16_asm(a, b, out, None, split_k, name, False)

            if backend == "aiter_opus":
                from aiter.ops.opus import gemm_op_a16w16 as opus

                kernel_id = cfg.get("kernelId")
                if kernel_id is None:
                    return None
                init = getattr(opus, "opus_gemm_workspace_init", None)
                if init:
                    init()
                a3, b3 = a.unsqueeze(0), b.unsqueeze(0)
                y = torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16)
                return lambda: opus.opus_gemm_a16w16_tune(
                    a3, b3, y, bias=None, kernelId=kernel_id, splitK=cfg.get("splitK", 1)
                )

            if backend == "aiter_flydsl":
                import aiter.ops.flydsl.gemm_kernels as fly

                return lambda: fly.flydsl_hgemm(
                    a,
                    b,
                    bias=None,
                    kernel_family="hgemm",
                    tile_m=cfg.get("tile_m"),
                    tile_n=cfg.get("tile_n"),
                    tile_k=cfg.get("tile_k"),
                    split_k=cfg.get("split_k", 1),
                    block_m_warps=cfg.get("block_m_warps", 1),
                    block_n_warps=cfg.get("block_n_warps", 1),
                    block_k_warps=cfg.get("block_k_warps", 1),
                    stages=cfg.get("stages", 4),
                    async_copy=cfg.get("async_copy", True),
                    b_to_lds=cfg.get("b_to_lds", True),
                    b_preshuffle=False,
                    c_to_lds=False,
                )
        except Exception as exc:  # noqa: BLE001 - undispatchable is data
            log.info("tier3: cannot dispatch %s: %r", backend, exc)
            return None

        # Declared above but not built here: a gap in this module, not in the
        # candidate. Named as such so it is not mistaken for a bad proposal.
        log.warning("tier3: backend %r is advertised in the mandate but not implemented", backend)
        return None

    def _is_correct(self, key: tuple[int, int, int], cand: dict[str, Any]) -> bool:
        torch = self._torch()
        m, n, k = key
        saved = self._operands.get(key)
        worst = 0.0
        try:
            for _ in range(CORRECTNESS_TRIALS):
                self._operands[key] = (
                    torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                    torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
                )
                a, b = self._operands[key]
                call = self._build(key, cand)
                if call is None:
                    return False
                got = call()
                torch.cuda.synchronize()
                if got is None:
                    return False
                ref = torch.matmul(a.float(), b.float().t())
                worst = max(worst, relative_error(got, ref))
        except Exception as exc:  # noqa: BLE001 - a kernel that raises is wrong
            log.warning("tier3: correctness check raised, rejecting: %r", exc)
            return False
        finally:
            if saved is not None:
                self._operands[key] = saved
            else:
                self._operands.pop(key, None)

        if worst > MAX_RELATIVE_ERROR:
            log.error(
                "tier3: rejecting %s %s -- worst error over %d fresh inputs was %.4g, above the %.3g limit",
                cand.get("backend"),
                str(cand.get("config"))[:60],
                CORRECTNESS_TRIALS,
                worst,
                MAX_RELATIVE_ERROR,
            )
            return False
        return True


def relative_error(got: Any, ref: Any) -> float:
    """Largest deviation, against the magnitude of the reference as a whole."""
    return float((got.float() - ref).abs().max() / ref.abs().mean())


def aiter_honours_kernel_pair(kernel_name_1: str, kernel_name_2: str) -> bool:
    """Return whether aiter dispatches this pair instead of silently skipping it.

    At least one member must be FlyDSL; logging happens before this guard, so
    runtime read-back cannot detect rejection.
    """
    return kernel_name_1.startswith(FLYDSL_KERNEL_PREFIX) or kernel_name_2.startswith(FLYDSL_KERNEL_PREFIX)


#: The tile a FlyDSL kernel name carries, as ``_t<M>x<N>x<K>``.
FLYDSL_TILE = re.compile(r"_t(\d+)x(\d+)x(\d+)")


def flydsl_tile(kernel_name: str) -> tuple[int, int, int] | None:
    """Return a FlyDSL ``(M, N, K)`` tile, or None for other families."""
    if not kernel_name.startswith(FLYDSL_KERNEL_PREFIX):
        return None
    found = FLYDSL_TILE.search(kernel_name)
    return (int(found.group(1)), int(found.group(2)), int(found.group(3))) if found else None


def flydsl_pair_misconfigured(kernel_name_1: str, kernel_name_2: str, block_m: int, inter_dim: int) -> str:
    """Explain silent FlyDSL substitution or row-layout mismatch.

    Stage tiles must divide ``inter_dim`` or aiter halves them, and ``block_m``
    must match both M tiles. Return "" when neither rejection applies.
    """
    tile_1, tile_2 = flydsl_tile(kernel_name_1), flydsl_tile(kernel_name_2)
    if tile_1 and inter_dim % tile_1[1]:
        return f"stage-1 tile N={tile_1[1]} does not divide inter_dim={inter_dim}; aiter would run a smaller tile"
    if tile_2 and inter_dim % tile_2[2]:
        return f"stage-2 tile K={tile_2[2]} does not divide inter_dim={inter_dim}; aiter would run a smaller tile"
    for which, tile in (("stage-1", tile_1), ("stage-2", tile_2)):
        if tile and tile[0] != block_m:
            return f"block_m={block_m} is not the {which} tile M={tile[0]}; the token sort would not match the indexing"
    return ""


def mean_error(got: Any, ref: Any) -> float:
    """Return average deviation normalized by the reference's global magnitude."""
    return float((got.float() - ref).abs().mean() / ref.abs().mean())
