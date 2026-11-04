"""A directions map generated from a host's root profile, with the v2 pack's roadmap as the template.

A host that runs several Arbors from one map needs that map before its branches open. A
dossier pass measures one (``component.stop_at: "map"``); this writes one at once, with no GPU
time, from what the host already has after its baseline: the op shares of the root's profile,
the file it gives Arbor as ``component.profile``: minimal directions from the host's profile,
with the v2 pack as a template.

- **The minimal directions** are the profile's op families, as Hyperloom's MB reads them for its
  open areas: each op is put in the first family whose pattern its name matches
  (``hyperloom_multibranch/blackboard/families.py`` OP_FAMILIES, copied below), with collectives
  first, which MB's families leave out. Ops of no family are reported, not mapped. A profile of
  kernel names maps too (MB gives its kernel shares when TraceLens' op decomposition degrades on a
  CUDA-graph trace): the families also match hipBLASLt's Tensile GEMMs (``Cijk_``), vLLM's skinny
  GEMM (``wvSplitK``), its KV-cache write (``reshape_and_cache``) and its prefix-prefill attention
  (``_fwd_kernel``); copies and memcpys stay unmapped.
- **The template** is the v2 pack's roadmap (``packs/agentx-dsv41-2``, constraints section 3),
  model-independent: each family is one direction, titled as that roadmap frames the gap. The
  small kernels (norm, RoPE, activation, elementwise) are one direction, as its X3 fusion pack.
- **Rank** follows the family's share of GPU time, largest first. **Ceiling** is its Amdahl
  ceiling, the most the run could gain if the family's time went to zero (MB's
  ``amdahl_ceiling_pct``). A family under ``min_share`` percent is left out.

    python3 -m arbor.directions generate --profile PROFILE --out MAP [--min-share PCT]

PROFILE is ``{"unit": "percent", "shares": {<op name>: <share>}, ...}``, as MB writes it
(``root_profile``); shares as fractions (summing to about 1) are read as such. MAP is
``{"directions": [...]}``, every direction open, which ``split`` and ``component.plan`` take.
Exit 0: written. Exit 1: the profile cannot be used (unreadable, no shares, or nothing above
``min_share``), and nothing is written. Exit 2: a usage error.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_MIN_SHARE = 2.0

# (family, op-name pattern), the more specific first: collectives (not one of MB's), then MB's.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("collectives", r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|cross_?device"),
    ("linear attention", r"gdn|gated_delta|delta_rule|causal_conv1d|conv1d|mamba|linear[ _-]attention"),
    ("moe", r"moe|expert|router|gating|topk_?softmax|softmax_?topk"),
    ("attention", r"attn|attention|paged_attention|\bpa_|\bmla\b|\bmha\b|flash|ll4mi"
                  r"|reshape_and_cache|^_fwd_kernel"),
    ("gemm", r"gemm|matmul|blockscale|a8w8|a16w16|qkv_proj|o_proj|gate_up|down_proj|split-?k|\bck\b"
             r"|hipblaslt|aten::(?:addmm|bmm|mm)\b|^cijk_|wvsplitk|parallellinear"),
    ("norm", r"rmsnorm|layernorm|qk-?norm|\bnorm\b"),
    ("rope", r"rope|rotary"),
    ("activation", r"silu|gelu|swiglu|_and_mul|activation"),
    ("sampling", r"sampl|logits"),
    ("elementwise", r"elementwise|residual"),
)
def _boundaries(pat: str) -> str:
    """MB's pattern with each ``\\b`` read as an alphanumeric boundary. Op names join words with
    ``_``, which ``\\b`` counts as part of a word: in MB ``\\bmla\\b`` misses ``mla_decode_fwd``."""
    opening = re.sub(r"\\b(?=[A-Za-z0-9_(\[])", "(?<![a-z0-9])", pat)
    return opening.replace(r"\b", "(?![a-z0-9])")


_FAMILY_RES = tuple((name, re.compile(_boundaries(pat), re.IGNORECASE)) for name, pat in FAMILIES)

# Each family's direction, framed as the v2 pack's roadmap frames that gap: (id, title).
TEMPLATE: dict[str, tuple[str, str]] = {
    "attention": ("m-attn", "Attention: the kernel each served shape dispatches, prefill and decode, "
                            "and the metadata it builds per step (v2 roadmap B: selection by shape)"),
    "linear attention": ("m-linattn", "Linear attention: the recurrent and conv kernels on the decode "
                                      "path, and their state layout"),
    "moe": ("m-moe", "MoE: expert GEMMs in the checkpoint's own datatype, routing and top-k, and TP "
                     "against EP for the served batches (v2 roadmap A2, D2)"),
    "gemm": ("m-gemm", "Dense GEMMs: tuned kernels for the served shapes, the scale layout, and no "
                       "silent fallback (v2 roadmap A1)"),
    "small": ("m-fuse", "Small kernels: norm, RoPE, activation and elementwise folded into their "
                        "producers, fewer launches at small batch (v2 roadmap A3, X3)"),
    "sampling": ("m-sampling", "Sampling and logits: the path after the forward pass"),
    "collectives": ("m-comm", "Collectives: all-reduce, all-gather and reduce-scatter on the custom "
                              "paths, overlapped or fused with their neighbours (v2 roadmap D)"),
}
SMALL = ("norm", "rope", "activation", "elementwise")


# A kernel name as MB's profile writes it may end in the module that launched it, in brackets:
# ``ck::kernel_gemm_... [Qwen3Attention.QKVParallelLinear]``.
_CALLER = re.compile(r"^(.*?)\s*\[([^\]]*)\]\s*$")


def family(op: str) -> str | None:
    """The op's family, the first whose pattern its name matches; None for none. A name that ends
    in its caller, ``kernel [Module.Layer]``, is placed by the kernel, and only when the kernel
    matches none by the caller, its innermost part first: the GEMM of
    ``[Qwen3Attention.QKVParallelLinear]`` is a GEMM, and so are ``ck_tile::kentry`` and a quant
    kernel launched by a ``QKVParallelLinear``, though their module's name says Attention."""
    named = _CALLER.match(op)
    names = (named.group(1), *reversed(named.group(2).split("."))) if named else (op,)
    for part in names:
        for name, rx in _FAMILY_RES:
            if rx.search(part):
                return name
    return None


def amdahl_ceiling_pct(share_pct: float) -> float:
    """The most, in percent, that a region worth ``share_pct`` of the time can gain: made free."""
    return float("inf") if share_pct >= 100.0 else 100.0 * share_pct / (100.0 - share_pct)


def read_shares(profile: dict) -> dict[str, float]:
    """The profile's op shares in percent. Raises ValueError when it has none usable."""
    shares = profile.get("shares") if isinstance(profile, dict) else None
    if not isinstance(shares, dict) or not shares:
        raise ValueError("the profile has no shares")
    vals = {str(k): float(v) for k, v in shares.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
    if not vals:
        raise ValueError("the profile's shares are not positive numbers")
    unit = str(profile.get("unit") or "").lower()
    if unit != "percent" and sum(vals.values()) <= 1.5:    # fractions of the whole
        vals = {k: v * 100.0 for k, v in vals.items()}
    return vals


def generate(profile: dict, min_share: float = DEFAULT_MIN_SHARE) -> tuple[list[dict], dict]:
    """The map's directions, ranked, and a summary ``{groups, unmapped}`` for the log.
    Raises ValueError when no family reaches ``min_share``."""
    shares = read_shares(profile)
    groups: dict[str, list[tuple[str, float]]] = {}
    unmapped: list[tuple[str, float]] = []
    for op, pct in shares.items():
        fam = family(op)
        if fam is None:
            unmapped.append((op, pct))
            continue
        groups.setdefault("small" if fam in SMALL else fam, []).append((op, pct))
    totals = {g: sum(p for _, p in ops) for g, ops in groups.items()}
    kept = sorted((g for g in totals if totals[g] >= min_share), key=lambda g: (-totals[g], g))
    if not kept:
        raise ValueError(f"no op family reaches {min_share:g} % of the profile")
    source = ", ".join(str(profile[k]) for k in ("source", "commit") if profile.get(k)) or "the host"
    directions = []
    for rank, g in enumerate(kept, 1):
        did, title = TEMPLATE[g]
        top = sorted(groups[g], key=lambda o: -o[1])[:3]
        ceiling = amdahl_ceiling_pct(totals[g])
        directions.append({
            "id": did, "kind": "increment", "title": title, "status": "open", "rank": rank,
            "ceiling": "unbounded" if ceiling == float("inf") else f"<= {ceiling:.0f}% (Amdahl)",
            "why": (f"root profile ({source}): {totals[g]:.1f} % of GPU time in {len(groups[g])} ops; "
                    f"largest {', '.join(f'{o} {p:.1f} %' for o, p in top)}")[:400],
        })
    return directions, {"groups": totals, "unmapped": sorted(unmapped, key=lambda o: -o[1])}


def main(argv: list[str]) -> int:
    import argparse
    from arbor import component
    from arbor.directions import _entry
    ap = argparse.ArgumentParser(prog="python3 -m arbor.directions generate",
                                 description="A directions map from a host's root profile, with the v2 "
                                             "pack's roadmap as the template.")
    ap.add_argument("--profile", required=True, help="the root profile: {unit, shares: {op: share}}")
    ap.add_argument("--out", required=True, help="the map to write: {\"directions\": [...]}")
    ap.add_argument("--min-share", type=float, default=DEFAULT_MIN_SHARE,
                    help=f"leave out an op family under this percent of GPU time ({DEFAULT_MIN_SHARE:g})")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    try:
        profile = json.loads(Path(args.profile).read_text())
        directions, summary = generate(profile, args.min_share)
    except (OSError, ValueError) as exc:
        print(f"{args.profile}: no map generated: {exc}", file=sys.stderr)
        return 1
    for d in directions:                         # each must be a direction and a plan Arbor accepts
        _, err = _entry(d, listed=True)
        if err:
            print(f"generated an invalid direction: {err}", file=sys.stderr)
            return 1
    bad = component.plan_entries({"component": {"plan": directions}})[1]
    if bad:
        print("generated an invalid plan: " + "; ".join(bad), file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"directions": directions}, indent=1) + "\n")
    for d in directions:
        print(f"{d['rank']}. {d['id']}: {summary['groups'][_group_of(d['id'])]:.1f} %, ceiling {d['ceiling']}")
    if summary["unmapped"]:
        rest = sum(p for _, p in summary["unmapped"])
        print(f"({len(summary['unmapped'])} ops in no family, {rest:.1f} %: "
              + ", ".join(o for o, _ in summary["unmapped"][:5]) + ")")
    return 0


def _group_of(did: str) -> str:
    return next(g for g, (i, _) in TEMPLATE.items() if i == did)
