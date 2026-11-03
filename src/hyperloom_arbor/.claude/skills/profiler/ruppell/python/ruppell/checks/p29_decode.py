"""Runner Plan 29 §4.6: on the same decode inputs, the ROCm sparse indexer's
outputs with a Plan 29 flag equal (B3, B1) or match (B2, per-row Jaccard)
those without it, and C4's reused top-k CSR is the one its layer would build.

RUPPELL_CHECK=p29_decode patches, in every vLLM process, before the model is
built:

1. SparseAttnIndexer.forward_hip. On a real pure-decode step (attention
   metadata present, no graph capture, decodes and no prefill for the layer;
   so the server runs eagerly: decode replays FULL graphs otherwise) it calls
   the indexer op several times on the same inputs:
   - once per flag in P29_CHECK_FLAGS, that flag alone on (B3
     VLLM_DSV41_CANDIDATE_LIVE_SIZED and B2 VLLM_ROCM_CANDIDATE_PAGED_SCORER
     are read inside the op at call time);
   - with P29_CHECK_FLAT=1 and native (B, next_n) decode metadata (B1 on at
     start-up), once with that metadata re-laid flat: one row per query token,
     the per-token lengths, each request's block-table row repeated, decode
     length 1, which is what the pin's builder makes ("FLAT");
   - twice as built: R2, then R1 last, so the buffers the model reads hold the
     server's own configuration's output.
   Re-running is safe for p28_indexer's reasons: the K insert writes the same
   values to the same slots, the op resets its top-k rows, and the
   candidate-writing layer rewrites candidate_blocks from the same logits.
   Compared over the decode rows: the top-k indices exactly, as per-row sorted
   sets, and (when the sets differ) as per-row Jaccard of the valid (>= 0)
   entries; on the candidate-writing layer, candidate_blocks the same ways;
   and every dense paged-logits call's fingerprint, the int64 sum of the
   logits' bit patterns over each row's visible columns (row b*n+j of a
   native call sees L_b - n + j + 1 columns: AITER's own mask). R2 against R1
   is the op's own run-to-run behaviour, the floor for the flags.

2. DeepseekV41ROCMAiterMLAAttention._forward_decode, with P29_CHECK_CSR=1
   (C4, VLLM_DSV41_CSR_PER_INDEX_GROUP on): before each call that reads the
   top-k it builds the layer's global top-k CSR afresh from the same inputs;
   after the call, if the layer reused its group's memo, the memo's CSR must
   equal the fresh one exactly (lens, indptr, and the packed slots up to
   indptr[-1]).

One JSON line per checked call, $P29_CHECK_OUT/idx_<pid>.jsonl and
csr_<pid>.jsonl; at most P29_CHECK_MAX indexer calls and P29_CHECK_MAX_CSR
attention calls per process.
"""

import dataclasses
import json
import os
import time

FLAGS = [f for f in os.environ.get("P29_CHECK_FLAGS", "").split() if f]
FLAT = os.environ.get("P29_CHECK_FLAT", "0") == "1"
CSR = os.environ.get("P29_CHECK_CSR", "0") == "1"
MAX_CALLS = int(os.environ.get("P29_CHECK_MAX", "3000"))
MAX_CSR = int(os.environ.get("P29_CHECK_MAX_CSR", "3000"))
OUT = os.environ.get("P29_CHECK_OUT", "/work/p29_check")

_state = {"calls": 0, "csr_calls": 0, "fh": {}, "record": None}


def _out(kind):
    fh = _state["fh"].get(kind)
    if fh is None:
        os.makedirs(OUT, exist_ok=True)
        fh = open(os.path.join(OUT, f"{kind}_{os.getpid()}.jsonl"), "a", buffering=1)
        _state["fh"][kind] = fh
    return fh


def _cmp(torch, a, b):
    """exact; same per-row set; per-row Jaccard of the valid entries when not."""
    exact = bool(torch.equal(a, b))
    if a.dim() != 2:
        return {"exact": exact}
    same_set = exact
    if not exact:
        sa, _ = torch.sort(a, dim=-1)
        sb, _ = torch.sort(b, dim=-1)
        same_set = bool(torch.equal(sa, sb))
    res = {"exact": exact, "set": same_set}
    if not same_set:
        A, B = a.cpu().tolist(), b.cpu().tolist()
        jac = []
        for ra, rb in zip(A, B):
            x = {v for v in ra if v >= 0}
            y = {v for v in rb if v >= 0}
            u = len(x | y)
            jac.append(1.0 if u == 0 else len(x & y) / u)
        jac.sort()
        n = len(jac)
        # median, p1 and min over rows, row count, rows with any difference
        res["jaccard"] = [jac[n // 2], jac[(n - 1) // 100], jac[0], n, sum(j < 1.0 for j in jac)]
    return res


def _flat_decode(torch, dmd):
    """The pin's flattened decode metadata for a native (B, next_n) one."""
    sl = dmd.seq_lens
    if sl.dim() != 2 or sl.shape[1] <= 1:
        return None
    b, n = sl.shape
    return dataclasses.replace(
        dmd,
        seq_lens=sl.reshape(-1, 1).contiguous(),
        block_table=dmd.block_table[:b].repeat_interleave(n, dim=0).contiguous(),
        decode_lens=torch.ones(b * n, dtype=dmd.decode_lens.dtype, device=dmd.decode_lens.device),
        requires_padding=False,
    )


def _patch_indexer(mod):
    import torch
    import vllm.envs as envs
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as ops
    from vllm.forward_context import get_forward_context

    C = mod.SparseAttnIndexer
    if getattr(C.forward_hip, "_p29_check", False):
        return
    orig = C.forward_hip
    unknown = [f for f in FLAGS if f not in envs.environment_variables]
    if unknown:
        print(f"[p29_check] flags not declared by this vLLM: {unknown}", flush=True)

    if not getattr(ops.rocm_fp8_paged_mqa_logits, "_p29_check", False):
        orig_logits = ops.rocm_fp8_paged_mqa_logits

        def rocm_fp8_paged_mqa_logits(*a, **kw):
            out = orig_logits(*a, **kw)
            if _state["record"] is not None:
                q, ctx = a[0], a[3]
                b, n = q.shape[0], q.shape[1]
                ctx = ctx.reshape(-1).long()
                if n > 1:
                    j = torch.arange(n, device=out.device)
                    vis = (ctx[:b, None] - n + j[None, :] + 1).clamp_(min=0).reshape(-1)
                else:
                    vis = ctx[:b]
                width = max(int(vis.max().item()) if vis.numel() else 0, 1)
                sub = out[: vis.numel(), :width]
                cols = torch.arange(width, device=out.device)
                mask = cols[None, :] < vis[:, None]
                bits = torch.where(mask, sub.view(torch.int32), 0)
                _state["record"].append([int(bits.sum(dtype=torch.int64)), int(mask.sum())])
                del cols, mask, bits, sub
            return out

        rocm_fp8_paged_mqa_logits._p29_check = True
        ops.rocm_fp8_paged_mqa_logits = rocm_fp8_paged_mqa_logits

    def outputs(self, n):
        outs = {"topk": self.topk_indices_buffer[:n].clone()}
        if getattr(self, "candidate_write", False) and self.candidate_blocks is not None:
            outs["candidates"] = self.candidate_blocks[:n].clone()
        return outs

    def run(self, n, args, flag=None, flat=None, lmd=None):
        saved = None
        if flag is not None:
            setattr(envs, flag, True)
        if flat is not None:
            saved = lmd.decode
            lmd.decode = flat
        _state["record"] = []
        try:
            ret = orig(self, *args)
        finally:
            if flag is not None:
                delattr(envs, flag)
            if saved is not None:
                lmd.decode = saved
            fps, _state["record"] = _state["record"], None
        return ret, outputs(self, n), fps

    def forward_hip(self, hidden_states, q_quant, k, weights):
        args = (hidden_states, q_quant, k, weights)
        if _state["calls"] >= MAX_CALLS or torch.cuda.is_current_stream_capturing():
            return orig(self, *args)
        md = get_forward_context().attn_metadata
        lmd = md.get(self.k_cache.prefix) if isinstance(md, dict) else None
        if (
            lmd is None
            or getattr(lmd, "num_decodes", 0) <= 0
            or getattr(lmd, "num_prefills", 0) > 0
            or getattr(lmd, "decode", None) is None
        ):
            return orig(self, *args)
        _state["calls"] += 1
        n = int(lmd.num_decode_tokens)
        t0 = time.time()
        variants = {}
        for flag in FLAGS:
            if flag in envs.environment_variables:
                variants[flag] = run(self, n, args, flag=flag)[1:]
        if FLAT:
            fl = _flat_decode(torch, lmd.decode)
            if fl is not None:
                variants["FLAT"] = run(self, n, args, flat=fl, lmd=lmd)[1:]
        _, r2, f2 = run(self, n, args)
        ret, r1, f1 = run(self, n, args)
        torch.cuda.synchronize()
        dmd = lmd.decode
        rec = {
            "layer": self.k_cache.prefix,
            "tokens": n,
            "decodes": int(lmd.num_decodes),
            "native": bool(dmd.seq_lens.dim() == 2 and dmd.seq_lens.shape[1] > 1),
            "max_ctx": int(dmd.seq_lens.max().item()),
            "candidate_write": bool(getattr(self, "candidate_write", False)),
            "candidate_read": self.candidate_blocks is not None
            and not getattr(self, "candidate_write", False),
        }
        rec["R2"] = {k: _cmp(torch, r2[k], r1[k]) for k in r1}
        rec["R2"]["logits"] = f2 == f1
        for name, (outs, fv) in variants.items():
            rec[name] = {k: _cmp(torch, outs[k], r1[k]) for k in r1 if k in outs}
            rec[name]["logits"] = (fv == f1) if len(fv) == len(f1) else None
            rec[name]["logits_calls"] = len(fv)
        rec["logits_calls"] = len(f1)
        rec["ms"] = round((time.time() - t0) * 1e3, 1)
        _out("idx").write(json.dumps(rec) + "\n")
        return ret

    forward_hip._p29_check = True
    C.forward_hip = forward_hip
    print(
        f"[p29_check] patched SparseAttnIndexer.forward_hip in pid {os.getpid()}: flags {FLAGS}, "
        f"flat {FLAT}, at most {MAX_CALLS} calls, to {OUT}",
        flush=True,
    )


def _patch_attn(mod):
    import torch
    import vllm.envs as envs

    C = getattr(mod, "DeepseekV41ROCMAiterMLAAttention", None)
    if C is None or not CSR or getattr(C._forward_decode, "_p29_check", False):
        return
    orig = C._forward_decode
    build = mod.compute_global_topk_ragged_indices_and_indptr

    def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
        if (
            swa_only
            or attn_metadata is None
            or _state["csr_calls"] >= MAX_CSR
            or torch.cuda.is_current_stream_capturing()
        ):
            return orig(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output)
        _state["csr_calls"] += 1
        nd = swa_metadata.num_decodes
        ndt = swa_metadata.num_decode_tokens
        bs = attn_metadata.block_size // self.compress_ratio
        bt = attn_metadata.block_table[:nd]
        fresh = build(
            self.topk_indices_buffer[:ndt],
            swa_metadata.token_to_req_indices,
            bt,
            bs,
            swa_metadata.is_valid_token[:ndt],
        )
        key = (self.index_source_layer_id, ndt, nd, bs, bt.data_ptr(), self.topk_indices_buffer.data_ptr())
        before = mod._DECODE_CSR_MEMO["key"]
        ret = orig(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output)
        reused = bool(envs.VLLM_DSV41_CSR_PER_INDEX_GROUP) and not self.is_index_source and before == key
        rec = {
            "layer": getattr(self, "layer_id", None),
            "index_source": self.index_source_layer_id,
            "is_source": bool(self.is_index_source),
            "tokens": int(ndt),
            "reused": reused,
        }
        if reused:
            mr, mi, ml = mod._DECODE_CSR_MEMO["csr"]
            fr, fi, fl = fresh
            nv = int(fi[-1].item()) if fi.numel() else 0
            rec["equal"] = (
                bool(torch.equal(mi, fi)) and bool(torch.equal(ml, fl)) and bool(torch.equal(mr[:nv], fr[:nv]))
            )
        _out("csr").write(json.dumps(rec) + "\n")
        return ret

    _forward_decode._p29_check = True
    C._forward_decode = _forward_decode
    print(f"[p29_check] patched {C.__name__}._forward_decode in pid {os.getpid()}: CSR reuse check", flush=True)


_TARGETS = {
    "vllm.model_executor.layers.sparse_attn_indexer": _patch_indexer,
    "vllm.models.deepseek_v4_1.amd.rocm": _patch_attn,
}


def install():
    """Patch each target module when it is imported (or now, if it was)."""
    import importlib.abc
    import importlib.util
    import sys

    pending = {}
    for name, fn in _TARGETS.items():
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            pending[name] = fn
    if not pending:
        return

    class _Finder(importlib.abc.MetaPathFinder):
        busy = set()

        def find_spec(self, fullname, path, target=None):
            if fullname not in pending or fullname in self.busy:
                return None
            self.busy.add(fullname)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self.busy.discard(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module
            fn = pending[fullname]

            def exec_module(module):
                orig_exec(module)
                fn(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
