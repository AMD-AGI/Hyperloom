"""Runner Plan 28 §4.6 for §4.4a and §4.4c: on the same inputs, the ROCm sparse
indexer's outputs with a flag on are those with every flag off.

RUPPELL_CHECK=p28_indexer patches vLLM's SparseAttnIndexer.forward_hip before
the model is built. On a real step (attention metadata present, no graph
capture) whose batch holds a prefill for the layer, it calls the indexer op
several times on the same inputs:

  - once per flag in P28_CHECK_FLAGS, that flag alone on;
  - twice with every flag off (R2, then R1 last, so the buffers the model
    reads hold the server's own configuration's output).

Re-running the op is safe: its cache insert writes the same values to the same
slots, it resets topk_indices_buffer itself, and a candidate-writing layer
writes candidate_blocks from the same logits. The op's outputs are
topk_indices_buffer[:num_tokens] and, on a candidate-writing layer,
candidate_blocks; the sparse attention reads nothing else from it, so equal
outputs mean an equal attention output.

Each output is compared exactly (torch.equal) and as per-row sorted sets (the
top-k may order or break ties differently between runs on identical logits).
R1 against R2 is the op's own run-to-run behaviour, the floor for the flags.

The prefill logits each run produces are fingerprinted too: every
rocm_fp8_mqa_logits call inside the op (one per chunk) records the int64 sum of
the logits' bit patterns over each row's [ks, ke), and the count of those
columns. That is the top-k's whole input (it reads nothing outside [ks, ke)),
so equal fingerprints settle a flag whatever the top-k's tie-breaking does.

One JSON line per checked call in $P28_CHECK_OUT/check_<pid>.jsonl, at most
P28_CHECK_MAX calls per process (default 3000). Lines record the layer, the
token count, prefill chunk count, sub-chunks with skip_kv_gather, and the
equalities.
"""

import json
import os
import time

FLAGS = [f for f in os.environ.get(
    "P28_CHECK_FLAGS", "VLLM_ROCM_INDEXER_SKIP_REGATHER VLLM_ROCM_INDEXER_SKIP_LOGITS_FILL").split() if f]
MAX_CALLS = int(os.environ.get("P28_CHECK_MAX", "3000"))
OUT = os.environ.get("P28_CHECK_OUT", "/work/p28_check")

_state = {"calls": 0, "fh": None, "patched": False, "record": None}


def _out():
    if _state["fh"] is None:
        os.makedirs(OUT, exist_ok=True)
        _state["fh"] = open(os.path.join(OUT, f"check_{os.getpid()}.jsonl"), "a", buffering=1)
    return _state["fh"]


def _compare(torch, a, b):
    exact = bool(torch.equal(a, b))
    if exact:
        return True, True
    if a.dim() == 2:
        sa, _ = torch.sort(a, dim=-1)
        sb, _ = torch.sort(b, dim=-1)
        return False, bool(torch.equal(sa, sb))
    return False, False


def _patch(mod):
    import torch
    import vllm.envs as envs
    from vllm.forward_context import get_forward_context

    C = mod.SparseAttnIndexer
    if getattr(C.forward_hip, "_p28_check", False):
        return
    orig = C.forward_hip
    unknown = [f for f in FLAGS if f not in envs.environment_variables]
    if unknown:
        print(f"[p28_check] flags not declared by this vLLM: {unknown}", flush=True)

    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as ops

    if not getattr(ops.rocm_fp8_mqa_logits, "_p28_check", False):
        orig_logits = ops.rocm_fp8_mqa_logits

        def rocm_fp8_mqa_logits(*a, **kw):
            out = orig_logits(*a, **kw)
            if _state["record"] is not None:
                ks, ke = a[3], a[4]
                cols = torch.arange(out.shape[1], device=out.device)
                mask = (cols[None, :] >= ks[:, None].long()) & (cols[None, :] < ke[:, None].long())
                bits = torch.where(mask, out.view(torch.int32), 0)
                _state["record"].append([int(bits.sum(dtype=torch.int64)), int(mask.sum())])
                del cols, mask, bits
            return out

        rocm_fp8_mqa_logits._p28_check = True
        ops.rocm_fp8_mqa_logits = rocm_fp8_mqa_logits

    def outputs(self, n):
        outs = [self.topk_indices_buffer[:n].clone()]
        if getattr(self, "candidate_write", False) and self.candidate_blocks is not None:
            outs.append(self.candidate_blocks.clone())
        return outs

    def run(self, n, args, flag=None):
        if flag is not None:
            setattr(envs, flag, True)
        _state["record"] = []
        try:
            ret = orig(self, *args)
        finally:
            if flag is not None:
                delattr(envs, flag)
            fps, _state["record"] = _state["record"], None
        return ret, outputs(self, n), fps

    def forward_hip(self, hidden_states, q_quant, k, weights):
        args = (hidden_states, q_quant, k, weights)
        if _state["calls"] >= MAX_CALLS or torch.cuda.is_current_stream_capturing():
            return orig(self, *args)
        md = get_forward_context().attn_metadata
        lmd = md.get(self.k_cache.prefix) if isinstance(md, dict) else None
        if lmd is None or getattr(lmd, "num_prefills", 0) <= 0 or getattr(lmd, "prefill", None) is None:
            return orig(self, *args)
        _state["calls"] += 1
        n = hidden_states.shape[0]
        chunks = lmd.prefill.chunks
        t0 = time.time()
        variants = {}
        for flag in FLAGS:
            if flag in envs.environment_variables:
                variants[flag] = run(self, n, args, flag)[1:]
        _, r2, f2 = run(self, n, args)
        ret, r1, f1 = run(self, n, args)
        torch.cuda.synchronize()
        rec = {
            "layer": self.k_cache.prefix,
            "tokens": n,
            "prefills": int(lmd.num_prefills),
            "decodes": int(getattr(lmd, "num_decodes", 0)),
            "chunks": len(chunks),
            "skip_kv_gather_chunks": sum(1 for c in chunks if getattr(c, "skip_kv_gather", False)),
            "max_chunk_width": max((int(c.total_seq_lens) for c in chunks), default=0),
            "candidate_write": bool(getattr(self, "candidate_write", False)),
            "candidate_read": self.candidate_blocks is not None and not getattr(self, "candidate_write", False),
        }
        names = ["topk"] + (["candidates"] if len(r1) > 1 else [])
        rec["R2"] = {nm: _compare(torch, a, b) for nm, a, b in zip(names, r2, r1)}
        rec["R2"]["logits"] = [f2 == f1, f2 == f1]
        for flag, (outs, fv) in variants.items():
            rec[flag] = {nm: _compare(torch, a, b) for nm, a, b in zip(names, outs, r1)}
            rec[flag]["logits"] = [fv == f1, fv == f1]
        rec["logits_calls"] = len(f1)
        rec["ms"] = round((time.time() - t0) * 1e3, 1)
        _out().write(json.dumps(rec) + "\n")
        return ret

    forward_hip._p28_check = True
    C.forward_hip = forward_hip
    print(f"[p28_check] patched SparseAttnIndexer.forward_hip in pid {os.getpid()}: flags {FLAGS}, "
          f"at most {MAX_CALLS} calls, to {OUT}", flush=True)


def install():
    """Patch the indexer layer when its module is imported (or now, if it was)."""
    import importlib.abc
    import importlib.util
    import sys

    name = "vllm.model_executor.layers.sparse_attn_indexer"
    if name in sys.modules:
        _patch(sys.modules[name])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        busy = False

        def find_spec(self, fullname, path, target=None):
            if fullname != name or self.busy:
                return None
            self.busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self.busy = False
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
