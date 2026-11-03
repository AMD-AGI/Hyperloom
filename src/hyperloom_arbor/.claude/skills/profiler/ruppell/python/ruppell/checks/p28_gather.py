"""Runner Plan 28 §4.6 for §4.4b: with VLLM_ROCM_DSV41_PREFILL_GATHER_REUSE on,
each compressed layer's attention output in a mixed step is bitwise the output
with it off, on the same inputs.

RUPPELL_CHECK=p28_gather patches DeepseekV41ROCMAiterMLAAttention.forward_mqa
(vllm.models.deepseek_v4_1.amd.rocm). On a real step (attention metadata, no
graph capture) with a prefill, on a layer with a compressed cache, it runs
forward_mqa with the flag on, keeps the output, then runs it with the flag off
(the output the model reads) and compares. forward_mqa only reads the caches
and writes `output`. The flag-on runs see the reuse state earlier layers'
flag-on runs left, so later layers of a group exercise the skipped gather and
the cached CSR; the flag-off runs use the shared workspace, which the reuse
buffer does not share.

One JSON line per checked call in $P28_CHECK_OUT/gather_<pid>.jsonl, at most
P28_CHECK_MAX calls per process (default 3000): the layer, prefill requests,
whether reuse was taken, whether the compressed gather and the CSR were reused,
and equality (NaN-aware) with the largest absolute difference.
"""

import json
import os
import time

FLAG = "VLLM_ROCM_DSV41_PREFILL_GATHER_REUSE"
MAX_CALLS = int(os.environ.get("P28_CHECK_MAX", "3000"))
OUT = os.environ.get("P28_CHECK_OUT", "/work/p28_check")
MODULE = "vllm.models.deepseek_v4_1.amd.rocm"

_state = {"calls": 0, "fh": None}


def _out():
    if _state["fh"] is None:
        os.makedirs(OUT, exist_ok=True)
        _state["fh"] = open(os.path.join(OUT, f"gather_{os.getpid()}.jsonl"), "a", buffering=1)
    return _state["fh"]


def _patch(mod):
    import torch
    import vllm.envs as envs
    from vllm.forward_context import get_forward_context

    C = mod.DeepseekV41ROCMAiterMLAAttention
    if getattr(C.forward_mqa, "_p28_check", False):
        return
    orig = C.forward_mqa
    reuse = getattr(mod, "_PREFILL_GATHER_REUSE", None)
    if reuse is None or FLAG not in envs.environment_variables:
        print(f"[p28_check] {FLAG} is not in this vLLM; gather check not installed", flush=True)
        return

    def forward_mqa(self, q, kv, positions, output):
        md = get_forward_context().attn_metadata
        if (_state["calls"] >= MAX_CALLS or torch.cuda.is_current_stream_capturing()
                or not isinstance(md, dict) or self.compress_ratio == 0):
            return orig(self, q, kv, positions, output)
        swa = md.get(self.swa_cache_layer.prefix)
        if swa is None or getattr(swa, "num_prefills", 0) <= 0:
            return orig(self, q, kv, positions, output)
        _state["calls"] += 1
        rocm_md = md.get(self.compressed_cache_prefix)
        t0 = time.time()
        same_step = reuse.meta is rocm_md and reuse.meta is not None
        had_compressed = same_step and reuse.have_compressed
        had_csr = same_step and self.index_source_layer_id in reuse.csr
        setattr(envs, FLAG, True)
        try:
            orig(self, q, kv, positions, output)
        finally:
            delattr(envs, FLAG)
        taken = reuse.meta is rocm_md
        var = output.clone()
        orig(self, q, kv, positions, output)
        torch.cuda.synchronize()
        both_nan = torch.isnan(var) & torch.isnan(output)
        equal = bool(((var == output) | both_nan).all())
        diff = float((var.float() - output.float()).nan_to_num(0, 0, 0).abs().max()) if not equal else 0.0
        rec = {
            "layer": self.prefix,
            "kv_source": self.compressed_cache_prefix,
            "index_source": self.index_source_layer_id,
            "ratio": self.compress_ratio,
            "prefills": int(swa.num_prefills),
            "prefill_tokens": int(swa.num_prefill_tokens),
            "decodes": int(swa.num_decodes),
            "reuse_taken": bool(taken),
            "compressed_reused": bool(had_compressed and taken),
            "csr_reused": bool(had_csr and taken),
            "M": reuse.M if taken else None,
            "N": reuse.N if taken else None,
            "equal": equal,
            "max_abs_diff": diff,
            "nan": int(torch.isnan(output).sum()),
            "ms": round((time.time() - t0) * 1e3, 1),
        }
        _out().write(json.dumps(rec) + "\n")
        return None

    forward_mqa._p28_check = True
    C.forward_mqa = forward_mqa
    print(f"[p28_check] patched DeepseekV41ROCMAiterMLAAttention.forward_mqa in pid {os.getpid()}: "
          f"{FLAG} on then off, at most {MAX_CALLS} calls, to {OUT}", flush=True)


def install():
    import importlib.abc
    import importlib.util
    import sys

    if MODULE in sys.modules:
        _patch(sys.modules[MODULE])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        busy = False

        def find_spec(self, fullname, path, target=None):
            if fullname != MODULE or self.busy:
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
