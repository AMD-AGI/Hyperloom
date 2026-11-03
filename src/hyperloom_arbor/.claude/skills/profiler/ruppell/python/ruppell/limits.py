"""What a ruppell trace cannot be trusted for, in machine-readable form.

Every output (the Chrome trace, the TraceLens files, the census summary)
carries `limitations`: a list of {"id", "applies_to", "text"}, the ones that
hold for that run, and collective kernels carry `ruppell_limits` naming the
ones that apply to them. A reader that drops the metadata still has
LIMITATIONS.md beside the TraceLens files. The texts are docs/limitations.md's,
shortened; that page says why.

Collectives are named by kernel, one classifier for every output and check.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# RCCL/NCCL, vLLM's and AITER's custom all-reduce, quick reduce, MSCCL (RCCL's
# mscclKernel_*, DLRMv4's input all-to-all) and mscclpp, and the
# one-shot/two-shot all-reduce kernels FlashInfer and TRT-LLM ship
COLLECTIVE = re.compile(
    r"nccl|rccl|cross_device_reduce|all_?reduce|allgather|all_gather|reduce_?scatter|all_?to_?all|"
    r"quick_?reduce|msccl|custom_ar|one_?shot|two_?shot|lamport|mnnvl", re.IGNORECASE)


def is_collective(kernel_name: str) -> bool:
    return bool(COLLECTIVE.search(kernel_name or ""))


LIMITS: Dict[str, Dict[str, str]] = {
    "collective-wait": {
        "applies_to": "collective kernels",
        "text": ("A collective kernel's time is transfer plus waiting for the slowest rank. Tracing changes which "
                 "rank arrives last (M3, jobs 49062 and 49794), so per-rank collective time and rank skew in a "
                 "traced run do not describe the untraced server. Use the minimum across ranks per launch "
                 "(the transfer estimate) for collective cost; do not rank optimization sites by collective wait."),
    },
    "traced-latency": {
        "applies_to": "latency and throughput",
        "text": ("Tracing adds host time per traced dispatch (about 2 us on ROCm). Compare kernel time within a "
                 "run, not a traced run's latency or throughput with an untraced baseline."),
    },
    "detail-sampled": {
        "applies_to": "kernel records",
        "text": ("Detail sampling (RUPPELL_DETAIL=K/N): kernels are recorded in sampled steps only, plus startup, "
                 "capture and warmup. Totals over the run are over the sampled steps."),
    },
    "synthetic-cpu-intervals": {
        "applies_to": "host events of graph-replayed kernels",
        "text": ("A graph replay is one launch call; its interval is divided among the replay's kernels in node "
                 "order. Those CPU intervals are synthetic; eager launches keep their real ones."),
    },
    "module-ops": {
        "applies_to": "op names and shapes",
        "text": ("Without the CPU op tree, ops are modules (class or path) and the shape is the first input's; "
                 "FLOP and roofline models keyed on aten ops do not apply."),
    },
    "incomplete-records": {
        "applies_to": "the last seconds of the processes listed as incomplete_processes",
        "text": ("These processes ended without a final flush and without completing their adapter's flush at "
                 "shutdown: killed after the tool's last periodic flush (RUPPELL_FLUSH_S, default 5 s), they "
                 "lost what they recorded since, and their dropped_records does not count it. Their last steps "
                 "are missing or cut short, and ranks then disagree on their engine steps. vLLM kills its "
                 "workers this way with --shutdown-timeout 0 (its default): serve traced with "
                 "--shutdown-timeout 60, or stop after two flush intervals of idle."),
    },
    # birdeye's reports (ruppell/birdeye, docs/birdeye.md)
    "comm-by-name": {
        "applies_to": "communication time in birdeye's timeline and collective tables",
        "text": ("Collectives are recognized by kernel name (COLLECTIVE above, or TraceLens's nccl/rccl/"
                 "cross_device_reduce with --comm tracelens). A collective kernel whose name matches no pattern "
                 "counts as computation, and exposed communication is then understated."),
    },
    "categories-by-name": {
        "applies_to": "kernel categories (Parent op category, category_summary, ops_summary_by_category)",
        "text": ("Kernel categories come from kernel names (ruppell/birdeye/categories.py), never from what the "
                 "kernel computes: a kernel whose name gives no hint is 'other', and every top-k kernel is 'MoE' "
                 "whatever called it. Compare categories across runs, not as exact accounting."),
    },
    "gemm-needs-shapes": {
        "applies_to": "birdeye's GEMM table",
        "text": ("FLOPs and bytes need each GEMM op's input dims and dtypes: the CPU op tree (RUPPELL_CPU_OPS=1) "
                 "or a PyTorch profiler trace recorded with shapes. Without them the table is not written."),
    },
    "memcpy-unrecorded": {
        "applies_to": "memcpy and memset time of ruppell runs without copies.bin",
        "text": ("This run has no copy records (ruppell before copies.bin, 2026-10-20, or RUPPELL_COPIES=0): on "
                 "CUDA no memcpy, memset or graph memory node, on ROCm no copy the DMA engines ran (copy and fill "
                 "kernels are recorded). Their time counts as idle, and exposed and total memcpy time are "
                 "understated (0 on CUDA)."),
    },
    "counter-steps": {
        "applies_to": "times of ruppell runs with hardware counters (gray-parrot, RUPPELL_COUNTERS)",
        "text": ("Counting serializes a counter step's kernels, so the counter steps (RUPPELL_COUNTER_STEPS; "
                 "steps.tsv's `counters` field) are left out of every time: the engine's report, the census, "
                 "birdeye's tables and the exports. Their kernels are still recorded and attributed, and carry "
                 "the counters. Spans that include them (the timeline's total) count their time as idle. The step "
                 "after each is flagged, and left out too with RUPPELL_UNTIMED_AFTER_COUNTER=1. On ROCm, once a "
                 "process has counted, every later launch costs the launching thread more (graph launches about 2x "
                 "on MI350X, runner plan 03 probe A3c): its timing steps are not those of a run without counters."),
    },
    "counters-per-step-cuda": {
        "applies_to": "hardware counters of ruppell runs on CUDA (gray-parrot)",
        "text": ("On CUDA a counter step is counted as one CUPTI Range Profiler user range: each value is the "
                 "step's total (or its rate over the step), not a kernel's. Per kernel is not used: CUPTI 130001's "
                 "auto range skipped CUDA graph launches outright (B200, runner plan 04, probe A3); per-kernel "
                 "numbers come from Nsight Compute on a kernel bench. The profiler runs one WaitNs kernel per "
                 "counter step, outside every step, which the analysis names and leaves out. A counter step costs "
                 "the program 60-120 ms of host time, mostly the Range Profiler's start, which grows with the "
                 "program: at 1/100 a served Qwen3-8B lost 10 % of its throughput and its TTFT rose, so the "
                 "default on CUDA is 1/1000. Switching groups costs 15-55 ms more, so a group is kept for "
                 "RUPPELL_COUNTER_GROUP_STEPS counter steps (default 10), and a set's groups are all read only "
                 "after groups x 10 counter steps. tcgen05's fp8/fp6/fp4 tensor-op counter takes two passes "
                 "alone, so no counter step counts it."),
    },
    "device-samples-cuda": {
        "applies_to": "device activity of ruppell runs on CUDA (gray-parrot, RUPPELL_DEVICE_COUNTERS)",
        "text": ("Device activity is CUPTI PM sampling: the whole GPU, every process's work, at the sample interval. "
                 "A step gets the samples whose midpoint lies in its GPU span, so only steps with kernel records "
                 "(the detail sample) have any. It cannot run beside per-step counters, and at 10 kHz its thread uses "
                 "about 0.65 of a host core (B200, runner plan 04 Part D). Against Nsight Systems' GPU metrics it reads "
                 "within a few percent, a little lower."),
    },
    "software-timestamps": {
        "applies_to": "kernel durations of ruppell runs on CUDA",
        "text": ("On CUDA ruppell takes CUPTI's software timestamps (hardware ones lost graph-replayed records on "
                 "CUPTI 130001). Against Blackwell's hardware timestamps (nsys -t cuda) each kernel reads 0.2-0.5 us "
                 "longer: 1.1-1.3x on 1-3 us kernels, +2-6 % of kernel time on serving and training; against nsys "
                 "with software timestamps (-t cuda-sw), within 0.2 % (B200, runner ruppell-accuracy). Compare "
                 "short kernels' durations with a hardware-timestamped profile with that offset in mind."),
    },
    "nvtx-capture-time": {
        "applies_to": "NVTX (ROCTX) ranges of graph-replayed kernels",
        "text": ("A replayed kernel takes the range that was open when its graph was captured (its launch then), "
                 "not the one around the replay: a range pushed around each graph launch covers no replayed "
                 "kernel. Nsight Systems' NVTX projection (nvtx_gpu_proj_sum with --cuda-graph-trace=node, "
                 "2026.3.2 on B200) does the opposite: it counts a replay under the range around its launch, and "
                 "nothing under the ranges open at capture. The two agree on eagerly launched kernels. Start/end "
                 "ranges and marks attribute no kernel (nsys projects start/end ranges too), and ranges attribute "
                 "kernels, not copies."),
    },
    "nvtx-not-injected": {
        "applies_to": "NVTX ranges of ruppell runs on CUDA",
        "text": ("RUPPELL_NVTX was set but NVTX was not injected through ruppell (meta.json nvtx.source "
                 "\"not injected\"): another tool held NVTX_INJECTION64_PATH (nsys), the program made no NVTX "
                 "call, or it was not started by `ruppell run`. No range was recorded, whatever the program "
                 "annotated."),
    },
}


def for_run(meta: Optional[dict] = None, *, export: Optional[str] = None, cpu_ops: bool = False,
            incomplete: bool = False, copies: Optional[bool] = None) -> List[dict]:
    """The limitations that hold for a run (its tool meta) and an output kind
    (export="torch" for the PyTorch-profiler form); `incomplete` when a process
    with engine steps lost its last records (census incomplete_processes);
    `copies` False when the run has no copies.bin (None: not known here)."""
    meta = meta or {}
    ids = ["collective-wait", "traced-latency"]
    if meta.get("detail_toggles") or meta.get("detail_sampling"):
        ids.append("detail-sampled")
    if meta.get("platform") == "cuda":
        ids.append("software-timestamps")
        c = meta.get("counters")
        if isinstance(c, dict) and isinstance(c.get("device"), dict):
            ids.append("device-samples-cuda")
    if copies is False:
        ids.append("memcpy-unrecorded")
    nv = meta.get("nvtx") or {}
    if nv.get("enabled"):
        ids.append("nvtx-capture-time")
        if "not injected" in str(nv.get("source", "")):
            ids.append("nvtx-not-injected")
    if export == "torch":
        ids.append("synthetic-cpu-intervals")
    if not cpu_ops:
        ids.append("module-ops")
    if incomplete:
        ids.append("incomplete-records")
    return [{"id": i, **LIMITS[i]} for i in ids]


def markdown(limits: List[dict]) -> str:
    lines = ["# What these traces cannot be trusted for", "",
             "Written by ruppell beside its exported files; the same list is in each file's",
             "`otherData.ruppell.limitations`, and collective kernels carry `ruppell_limits`.", ""]
    for x in limits:
        lines += [f"## {x['id']} ({x['applies_to']})", "", x["text"], ""]
    return "\n".join(lines)
