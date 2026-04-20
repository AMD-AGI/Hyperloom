# Action: Communication Optimization

Optimize inter-GPU and inter-node communication: topology-aware algorithm selection,
compute-communication overlap, collective tuning, and multi-node scaling. This action
becomes increasingly important as GPU count grows (critical at >1 node, dominant at
100s-1000s of GPUs).

## Inputs
- Profile trace showing communication kernels and their GPU time %
- Cluster topology (intra-node interconnect, inter-node fabric)
- Current TP/PP/DP/EP configuration
- `state.tier_breakdown['T4_communication']` from profile

## Procedure

### Step 1: Communication audit

Identify all communication operations and their cost:

```bash
# a) Extract communication kernels from trace
# Common patterns by library:
#   RCCL/NCCL: ncclAllReduce, ncclAllGather, ncclReduceScatter, ncclAlltoAll
#   Custom: allreduce_*, all_gather_*, reduce_scatter_*
#   MPI: MPI_Allreduce, MPI_Allgather

python3 -c "
import json

# Parse profiler trace for comm kernels
# (framework-specific trace parsing)

# Categorize by type:
# allreduce  → attention output, MoE expert aggregation
# allgather  → tensor parallel weight gathering
# reduce_scatter → gradient reduction
# all-to-all → expert routing in MoE

comm_ops = [
    # {'name': 'ncclAllReduce', 'gpu_time_ms': 1.2, 'count': 48, 'message_size_bytes': ...}
]
total_comm_pct = sum(op['gpu_time_ms'] for op in comm_ops) / total_gpu_time * 100
print(f'Total communication: {total_comm_pct:.1f}% GPU time')

# Group by operation type
from collections import defaultdict
by_type = defaultdict(list)
for op in comm_ops:
    by_type[op['type']].append(op)
for typ, ops in by_type.items():
    time = sum(o['gpu_time_ms'] for o in ops)
    print(f'  {typ}: {time:.1f}ms ({len(ops)} calls)')
"
```

### Step 2: Topology discovery

```bash
# a) Discover GPU-to-GPU connectivity
# For AMD ROCm:
rocm-smi --showtopoweight 2>/dev/null || echo "No rocm-smi"

# For NVIDIA:
nvidia-smi topo -m 2>/dev/null || echo "No nvidia-smi"

# b) Discover inter-node fabric
# Check for InfiniBand / RoCE
ibstat 2>/dev/null | head -20 || echo "No IB"

# Check NCCL/RCCL topology detection
echo "$NCCL_TOPO_FILE"
echo "$RCCL_TOPO_FILE"

# c) Record topology
python3 -c "
topology = {
    'intra_node': {
        'type': 'xGMI | NVLink | PCIe',
        'bandwidth_gb_s': 0,  # bidirectional
        'gpus_per_node': 0,
    },
    'inter_node': {
        'type': 'InfiniBand | RoCE | TCP',
        'bandwidth_gb_s': 0,
        'latency_us': 0,
        'num_nodes': 0,
    },
}
print(topology)
"
```

### Step 3: Algorithm selection tuning

Different algorithms are optimal for different message sizes and topologies:

```bash
# a) Check current NCCL/RCCL algorithm selection
echo "NCCL_ALGO=$NCCL_ALGO"
echo "RCCL_ALGO=$RCCL_ALGO"
echo "NCCL_PROTO=$NCCL_PROTO"

# b) For each communication operation, determine optimal algorithm:
#
# Small messages (<256KB):
#   - Ring: good latency, poor bandwidth utilization
#   - Tree: lower latency at scale
#
# Large messages (>1MB):
#   - Ring: good bandwidth utilization
#   - Recursive halving-doubling: balanced
#   - Direct: if topology supports it (e.g., fully connected)
#
# MoE All-to-All:
#   - Custom implementations often beat generic NCCL all-to-all
#   - Check framework for fused expert routing + comm

# c) Try different algorithm/protocol combinations
ALGO_OPTIONS=("Tree" "Ring" "CollnetDirect" "CollnetChain")
PROTO_OPTIONS=("Simple" "LL" "LL128")

for algo in "${ALGO_OPTIONS[@]}"; do
    for proto in "${PROTO_OPTIONS[@]}"; do
        echo "Testing NCCL_ALGO=$algo NCCL_PROTO=$proto"
        NCCL_ALGO=$algo NCCL_PROTO=$proto bash "$SKILL_ROOT/scripts/run_benchmark.sh"
    done
done
```

### Step 4: Compute-communication overlap

```bash
# a) Check if the framework supports overlap
# Common overlap mechanisms:
#   - Pipeline parallelism: overlaps forward/backward between stages
#   - Tensor parallelism: overlaps allreduce with next layer's compute
#   - Expert parallelism: overlaps all-to-all with expert compute
#   - Data parallelism: overlaps gradient allreduce with backward

# b) Check for overlap-related flags in the framework
rg "overlap|async_op|async_comm|pipeline_overlap" --type py "$FRAMEWORK_ROOT/" | head -20

# c) Enable overlap if available
# Framework-specific flags:
# sglang: --overlap-scheduler
# vllm: various internal overlap mechanisms
# megatron: --overlap-grad-reduce, --overlap-param-gather
```

### Step 5: Multi-node scaling optimizations (if num_nodes > 1)

```bash
# a) Check parallelism strategy
# For large-scale deployment:
#   TP (tensor parallel): within a node (high-bandwidth intra-node)
#   EP (expert parallel): across nodes for MoE (all-to-all)
#   PP (pipeline parallel): across nodes (point-to-point)
#   DP (data parallel): across nodes (allreduce)

echo "Current: TP=$TP PP=$PP DP=$DP EP=$EP"

# b) Evaluate if current strategy is topology-optimal
# Rule of thumb:
#   TP should NOT cross node boundaries (bandwidth-hungry)
#   EP all-to-all benefits from high-bandwidth interconnect
#   PP is latency-sensitive, keep pipeline bubbles small

# c) Test alternative parallelism configurations if topology allows
# (only if the framework supports dynamic reconfiguration)
```

### Step 6: Collective tuning environment

```bash
# NCCL/RCCL tuning knobs:
export NCCL_MIN_NCHANNELS=4          # min channels for collectives
export NCCL_MAX_NCHANNELS=16         # max channels
export NCCL_BUFFSIZE=8388608         # 8MB buffer per channel
export NCCL_NET_GDR_LEVEL=5          # GPU Direct RDMA level
export NCCL_IB_GID_INDEX=3           # IB GID index for RoCE
export NCCL_SOCKET_NTHREADS=4        # TCP transport threads
export NCCL_NSOCKS_PERTHREAD=4       # sockets per thread

# Framework-specific:
export RCCL_MSCCL_ENABLE=1           # MS collective communication library
export RCCL_ENABLE_HIPGRAPH=1        # fuse comm ops into HIP graph

# Test each knob individually, then combine winners
```

## Outputs
- Communication audit report
- Topology map
- Optimized NCCL/RCCL environment variables
- E2E benchmark results with comm optimizations
- Updated `state.tier_breakdown['T4_communication']`

## Heuristic Update

- **Comm optimization gains >2%:** Boost to try additional comm strategies 1.5×
- **Comm optimization gains ≤0%:** Reduce comm-optimization score by 0.5×.
  If total comm is <5% GPU time, zero the score (compute-bound, not comm-bound).
- **Multi-node specific gains:** Log topology + algorithm combination to KB.
  This is high-value knowledge for scaling to similar clusters.
- **Overlap enabled successfully:** Log to KB. Boost framework-level optimizations.

## Scaling Notes

Communication optimization has non-linear scaling impact:
- **1 node (8 GPUs):** Comm is typically <10% GPU time. Focus elsewhere.
- **2-8 nodes (16-64 GPUs):** Comm starts mattering. Algorithm selection helps.
- **16+ nodes (128+ GPUs):** Comm can be 20-40% of time. Critical to optimize.
- **100+ nodes (1000+ GPUs):** Comm dominates. Topology-aware routing, overlap,
  and collective algorithm selection are the primary optimization vectors.

Score comm-optimization higher when `num_nodes > 1` and progressively higher
as node count increases.
