#!/bin/bash
# Run this first on the new node to restore tuned configs and verify environment
set -e

echo "=== Restoring tuned aiter configs ==="
AITER_CONFIGS="/sgl-workspace/aiter/aiter/configs"
BACKUP_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization"

# Restore tuned GEMM configs
if [ -f "$BACKUP_DIR/aiter_a8w8_blockscale_tuned_gemm_merged.csv" ]; then
    cp "$BACKUP_DIR/aiter_a8w8_blockscale_tuned_gemm_merged.csv" "$AITER_CONFIGS/a8w8_blockscale_tuned_gemm.csv"
    echo "  Restored a8w8_blockscale_tuned_gemm.csv ($(wc -l < "$AITER_CONFIGS/a8w8_blockscale_tuned_gemm.csv") lines)"
fi

# Restore tuned FMoE configs
if [ -f "$BACKUP_DIR/aiter_tuned_fmoe_merged.csv" ]; then
    cp "$BACKUP_DIR/aiter_tuned_fmoe_merged.csv" "$AITER_CONFIGS/tuned_fmoe.csv"
    echo "  Restored tuned_fmoe.csv ($(wc -l < "$AITER_CONFIGS/tuned_fmoe.csv") lines)"
fi

# Also copy to /tmp/aiter_configs if it exists
mkdir -p /tmp/aiter_configs 2>/dev/null || true
cp "$AITER_CONFIGS/a8w8_blockscale_tuned_gemm.csv" /tmp/aiter_configs/ 2>/dev/null || true
cp "$AITER_CONFIGS/tuned_fmoe.csv" /tmp/aiter_configs/ 2>/dev/null || true
echo "  Also copied to /tmp/aiter_configs/"

# Kill any leftover processes
pkill -9 -f sglang 2>/dev/null || true
echo ""
echo "=== Verify GLM-5 tuned shapes ==="
python3 -c "
import pandas as pd
# Check GEMM
df = pd.read_csv('$AITER_CONFIGS/a8w8_blockscale_tuned_gemm.csv')
glm5 = df[(df['N']==6144) & (df['K'].isin([6144,3072]))]
print(f'GEMM: {len(glm5)} GLM-5 shapes in {len(df)} total')
# Check FMoE
df2 = pd.read_csv('$AITER_CONFIGS/tuned_fmoe.csv')
glm5_moe = df2[(df2['model_dim']==6144) & (df2['expert']==257)]
print(f'FMoE: {len(glm5_moe)} GLM-5 shapes in {len(df2)} total')
"

echo ""
echo "=== Environment check ==="
echo "GPUs: $(rocm-smi --showid 2>/dev/null | grep -c GPU || echo 'N/A')"
echo "Model: $(ls /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX/models/zai-org/GLM-5-FP8/config.json 2>/dev/null && echo 'FOUND' || echo 'MISSING')"
echo ""
echo "=== Ready to resume! ==="
echo "Read RESUME_STATE.md for full context"
echo "Next: run combined_best experiment (NSA aiter + mixed-chunk + ds=16)"
