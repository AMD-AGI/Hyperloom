#!/bin/bash
# Generalized DETERMINISTIC kernel-usage proof (runs in container). For each kept tuned shape in
# $OUTDIR/tune/results/final_results.csv, bench merged vs stock lib with the shape's EXACT types and check
# the merged lib selects a different (tuned) kernel. A shape that "TUNED-SELECTED" fires the tuned kernel
# every time serving dispatches it. Writes $OUTDIR/selection_check.txt. Independent of trace-window luck.
set -uo pipefail
WS="${WS:?set WS to your geko_agentx workspace root}"
OUTDIR=${OUTDIR:?need OUTDIR}
HB=$WS/codebases/rocm-libraries/projects/hipblaslt
BENCH=$HB/build/release/clients/hipblaslt-bench
STOCK=$WS/codebases/rocm-libraries/build/projects/hipblaslt/Tensile/library/gfx950
MERGED=$OUTDIR/merged_devlib/library/gfx950
export HIP_VISIBLE_DEVICES=0
FR=$OUTDIR/tune/results/final_results.csv
OUT=$OUTDIR/selection_check.txt
[ -f "$FR" ] || { echo "SELECTION: no final_results.csv" | tee "$OUT"; exit 0; }
[ -d "$MERGED" ] || { echo "SELECTION: no merged lib" | tee "$OUT"; exit 0; }

mapfile -t ROWS < <(python3 -c "
import csv
r=list(csv.reader(open('$FR'))); h=r[0]
c={n:h.index(n) for n in ['m','n','k','transA','transB','a_type','b_type','c_type','d_type','compute_type']}
for row in r[1:]:
    cp=row[c['compute_type']].replace('c_','')
    print(row[c['m']],row[c['n']],row[c['k']],row[c['transA']],row[c['transB']],row[c['a_type']],row[c['b_type']],row[c['c_type']],row[c['d_type']],cp)
")
sel(){ HIPBLASLT_TENSILE_LIBPATH="$1" timeout 90 "$BENCH" -m "$2" -n "$3" -k "$4" --transA "$5" --transB "$6" \
  --a_type "$7" --b_type "$8" --c_type "$9" --d_type "${10}" --compute_type "${11}" \
  --print_kernel_info --device 0 2>&1 | grep -oiE 'Cijk_[A-Za-z0-9_]+' | head -1; }
: > "$OUT"
diff=0; same=0; err=0; tot=0
for line in "${ROWS[@]}"; do
  read -r M N K TA TB A B Cc D CP <<< "$line"; tot=$((tot+1))
  s=$(sel "$STOCK" "$M" "$N" "$K" "$TA" "$TB" "$A" "$B" "$Cc" "$D" "$CP")
  m=$(sel "$MERGED" "$M" "$N" "$K" "$TA" "$TB" "$A" "$B" "$Cc" "$D" "$CP")
  if [ -z "$s" ] || [ -z "$m" ]; then err=$((err+1)); echo "M=$M N=$N  BENCH_ERR" | tee -a "$OUT"
  elif [ "$s" != "$m" ]; then diff=$((diff+1)); echo "M=$M N=$N  TUNED-SELECTED  merged=MT${m##*UserArgs_MT}" | tee -a "$OUT"
  else same=$((same+1)); echo "M=$M N=$N  same-as-stock" | tee -a "$OUT"; fi
done
echo "SELECTION_SUMMARY: tuned-selected=$diff same-as-stock=$same bench_err=$err of $tot" | tee -a "$OUT"
echo "VERIFY_SELECTION_DONE"
