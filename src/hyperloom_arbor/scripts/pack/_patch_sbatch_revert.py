#!/usr/bin/env python3
"""One-shot: add REVERT_PATCHES support to campaign_cg.sbatch.

Kept in-tree rather than run as a shell one-liner so the edit is reviewable
and re-runnable. Idempotent: exits 0 without changing anything if the block
is already present.
"""
import pathlib
import sys

p = pathlib.Path("scripts/dsv41-campaign/campaign_cg.sbatch")
t = p.read_text()

if "REVERT_PATCHES" in t:
    print("already patched")
    sys.exit(0)

OLD = '''        log "  landed.patch present -- restoring THAT instead (it contains the seed)"
        apply_one "$MY_SESSION/landed.patch" "cumulative landed tree" \\
            || { log "FATAL: landed.patch did not apply; refusing to run against a"
                 log "       tree that does not match session.json's best_metric."; exit 1; }'''

NEW = '''        log "  landed.patch present -- restoring THAT instead (it contains the seed)"
        apply_one "$MY_SESSION/landed.patch" "cumulative landed tree" \\
            || { log "FATAL: landed.patch did not apply; refusing to run against a"
                 log "       tree that does not match session.json's best_metric."; exit 1; }

        # REVERT_PATCHES: iteration patches to REVERSE-apply after the
        # cumulative restore. landed.patch is a single cumulative diff, so a
        # KEEP cannot be removed from it by text surgery -- but each
        # iteration's own patch.diff is kept, and reversing it is exact.
        #
        # Used to drop a KEEP that should not have shipped. Iteration 35 put a
        # per-tuple dispatch census INSIDE the served decode path, booked it as
        # a class-A neutral, and cost p90 for no p50 gain. The pack's
        # secondary band has since been tightened to 2.5 so the gate refuses
        # that shape, but the tree still carries the change.
        #
        # SAFE BECAUSE best_metric IS UNAFFECTED: the last performance KEEP was
        # iteration 26 (358.28707) and iteration 35 was neutral, so reversing it
        # returns the tree to the state the bar already describes. Reversing a
        # patch that DID advance best_metric would leave session.json claiming a
        # number no tree produces -- do not do that without also clearing the bar.
        for rp in ${REVERT_PATCHES:-}; do
            src="$MY_SESSION/iter_${rp}/patch.diff"
            if [ ! -s "$src" ]; then
                log "FATAL: REVERT_PATCHES names iter_$rp but $src is missing."
                log "       Refusing to start with a tree that does not match intent."
                exit 1
            fi
            if docker exec -i "$CONTAINER" bash -lc "cd $W && git apply --reverse --whitespace=nowarn -" < "$src" 2>/dev/null; then
                docker exec "$CONTAINER" bash -lc "cd $W && git add -A && git commit -q -m 'revert: iteration $rp'" 2>/dev/null
                log "  REVERSED iteration $rp"
            else
                log "FATAL: could not reverse iter_$rp. A half-reverted tree is"
                log "       worse than either endpoint; refusing to continue."
                exit 1
            fi
        done'''

if OLD not in t:
    print("anchor not found", file=sys.stderr)
    sys.exit(1)

p.write_text(t.replace(OLD, NEW, 1))
print("REVERT_PATCHES added")
