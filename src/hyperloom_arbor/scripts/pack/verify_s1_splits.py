"""Evaluate the STOCK gfx950 split heuristic at the C4 DSpark verify shape.

Functions transcribed verbatim from the archived stock_base (which does not
move), not from the live container (which a V5 campaign is patching).
"""
import math

CU = 256  # verified: torch.cuda.get_device_properties(0).multi_processor_count


def _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k):
    main = math.ceil(math.ceil(avg_main_len / splits) / block_k) if avg_main_len > 0 else 0
    extra = math.ceil(math.ceil(avg_extra_len / splits) / block_k) if avg_extra_len > 0 else 0
    return main + extra


def num_splits_gfx950(num_queries, heads_blocks, avg_main_len=0.0,
                      avg_extra_len=0.0, block_k=32):
    base = max(1, num_queries * heads_blocks)
    cu = max(1, CU)
    target_workgroups = 2 * cu
    num_splits = min(32, max(1, math.ceil(target_workgroups / base)))
    gates = []
    if (base >= 16 and num_splits > 4
            and _decode_partial_iters(avg_main_len, avg_extra_len, 4, block_k) <= 3):
        return 4, base, ["fast-exit -> 4"]
    gates.append(f"fast-exit skipped (base>=16 is {base >= 16})")
    if 16 <= base < 64:
        one_wave_splits = max(1, cu // base)
        one_wave_iters = _decode_partial_iters(avg_main_len, avg_extra_len,
                                               one_wave_splits, block_k)
        target_waves = 1 if one_wave_iters <= 9 else 2
        num_splits = min(num_splits, max(1, target_waves * cu // base))
        gates.append("one-wave tuning APPLIED")
    else:
        gates.append(f"one-wave tuning skipped (16<=base<64 is {16 <= base < 64})")
    if base >= 16 and num_splits > 1:
        target_waves = (base * num_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(avg_main_len, avg_extra_len,
                                             num_splits, block_k)
        for splits in range(1, num_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                return splits, base, gates + [f"search -> {splits}"]
        gates.append("search found nothing")
    else:
        gates.append(f"search skipped (base>=16 is {base >= 16})")
    return num_splits, base, gates


NOPE, ROPE = 448, 64            # _DSV4_SPARSE_NOPE_DIM / _DSV4_SPARSE_ROPE_DIM
COMB = NOPE + ROPE              # 512
HEADS_PER_RANK = 64 // 4        # 64 attention heads, TP=4
BLOCK_H = 16
HEADS_BLOCKS = math.ceil(HEADS_PER_RANK / BLOCK_H)

print(f"heads/rank={HEADS_PER_RANK} block_h={BLOCK_H} -> heads_blocks={HEADS_BLOCKS}")
print(f"comb_dim = {NOPE} + {ROPE} = {COMB}\n")

for avg_main in (512.0,):
    print(f"--- avg_main_len = {avg_main} (top-k selection width), block_k=32 ---")
    for nq, label in ((6, "C4 verify: 1 req x 6 rows"),
                      (12, "C4: 2 reqs x 6 rows"),
                      (24, "4 reqs x 6 rows"),
                      (192, "C32 verify: 32 reqs x 6 rows")):
        s, base, gates = num_splits_gfx950(nq, HEADS_BLOCKS, avg_main)
        iters = _decode_partial_iters(avg_main, 0.0, s, 32)
        lanes = math.ceil(avg_main / s)
        acc_mb = nq * s * HEADS_PER_RANK * COMB * 4 / 1e6
        print(f"  nq={nq:3d} base={base:3d} -> splits={s:2d} iters={iters} "
              f"tokens/split={lanes:5.1f} part_acc={acc_mb:6.2f} MB   [{label}]")
        for g in gates:
            print(f"        {g}")

print("\n--- what a smaller split count would give at the C4 verify shape ---")
for s in (32, 16, 8, 4, 2, 1):
    iters = _decode_partial_iters(512.0, 0.0, s, 32)
    acc_mb = 6 * s * HEADS_PER_RANK * COMB * 4 / 1e6
    waves = (6 * s + CU - 1) // CU
    print(f"  splits={s:2d}  iters={iters}  waves={waves}  "
          f"tokens/split={math.ceil(512/s):5.1f}  part_acc={acc_mb:5.2f} MB")
