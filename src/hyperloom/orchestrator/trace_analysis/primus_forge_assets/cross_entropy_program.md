# Task: the LM head's cross-entropy of a Primus training step, forward and backward, on {gpu}

Primus' projection of `{config}` ({deployment}) puts this kernel at {share_pct:.1f}% of the iteration
({est_ms:.1f} ms of {iteration_ms:.1f} ms): {phase_lines}. The times come from a measured profile of the stack.

## What it computes

`kernel.py` exposes `forward(logits, target) -> (loss, ctx)` and `backward(ctx, grad_loss) -> grad_logits`;
its docstring is the contract. `logits` is `[tokens, vocab]` bf16 (vocab = {vocab}), `target` `[tokens]`
int64, `loss` `[tokens]` fp32, `grad_logits` `[tokens, vocab]` bf16 = `(softmax(logits) - onehot(target)) *
grad_loss[:, None]`.

The starting `kernel.py` is what Primus runs today: Megatron-LM's native fused path, which makes an fp32 copy
of the logits, passes over it several times and keeps the fp32 softmax for the backward. The logits are dead
once the loss is computed: the forward may keep them, and the backward may overwrite them in place and return
them as the gradient (as Transformer Engine's `parallel_cross_entropy` does). Nothing else may change: the
loss and the gradient are compared against an fp32 reference.

## How it is scored

`driver.py` (do not edit it) checks every candidate on {check_cases} and times forward plus backward on
{cases}. The score is Megatron's time over the candidate's, per case. `phase_ms:` lines show the two
directions apart; both count, since Primus charges each its own share.

The floor is HBM traffic: the bf16 logits are {logits_gb:.1f} GB at {tokens_max} tokens, so one read in the
forward and a read plus a write in the backward is about {sol_ms:.1f} ms at {hbm_tbps:.1f} TB/s.

## Where to look

Triton on {gpu} ({arch}). An online-softmax forward that reads each row once and saves only the row's
log-sum-exp, and a backward that recomputes the softmax from it and writes the gradient, is the shape a
fast kernel usually takes; vocab = {vocab} does not fit one tile, so each row is split across programs or
walked in blocks.
