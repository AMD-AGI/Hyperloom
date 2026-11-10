"""Cross-entropy over the LM head's logits, forward and backward, as Megatron's native fused path runs it.

This is the starting kernel: Megatron-LM's ``fused_vocab_parallel_cross_entropy`` at tensor-parallel size 1
(``cross_entropy_loss_fusion: true`` with ``cross_entropy_fusion_impl: native``, Primus' default). It casts the
bf16 logits to fp32, takes the row max, exponentiates in place, sums, and keeps the fp32 softmax for the
backward, which subtracts one at the target, scales by the incoming gradient and casts back to bf16.

Interface (the driver calls only these two):

* ``forward(logits, target) -> (loss, ctx)``: ``logits`` is ``[tokens, vocab]`` bf16, contiguous; ``target`` is
  ``[tokens]`` int64. ``loss`` is ``[tokens]`` fp32, ``log(sum(exp(logits))) - logits[target]`` per row. ``ctx``
  is whatever the backward needs.
* ``backward(ctx, grad_loss) -> grad_logits``: ``grad_loss`` is ``[tokens]`` fp32; ``grad_logits`` is
  ``[tokens, vocab]`` bf16, ``(softmax(logits) - onehot(target)) * grad_loss[:, None]``.

The logits are dead after the loss: the forward may keep them in ``ctx`` and the backward may overwrite them
in place and return them as ``grad_logits``.
"""

import torch


@torch.compile
def _logits_max(logits):
    logits = logits.float()
    return logits, torch.max(logits, dim=-1)[0]


@torch.compile
def _predicted_and_exp(logits, target, logits_max):
    rows = torch.arange(logits.size(0), device=logits.device)
    predicted = logits[rows, target] - logits_max
    logits.sub_(logits_max.unsqueeze(-1))
    torch.exp(logits, out=logits)
    return predicted, logits.sum(dim=-1)


@torch.compile
def _loss_and_softmax(exp_logits, predicted, sum_exp):
    loss = torch.log(sum_exp) - predicted
    exp_logits.div_(sum_exp.unsqueeze(-1))
    return exp_logits, loss


@torch.compile
def _gradients(softmax, grad_loss, target):
    rows = torch.arange(softmax.size(0), device=softmax.device)
    softmax[rows, target] -= 1.0
    softmax.mul_(grad_loss.unsqueeze(-1))
    return softmax


def forward(logits, target):
    fp32, logits_max = _logits_max(logits)
    predicted, sum_exp = _predicted_and_exp(fp32, target, logits_max)
    softmax, loss = _loss_and_softmax(fp32, predicted, sum_exp)
    return loss, (softmax, target, logits.dtype)


def backward(ctx, grad_loss):
    softmax, target, dtype = ctx
    return _gradients(softmax, grad_loss, target).to(dtype)
