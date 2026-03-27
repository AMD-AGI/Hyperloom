###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch
from megatron.core.transformer.moe.moe_utils import apply_router_token_dropping
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from primus.backends.megatron.core.extensions.logits_processor import fused_softcap


class PrimusTopKRouter(TopKRouter):
    """Balanced route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)

    def fused_router_and_auxiliary_loss(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            import primus_turbo.pytorch as pt
        except ImportError as e:
            raise ImportError("Failed to import 'primus_turbo'. Please make sure it is installed. ") from e

        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        scores_for_aux_loss, probs, routing_map = pt.ops.fused_group_topk_routing_with_aux_score(
            logits,
            self.config.moe_router_topk,
            self.config.moe_router_num_groups,
            self.config.moe_router_group_topk,
            self.config.moe_router_score_function,
            self.config.moe_router_topk_scaling_factor,
        )

        routing_map = routing_map.bool()

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # todo: fuse the following logic into the turbo fused_group_topk_routing_with_aux_score OP
            # (the routing_map here differs from the one in the OP output is regarding less of group limit)
            if self.config.moe_router_num_groups is None or self.config.moe_router_num_groups <= 1:
                routing_map_for_aux_loss = routing_map
            else:
                _, top_indices_for_aux_loss = torch.topk(scores_for_aux_loss, k=self.topk, dim=1)
                routing_map_for_aux_loss = (
                    torch.zeros_like(logits).int().scatter(1, top_indices_for_aux_loss, 1).bool()
                )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        # Update expert bias and tokens_per_expert
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        args = get_args()

        if args.router_logit_softcapping is not None and args.router_logit_softcapping > 0.0:
            # grok2 router logit softcapping
            fused_softcap(logits, args.router_logit_softcapping)

        if args.enable_primus_turbo and args.moe_use_fused_router_with_aux_score:
            scores, routing_map = self.fused_router_and_auxiliary_loss(logits)
        else:
            scores, routing_map = super().routing(logits)

        assert routing_map.dtype == torch.bool, "routing_map should be boolean"
        # profile for moe
        if args.moe_router_force_load_balancing:
            indices = (
                torch.arange(routing_map.size(0) * self.topk, device=routing_map.device).view(
                    routing_map.size(0), self.topk
                )
                % self.num_experts
            )
            row = torch.arange(routing_map.size(0), device=routing_map.device).repeat_interleave(self.topk)
            col = indices.view(-1)
            routing_map = torch.zeros_like(routing_map, dtype=torch.bool).index_put_(
                (row, col), torch.ones(1, device=routing_map.device, dtype=torch.bool)
            )

        return scores, routing_map
