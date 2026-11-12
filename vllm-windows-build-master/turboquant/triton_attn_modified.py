# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    get_turboquant_norm_lut,
    turboquant_decode_attention_fwd,
)
from vllm.v1.attention.ops.triton_turboquant_kv_update import (
    turboquant_write_packed_kv,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.ops.turboquant_kv_cache import (
    TurboQuantLayout,
    get_turboquant_bits,
    get_turboquant_centroids,
    get_turboquant_group_dims,
    get_turboquant_layout,
    get_turboquant_mse_inverse_transform_matrix,
    get_turboquant_mse_to_qjl_matrix,
    get_turboquant_mse_transform_matrix,
    get_turboquant_packed_dim,
    get_turboquant_qjl_inverse_transform_matrix,
    get_turboquant_qjl_matrix,
    get_turboquant_qjl_transform_matrix,
    get_turboquant_rotation,
    is_turboquant_kv_cache,
)
from vllm.v1.attention.ops.turboquant_metadata import (
    TurboQuantLayerMetadata,
    TurboQuantMetadata,
    discover_turboquant_metadata_path,
    load_turboquant_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
GB10_CAPABILITY = DeviceCapability(12, 1)
# The Triton prefill kernel already handles power-of-two head sizes up to 256.
# Keep TurboQuant prefill on that kernel for common Qwen 27B/35B-class models
# instead of dropping to the slow Python fallback on the first prompt.
TURBOQUANT_TRITON_PREFILL_MAX_HEAD_SIZE = 256


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    turboquant_seq_ids: torch.Tensor | None = None
    turboquant_token_seq_lens: torch.Tensor | None = None
    turboquant_query_positions: torch.Tensor | None = None
    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: torch.Tensor | None = None
    causal: bool = True

    @property
    def mm_prefix_range_tensor(self) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        # TODO(Isotr0py): Move to model runner's attention metadata
        # preparation to avoid duplicate computation.
        if self.mm_prefix_range is None:
            return None

        num_seqs = self.seq_lens.shape[0]
        device = self.seq_lens.device

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            self.mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Create 2D tensors with shape (num_ranges, 2) for each sequence
        range_tensors = [
            torch.tensor(r, dtype=torch.int32, device=device).view(-1, 2)
            for r in range_lists
        ]

        return torch.nested.nested_tensor(
            range_tensors, layout=torch.jagged
        ).to_padded_tensor(0)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        cache_dtype_str = getattr(kv_cache_spec, "cache_dtype_str", None)
        if cache_dtype_str is None and hasattr(kv_cache_spec, "kv_cache_specs"):
            cache_dtype_strs = {
                getattr(spec, "cache_dtype_str", None)
                for spec in kv_cache_spec.kv_cache_specs.values()
            }
            if any(
                dtype is not None and is_turboquant_kv_cache(dtype)
                for dtype in cache_dtype_strs
            ):
                return AttentionCGSupport.NEVER
        elif cache_dtype_str is not None and is_turboquant_kv_cache(cache_dtype_str):
            return AttentionCGSupport.NEVER

        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
            encoder_seq_lens=common_attn_metadata.encoder_seq_lens,
            encoder_seq_lens_cpu=(
                None
                if common_attn_metadata.encoder_seq_lens_cpu is None
                else torch.as_tensor(common_attn_metadata.encoder_seq_lens_cpu)
            ),
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "turboquant25",
        "turboquant35",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if is_turboquant_kv_cache(cache_dtype_str):
            bits = get_turboquant_bits(cache_dtype_str)
            packed_dim = get_turboquant_packed_dim(head_size, bits)
            return (num_blocks, 2, block_size, num_kv_heads, packed_dim)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        if include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)

        # (num_blocks, 2, block_size, num_kv_heads, head_size)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype is None or not is_turboquant_kv_cache(kv_cache_dtype):
            return None
        if not current_platform.is_cuda():
            return "TurboQuant KV cache requires CUDA"
        if device_capability != GB10_CAPABILITY:
            return "TurboQuant KV cache requires NVIDIA GB10 / SM121"
        return None


class TritonAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        turboquant_layer_name: str | None = None,
        turboquant_model_name: str | None = None,
        turboquant_metadata_path: str | None = None,
        turboquant_metadata: TurboQuantMetadata | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()
        self.turboquant_bits = (
            get_turboquant_bits(kv_cache_dtype)
            if is_turboquant_kv_cache(kv_cache_dtype)
            else None
        )
        self._turboquant_tables: dict[
            tuple[str, int | None],
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                dict[int, torch.Tensor],
                torch.Tensor,
                TurboQuantLayout,
            ],
        ] = {}
        self._turboquant_masks: dict[
            tuple[str, int | None],
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}
        self._turboquant_query_group_indices: dict[
            tuple[str, int | None],
            tuple[
                torch.Tensor,
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}
        self._turboquant_update_tables: dict[
            tuple[str, int | None],
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}
        self._turboquant_decode_tables: dict[
            tuple[str, int | None],
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}
        self._turboquant_metadata = turboquant_metadata
        self._turboquant_layer_name = turboquant_layer_name
        self._turboquant_layer_metadata: TurboQuantLayerMetadata | None = None

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.supports_quant_query_input = current_platform.is_cuda()

        if self.turboquant_bits is not None:
            capability = current_platform.get_device_capability()
            if not current_platform.is_cuda() or capability != GB10_CAPABILITY:
                raise RuntimeError("TurboQuant KV cache requires NVIDIA GB10 / SM121.")
            if turboquant_layer_name is None:
                raise ValueError(
                    "TurboQuant KV cache requires the attention layer name for "
                    "metadata lookup."
                )
            if self._turboquant_metadata is None:
                resolved_metadata_path = discover_turboquant_metadata_path(
                    turboquant_model_name,
                    turboquant_metadata_path,
                )
                if resolved_metadata_path is None:
                    raise ValueError(
                        "TurboQuant KV cache requires metadata. Pass "
                        "`turboquant_metadata_path` or place `turboquant_kv.json` "
                        "under the local model path."
                    )
                self._turboquant_metadata = load_turboquant_metadata(
                    resolved_metadata_path
                )
                logger.info_once(
                    "Resolved TurboQuant metadata from %s.",
                    resolved_metadata_path,
                    scope="local",
                )
            if self._turboquant_metadata.recipe != self.kv_cache_dtype:
                raise ValueError(
                    "TurboQuant metadata recipe does not match kv_cache_dtype: "
                    f"{self._turboquant_metadata.recipe} vs {self.kv_cache_dtype}."
                )
            if self._turboquant_metadata.head_size != self.head_size:
                raise ValueError(
                    "TurboQuant metadata head_size does not match layer head size: "
                    f"{self._turboquant_metadata.head_size} vs {self.head_size}."
                )
            layer_metadata = self._turboquant_metadata.get_layer(turboquant_layer_name)
            self._turboquant_layer_metadata = layer_metadata
            if len(layer_metadata.key.high_precision_indices) != self.num_kv_heads:
                raise ValueError(
                    "TurboQuant metadata key KV head count does not match layer: "
                    f"{len(layer_metadata.key.high_precision_indices)} vs "
                    f"{self.num_kv_heads}."
                )
            if len(layer_metadata.value.high_precision_indices) != self.num_kv_heads:
                raise ValueError(
                    "TurboQuant metadata value KV head count does not match "
                    f"layer: {len(layer_metadata.value.high_precision_indices)} vs "
                    f"{self.num_kv_heads}."
                )
            logger.info_once(
                "TurboQuant enabled for layer %s with %s on Triton/GB10.",
                turboquant_layer_name,
                self.kv_cache_dtype,
                scope="local",
            )

    def _get_turboquant_tables(
        self,
        device: torch.device,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        dict[int, torch.Tensor],
        torch.Tensor,
        TurboQuantLayout,
    ]:
        assert self.turboquant_bits is not None
        cache_key = (device.type, device.index)
        tables = self._turboquant_tables.get(cache_key)
        if tables is None:
            group_dims = get_turboquant_group_dims(self.head_size, self.kv_cache_dtype)
            layout = get_turboquant_layout(self.kv_cache_dtype, self.head_size)
            centroid_dims = {
                group.mse_bits: group.dim
                for group in layout.groups
                if group.mse_bits > 0
            }
            centroids = {
                mse_bits: get_turboquant_centroids(
                    device,
                    dim,
                    mse_bits,
                )
                for mse_bits, dim in centroid_dims.items()
            }
            tables = (
                (
                    get_turboquant_rotation(device, group_dims[0], seed_offset=101),
                    get_turboquant_rotation(device, group_dims[1], seed_offset=211),
                ),
                (
                    get_turboquant_qjl_matrix(device, group_dims[0], seed_offset=307),
                    get_turboquant_qjl_matrix(device, group_dims[1], seed_offset=401),
                ),
                centroids,
                get_turboquant_norm_lut(device),
                layout,
            )
            self._turboquant_tables[cache_key] = tables
        return tables

    def _validate_turboquant_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TurboQuant KV cache requires CUDA.")
        if torch.cuda.get_device_capability(device) != (12, 1):
            raise RuntimeError("TurboQuant KV cache requires NVIDIA GB10 / SM121.")

    def _get_turboquant_update_tables(
        self,
        device: torch.device,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        cache_key = (device.type, device.index)
        tables = self._turboquant_update_tables.get(cache_key)
        if tables is not None:
            return tables

        group_dims = get_turboquant_group_dims(self.head_size, self.kv_cache_dtype)
        mse_seed_offsets = (101, 211)
        qjl_seed_offsets = (307, 401)
        tables = (
            tuple(
                get_turboquant_mse_transform_matrix(device, dim, seed_offset)
                for dim, seed_offset in zip(group_dims, mse_seed_offsets, strict=True)
            ),
            tuple(
                get_turboquant_qjl_transform_matrix(device, dim, seed_offset)
                for dim, seed_offset in zip(group_dims, qjl_seed_offsets, strict=True)
            ),
            tuple(
                get_turboquant_mse_to_qjl_matrix(
                    device,
                    dim,
                    mse_seed_offset,
                    qjl_seed_offset,
                )
                for dim, mse_seed_offset, qjl_seed_offset in zip(
                    group_dims,
                    mse_seed_offsets,
                    qjl_seed_offsets,
                    strict=True,
                )
            ),
        )
        self._turboquant_update_tables[cache_key] = tables
        return tables

    def _get_turboquant_decode_tables(
        self,
        device: torch.device,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        cache_key = (device.type, device.index)
        tables = self._turboquant_decode_tables.get(cache_key)
        if tables is not None:
            return tables

        group_dims = get_turboquant_group_dims(self.head_size, self.kv_cache_dtype)
        mse_seed_offsets = (101, 211)
        qjl_seed_offsets = (307, 401)
        tables = (
            tuple(
                get_turboquant_mse_inverse_transform_matrix(device, dim, seed_offset)
                for dim, seed_offset in zip(group_dims, mse_seed_offsets, strict=True)
            ),
            tuple(
                get_turboquant_qjl_inverse_transform_matrix(device, dim, seed_offset)
                for dim, seed_offset in zip(group_dims, qjl_seed_offsets, strict=True)
            ),
        )
        self._turboquant_decode_tables[cache_key] = tables
        return tables

    def _ensure_turboquant_masks(
        self,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (device.type, device.index)
        masks = self._turboquant_masks.get(cache_key)
        if masks is not None:
            return masks
        if self._turboquant_layer_metadata is None:
            raise RuntimeError("TurboQuant metadata is not initialized.")
        masks = (
            self._turboquant_layer_metadata.key.get_group_indices(
                device=device,
                head_size=self.head_size,
                kv_cache_dtype=self.kv_cache_dtype,
            ),
            self._turboquant_layer_metadata.value.get_group_indices(
                device=device,
                head_size=self.head_size,
                kv_cache_dtype=self.kv_cache_dtype,
            ),
        )
        self._turboquant_masks[cache_key] = masks
        return masks

    def _get_turboquant_query_group_indices(
        self,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        cache_key = (device.type, device.index)
        indices = self._turboquant_query_group_indices.get(cache_key)
        if indices is not None:
            return indices

        key_masks, value_masks = self._ensure_turboquant_masks(device)
        kv_head_for_query_head = (
            torch.arange(self.num_heads, device=device, dtype=torch.int64)
            // self.num_queries_per_kv
        )
        indices = (
            kv_head_for_query_head,
            tuple(group.index_select(0, kv_head_for_query_head) for group in key_masks),
            tuple(
                group.index_select(0, kv_head_for_query_head) for group in value_masks
            ),
        )
        self._turboquant_query_group_indices[cache_key] = indices
        return indices

    def _build_turboquant_token_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        *,
        kv_lens: torch.Tensor | None = None,
        query_position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_lens is None and attn_metadata.turboquant_seq_ids is not None:
            assert attn_metadata.turboquant_token_seq_lens is not None
            assert attn_metadata.turboquant_query_positions is not None
            return (
                attn_metadata.turboquant_seq_ids,
                attn_metadata.turboquant_token_seq_lens,
                attn_metadata.turboquant_query_positions,
            )

        if attn_metadata.turboquant_seq_ids is None or kv_lens is not None:
            query_lens = (
                attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
            )
            seq_ids = torch.repeat_interleave(
                torch.arange(
                    attn_metadata.seq_lens.shape[0],
                    device=attn_metadata.seq_lens.device,
                    dtype=torch.int32,
                ),
                query_lens,
            )
            base_kv_lens = (
                kv_lens
                if kv_lens is not None
                else (
                    attn_metadata.encoder_seq_lens
                    if (
                        not attn_metadata.causal
                        and attn_metadata.encoder_seq_lens is not None
                    )
                    else attn_metadata.seq_lens
                )
            )
            token_kv_lens = base_kv_lens.index_select(0, seq_ids.to(torch.int64)).to(
                torch.int32
            )
            if attn_metadata.causal:
                token_offsets = torch.arange(
                    attn_metadata.num_actual_tokens,
                    device=attn_metadata.query_start_loc.device,
                    dtype=torch.int32,
                ) - attn_metadata.query_start_loc.index_select(
                    0, seq_ids.to(torch.int64)
                )
                query_positions = (
                    attn_metadata.seq_lens.index_select(0, seq_ids.to(torch.int64))
                    - query_lens.index_select(0, seq_ids.to(torch.int64))
                    + token_offsets
                    + query_position_offset
                ).to(torch.int32)
            else:
                query_positions = torch.zeros(
                    attn_metadata.num_actual_tokens,
                    dtype=torch.int32,
                    device=attn_metadata.query_start_loc.device,
                )
            if kv_lens is None and query_position_offset == 0:
                attn_metadata.turboquant_seq_ids = seq_ids
                attn_metadata.turboquant_token_seq_lens = token_kv_lens
                attn_metadata.turboquant_query_positions = query_positions
            return seq_ids, token_kv_lens, query_positions

        assert attn_metadata.turboquant_query_positions is not None
        return (
            attn_metadata.turboquant_seq_ids,
            attn_metadata.turboquant_token_seq_lens,
            attn_metadata.turboquant_query_positions,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)
        if is_turboquant_kv_cache(self.kv_cache_dtype):
            return self._forward_turboquant(
                query=query[:num_actual_tokens],
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],
                attn_metadata=attn_metadata,
            )
        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)

        if is_turboquant_kv_cache(self.kv_cache_dtype):
            if kv_cache.numel() == 0:
                return
            self._validate_turboquant_device(key.device)
            if not (slot_mapping >= 0).any().item():
                return

            assert self.turboquant_bits is not None
            _, _, centroids, _, layout = self._get_turboquant_tables(key.device)
            key_masks, value_masks = self._ensure_turboquant_masks(key.device)
            mse_transform_matrices, qjl_transform_matrices, mse_to_qjl_matrices = (
                self._get_turboquant_update_tables(key.device)
            )
            turboquant_write_packed_kv(
                x=key,
                cache=key_cache,
                slot_mapping=slot_mapping,
                layout=layout,
                group_indices=key_masks,
                mse_transform_matrices=mse_transform_matrices,
                qjl_transform_matrices=qjl_transform_matrices,
                mse_to_qjl_matrices=mse_to_qjl_matrices,
                centroids=centroids,
            )
            turboquant_write_packed_kv(
                x=value,
                cache=value_cache,
                slot_mapping=slot_mapping,
                layout=layout,
                group_indices=value_masks,
                mse_transform_matrices=mse_transform_matrices,
                qjl_transform_matrices=qjl_transform_matrices,
                mse_to_qjl_matrices=mse_to_qjl_matrices,
                centroids=centroids,
            )
            return

        # Reshape the input keys and values and store them in the cache.
        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # triton kernel does not support uint8 kv_cache
            #  (because some explicit casts (e.g. float8_e4m3fnuz)
            #   are not supported)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        # TurboQuant still applies RoPE in PyTorch and then performs the fused
        # quantize-and-store KV update. Do not advertise RoPE+KV fusion support
        # until a real fused TurboQuant kernel exists.
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if is_turboquant_kv_cache(self.kv_cache_dtype):
            rotary_dtype = query.dtype
            rotary_cache = cos_sin_cache.to(query.device, dtype=rotary_dtype)
            rotary_dim = rotary_cache.shape[-1]
            rotated_query, rotated_key = RotaryEmbedding.forward_static(
                positions=positions,
                query=query,
                key=key,
                head_size=query.shape[-1],
                rotary_dim=rotary_dim,
                cos_sin_cache=rotary_cache,
                is_neox_style=is_neox,
            )
            query.copy_(rotated_query)
            assert rotated_key is not None
            key.copy_(rotated_key)
            self.do_kv_cache_update(layer, key, value, kv_cache, layer_slot_mapping)
            return

        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )

    def _fallback_turboquant_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        kv_seq_lens_cpu = (
            attn_metadata.encoder_seq_lens_cpu
            if (
                not attn_metadata.causal
                and attn_metadata.encoder_seq_lens_cpu is not None
            )
            else attn_metadata.seq_lens_cpu
        )
        kv_head_for_query_head = (
            torch.arange(self.num_heads, device=query.device, dtype=torch.int64)
            // self.num_queries_per_kv
        )
        left_window, right_window = self.sliding_window
        key_cursor = 0

        for seq_idx, kv_seq_len in enumerate(kv_seq_lens_cpu.tolist()):
            q_start = int(attn_metadata.query_start_loc_cpu[seq_idx].item())
            q_end = int(attn_metadata.query_start_loc_cpu[seq_idx + 1].item())
            q_len = q_end - q_start
            if q_len == 0:
                key_cursor += kv_seq_len
                continue

            seq_query = query[q_start:q_end]
            seq_key = key[key_cursor : key_cursor + kv_seq_len]
            seq_value = value[key_cursor : key_cursor + kv_seq_len]
            key_cursor += kv_seq_len

            q_states = seq_query.permute(1, 0, 2).to(torch.float32)
            k_states = (
                seq_key.permute(1, 0, 2)
                .index_select(0, kv_head_for_query_head)
                .to(torch.float32)
            )
            v_states = (
                seq_value.permute(1, 0, 2)
                .index_select(0, kv_head_for_query_head)
                .to(torch.float32)
            )

            logits = torch.einsum("hqd,hkd->hqk", q_states, k_states) * self.scale
            key_positions = torch.arange(kv_seq_len, device=query.device)
            if attn_metadata.causal:
                query_positions = torch.arange(
                    kv_seq_len - q_len,
                    kv_seq_len,
                    device=query.device,
                )
                allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                if left_window != -1:
                    allowed &= key_positions.unsqueeze(0) >= (
                        query_positions.unsqueeze(1) - left_window
                    )
                if right_window != -1:
                    allowed &= key_positions.unsqueeze(0) <= (
                        query_positions.unsqueeze(1) + right_window
                    )
                if attn_metadata.mm_prefix_range is not None:
                    for range_start, range_end in attn_metadata.mm_prefix_range.get(
                        seq_idx, []
                    ):
                        q_in_range = (query_positions >= range_start) & (
                            query_positions <= range_end
                        )
                        k_in_range = (key_positions >= range_start) & (
                            key_positions <= range_end
                        )
                        allowed |= q_in_range[:, None] & k_in_range[None, :]
                logits = logits.masked_fill(
                    ~allowed.unsqueeze(0),
                    float("-inf"),
                )

            if self.logits_soft_cap:
                logits = self.logits_soft_cap * torch.tanh(
                    logits / self.logits_soft_cap
                )

            if self.sinks is not None:
                sink_logits = self.sinks[:, None, None].to(torch.float32).expand(
                    -1, q_len, 1
                )
                zero_value = torch.zeros(
                    (self.num_heads, 1, self.head_size),
                    dtype=torch.float32,
                    device=query.device,
                )
                logits = torch.cat((logits, sink_logits), dim=-1)
                v_states = torch.cat((v_states, zero_value), dim=1)

            attn = torch.softmax(logits, dim=-1)
            seq_output = torch.einsum("hqk,hkd->hqd", attn, v_states)
            output[q_start:q_end].copy_(seq_output.permute(1, 0, 2).to(output.dtype))

        return output

    def _can_use_turboquant_dense_prefill(
        self,
        query: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> bool:
        if (
            not attn_metadata.causal
            or attn_metadata.use_cascade
            or self.logits_soft_cap != 0
            or attn_metadata.mm_prefix_range is not None
            or self.sinks is not None
            or query.shape[-1] > TURBOQUANT_TRITON_PREFILL_MAX_HEAD_SIZE
        ):
            return False

        if attn_metadata.max_query_len != attn_metadata.max_seq_len:
            return False

        query_lens_cpu = (
            attn_metadata.query_start_loc_cpu[1:] - attn_metadata.query_start_loc_cpu[:-1]
        )
        return torch.equal(query_lens_cpu, attn_metadata.seq_lens_cpu)

    def _forward_turboquant(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        # Pure prompt-prefill batches do not need to read back quantized KV from
        # cache. Reuse the dense Triton prefill kernel so first-token latency is
        # dominated by attention work instead of TurboQuant decode setup.
        if self._can_use_turboquant_dense_prefill(query, attn_metadata):
            context_attention_fwd(
                q=query,
                k=key,
                v=value,
                o=output,
                b_start_loc=attn_metadata.query_start_loc,
                b_seq_len=attn_metadata.seq_lens,
                max_input_len=attn_metadata.max_query_len,
                is_causal=attn_metadata.causal,
                softmax_scale=self.scale,
                sliding_window_q=self.sliding_window[0],
                sliding_window_k=self.sliding_window[1],
            )
            return output

        if key_cache.numel() == 0 or value_cache.numel() == 0:
            # TurboQuant-native prefill is not implemented yet. When there is no
            # KV cache backing tensor, fall back to an eager live-K/V reference.
            self._fallback_turboquant_attention(
                query=query,
                key=key,
                value=value,
                output=output,
                attn_metadata=attn_metadata,
            )
            return output

        assert self.turboquant_bits is not None
        self._validate_turboquant_device(query.device)
        rotations, qjl_matrices, centroids, norm_lut, _ = self._get_turboquant_tables(
            query.device
        )
        key_masks, value_masks = self._ensure_turboquant_masks(query.device)
        token_seq_ids, token_kv_lens, token_query_positions = (
            self._build_turboquant_token_metadata(attn_metadata)
        )
        kv_head_for_query_head, key_query_group_indices, value_query_group_indices = (
            self._get_turboquant_query_group_indices(query.device)
        )
        value_mse_inverse_matrices, value_qjl_inverse_matrices = (
            self._get_turboquant_decode_tables(query.device)
        )
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        if attn_metadata.use_cascade:
            if self.sliding_window != (-1, -1):
                raise NotImplementedError(
                    "TurboQuant cascade attention does not support sliding "
                    "window attention yet."
                )
            if attn_metadata.mm_prefix_range is not None:
                raise NotImplementedError(
                    "TurboQuant cascade attention does not support mm-prefix "
                    "ranges yet."
                )
            common_prefix_len = attn_metadata.common_prefix_len
            block_size = key_cache.shape[1]
            assert common_prefix_len % block_size == 0
            num_common_blocks = common_prefix_len // block_size

            prefix_output = torch.empty_like(output)
            prefix_lse = torch.empty(
                (query.shape[1], query.shape[0]),
                dtype=torch.float32,
                device=query.device,
            )
            prefix_query_start_loc = torch.tensor(
                [0, query.shape[0]], dtype=torch.int32, device=query.device
            )
            prefix_seq_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=query.device
            )
            prefix_seq_ids = torch.zeros(
                query.shape[0], dtype=torch.int32, device=query.device
            )
            prefix_kv_lens = torch.full(
                (query.shape[0],),
                common_prefix_len,
                dtype=torch.int32,
                device=query.device,
            )
            prefix_query_positions = torch.zeros(
                query.shape[0], dtype=torch.int32, device=query.device
            )
            turboquant_decode_attention_fwd(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table[:1, :num_common_blocks],
                query_start_loc=prefix_query_start_loc,
                seq_lens=prefix_seq_lens,
                key_group_indices=key_masks,
                value_group_indices=value_masks,
                key_rotations=rotations,
                key_qjl_matrices=qjl_matrices,
                value_rotations=rotations,
                value_qjl_matrices=qjl_matrices,
                centroids=centroids,
                norm_lut=norm_lut,
                softmax_scale=self.scale,
                kv_cache_dtype=self.kv_cache_dtype,
                token_seq_ids=prefix_seq_ids,
                token_kv_lens=prefix_kv_lens,
                token_query_positions=prefix_query_positions,
                kv_head_for_query_head=kv_head_for_query_head,
                key_query_group_indices=key_query_group_indices,
                value_query_group_indices=value_query_group_indices,
                value_mse_inverse_matrices=value_mse_inverse_matrices,
                value_qjl_inverse_matrices=value_qjl_inverse_matrices,
                causal=False,
                sinks=self.sinks,
                logits_soft_cap=self.logits_soft_cap,
                output_lse=prefix_lse,
                out=prefix_output,
            )

            assert attn_metadata.suffix_kv_lens is not None
            suffix_output = torch.empty_like(output)
            suffix_lse = torch.empty(
                (query.shape[1], query.shape[0]),
                dtype=torch.float32,
                device=query.device,
            )
            (
                suffix_seq_ids,
                suffix_kv_lens,
                suffix_query_positions,
            ) = self._build_turboquant_token_metadata(
                attn_metadata,
                kv_lens=attn_metadata.suffix_kv_lens,
                query_position_offset=-common_prefix_len,
            )
            turboquant_decode_attention_fwd(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table[:, num_common_blocks:],
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.suffix_kv_lens,
                key_group_indices=key_masks,
                value_group_indices=value_masks,
                key_rotations=rotations,
                key_qjl_matrices=qjl_matrices,
                value_rotations=rotations,
                value_qjl_matrices=qjl_matrices,
                centroids=centroids,
                norm_lut=norm_lut,
                softmax_scale=self.scale,
                kv_cache_dtype=self.kv_cache_dtype,
                token_seq_ids=suffix_seq_ids,
                token_kv_lens=suffix_kv_lens,
                token_query_positions=suffix_query_positions,
                kv_head_for_query_head=kv_head_for_query_head,
                key_query_group_indices=key_query_group_indices,
                value_query_group_indices=value_query_group_indices,
                value_mse_inverse_matrices=value_mse_inverse_matrices,
                value_qjl_inverse_matrices=value_qjl_inverse_matrices,
                causal=attn_metadata.causal,
                sinks=None,
                logits_soft_cap=self.logits_soft_cap,
                output_lse=suffix_lse,
                out=suffix_output,
            )
            merge_attn_states(
                output, prefix_output, prefix_lse, suffix_output, suffix_lse
            )
            return output

        turboquant_decode_attention_fwd(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=(
                attn_metadata.encoder_seq_lens
                if (
                    not attn_metadata.causal
                    and attn_metadata.encoder_seq_lens is not None
                )
                else attn_metadata.seq_lens
            ),
            key_group_indices=key_masks,
            value_group_indices=value_masks,
            key_rotations=rotations,
            key_qjl_matrices=qjl_matrices,
            value_rotations=rotations,
            value_qjl_matrices=qjl_matrices,
            centroids=centroids,
            norm_lut=norm_lut,
            softmax_scale=self.scale,
            kv_cache_dtype=self.kv_cache_dtype,
            token_seq_ids=token_seq_ids,
            token_kv_lens=token_kv_lens,
            token_query_positions=token_query_positions,
            kv_head_for_query_head=kv_head_for_query_head,
            key_query_group_indices=key_query_group_indices,
            value_query_group_indices=value_query_group_indices,
            value_mse_inverse_matrices=value_mse_inverse_matrices,
            value_qjl_inverse_matrices=value_qjl_inverse_matrices,
            causal=attn_metadata.causal,
            sliding_window=self.sliding_window,
            sinks=self.sinks,
            mm_prefix_range=mm_prefix_range_tensor,
            logits_soft_cap=self.logits_soft_cap,
            out=output,
        )
        return output
