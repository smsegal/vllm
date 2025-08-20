# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from vllm.attention.layer import Attention
from vllm.config import (
    CompilationLevel,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import bind_kv_cache

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class DraftModelProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.target_model_config = self.vllm_config.model_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.max_model_len = self.target_model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        self.device = device
        self.runner = runner

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.target_model_config.dtype
        cache_config = self.vllm_config.cache_config
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype
            ]

        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens
        )
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.token_arange_np = np.arange(self.max_num_tokens)

        self.use_cuda_graph = (
            self.vllm_config.compilation_config.level
            == CompilationLevel.PIECEWISE
            and not self.vllm_config.model_config.enforce_eager
        )

        assert (
            self.vllm_config.compilation_config.cudagraph_capture_sizes
            is not None
        )
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            )
        )

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Initialize sampler for speculative token generation
        self.sampler = Sampler()
        # Placeholder for loaded draft model (set in load_model)
        self.model: Any = None
        self.attn_layer_names: list[str] = []
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.arange = torch.arange(
            # We need +1 here because the arange is used to set query_start_loc,
            # which has one more element than batch_size.
            max_batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        # Draft-specific context/builder and KV cache will be initialized in load_model().
        self.draft_vllm_config: Optional[VllmConfig] = None
        self.draft_attn_metadata_builder: Optional[Any] = None
        self._draft_runner_kv_caches: list[torch.Tensor] = []
        self.draft_block_size: int = self.block_size
        self.draft_max_model_len: int = self.max_model_len

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None
        # Use draft-specific config if available
        vllm_config_local = self.draft_vllm_config or self.vllm_config
        block_size = getattr(self, "draft_block_size", None) or self.block_size
        max_model_len = (
            getattr(self, "draft_max_model_len", None) or self.max_model_len
        )

        # Build attention metadata with the correct (draft) builder.
        assert self.draft_attn_metadata_builder is not None, (
            "DraftModelProposer.load_model() must be called before propose; "
            "draft_attn_metadata_builder is not initialized."
        )
        attn_metadata = self.draft_attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = vllm_config_local.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        # TODO: multimodal support
        inputs_embeds = None
        input_ids = self.input_ids[:num_input_tokens]

        with set_forward_context(
            per_layer_attn_metadata,
            vllm_config_local,
            num_tokens=num_input_tokens,
        ):
            last_hidden_states = self.model(
                input_ids=input_ids,
                positions=self.positions[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        positions = target_positions[last_token_indices]

        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = vllm_config_local.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size
        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions >= max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Increment the sequence lengths.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (
                block_ids * block_size + clamped_positions % block_size
            )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            inputs_embeds = None
            input_ids = self.input_ids[:input_batch_size]

            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                vllm_config_local,
                num_tokens=input_batch_size,
            ):
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self.positions[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
            logits = self.model.compute_logits(
                last_hidden_states[:batch_size], None
            )
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        Prepare inputs for the draft model by reconstructing the full sequences
        after accounting for rejected tokens (similar to EAGLE's prepare_inputs).

        Returns:
          - Updated CommonAttentionMetadata with adjusted query_start_loc/seq_lens
          - token_indices: indices into the flattened token buffer that represent
            the reconstructed sequences to feed into the draft model.
        """
        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu - num_rejected_tokens
        )

        # [0, q1, q1 + q2, ...] -> [q1, q2, ...]
        new_query_len_per_req = (
            query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        )
        # [q1, q2, ...] -> [q1 - n1, q2 - n2, ...]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # Build new query_start_loc on CPU (pinned)
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])
        total_num_tokens = int(new_query_start_loc_np[-1])

        # Compute per-token offsets within each request segment: [0.., 0.., ...]
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        token_offsets = (
            self.token_arange_np[:total_num_tokens]
            - new_query_start_locs_expanded
        )

        # Expand old starting locations to align with new per-request token counts
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )

        # Final flattened indices into the original token buffer
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(
            device, non_blocking=True
        )

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(
                device, non_blocking=True
            ),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=int(new_query_len_per_req.max().item()),
            max_seq_len=int(new_seq_lens_cpu.max().item()),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
        )

        return spec_common_attn_metadata, token_indices

    def load_model(self) -> None:
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("draft_model"):
            # needed to avoid duplicate names in the compilation context, crashing the attention metadata
            vllm_config_seperate_forward_context = copy.deepcopy(
                self.vllm_config
            )
            vllm_config_seperate_forward_context.compilation_config.static_forward_context = {}
            vllm_config_seperate_forward_context.model_config = (
                self.draft_model_config
            )
            self.model = get_model(
                vllm_config=vllm_config_seperate_forward_context,
                model_config=self.draft_model_config,
            )
            self.attn_layer_names = list(
                set(
                    get_layers_from_vllm_config(
                        vllm_config_seperate_forward_context, Attention
                    ).keys()
                )
            )
            # Save draft-specific vllm_config
            self.draft_vllm_config = vllm_config_seperate_forward_context
            # Build a draft-specific attention metadata builder and KV caches
            layers = get_layers_from_vllm_config(
                vllm_config_seperate_forward_context, Attention
            )
            if len(self.attn_layer_names) > 0:
                first_layer = layers[self.attn_layer_names[0]]
                attn_backend = first_layer.attn_backend
                # Build KV cache spec based on the draft model config
                draft_dtype = self.draft_model_config.dtype
                if isinstance(draft_dtype, str):
                    draft_dtype_t = STR_DTYPE_TO_TORCH_DTYPE[draft_dtype]
                else:
                    draft_dtype_t = draft_dtype
                kv_cache_spec = FullAttentionSpec(
                    block_size=vllm_config_seperate_forward_context.cache_config.block_size,
                    num_kv_heads=self.draft_model_config.get_num_kv_heads(
                        self.parallel_config
                    ),
                    head_size=self.draft_model_config.get_head_size(),
                    dtype=draft_dtype_t,
                    use_mla=self.draft_model_config.use_mla,
                    sliding_window=self.draft_model_config.get_sliding_window(),
                )
                builder_cls = attn_backend.get_builder_cls()
                self.draft_attn_metadata_builder = builder_cls(
                    kv_cache_spec=kv_cache_spec,
                    layer_names=self.attn_layer_names,
                    vllm_config=vllm_config_seperate_forward_context,
                    device=self.device,
                )
                # Initialize draft-specific KV cache tensors and bind to the draft model layers.
                if (
                    hasattr(self.runner, "kv_cache_config")
                    and self.runner.kv_cache_config is not None
                ):
                    num_blocks = self.runner.kv_cache_config.num_blocks
                else:
                    # Fallback: allocate for the full model length.
                    bs = kv_cache_spec.block_size
                    mlen = self.draft_model_config.max_model_len
                    num_blocks = max(1, (mlen + bs - 1) // bs)
                # Determine kv-cache dtype for the draft model.
                cache_dtype = vllm_config_seperate_forward_context.cache_config.cache_dtype
                if cache_dtype == "auto":
                    draft_kv_dtype = draft_dtype_t
                else:
                    draft_kv_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
                # Allocate and reshape per-layer KV caches.
                kv_caches: dict[str, torch.Tensor] = {}
                try:
                    stride_order = attn_backend.get_kv_cache_stride_order()
                except Exception:
                    stride_order = None
                for layer_name in self.attn_layer_names:
                    kv_shape = attn_backend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                    )
                    raw = torch.zeros(
                        int(num_blocks * kv_cache_spec.page_size_bytes),
                        dtype=torch.int8,
                        device=self.device,
                    )
                    if stride_order is not None:
                        kv_cache_shape = tuple(
                            kv_shape[i] for i in stride_order
                        )
                        inv_order = [
                            stride_order.index(i)
                            for i in range(len(stride_order))
                        ]
                        tensor = (
                            raw.view(draft_kv_dtype)
                            .view(kv_cache_shape)
                            .permute(*inv_order)
                        )
                    else:
                        tensor = raw.view(draft_kv_dtype).view(kv_shape)
                    kv_caches[layer_name] = tensor
                # Bind KV caches to the draft model forward context so attention layers can access them.
                self._draft_runner_kv_caches = []
                bind_kv_cache(
                    kv_caches,
                    vllm_config_seperate_forward_context.compilation_config.static_forward_context,
                    self._draft_runner_kv_caches,
                )
                # Cache draft-specific constants.
                self.draft_block_size = kv_cache_spec.block_size
                self.draft_max_model_len = self.draft_model_config.max_model_len
            logger.info("LOADED DRAFT MODEL %s", self.draft_model_config.model)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int):
        with set_forward_context(
            None,
            self.draft_vllm_config
            if getattr(self, "draft_vllm_config", None) is not None
            else self.vllm_config,
            num_tokens=num_tokens,
        ):
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None

            self.model(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
