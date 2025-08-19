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
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

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

        # FIXME: need to consider multiple kv_cache_groups
        attn_metadata = self.runner.attn_groups[0][
            0
        ].metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        # TODO: multimodal support
        inputs_embeds = None
        input_ids = self.input_ids[:num_input_tokens]

        with set_forward_context(
            None,
            self.vllm_config,
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
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
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
            exceeds_max_model_len = positions >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Increment the sequence lengths.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (
                block_ids * self.block_size
                + clamped_positions % self.block_size
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
                None,
                self.vllm_config,
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

    def propose_old(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # Base per-sequence boundaries
        query_start = common_attn_metadata.query_start_loc
        seq_starts = query_start[:-1]
        seq_ends = query_start[1:]
        batch_size = seq_starts.shape[0]

        # Helper: rebuild flattened tokens with interleaved speculative tokens
        extra_tokens: list[list[int]] = [[] for _ in range(batch_size)]

        def rebuild() -> tuple[torch.Tensor, int]:
            ptr = 0
            last_indices_list = []
            for i in range(batch_size):
                s = int(seq_starts[i])
                e = int(seq_ends[i])
                length = e - s
                base_slice = target_token_ids[s:e]
                n_extra = len(extra_tokens[i])
                total_len = length + n_extra
                # Copy base tokens
                self.input_ids[ptr : ptr + length] = base_slice
                base_pos = target_positions[s:e]
                self.positions[ptr : ptr + length] = base_pos
                # Append speculative tokens & their positions
                if n_extra:
                    self.input_ids[ptr + length : ptr + total_len] = (
                        torch.tensor(
                            extra_tokens[i],
                            dtype=torch.int32,
                            device=self.device,
                        )
                    )
                    last_pos = int(base_pos[length - 1]) if length > 0 else -1
                    new_pos = torch.arange(
                        last_pos + 1,
                        last_pos + 1 + n_extra,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    self.positions[ptr + length : ptr + total_len] = new_pos
                last_indices_list.append(ptr + total_len - 1)
                ptr += total_len
            return torch.tensor(
                last_indices_list, device=self.device, dtype=torch.int64
            ), ptr

        last_indices, total_tokens = rebuild()
        if (
            self.use_cuda_graph
            and total_tokens <= self.cudagraph_batch_sizes[-1]
        ):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(total_tokens)
        else:
            num_input_tokens = total_tokens
        with set_forward_context(
            None, self.vllm_config, num_tokens=num_input_tokens
        ):
            ret = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                inputs_embeds=None,
            )
        hidden = ret[0] if isinstance(ret, tuple) else ret
        # Pass None: logits processor only needs sampling metadata fields
        logits_all = self.model.compute_logits(hidden, None)
        current_logits = logits_all.index_select(0, last_indices)

        sampling_md = replace(
            sampling_metadata,
            output_token_ids=[
                lst.copy() for lst in sampling_metadata.output_token_ids
            ],
        )
        k = int(self.num_speculative_tokens or 0)
        if k <= 0:
            return torch.empty(
                (batch_size, 0), dtype=torch.int64, device=self.device
            )

        collected: list[torch.Tensor] = []
        for step in range(k):
            sampled = self.sampler(current_logits, sampling_md)
            step_tokens = sampled.sampled_token_ids.squeeze(-1).long()  # [B]
            collected.append(step_tokens.view(-1, 1))
            for i, tok in enumerate(step_tokens.tolist()):
                sampling_md.output_token_ids[i].append(int(tok))
                extra_tokens[i].append(int(tok))
            if step + 1 == k:
                break
            last_indices, total_tokens = rebuild()
            if (
                self.use_cuda_graph
                and total_tokens <= self.cudagraph_batch_sizes[-1]
            ):
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    total_tokens
                )
            else:
                num_input_tokens = total_tokens
            with set_forward_context(
                None, self.vllm_config, num_tokens=num_input_tokens
            ):
                ret = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    inputs_embeds=None,
                )
            hidden = ret[0] if isinstance(ret, tuple) else ret
            logits_all = self.model.compute_logits(hidden, None)
            current_logits = logits_all.index_select(0, last_indices)
        return torch.cat(collected, dim=1)

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
            max_query_len=new_query_len_per_req.max().item(),
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
            logger.info("LOADED DRAFT MODEL %s", self.draft_model_config.model)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int):
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None

            self.model(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
