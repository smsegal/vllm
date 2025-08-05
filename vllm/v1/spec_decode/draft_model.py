# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import TYPE_CHECKING

from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.sampler import Sampler
from dataclasses import replace

import torch
import numpy as np

from vllm.logger import init_logger
from vllm.config import CompilationLevel, VllmConfig
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


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

        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens
        )
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

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

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # Number of tokens across the batch (flattened)
        num_tokens = target_token_ids.shape[0]
        # Indices of the last token for each request in the flattened view
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        # Copy full sequence ids/positions into persistent buffers
        self.input_ids[:num_tokens] = target_token_ids
        self.positions[:num_tokens] = target_positions

        # Pad for CUDA graphs if enabled and within captured size
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens

        # Run the draft model on the full sequence to get logits for the last tokens
        with set_forward_context(
            None, self.vllm_config, num_tokens=num_input_tokens
        ):
            ret = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                inputs_embeds=None,
            )

        # Assume the draft model forward returns logits; support tuple or tensor
        if isinstance(ret, tuple):
            logits_all = ret[0]
        else:
            logits_all = ret
        assert isinstance(logits_all, torch.Tensor)

        logits = logits_all[last_token_indices]

        # Prepare a local sampling metadata with independent output_token_ids
        sampling_md = replace(
            sampling_metadata,
            output_token_ids=[
                lst.copy() for lst in sampling_metadata.output_token_ids
            ],
        )

        # If no speculative tokens are requested, return an empty tensor with correct shape
        k = (
            int(self.num_speculative_tokens)
            if self.num_speculative_tokens is not None
            else 0
        )
        batch_size = logits.shape[0]
        if k <= 0:
            return torch.empty(
                (batch_size, 0), dtype=torch.int64, device=logits.device
            )

        draft_token_ids_list = []
        # Track growing sequences - start with original sequences
        current_num_tokens = num_tokens
        current_logits = logits

        # Generate all speculative tokens in unified loop
        for step in range(k):
            # For steps after the first, append new tokens and run model forward pass
            if step > 0:
                # Append new tokens to the sequences
                query_start_loc = common_attn_metadata.query_start_loc
                # Vectorized append of new tokens/positions at the end of each sequence
                per_req_offsets = (
                    torch.arange(
                        batch_size,
                        device=query_start_loc.device,
                        dtype=query_start_loc.dtype,
                    )
                    + 1
                ) * step
                seq_end_indices = query_start_loc[1:] - 1 + per_req_offsets
                self.input_ids[seq_end_indices] = next_token_ids.to(torch.int32)
                base_positions = target_positions[last_token_indices]
                new_positions = base_positions + step
                clamped_positions = torch.where(
                    new_positions >= self.max_model_len,
                    torch.zeros_like(new_positions),
                    new_positions,
                )
                self.positions[seq_end_indices] = clamped_positions

                current_num_tokens = num_tokens + step * batch_size

                if (
                    self.use_cuda_graph
                    and current_num_tokens <= self.cudagraph_batch_sizes[-1]
                ):
                    input_num_tokens = self.vllm_config.pad_for_cudagraph(
                        current_num_tokens
                    )
                else:
                    input_num_tokens = current_num_tokens

                with set_forward_context(
                    None, self.vllm_config, num_tokens=input_num_tokens
                ):
                    ret = self.model(
                        input_ids=self.input_ids[:input_num_tokens],
                        positions=self.positions[:input_num_tokens],
                        inputs_embeds=None,
                    )

                if isinstance(ret, tuple):
                    logits_all_step = ret[0]
                else:
                    logits_all_step = ret

                # Extract logits for the last token of each sequence
                current_logits = logits_all_step[seq_end_indices]

            sampled = self.sampler(current_logits, sampling_md)
            next_token_ids = sampled.sampled_token_ids.squeeze(-1).long()
            # Update histories
            next_token_list = next_token_ids.tolist()
            for i, tok in enumerate(next_token_list):
                sampling_md.output_token_ids[i].append(int(tok))

            draft_token_ids_list.append(next_token_ids.view(-1, 1))

        # [batch_size, num_speculative_tokens]
        return torch.cat(draft_token_ids_list, dim=1)

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
            np.arange(total_num_tokens, dtype=new_query_start_loc_np.dtype)
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
            logger.info(f"LOADED DRAFT MODEL {self.draft_model_config.model}")

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
