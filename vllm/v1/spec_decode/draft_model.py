# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

from typing import TYPE_CHECKING, Any, Optional, cast

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
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.sample.sampler import Sampler
from collections import defaultdict
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    ChunkedLocalAttentionSpec,
    KVCacheSpec,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config,
    get_uniform_page_size,
    is_kv_cache_type_uniform,
)
from vllm.v1.worker.utils import bind_kv_cache, initialize_kv_cache_for_kv_sharing, AttentionGroup

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
        # KV config and attention groups constructed at load time.
        self.draft_kv_cache_config: Optional[Any] = None
        self.draft_attn_groups: list[list[AttentionGroup]] = []
        self.shared_kv_cache_layers_draft: dict[str, str] = {}
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

        # Prefill the true window (no shift) to rebuild KV coherently.
        use_tmp_for_prefill = num_tokens > self.max_num_tokens
        if use_tmp_for_prefill:
            # Directly use the provided window tokens on device.
            prefill_input_ids = target_token_ids
        else:
            # Copy to the persistent buffer so we can pad for CUDA graph if needed.
            self.input_ids[:num_tokens] = target_token_ids
            prefill_input_ids = self.input_ids

        assert self.runner is not None
        # Use draft-specific config if available
        vllm_config_local = self.draft_vllm_config or self.vllm_config
        block_size = getattr(self, "draft_block_size", None) or self.block_size
        max_model_len = (
            getattr(self, "draft_max_model_len", None) or self.max_model_len
        )

        # Build per-group attention metadata using draft-specific builders, then
        # construct per-layer mapping.
        assert hasattr(self, "draft_attn_groups") and self.draft_attn_groups, (
            "DraftModelProposer.load_model() must be called before propose; "
            "draft_attn_groups is not initialized."
        )
        per_layer_attn_metadata: dict[str, Any] = {}
        group_metas: list[Any] = []
        for kv_attn_groups in self.draft_attn_groups:
            for attn_group in kv_attn_groups:
                meta = attn_group.metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata, draft_index=0
                )
                group_metas.append(meta)
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = meta

        if (
            self.use_cuda_graph
            and (not use_tmp_for_prefill)
            and num_tokens <= self.cudagraph_batch_sizes[-1]
        ):
            num_input_tokens = vllm_config_local.pad_for_cudagraph(num_tokens)
            # copy inputs to buffer for cudagraph
            self.positions[:num_tokens] = target_positions
            positions_in = self.positions[:num_input_tokens]
            input_ids = prefill_input_ids[:num_input_tokens]
            # Pad slot_mapping tail with -1 (PADDING_SLOT_ID) for CUDA graph safety.
            if num_input_tokens > num_tokens:
                for meta in group_metas:
                    sm = meta.slot_mapping
                    pad_len = num_input_tokens - num_tokens
                    pad = torch.full(
                        (pad_len,),
                        PADDING_SLOT_ID,
                        dtype=sm.dtype,
                        device=sm.device,
                    )
                    meta.slot_mapping = torch.cat([sm[:num_tokens], pad], dim=0)
        else:
            num_input_tokens = num_tokens
            positions_in = target_positions
            input_ids = prefill_input_ids[:num_input_tokens]

        # True-window prefill (writes KV at correct slots for the window). Neccessary?
        with set_forward_context(
            per_layer_attn_metadata,
            vllm_config_local,
            num_tokens=num_input_tokens,
        ):
            _ = self.model(
                input_ids=input_ids,
                positions=positions_in,
                inputs_embeds=None,
            )

        # Start decode from the last position of each request.
        positions = target_positions[last_token_indices]

        # Generate draft tokens via explicit decode loop.
        draft_token_ids_list: list[torch.Tensor] = []

        if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = vllm_config_local.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size
        for meta in group_metas:
            meta.num_actual_tokens = batch_size
            meta.max_query_len = 1
            meta.query_start_loc = self.arange[: batch_size + 1]
        for i in range(self.num_speculative_tokens):
            # Update the inputs.
            # tensor.argmax() returns int64 by default; cast to int32 is crucial when compiled.
            if i == 0:
                input_ids = next_token_ids.int()
            else:
                input_ids = draft_token_ids_list[-1].int()
            positions += 1

            # Handle positions exceeding max model length (ignore by masking).
            exceeds_max_model_len = positions >= max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Increment the sequence lengths and update slot mappings per group.
            for meta in group_metas:
                meta.max_seq_len += 1
                meta.seq_lens += 1
                meta.max_seq_len = min(meta.max_seq_len, max_model_len)
                # For the requests that exceed the max model length, set seq len to 1 to reduce overhead.
                meta.seq_lens.masked_fill_(exceeds_max_model_len, 1)

                # Compute the slot mapping for this decode step.
                block_numbers = clamped_positions // block_size
                # Backends may expose block table as block_table or block_table_tensor
                blk_tbl = getattr(meta, "block_table", None)
                if blk_tbl is None:
                    blk_tbl = getattr(meta, "block_table_tensor")
                block_ids = blk_tbl.gather(
                    dim=1, index=block_numbers.view(-1, 1)
                ).view(-1)
                meta.slot_mapping = (
                    block_ids * block_size + clamped_positions % block_size
                )
                # Mask out the slot mappings that exceed the max model length.
                meta.slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

                # Pad slot_mapping tail with -1 for CUDA graph capture safety.
                if input_batch_size > batch_size:
                    sm = meta.slot_mapping
                    pad_len = input_batch_size - batch_size
                    pad = torch.full(
                        (pad_len,),
                        PADDING_SLOT_ID,
                        dtype=sm.dtype,
                        device=sm.device,
                    )
                    meta.slot_mapping = torch.cat([sm[:batch_size], pad], dim=0)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            inputs_embeds = None
            input_ids = self.input_ids[:input_batch_size]

            # Run the model for this decode step.
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

    def prepare_inputs_eagle(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
        block_table: BlockTable,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu - num_rejected_tokens
        )

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = (
            query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        )
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens]
            - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                   // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,         // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2]  // req 3
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

    def prepare_inputs(
        self,
        # Number of tokens per request (window length), [batch_size] on CPU
        window_seq_lens_cpu: torch.Tensor,
        # Flattened indices into the runner's CPU token buffer representing the
        # concatenated window tokens for all requests (length = sum(Li)), on CPU
        token_indices_cpu: torch.Tensor,
        # Block table to compute slot mapping (KV group 0)
        block_table: BlockTable,
    ) -> CommonAttentionMetadata:
        """
        Prepare attention metadata for a coherent per-request context window by
        computing a dynamic slot_mapping directly from the block table. This avoids
        using BlockTable.compute_slot_mapping(), which writes into a fixed-size buffer
        limited by max_num_batched_tokens and can overflow when the window spans more
        tokens than that cap.

        This method ensures the draft KV matches the target's current block table
        mapping for the effective target context (e.g., full context or a configured
        sliding window).

        Args:
          window_seq_lens_cpu: [num_reqs] tensor of Li values (per-request window
            lengths) on CPU.
          token_indices_cpu: [sum(Li)] tensor of flattened indices into the
            runner's CPU token matrix (row-major) selecting the window tokens.
          block_table: KV-group-0 BlockTable for block table tensors.

        Returns:
          CommonAttentionMetadata for the window (KV group 0).
        """
        assert isinstance(window_seq_lens_cpu, torch.Tensor)
        assert isinstance(token_indices_cpu, torch.Tensor)

        # Derive per-token request indices and positions within each request.
        num_reqs = window_seq_lens_cpu.shape[0]
        window_seq_lens_np = window_seq_lens_cpu.numpy().astype(
            np.int32, copy=False
        )

        # print(f"[DEBUG] DraftModelProposer.prepare_inputs: num_reqs={num_reqs}, "
        #       f"window_seq_lens={window_seq_lens_np}, "
        #       f"token_indices_cpu.shape={token_indices_cpu.shape}")

        # Build query_start_loc on CPU (pinned).
        query_start_loc_cpu = torch.zeros(
            (num_reqs + 1,),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        qsl_np = query_start_loc_cpu.numpy()
        np.cumsum(window_seq_lens_np, out=qsl_np[1:])
        total_num_tokens = int(qsl_np[-1])

        # print(f"[DEBUG] DraftModelProposer.prepare_inputs: total_num_tokens={total_num_tokens}, "
        #       f"query_start_loc={qsl_np}")

        # Per-token request index and per-token position within request.
        req_indices_np = np.empty(total_num_tokens, dtype=np.int32)
        positions_np = np.empty(total_num_tokens, dtype=np.int64)
        for i in range(num_reqs):
            Li = int(window_seq_lens_np[i])
            if Li == 0:
                continue
            start = qsl_np[i]
            end = qsl_np[i + 1]
            req_indices_np[start:end] = i
            # Positions within each request are [0..Li-1]
            positions_np[start:end] = np.arange(Li, dtype=np.int64)

        # Compute a dynamic slot mapping without using the BlockTable's fixed-size buffer.
        # This mirrors BlockTable.compute_slot_mapping but writes into a per-window tensor.
        block_size = block_table.block_size
        max_blocks_per_req = block_table.max_num_blocks_per_req
        block_table_np = block_table.block_table_np

        block_table_indices = req_indices_np.astype(
            np.int64
        ) * max_blocks_per_req + (positions_np // block_size)
        block_numbers = block_table_np.ravel()[block_table_indices]
        block_offsets = positions_np % block_size
        slot_mapping_np = (
            block_numbers.astype(np.int64) * block_size + block_offsets
        )
        slot_mapping_gpu = torch.from_numpy(slot_mapping_np).to(
            self.device, non_blocking=True
        )

        # Build CommonAttentionMetadata (KV group 0).
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_cpu.to(
                self.device, non_blocking=True
            ),
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=window_seq_lens_cpu.to(self.device, non_blocking=True),
            seq_lens_cpu=window_seq_lens_cpu,
            num_computed_tokens_cpu=window_seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=int(window_seq_lens_np.max(initial=0)),
            max_seq_len=int(window_seq_lens_np.max(initial=0)),
            block_table_tensor=block_table.get_device_tensor()[:num_reqs],
            slot_mapping=slot_mapping_gpu,
            causal=True,
        )
        # print(f"[DEBUG] DraftModelProposer.prepare_inputs: returning metadata with "
        #       f"num_actual_tokens={common_attn_metadata.num_actual_tokens}")
        return common_attn_metadata

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

            layers = get_layers_from_vllm_config(
                vllm_config_seperate_forward_context, Attention
            )
            self.attn_layer_names = list(set(layers.keys()))

            # Save draft-specific vllm_config
            self.draft_vllm_config = vllm_config_seperate_forward_context
            # Build draft-specific KV cache spec for all attention layers and allocate KV using kv_cache_utils.
            if len(self.attn_layer_names) > 0:
                # Determine draft model dtype
                draft_dtype = self.draft_model_config.dtype
                if isinstance(draft_dtype, str):
                    draft_dtype_t = STR_DTYPE_TO_TORCH_DTYPE[draft_dtype]
                else:
                    draft_dtype_t = draft_dtype

                # Build per-layer kv_cache_spec mirroring GPUModelRunner.get_kv_cache_spec
                kv_cache_spec: dict[str, KVCacheSpec] = {}
                self.shared_kv_cache_layers_draft: dict[str, str] = {}
                block_size = (
                    vllm_config_seperate_forward_context.cache_config.block_size
                )
                use_mla = self.draft_model_config.use_mla
                for layer_name, attn_module in layers.items():
                    kv_tgt_layer = getattr(
                        attn_module, "kv_sharing_target_layer_name", None
                    )
                    if kv_tgt_layer is not None:
                        self.shared_kv_cache_layers_draft[layer_name] = kv_tgt_layer
                        continue
                    if attn_module.attn_type == AttentionType.DECODER:
                        if attn_module.sliding_window is not None:
                            kv_cache_spec[layer_name] = SlidingWindowSpec(
                                block_size=block_size,
                                num_kv_heads=attn_module.num_kv_heads,
                                head_size=attn_module.head_size,
                                dtype=draft_dtype_t,
                                sliding_window=attn_module.sliding_window,
                                use_mla=use_mla,
                            )
                        elif getattr(self.draft_model_config, "attention_chunk_size", None) is not None and isinstance(
                            attn_module, ChunkedLocalAttention
                        ):
                            kv_cache_spec[layer_name] = ChunkedLocalAttentionSpec(
                                block_size=block_size,
                                num_kv_heads=attn_module.num_kv_heads,
                                head_size=attn_module.head_size,
                                dtype=draft_dtype_t,
                                attention_chunk_size=int(self.draft_model_config.attention_chunk_size or 0),
                                use_mla=use_mla,
                            )
                        else:
                            kv_cache_spec[layer_name] = FullAttentionSpec(
                                block_size=block_size,
                                num_kv_heads=attn_module.num_kv_heads,
                                head_size=attn_module.head_size,
                                dtype=draft_dtype_t,
                                use_mla=use_mla,
                            )
                    elif attn_module.attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
                        continue
                    else:
                        raise ValueError(
                            f"Unknown attention type: {attn_module.attn_type}"
                        )

                # Compute available_memory to match target num_blocks semantics
                if hasattr(self.runner, "kv_cache_config") and self.runner.kv_cache_config is not None:
                    desired_num_blocks = self.runner.kv_cache_config.num_blocks
                else:
                    bs = block_size
                    mlen = self.draft_model_config.max_model_len
                    desired_num_blocks = max(1, (mlen + bs - 1) // bs)

                if kv_cache_spec:
                    page_size = get_uniform_page_size(kv_cache_spec)
                    if is_kv_cache_type_uniform(kv_cache_spec):
                        layers_divisor = len(kv_cache_spec)
                    else:
                        # Hybrid (uniform page size) case: group size is minimum count per type
                        same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
                        for ln, spec in kv_cache_spec.items():
                            same_type_layers[spec].append(ln)
                        layers_divisor = min(len(v) for v in same_type_layers.values())
                    available_memory = desired_num_blocks * page_size * layers_divisor
                else:
                    available_memory = 0

                # Build KV cache config for the draft model
                self.draft_kv_cache_config = get_kv_cache_config(
                    vllm_config_seperate_forward_context, kv_cache_spec, available_memory
                )

                # Create per-KV-group attention groups partitioned by backend type
                self.draft_attn_groups: list[list[AttentionGroup]] = []
                for kv_group in self.draft_kv_cache_config.kv_cache_groups:
                    layer_names = kv_group.layer_names
                    layers_in_group = get_layers_from_vllm_config(
                        vllm_config_seperate_forward_context, Attention, layer_names
                    )
                    by_backend: dict[type, list[str]] = defaultdict(list)
                    for lname, layer in layers_in_group.items():
                        backend = layer.get_attn_backend()
                        by_backend[backend].append(lname)

                    kv_attn_groups: list[AttentionGroup] = []
                    for backend, ln_subset in by_backend.items():
                        builder_cls: Any = backend.get_builder_cls()
                        builder = builder_cls(
                            kv_cache_spec=kv_group.kv_cache_spec,
                            layer_names=ln_subset,
                            vllm_config=vllm_config_seperate_forward_context,
                            device=self.device,
                        )
                        kv_attn_groups.append(
                            AttentionGroup(
                                backend=backend,
                                metadata_builder=builder,
                                layer_names=ln_subset,
                            )
                        )
                    self.draft_attn_groups.append(kv_attn_groups)

                # Allocate per KVCacheTensor raw buffers like GPUModelRunner._allocate_kv_cache_tensors
                kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
                for kv_tensor in self.draft_kv_cache_config.kv_cache_tensors:
                    raw = torch.zeros(kv_tensor.size, dtype=torch.int8, device=self.device)
                    for layer_name in kv_tensor.shared_by:
                        kv_cache_raw_tensors[layer_name] = raw

                # Reshape per-layer KV caches consistent with backend stride orders
                kv_caches: dict[str, torch.Tensor] = {}
                for group_idx, kv_attn_groups in enumerate(self.draft_attn_groups):
                    kv_spec = self.draft_kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
                    kv_attn_spec = cast(AttentionSpec, kv_spec)
                    for attn_group in kv_attn_groups:
                        backend = attn_group.backend
                        try:
                            stride_order = backend.get_kv_cache_stride_order()
                        except Exception:
                            stride_order = None
                        for layer_name in attn_group.layer_names:
                            raw = kv_cache_raw_tensors[layer_name]
                            assert raw.numel() % kv_attn_spec.page_size_bytes == 0
                            num_blocks = raw.numel() // kv_attn_spec.page_size_bytes
                            kv_shape = backend.get_kv_cache_shape(
                                num_blocks,
                                kv_attn_spec.block_size,
                                kv_attn_spec.num_kv_heads,
                                kv_attn_spec.head_size,
                            )
                            dtype = kv_attn_spec.dtype
                            if stride_order is not None:
                                kv_cache_shape = tuple(kv_shape[i] for i in stride_order)
                                inv_order = [stride_order.index(i) for i in range(len(stride_order))]
                                tensor = raw.view(dtype).view(kv_cache_shape).permute(*inv_order)
                            else:
                                tensor = raw.view(dtype).view(kv_shape)
                            kv_caches[layer_name] = tensor

                # Apply KV sharing to reuse buffers and update groups
                if getattr(self, "shared_kv_cache_layers_draft", None):
                    initialize_kv_cache_for_kv_sharing(
                        shared_kv_cache_layers=self.shared_kv_cache_layers_draft,
                        kv_cache_groups=self.draft_kv_cache_config.kv_cache_groups,
                        kv_caches=kv_caches,
                        attn_groups=self.draft_attn_groups,
                    )

                # Bind KV caches to the draft model forward context so attention layers can access them.
                self._draft_runner_kv_caches = []
                bind_kv_cache(
                    kv_caches,
                    vllm_config_seperate_forward_context.compilation_config.static_forward_context,
                    self._draft_runner_kv_caches,
                )

                # Cache draft-specific constants.
                self.draft_block_size = block_size
                self.draft_max_model_len = self.draft_model_config.max_model_len

            logger.info("LOADED DRAFT MODEL %s", self.draft_model_config.model)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int):
        assert self.draft_vllm_config is not None
        with set_forward_context(
            None,
            self.draft_vllm_config,
            num_tokens=num_tokens,
        ):
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None

            self.model(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
