# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from unittest import mock

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform
from tests.v1.attention.utils import (
    BatchSpec,
    _Backend,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.v1.spec_decode.draft_model import DraftModelProposer

# Target and draft use distinct models to exercise separate-config path.
model_dir = "meta-llama/Llama-3.1-8B-Instruct"
draft_model_dir = "meta-llama/Llama-3.2-1B-Instruct"


def _create_proposer(
    num_speculative_tokens: int,
) -> DraftModelProposer:
    """
    Helper mirroring the pattern used in test_eagle to assemble a DraftModelProposer.
    """
    target_model_config = ModelConfig(
        model=model_dir,  # type: ignore[arg-type]
        runner="generate",  # type: ignore[arg-type]
        max_model_len=128,  # type: ignore[arg-type]
    )

    speculative_config = SpeculativeConfig(  # type: ignore[arg-type]
        target_model_config=target_model_config,  # type: ignore[arg-type]
        target_parallel_config=ParallelConfig(),  # type: ignore[arg-type]
        model=draft_model_dir,  # distinct draft model id  # type: ignore[arg-type]
        num_speculative_tokens=num_speculative_tokens,  # type: ignore[arg-type]
        method="draft_model",  # Explicitly set.  # type: ignore[arg-type]
    )  # type: ignore[arg-type]

    vllm_config = VllmConfig(  # type: ignore[arg-type]
        model_config=target_model_config,  # type: ignore[arg-type]
        cache_config=CacheConfig(),  # type: ignore[arg-type]
        speculative_config=speculative_config,  # type: ignore[arg-type]
        device_config=DeviceConfig(device=current_platform.device_type),  # type: ignore[arg-type]
        parallel_config=ParallelConfig(),  # type: ignore[arg-type]
        load_config=LoadConfig(),  # type: ignore[arg-type]
        scheduler_config=SchedulerConfig(),  # type: ignore[arg-type]
    )  # type: ignore[arg-type]

    # runner is required (unlike EagleProposer where tests inject later only).
    # We will attach a mock runner (with attn groups) in each test before propose().
    proposer = DraftModelProposer(
        vllm_config=vllm_config,
        device=torch.device(current_platform.device_type),
        runner=mock.MagicMock(),
    )
    return proposer


def test_prepare_inputs():
    """
    Validate prepare_inputs

    cu_target_query_lens: [0, a, a + b, a + b + c]
    num_rejected_tokens: [n1, n2, n3]
    num_tokens_per_req: [a - n1, b - n2, c - n3]
    cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    token_indices: [0, 1, ..., a - n1 - 1,
                    a, a + 1, ..., a + b - n2 - 1,
                    a + b, a + b + 1, ..., a + b + c - n3 - 1]
    Setup:
      query_lens = [4, 7, 5]
      num_rejected_tokens = [1, 3, 2]
    Expected:
      num_tokens_per_req = [3, 4, 3]
      cumulative = [0, 3, 7, 10]
      token indices chosen from original flattened sequence:
        [0,1,2, 4,5,6,7, 11,12,13]
    """
    device = torch.device(current_platform.device_type)

    batch_spec = BatchSpec(
        seq_lens=[4, 7, 5],
        query_lens=[4, 7, 5],
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    proposer = _create_proposer(num_speculative_tokens=1)

    num_rejected_tokens = torch.tensor([1, 3, 2], dtype=torch.int32)

    expected_cu_num_tokens = torch.tensor(
        [0, 3, 7, 10], dtype=torch.int32, device=device
    )
    expected_token_indices = torch.tensor(
        [0, 1, 2, 4, 5, 6, 7, 11, 12, 13],
        dtype=torch.int32,
        device=device,
    )

    updated_metadata, token_indices = proposer.prepare_inputs(
        common_attn_metadata, num_rejected_tokens
    )

    assert torch.equal(updated_metadata.query_start_loc, expected_cu_num_tokens)
    assert token_indices.shape[0] == expected_cu_num_tokens[-1].item()
    assert torch.equal(token_indices, expected_token_indices)


@mock.patch("vllm.v1.spec_decode.draft_model.get_model")
@mock.patch("vllm.v1.spec_decode.draft_model.get_layers_from_vllm_config")
def test_load_model(mock_get_layers, mock_get_model):
    """
    Ensure load_model:
      - calls underlying get_model
      - collects attention layer names into proposer.attn_layer_names
    """
    proposer = _create_proposer(num_speculative_tokens=2)

    # Mock layers returned by get_layers_from_vllm_config for the cloned config.
    mock_layers = {
        "layer.attn.0": mock.MagicMock(),
        "layer.attn.1": mock.MagicMock(),
    }
    mock_get_layers.return_value = mock_layers

    # Mock model instance
    mock_model = mock.MagicMock()
    mock_get_model.return_value = mock_model

    # Execute
    proposer.load_model()

    mock_get_model.assert_called_once()
    # Order not guaranteed, compare as sets
    assert set(proposer.attn_layer_names) == set(mock_layers.keys())
    assert proposer.model is mock_model


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 6, 8])
@pytest.mark.parametrize(
    "attn_backend", get_attn_backend_list_based_on_platform()
)
def test_propose(num_speculative_tokens, attn_backend, monkeypatch):
    """
    We speculate N tokens by performing N sequential forward passes.
    """
    if attn_backend == "TREE_ATTN":
        pytest.skip("TREE_ATTN not supported for DraftModelProposer.")

    # Some backends may not support multi-token speculative decode in other pathways;
    # For draft_model we only rely on basic metadata operations, so skip only known unsupported combos.
    if (
        attn_backend == "TRITON_ATTN_VLLM_V1"
        and not current_platform.is_rocm()
        and num_speculative_tokens > 1
    ):
        pytest.skip(
            "Skip multi-token TRITON_ATTN_VLLM_V1 on non-ROCm for consistency with eagle tests"
        )

    if attn_backend == "FLASH_ATTN_VLLM_V1" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    device = torch.device(current_platform.device_type)

    # Batch with two sequences of differing lengths.
    seq_len_1 = 5
    seq_len_2 = 3
    seq_lens = [seq_len_1, seq_len_2]
    total_tokens = sum(seq_lens)
    batch_size = len(seq_lens)
    vocab_size = 100
    hidden_size = 16  # arbitrary for test; DraftModelProposer does not rely on hidden_size attribute.

    proposer = _create_proposer(num_speculative_tokens=num_speculative_tokens)

    # We'll bypass real load_model; set attn_layer_names manually.
    proposer.attn_layer_names = ["layer.attn.0"]

    # Attention metadata builder for chosen backend.
    if attn_backend == "FLASH_ATTN_VLLM_V1":
        attn_metadata_builder_cls, _ = get_attention_backend(
            _Backend.FLASH_ATTN_VLLM_V1
        )
    elif attn_backend == "TRITON_ATTN_VLLM_V1":
        attn_metadata_builder_cls, _ = get_attention_backend(
            _Backend.TRITON_ATTN_VLLM_V1
        )
    else:
        raise ValueError(f"Unsupported attention backend: {attn_backend}")

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )

    # Attach the metadata builder to a mock runner.
    proposer.runner.attn_groups.append([mock.MagicMock()])
    proposer.runner.attn_groups[0][0].metadata_builder = attn_metadata_builder

    # Create flattened target tokens / positions.
    target_token_ids = torch.randint(
        0, vocab_size, (total_tokens,), device=device, dtype=torch.int32
    )
    target_positions = torch.cat(
        [
            torch.arange(seq_len_1, device=device),
            torch.arange(seq_len_2, device=device),
        ]
    )
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), device=device, dtype=torch.int32
    )

    batch_spec = BatchSpec(
        seq_lens=seq_lens,
        query_lens=seq_lens,
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )
    sampling_metadata = mock.MagicMock()

    # Deterministic model mock for multi-step speculation:
    # Desired produced token sequence per batch element i:
    #   step j -> base_token_ids[i] + j
    # So for num_speculative_tokens=3:
    #   sequence 0: 42,43,44
    #   sequence 1: 60,61,62
    base_token_ids = [42, 60]
    forward_call_count = {"count": 0}

    def forward_side_effect(
        *, input_ids=None, positions=None, inputs_embeds=None
    ):
        # Return dummy hidden states sized (#tokens, hidden_size)
        if input_ids is not None:
            length = input_ids.shape[0]
        else:
            length = positions.shape[0]
        return torch.zeros(length, hidden_size, device=device)

    def compute_logits_side_effect(sample_hidden_states, _):
        step = forward_call_count["count"]
        logits = torch.full(
            (sample_hidden_states.shape[0], vocab_size),
            -1000.0,
            device=device,
        )
        for i, base in enumerate(base_token_ids):
            logits[i, base + step] = 1000.0
        forward_call_count["count"] += 1
        return logits

    model_mock = mock.MagicMock()
    model_mock.side_effect = forward_side_effect
    model_mock.compute_logits.side_effect = compute_logits_side_effect
    proposer.model = model_mock

    # Execute propose (sequential single-token forwards until num_speculative_tokens reached).
    draft_tokens = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        next_token_ids=next_token_ids,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    assert draft_tokens.shape == (batch_size, num_speculative_tokens)
    expected = torch.zeros(
        (batch_size, num_speculative_tokens), dtype=torch.int64, device=device
    )
    for i in range(batch_size):
        for j in range(num_speculative_tokens):
            expected[i, j] = base_token_ids[i] + j
    assert torch.equal(draft_tokens, expected)


def test_propose_single_token_fast_path():
    """
    Explicit single-token test (num_speculative_tokens = 1) verifying early exit branch.
    """
    device = torch.device(current_platform.device_type)
    proposer = _create_proposer(num_speculative_tokens=1)
    proposer.attn_layer_names = ["layer.attn.0"]

    # Simple backend (choose FLASH if available else first).
    attn_metadata_builder_cls, _ = get_attention_backend(
        _Backend.FLASH_ATTN_VLLM_V1
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )
    proposer.runner.attn_groups.append([mock.MagicMock()])
    proposer.runner.attn_groups[0][0].metadata_builder = attn_metadata_builder

    seq_lens = [3, 2]
    total_tokens = sum(seq_lens)
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    target_token_ids = torch.randint(
        0, 50, (total_tokens,), device=device, dtype=torch.int32
    )
    target_positions = torch.cat(
        [torch.arange(l, device=device) for l in seq_lens]
    )
    next_token_ids = torch.tensor([7, 9], device=device, dtype=torch.int32)
    sampling_metadata = mock.MagicMock()

    # Model mock returns zeros for hidden states, compute_logits picks a fixed token.
    model_mock = mock.MagicMock()
    hidden_size = 8
    model_mock.return_value = torch.zeros(
        total_tokens, hidden_size, device=device
    )

    def compute_logits(sample_hidden_states, _):
        vocab_size = 100
        logits = torch.full(
            (sample_hidden_states.shape[0], vocab_size), -1000.0, device=device
        )
        logits[0, 11] = 0.0
        logits[1, 22] = 0.0
        return logits

    model_mock.compute_logits.side_effect = compute_logits
    proposer.model = model_mock

    out = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        next_token_ids=next_token_ids,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )
    assert out.shape == (2, 1)
    assert out[0, 0].item() == 11
    assert out[1, 0].item() == 22
