# Standalone Draft Model Support - V1 Engine

Two parts:
1. Enabling the standalone draft model
2. Performing the same patch to return accepted_lengths per response, by modifying the OpenAI Response format.

## Enabling the draft model
Draft models get created inside `vllm/v1/worker/gpu_model_runner.py`.
They are loaded entirely on the last pipeline rank. All draft models must fit on a single GPU (with room for the target model layers).

Need to add a `Proposer` class, similar to those in:
* `vllm/v1/spec_decode/ngram_proposer.py`
* `vllm/v1/spec_decode/eagle.py`

There doesn't look to be any ABC or defined interface as of yet, but looks like proposers follow this implicit interface:

```python
class StandaloneDraftProposer:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Initialize with draft model config
        pass

    def propose(self,
                token_ids: torch.Tensor,  # [num_tokens] - the input token sequence
                sampling_metadata: SamplingMetadata) -> torch.Tensor:
        pass

    def load_model(self, draft_model: nn.Module): -> None:
        pass

    def dummy_run(self, num_tokens: int) -> None:
        # Warmup run with dummy token sequence
        pass
```

We'll also need a way to pass in the full token sequence. This looks like it's managed by `gpu_model_runner.py`, in particular: `GPUModelRUnner.propose_draft_token_ids`.
It looks like `sampled_token_ids` is the full input sequence. ~Will have to run and find out.~
Right now it's one token at a time. We need the initial prefill sequence and then additional verified tokens up to the current time step. 

## Returning Accepted Lengths with the chat completion.
