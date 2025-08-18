#!/usr/bin/env python3
"""
VLLM Speculative Decoding Demo Script
"""

from vllm import LLM, SamplingParams
import os


def build_conversations() -> list[list[dict[str, str]]]:
    """Return multiple conversations (each a list of chat messages).

    We use llm.generate with manually applied chat templates so that vLLM's
    tokenizer injects the proper system/user/assistant tokens.
    """
    return [
        [
            {
                "role": "user",
                "content": (
                    "How many B's are in the word 'blueberry'? Explain "
                    "counting succinctly."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Explain the difference between a CPU and a GPU in one "
                    "concise sentence."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Compute 12345 * 678. Show clear calculation steps then "
                    "give the final answer."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Write a haiku about speculative decoding accelerating "
                    "generation."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "List three potential risks of AI systems and one "
                    "mitigation for each."
                ),
            }
        ],
    ]


def init_llm():
    """Initialize and return the LLM with speculative decoding configuration."""
    return LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        enforce_eager=True,
        speculative_config=dict(
            method="draft_model",
            model="meta-llama/Llama-3.2-1B-Instruct",
            num_speculative_tokens=6,
        ),
    )


def run_batch_chat(
    llm: LLM,
    sampling_params: SamplingParams,
    conversations: list[list[dict[str, str]]],
    show_full: bool = False,
):
    print(
        f"Submitting {len(conversations)} conversations as a single batched "
        "chat() call..."
    )
    # Batched chat: list[list[message]]; suppress type checker variance issue.
    outputs = llm.chat(conversations, sampling_params=sampling_params)  # type: ignore[arg-type]
    for i, out in enumerate(outputs, start=1):
        print(f"\n=== Conversation {i} ===")
        if out.outputs:
            print(out.outputs[0].text)
        else:  # pragma: no cover - defensive
            print("(No text output)")
        if show_full:
            print("-- Full object --")
            print(out)


def main():
    # Force v1 engine (explicit for this demo)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Helpful for debugging kernel errors (may slow things down)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Sampling params (tweak as desired)
    sampling_params = SamplingParams(temperature=0.1, top_p=1.0, max_tokens=128)

    llm = init_llm()
    conversations = build_conversations()
    run_batch_chat(llm, sampling_params, conversations, show_full=False)


if __name__ == "__main__":
    main()
