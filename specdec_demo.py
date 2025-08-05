#!/usr/bin/env python3
"""
VLLM Speculative Decoding Demo Script
"""

from vllm import LLM, SamplingParams
import os


def main():
    # force v1 engine
    os.environ["VLLM_USE_V1"] = "1"
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.95, max_tokens=128
    )

    # Initialize LLM with speculative decoding configuration
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        enforce_eager=True,
        speculative_config=dict(
            method="draft_model",
            model="meta-llama/Llama-3.2-1B-Instruct",
            num_speculative_tokens=6,
        ),
    )

    # Generate response
    output = llm.chat(
        messages=[
            {
                "role": "user",
                "content": "How many B's in blueberry? Count using your fingers, no peeking!",
            },
        ],
        sampling_params=sampling_params,
    )

    # Print the response text
    print("Response:")
    print(output[0].outputs[0].text)

    # Print full output details for debugging
    print("\nFull output object:")
    print(output)


if __name__ == "__main__":
    main()
