#!/usr/bin/env python3
"""Speculative decoding demo using the OpenAI-compatible vLLM server.

This replaces the previous direct (in-process) `vllm.LLM` usage with a
client invoking the running server via its OpenAI-compatible REST API.

Run the server separately (see added VS Code launch config) with
`--speculative_config` pointing to a draft model, then execute this
client to issue batched chat completion requests.
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import cycle, islice
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def build_conversations() -> list[list[dict[str, str]]]:
    """Return multiple conversations (each a list of chat messages)."""
    return [
        [
            {
                "role": "user",
                "content": (
                    "You are an expert linguist and meticulous reasoning assistant. "
                    "First, restate the problem. Then count how many times the "
                    "letter 'B' (case-insensitive) appears in the word "
                    "'blueberry', explicitly showing each character with its "
                    "index. After counting, briefly explain whether frequency "
                    "analysis of single letters can ever be ambiguous in "
                    "English words and why or why not. Finish with a "
                    "one-sentence concise answer."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Provide a detailed yet clear comparison between modern "
                    "server-grade CPUs and data-center GPUs. Cover: (1) "
                    "architectural execution model (out-of-order, SIMD/SIMT, "
                    "warps), (2) memory hierarchy and bandwidth vs latency "
                    "trade-offs, (3) typical workload traits that favor each, "
                    "(4) power efficiency considerations for large-scale "
                    "inference, and (5) an illustrative micro-example (e.g., "
                    "matrix multiply) contrasting scheduling. Conclude with a "
                    "concise 2-sentence summary."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Carefully compute 12,345 * 678. Show the long multiplication "
                    "in a neatly formatted stepwise fashion, explain any carries. "
                    "Then verify using an alternative method (e.g., break 678 "
                    "into (700 - 22) or use distributive property). State the "
                    "final "
                    "product and give a quick sanity check based on order of "
                    "magnitude."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Compose THREE distinct haiku (traditional 5-7-5 syllable "
                    "structure) about speculative decoding speeding up large "
                    "language model text generation. Each haiku should "
                    "emphasize a different theme: (1) parallel drafting, (2) "
                    "verification and pruning, (3) user experience / latency. "
                    "After the haiku, add a one-paragraph prose explanation "
                    "(<=120 words) of how speculative decoding works "
                    "conceptually."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Identify five significant risks of deploying large AI "
                    "systems in production (e.g., hallucination, privacy "
                    "leakage, prompt injection, model drift, unfair bias). For "
                    "EACH: (a) 1-sentence risk description, (b) a short "
                    "real-world scenario, (c) one concrete mitigation with a "
                    "measurable control, (d) a key metric to monitor. Finish with "
                    "a compact table summarizing risk -> mitigation -> metric."
                ),
            }
        ],
    ]


def run_batch_chat(
    client: OpenAI,
    model: str | None,
    conversations: list[list[dict[str, str]]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_inflight: int,
) -> None:
    if model is None:
        models = client.models.list()
        if not models.data:
            raise RuntimeError("No models returned by server.")
        model = models.data[0].id
    print(
        (
            "Submitting {} conversations via OpenAI chat.completions "
            "to model '{}'..."
        ).format(len(conversations), model)
    )
    start_total = time.time()

    def _issue(idx_conv: tuple[int, list[dict[str, str]]]):
        idx, conv = idx_conv
        started = time.time()
        comp = client.chat.completions.create(
            model=model,
            messages=conv,  # type: ignore[arg-type]
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        if not hasattr(comp, "choices"):
            raise RuntimeError("Streaming object returned unexpectedly")
        text = str(comp.choices[0].message.content)  # type: ignore[union-attr]
        return idx, text, time.time() - started

    # Process in batches of size max_inflight while preserving order.
    for start in range(0, len(conversations), max_inflight):
        batch = conversations[start : start + max_inflight]
        batch_id = start // max_inflight + 1
        print(f"\n--- Batch {batch_id} ({len(batch)} requests) ---")
        with ThreadPoolExecutor(max_workers=max_inflight) as ex:
            futures = {
                ex.submit(_issue, (start + i + 1, conv)): (start + i + 1)
                for i, conv in enumerate(batch)
            }
            results: dict[int, tuple[str, float]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    got_idx, text, elapsed = fut.result()
                    results[got_idx] = (text, elapsed)
                except Exception as e:  # pragma: no cover
                    results[idx] = (f"<ERROR: {e}>", 0.0)
        # Emit in order for the batch.
        for idx in range(start + 1, start + 1 + len(batch)):
            text, elapsed = results[idx]
            print(f"\n=== Conversation {idx} ===")
            print(text)
            print(f"-- elapsed: {elapsed:.3f}s")

    print(f"\nCompleted in {time.time() - start_total:.3f}s total.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speculative decoding OpenAI client demo"
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        help="Server base URL (e.g. http://localhost:8000/v1)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (EMPTY for local server)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Explicit model id to use (defaults to first listed)",
    )
    p.add_argument(
        "--num-convs",
        type=int,
        default=16,
        help="Number of easy, repetitive conversations to submit (default 16)",
    )
    p.add_argument(
        "--list-prompts",
        action="store_true",
        help=(
            "List the first N prompts that would be sent (N=--num-convs) "
            "and exit"
        ),
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Continuously resend the batch in a loop until interrupted",
    )
    p.add_argument(
        "--loop-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between loop iterations (default 0)",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=16,
        help=(
            "Maximum number of parallel requests in flight at a time (batch "
            "size). Default 16"
        ),
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--wait-startup",
        action="store_true",
        help="Poll the server until ready before sending requests",
    )
    p.add_argument(
        "--wait-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for readiness when using --wait-startup",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Set defaults (client side; server must be started separately with same)
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    if getattr(args, "wait_startup", False):
        deadline = time.time() + args.wait_timeout
        attempt = 0
        while True:
            attempt += 1
            try:
                models = client.models.list()
                if models.data:
                    break
            except Exception:  # pragma: no cover
                pass
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Server not ready after {args.wait_timeout} seconds"
                )
            time.sleep(min(0.25 * attempt, 2.0))
    conversations = build_conversations()
    run_batch_chat(
        client=client,
        model=args.model,
        conversations=conversations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
