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

# ---------------------------------------------------------------------------
# Prompt templates chosen to be *highly predictable* so speculative decoding
# can accept long draft spans. Keep them short & deterministic.
# ---------------------------------------------------------------------------
TEMPLATES: list[str] = [
    # Repetitive tokens (single word) -> extremely predictable.
    (
        "Repeat the word 'hello' exactly 120 times separated by a single "
        "space, then on a new line write: COUNT=120."
    ),
    # Simple counting sequence.
    (
        "List the numbers 1 through 60 separated by commas only (no space). "
        "Finish with a newline saying LENGTH=60."
    ),
    # Alphabet repetition.
    (
        "Output the lowercase English alphabet repeated 4 times without "
        "separators (total length 26*4). Then state: DONE."
    ),
    # JSON numeric array.
    "Produce a JSON array of the integers from 1 to 50 inclusive.",
    # Line-numbered repetition.
    (
        "Create 30 lines. Each line i (starting at 1) must be: i: "
        "speculative decoding accelerates generation."
    ),
    # Even numbers.
    "List the first 70 even numbers separated by a single space.",
    # Fixed token block (A's) with grouping.
    (
        "Write 200 capital letter A characters. Insert a newline after every "
        "50 characters."
    ),
    # Poem with identical line pattern (predictable line starts).
    (
        "Generate a 20-line poem where every line starts with 'Speed gains:' "
        "followed by the line number."
    ),
    # Simple table.
    (
        "Create a 10-row markdown table with two columns: n and n_squared "
        "for n=1..10."
    ),
    # Repetition with periodic marker.
    (
        "Print the word 'data' 160 times; after every 40 occurrences add the "
        "token |MARK|."
    ),
    # Fibonacci small range (deterministic sequence).
    "List the first 25 Fibonacci numbers separated by spaces.",
    # Binary sequence.
    "Output a line of 128 characters alternating 0 and 1 starting with 0.",
    # Squares.
    "Give the squares of 1..40 as 'i^2=val' each on its own line.",
    # Multiplication table.
    (
        "Provide a 12x12 multiplication table; rows each start with row "
        "number then a colon then 12 products separated by spaces."
    ),
    # Simple predictable narrative.
    (
        "Write 15 sentences each exactly 'Speculative decoding reduces "
        "latency.' numbered (1) .. (15)."
    ),
    # JSON mapping.
    (
        "Return a JSON object mapping letters a through j to their 1-based "
        "positions."
    ),
]


def build_conversations(num: int) -> list[list[dict[str, str]]]:
    """Return ``num`` single-turn conversations cycling through TEMPLATES."""
    return [
        [{"role": "user", "content": prompt}]
        for prompt in islice(cycle(TEMPLATES), num)
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
    conversations = build_conversations(args.num_convs)

    if args.list_prompts:
        for i, conv in enumerate(conversations, 1):
            print(f"[{i}] {conv[0]['content']}")
        return

    def do_run():
        run_batch_chat(
            client=client,
            model=args.model,
            conversations=conversations,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            max_inflight=max(1, args.max_inflight),
        )

    if args.loop:
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n##### LOOP ITERATION {iteration} #####")
                do_run()
                if args.loop_delay > 0:
                    time.sleep(args.loop_delay)
        except KeyboardInterrupt:  # pragma: no cover
            print("\nLoop interrupted by user; exiting.")
    else:
        do_run()


if __name__ == "__main__":
    main()
