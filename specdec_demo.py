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
#
# TEMPLATES can contain either:
# 1. Strings (converted to single-turn conversations)
# 2. Lists of dicts in ChatML format for multi-turn conversations
# ---------------------------------------------------------------------------
TEMPLATES: list[str | list[dict[str, str]]] = [
    # # Repetitive tokens (single word) -> extremely predictable.
    # (
    #     "Repeat the word 'hello' exactly 120 times separated by a single "
    #     "space, then on a new line write: COUNT=120."
    # ),
    # # Simple counting sequence.
    # (
    #     "List the numbers 1 through 60 separated by commas only (no space). "
    #     "Finish with a newline saying LENGTH=60."
    # ),
    # # Alphabet repetition.
    # (
    #     "Output the lowercase English alphabet repeated 4 times without "
    #     "separators (total length 26*4). Then state: DONE."
    # ),
    # # JSON numeric array.
    # "Produce a JSON array of the integers from 1 to 50 inclusive.",
    # # Line-numbered repetition.
    # (
    #     "Create 30 lines. Each line i (starting at 1) must be: i: "
    #     "speculative decoding accelerates generation."
    # ),
    # # Even numbers.
    # "List the first 70 even numbers separated by a single space.",
    # # Fixed token block (A's) with grouping.
    # (
    #     "Write 200 capital letter A characters. Insert a newline after every "
    #     "50 characters."
    # ),
    # # Poem with identical line pattern (predictable line starts).
    # (
    #     "Generate a 20-line poem where every line starts with 'Speed gains:' "
    #     "followed by the line number."
    # ),
    # # Simple table.
    # (
    #     "Create a 10-row markdown table with two columns: n and n_squared "
    #     "for n=1..10."
    # ),
    # # Repetition with periodic marker.
    # (
    #     "Print the word 'data' 160 times; after every 40 occurrences add the "
    #     "token |MARK|."
    # ),
    # # Fibonacci small range (deterministic sequence).
    # "List the first 25 Fibonacci numbers separated by spaces.",
    # # Binary sequence.
    # "Output a line of 128 characters alternating 0 and 1 starting with 0.",
    # # Squares.
    # "Give the squares of 1..40 as 'i^2=val' each on its own line.",
    # # Multiplication table.
    # (
    #     "Provide a 12x12 multiplication table; rows each start with row "
    #     "number then a colon then 12 products separated by spaces."
    # ),
    # # Simple predictable narrative.
    # (
    #     "Write 15 sentences each exactly 'Speculative decoding reduces "
    #     "latency.' numbered (1) .. (15)."
    # ),
    # # JSON mapping.
    # (
    #     "Return a JSON object mapping letters a through j to their 1-based "
    #     "positions."
    # ),
    # Single-turn conversation (string format)
    # (
    #     "Analyze the following customer reviews from different sources for "
    #     "three different smartphones - the latest iPhone, Samsung Galaxy, "
    #     "and Google Pixel - and provide an overall rating for each phone on "
    #     "a scale of 1 to 10. Consider the following complex and contradictory "
    #     "reviews:\n- TechRadar's review of the latest iPhone: The new iPhone "
    #     "is a stunning triumph of engineering that sets a new bar for "
    #     "smartphone performance and camera quality. However, the incremental "
    #     "design and high price mean it lacks the 'wow' factor of previous "
    #     "iPhones. Still, its power and intelligence are unrivaled.\n- CNET's "
    #     "review of the latest Samsung Galaxy: The Samsung Galaxy phone has "
    #     "plenty of high points, including an amazing screen, fast performance, "
    #     "solid battery life and an impressive array of camera options. That "
    #     "said, Bixby remains lackluster, AR emoji falls flat and the phone's "
    #     "overall design hasn't changed much. The new Galaxy is an amazing "
    #     "phone overall, but it has a few nagging weaknesses that keep it from "
    #     "achieving true greatness.\n- The Verge's review of the latest Google "
    #     "Pixel: Google's Pixel packs cutting-edge specs, innovative AI-powered "
    #     "software, and a killer camera into a sleek design. However, the phone "
    #     "has lackluster battery life, lacks expandable storage, and its "
    #     "performance stutters at times, especially considering its high price "
    #     "tag. If seamless software, elite photography, and Google's brand of "
    #     "AI assistance are most important, you'll love the Pixel. But the "
    #     "overall experience isn't as well-rounded as some competitors. Return "
    #     "the answer as a JSON object with the overall ratings for each phone "
    #     "out of 10, to one decimal place."
    # ),
    # # Multi-turn conversation examples (ChatML format)
    # [
    #     {"role": "user", "content": "Count from 1 to 10."},
    #     {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
    #     {"role": "user", "content": "Now count from 11 to 20."},
    # ],
    # [
    #     {"role": "user", "content": "What is 2 + 2?"},
    #     {"role": "assistant", "content": "2 + 2 = 4"},
    #     {"role": "user", "content": "What is 4 + 4?"},
    #     {"role": "assistant", "content": "4 + 4 = 8"},
    #     {"role": "user", "content": "What is 8 + 8?"},
    # ],
    # [
    #     {"role": "user", "content": "List three colors."},
    #     {"role": "assistant", "content": "Red, blue, green."},
    #     {"role": "user", "content": "List three more colors."},
    # ],
    [
        {
            "role": "user",
            "content": "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        },
        {
            "role": "assistant",
            "content": "**Island of Aloha: Unveiling the Hidden Gems of Hawaii**\n\nAs I stepped off the plane in Honolulu, the warm tropical air enveloped me, and the sweet scent of plumeria filled my senses. I was finally in Hawaii, the ultimate paradise on earth. My recent trip to the Aloha State was a journey of discovery, a fusion of stunning natural beauty, rich cultural heritage, and unforgettable experiences.\n\n**Immersing in Hawaiian Culture**\n\nMy adventure began in Honolulu, where I delved into the city's vibrant cultural scene. I visited the Bishop Museum, the largest museum in Hawaii, which showcases the state's history, science, and art. The museum's exhibits took me on a journey through time, from the ancient Hawaiian civilization to the present day. I was fascinated by the intricate carvings, colorful textiles, and traditional instruments on display.\n\nNext, I headed to the historic Pearl Harbor and USS Arizona Memorial, a poignant tribute to the lives lost during the infamous attack in 1941. The memorial's somber atmosphere was a powerful reminder of the importance of peace and understanding.\n\n**Must-See Attractions**\n\nHawaii is home to some of the most breathtaking natural wonders in the world. I spent a day exploring the stunning Na Pali Coast on Kauai, where towering sea cliffs, hidden beaches, and lush green valleys left me awestruck. The scenic drive along the coast was a thrilling experience, with breathtaking views at every turn.\n\nOn the Big Island, I visited the Hawaii Volcanoes National Park, where I hiked through the Thurston Lava Tube and witnessed the awe-inspiring power of volcanic activity. The park's unique landscapes, including the Kilauea caldera and the Mauna Loa summit, were a testament to the island's geological history.\n\n**Beach Life and Water Activities**\n\nHawaii's beaches are legendary, and I was determined to experience them firsthand. I spent a day soaking up the sun on Waikiki Beach, where I watched surfers ride the waves and swam in the crystal-clear waters. I also took a snorkeling trip to Molokini Crater, a crescent-shaped volcanic crater and marine sanctuary, where I spotted colorful fish and coral.\n\n**Local Flavors and Cuisine**\n\nHawaiian cuisine is a fusion of traditional Hawaiian, Asian, and American flavors, and I was eager to try the local specialties. I indulged in poke bowls, laulau, and shave ice, and sampled the state's famous macadamia nuts and Kona coffee. I also visited a local farm, where I learned about sustainable farming practices and tasted fresh produce.\n\n**Conclusion**\n\nMy trip to Hawaii was a journey of discovery, a celebration of the island's rich culture, stunning natural beauty, and warm hospitality. From the historic sites to the breathtaking landscapes, and from the local cuisine to the beach life, I experienced the best of Hawaii. If you're planning a trip to the Aloha State, I highly recommend immersing yourself in the local culture, exploring the must-see attractions, and indulging in the local flavors. Mahalo (thank you) for joining me on this island adventure!\n\n**Practical Information**\n\n* Best time to visit: September to November or March to May\n* Must-try dishes: Poke bowls, laulau, shave ice, and Kona coffee\n* Must-see attractions: Na Pali Coast, Hawaii Volcanoes National Park, Pearl Harbor and USS Arizona Memorial\n* Cultural experiences: Bishop Museum, traditional Hawaiian hula dance performances, and local farm tours\n* Beach life: Waikiki Beach, Hanauma Bay, and Molokini Crater\n\n**Getting There**\n\n* Fly into Honolulu International Airport (HNL)\n* Rent a car or take public transportation to explore the islands\n* Book accommodations in advance, especially during peak season\n\n**Budget Breakdown**\n\n* Flights: $800 - $1,200\n* Accommodations: $1,000 - $2,000\n* Transportation: $200 - $500\n* Food and activities: $500 - $1,000\n\n**Tips and Tricks**\n\n* Book tours and activities in advance to avoid sold-out situations\n* Pack light, breathable clothing and comfortable shoes\n* Respect local customs and traditions, especially when visiting sacred sites\n* Don't forget sunscreen, sunglasses, and a hat to protect yourself from the sun\n\nI hope you've enjoyed this travel blog post about my recent trip to Hawaii. If you have any questions or comments, please feel free to share them below.",
        },
        {
            "role": "user",
            "content": "Rewrite your previous response. Start every sentence with the letter A.",
        },
    ],
    # Add your custom multi-turn conversations here in ChatML format:
    # [
    #     {"role": "user", "content": "Your first message"},
    #     {"role": "assistant", "content": "Assistant response"},
    #     {"role": "user", "content": "Your follow-up message"},
    # ],
]


def build_conversations(num: int) -> list[list[dict[str, str]]]:
    """Return ``num`` conversations cycling through TEMPLATES.

    TEMPLATES can contain:
    - Strings: converted to single-turn conversations with user role
    - List[dict]: multi-turn conversations in ChatML format (used as-is)
    """
    conversations = []
    for template in islice(cycle(TEMPLATES), num):
        if isinstance(template, str):
            # Single-turn: convert string to user message
            conversations.append([{"role": "user", "content": template}])
        elif isinstance(template, list):
            # Multi-turn: use ChatML conversation as-is
            conversations.append(template)
        else:
            raise ValueError(f"Invalid template type: {type(template)}")

    return conversations


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
        # Extract speculative decoding acceptance lengths if provided by server.
        acceptance_lengths: list[int] = []
        try:
            model_extra = getattr(comp, "model_extra", None)
            if isinstance(model_extra, dict):
                val = model_extra.get("acceptance_lengths")
                if isinstance(val, list):
                    acceptance_lengths = [int(x) for x in val]
        except Exception:
            acceptance_lengths = []
        return idx, text, time.time() - started, acceptance_lengths

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
            results: dict[int, tuple[str, float, list[int]]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    got_idx, text, elapsed, acc = fut.result()
                    results[got_idx] = (text, elapsed, acc)
                except Exception as e:  # pragma: no cover
                    results[idx] = (f"<ERROR: {e}>", 0.0, [])
        # Emit in order for the batch.
        for idx in range(start + 1, start + 1 + len(batch)):
            text, elapsed, acc = results[idx]
            print(f"\n=== Conversation {idx} ===")
            # print(text)
            print(f"-- elapsed: {elapsed:.3f}s")
            if acc:
                print(f"-- acceptance_lengths: {acc} (total={sum(acc)})")
            else:
                print("-- acceptance_lengths: []")

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
        help=(
            "Number of conversations to submit (default 16). "
            "Can be single-turn or multi-turn."
        ),
    )
    p.add_argument(
        "--list-prompts",
        action="store_true",
        help=(
            "List the first N conversations that would be sent (N=--num-convs) "
            "and exit. Shows both single-turn and multi-turn conversations."
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
            if len(conv) == 1 and conv[0]["role"] == "user":
                # Single-turn conversation
                print(f"[{i}] {conv[0]['content']}")
            else:
                # Multi-turn conversation
                print(f"[{i}] Multi-turn conversation:")
                for j, msg in enumerate(conv):
                    role_prefix = f"  {msg['role'].upper()}: "
                    content = msg["content"]
                    # Truncate long content for readability
                    if len(content) > 100:
                        content = content[:97] + "..."
                    print(f"{role_prefix}{content}")
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
