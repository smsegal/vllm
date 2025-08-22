# Speculative Decoding Acceptance Lengths and Draft Window Design — Summary

This document summarizes the design decisions, bugs fixed, and implementation
work to robustly track speculative decoding acceptance lengths per request and
stabilize acceptance rates over long runs by making the draft model use a
coherent context window that mirrors the target model.

## Goals

- Associate per-step speculative acceptance length with the correct request.
- Return a per-request list of acceptance lengths in OpenAI-compatible
  responses (one element per speculative block).
- Prevent late-run collapse of acceptance (all zeros) caused by stale/incorrect
  draft KV/cache state.
- Reuse vLLM’s existing attention/block-table machinery rather than duplicating
  logic or relying on ad-hoc resets.

## Key Concepts

- Acceptance length (per step): number of draft tokens accepted by rejection
  sampling for a given request during that decoding step (block).
- Per-request acceptance list: a growing list like `[4, 3, 6, 0, 1, 1, 6]`
  attached to the request, one entry per speculative block.
- Draft model coherence: the draft model must evaluate tokens in the context of
  a valid, up-to-date slice of the target’s sequence to produce useful
  proposals. The authoritative mapping of context to KV slots is the **block
  table** maintained for the target.

## Bugs / Issues Identified

1. Acceptance lengths appended for non-speculative steps
   - Cause: appending zeros when a request had no spec tokens scheduled or when
     a sampled token was discarded (partial prefill).
   - Fix: Only attach/append acceptance_length when the request had
     `scheduled_spec_decode_tokens` that step.

2. Logging crash in SpecDecoding metrics
   - Cause: format string mismatch (string argument fed to a `%f` placeholder).
   - Fix: Corrected the format arguments; eliminated the stray string.

3. Late-run acceptance collapse (all zeros)
   - Root cause: draft model KV/buffer state became stale or incoherent over
     many steps; propose() only used the current step’s slice, not a coherent
     recent context. As sequences grew, draft predictions drifted from target.

## Design: Coherent Draft Context Window

Principle: Each speculative proposal must be computed with the same effective
context as the target. We reconstruct a **per-request context window** (full
current context or a bounded sliding window) and re-encode it into the draft’s
KV every propose. We reuse the target’s block table to compute slot_mapping for
the draft window — this implicitly “clears/resets” draft KV without ad-hoc
memory operations.

### Why this works

- The target’s block table is the source of truth for where valid keys/values
  live. By rebuilding the draft KV from the most recent context window using
  the same slot mapping, the draft model always attends to the correct tokens,
  avoiding stale state and acceptance collapse.
- No periodic zeroing required; the correct window re-encoding updates KV slots
  consistently with the target’s cache lifecycle (including sliding/reuse).

## Implementation Highlights

### 1) Acceptance length plumbing

- In the rejection sampler, `analyze_speculation` already returns
  `accepted_count` per request (correctly unflattened via
  `SpecDecodeMetadata.cu_num_draft_tokens`).
- Runner (GPUModelRunner.execute_model):
  - Computes `acceptance_lengths = analyze_speculation(...).accepted_count`
    when speculative decoding is active.
  - Returns `acceptance_lengths` via `ModelRunnerOutput.acceptance_lengths`.
- Scheduler (update_from_output):
  - Attaches `acceptance_length` to `EngineCoreOutput` only when the request
    had `scheduled_spec_decode_tokens` that step.
- OutputProcessor:
  - Appends `EngineCoreOutput.acceptance_length` to the per-request
    `RequestState.acceptance_lengths` list if present.
  - The final `RequestOutput` includes `acceptance_lengths`.
- OpenAI Chat API:
  - The final aggregated response includes `acceptance_lengths` as a
    vLLM-specific field (will appear in `completion.model_extra` in typed SDKs).

### 2) Draft window (principled coherence)

- Runner (propose_draft_token_ids, method == "draft_model"):
  - Builds per-request window lengths Li (currently full sequence length;
    trivially changed to `min(W, seq_len)` for a bounded window).
  - Gathers the last Li tokens per request from the CPU token matrix into GPU
    tensors (`draft_input_ids`, `draft_positions`).
  - Delegates metadata construction to the drafter to avoid logic duplication.

- Drafter (DraftModelProposer.prepare_inputs):
  - New/renamed method that:
    - Computes `query_start_loc` for the window.
    - Derives per-token `req_indices` and `positions` (0..Li-1 within each
      request).
    - Reuses `BlockTable.compute_slot_mapping(req_indices, positions)` and
      `commit_slot_mapping(total)` to create a **GPU** `slot_mapping` for the
      window (KV group 0).
    - Returns a `CommonAttentionMetadata` (KV group 0) pointing to the correct
      block table tensor and slot mapping for the draft window.

- Draft propose():
  - Prefills the window using the returned metadata via `set_forward_context`.
  - Runs the chain decode for `num_speculative_tokens` using proper positions
    and the draft’s metadata builder (no ad-hoc clamping or padding hacks).

## Optional knobs (future)

- `speculative_config.draft_context_window`: bound Li by a fixed W for
  performance: `Li = min(W, seq_len_i)`.
- Chain length headroom per request: reduce speculative chain when near the
  model’s max window or sliding-window bounds to match the target’s boundary
  behavior precisely.

## Debugging/Validation Aids

- Runner asserts (debug mode):
  - If `scheduled_spec_decode_tokens[req_id]` exists, then
    `acceptance_length == len(generated_token_ids) - 1`.
- Temporary logs per proposal:
  - Request id, Li (window size), accepted_count.

## Summary of Changes (Code-Level)

- Acceptance-length propagation:
  - `ModelRunnerOutput.acceptance_lengths: list[int]`
  - Attach to per-request `EngineCoreOutput.acceptance_length` only for
    requests with speculative tokens that step.
  - Append to `RequestState.acceptance_lengths` and return on final output.
- Draft window preparation:
  - `DraftModelProposer.prepare_inputs(window_seq_lens_cpu, token_indices_cpu, block_table)`
    constructs a `CommonAttentionMetadata` for the draft window, reusing
    `BlockTable.compute_slot_mapping`.
  - Runner calls `prepare_inputs` and passes the metadata to `drafter.propose`.

## Expected Results

- Acceptance lists now reflect true speculative blocks only (no spurious zeros).
- Acceptance remains strong across long runs; no late-step collapse due to
  stale draft KV.
- Behavior stays in lockstep with the target’s cache lifecycle (block table),
  reusing existing vLLM code paths for slot mapping and metadata building.

---
Last updated: see Git history for the exact commit set that introduced
`ModelRunnerOutput.acceptance_lengths`, conditional attachment in the scheduler,
`RequestState.acceptance_lengths` accumulation, and the draft window
`prepare_inputs` redesign.