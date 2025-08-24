# Acceptance Length Degradation Examples

This document shows examples from falling_al.jsonl demonstrating the reduction in acceptance_length over time during speculative decoding.

## Example 1: Entry 1 (Good Performance)
**Accepted Lengths:** `[6, 0, 6, 6]`
**User Prompt:** Count word occurrences in Amazon paragraph
**Model Response:** "Here are the results:\n\nAmazon, 5\nriver, 4\nyou, 2"

## Example 2: Entry 50 (Degrading Performance) 
**Accepted Lengths:** `[4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 2, 0, 2, 0, 0, 1, ...]`
- Shows degradation with mostly 0s, occasional small acceptances (1-5 tokens)
- Much longer sequence with mostly failed predictions

## Example 3: Entry 100 (Complete Failure)
**Accepted Lengths:** All zeros for 58 consecutive predictions
- Complete failure - no tokens accepted at all

## Example 4: Entry 200 (Sustained Failure) 
**Accepted Lengths:** All zeros for 206 consecutive predictions
- Continues with complete failure over very long sequence

## Example 5: Entry 400 (Near End - Still Failing)
**Accepted Lengths:** `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- 23 consecutive failed predictions

## Example 6: Entry 476 (Final - Complete Failure)
**Accepted Lengths:** All zeros for 131 consecutive predictions  
- Ends with sustained complete failure

## Summary
The pattern shows clear degradation:
1. **Early:** Good performance with 6-token acceptances
2. **Middle-early:** Degraded performance with occasional 1-5 token acceptances
3. **Middle-late:** Complete failure with all-zero acceptance
4. **End:** Sustained complete failure through remainder of dataset

This demonstrates the "falling acceptance length" bug where speculative decoding performance degrades over time within a single session.