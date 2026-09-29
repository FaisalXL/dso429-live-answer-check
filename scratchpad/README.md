# Testing scripts

These scripts validate design decisions. They are not part of the
application. Run them against a local server. `efficacy_test.py` and
`multi_ref_test.py` run directly against the model.

- `efficacy_test.py`: Measures how cosine similarity tracks answer
  quality. Compares keyword-dense wrong answers against correctly-reasoned
  answers with different wording. Produced the recommendation against
  grading on the live score.
- `multi_ref_test.py`: Measures scoring against multiple reference
  phrasings using maximum similarity. Motivated the `REFERENCE_ANSWERS`
  dictionary structure.
- `load_test.py`: Sends 40 simultaneous connections at once. Measures
  round-trip latency under burst load.
- `sustained_load_test.py`: Sends 40 connections continuously for 8
  seconds. Measures throughput and latency under sustained load.
  Represents realistic exam conditions more closely than `load_test.py`.
- `regression_test.py`: Runs a fixed case bank against the live server.
  Cases include novel correct wording, wrong-logic answers, keyword
  lists, partial answers, and edge cases. Checks correctness and
  determinism. Tracks whether known issues persist after a change.
- `concept_decomposition_test.py`: Tests scoring against separate
  mechanism/risk concept references instead of one blended reference.
  Result: catches partial answers, but rejects a genuinely correct answer
  and does not stop false positives. Not adopted.
- `bigger_model_test.py`: Compares `all-MiniLM-L6-v2` against
  `all-mpnet-base-v2` on the same case bank. Result: mixed effect, and it
  lowers the score of a correct answer. Bigger model, same approach, does
  not fix the false-positive problem.
- `nli_entailment_test.py`: Tests an NLI cross-encoder (entailment /
  contradiction / neutral) instead of cosine similarity. Result: misses 2
  of 3 false-positive cases with high confidence. Not adopted.
- `llm_correctness_check_test.py`: Tests a single LLM call as a
  correctness check instead of a similarity metric. Result: correct on
  all tested cases, including every case the other approaches missed.
  Recommended approach for scoring, not live display.
- `jev_decisions_test.py`: Tests TypeSafe's Jev decisions API
  (`typesafe/jev-1.13`) as a correctness check, on the 5 hardest cases.
  Result: correct on all 5, with informative probabilities, not just a
  binary verdict.
- `jev_full_regression_test.py`: Runs the full 16-case bank (same cases
  as `regression_test.py`) through Jev. Result: correct on all 16. This
  includes the partial-answer case that every other tested approach got
  wrong.
- `jev_progressive_typing_test.py`: Sends a growing answer word-by-word,
  simulating live typing. Checks whether the score ramps smoothly or
  jumps. Result: ramps smoothly for a correct answer, stays flat and low
  for a wrong-logic answer throughout.
- `jev_sustained_load_test.py` / `jev_sustained_load_test_180ms.py`: 40
  simulated students, continuous requests for 8 seconds, at a 500ms and
  a 180ms interval. Result: zero errors at both, consistent latency
  distribution. Per-request latency (measured: 92ms-1.7s tail) can exceed
  the send interval. Responses can then arrive out of order. This is why
  the live integration in `backend/main.py` sequence-numbers requests and
  discards stale responses.
