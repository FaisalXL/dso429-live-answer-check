# Testing scripts behind the design decisions

Standalone scripts used to validate design choices before shipping them.
Not part of the app; run manually against a local server (or, for
`efficacy_test.py`/`multi_ref_test.py`, directly against the model) when
re-validating a change.

- `efficacy_test.py` — measures how well cosine similarity tracks actual
  answer quality (wrong-but-keyword-dense vs. correct-but-different-wording,
  etc). Motivated the "don't grade off the live number" recommendation in
  the main README.
- `multi_ref_test.py` — measures the effect of scoring against multiple
  reference-answer phrasings (max similarity) instead of one. Motivated
  adding `REFERENCE_ANSWERS` as a dict instead of a single string.
- `load_test.py` — 40 simultaneous connections firing at once (worst-case
  burst), measures round-trip latency.
- `sustained_load_test.py` — 40 connections continuously sending for 8
  seconds (worst-case sustained load), measures throughput and latency
  distribution. This is the more realistic stress test of the two.
