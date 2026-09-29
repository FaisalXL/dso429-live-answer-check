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
