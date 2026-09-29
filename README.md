# Live Answer Check

Students type an answer. A local embedding model computes cosine
similarity against multiple reference answers. The system displays the
highest score as a live meter. Below a threshold, students can request an
LLM hint. The system limits hints per student.

**Status:** Working demo. The question and reference answers are
placeholders. Replace `QUESTION_TEXT` and `REFERENCE_ANSWERS` in
`backend/main.py` with real midterm content before the exam.

## Setup

Run these commands:

```bash
cd extra-credit-pixel-display
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8001
```

Open http://localhost:8001 in a browser.

The first run downloads the embedding model (about 80MB). The system
caches the model after the first download.

## Configuration

Copy `.env.example` to `.env`. Set these values:

- `OPENROUTER_API_KEY`: Required for hints. The similarity meter works
  without this key.
- `OPENROUTER_MODEL`: Defaults to `openai/gpt-4o-mini`.
- `SIMILARITY_THRESHOLD`: A value from 0 to 1. Default: 0.55. This value
  sets when the hint button unlocks.
- `MAX_HINTS`: Maximum hints per student session. Default: 3.
- `DEMO_MODE`: Set to `true` to expose `/api/reveal-answer` and show the
  matched reference in the interface. Set to `false` for real students.
  This setting exposes the answer key when true.

## System design

**Similarity engine:** The system uses `sentence-transformers` with the
`all-MiniLM-L6-v2` model. The model runs locally. The system makes no
external calls for similarity scoring. The system embeds all reference
answers at startup. The system embeds each student update and compares it
against all references. The system reports the maximum similarity score.

**Multiple reference answers:** `REFERENCE_ANSWERS` in `backend/main.py`
stores several correct phrasings of the same answer. See "Known
limitations" for the reason.

**Live typing behavior:** The frontend sends a WebSocket update every
180ms during active typing. A standard debounce only updates after the
user stops typing; this system updates continuously instead. The frontend
animates the displayed score for smooth transitions.

**Concurrency:** The system runs each embedding call in a background
thread using `asyncio.to_thread`. The system limits PyTorch to one thread
per call using `torch.set_num_threads(1)`. This setting allows the thread
pool to parallelize calls across CPU cores.

Test conditions: 40 simultaneous connections, 8 seconds of continuous
typing, on an 8-core machine.
- Median latency: 89ms.
- 95th-percentile latency: 237ms.
- Throughput scales with CPU core count. Retest on the deployment host
  before the exam.

**Hints:** Hints require a button click. The system caps hints per
session. The hint prompt includes all reference phrasings.

## Known limitations

Cosine similarity measures topic proximity. It does not measure
correctness. Test results against this model and question:

| Case | Similarity |
|---|---|
| Reference answer itself | 100% |
| Wrong logic, dense reference vocabulary | 84% |
| Keyword list, no reasoning | 70% |
| Correct answer, independent wording, single reference | 62% |
| Correct answer, independent wording, multiple references | 80% |
| On-topic, does not answer the question | 25-28% |

Multiple reference answers reduce false negatives. A correctly-reasoned
answer now matches one of several valid phrasings.

Multiple reference answers do not reduce false positives. A wrong answer
with matching vocabulary still scores high. Adding reference phrasings
only raises scores. It cannot lower an incorrect score.

**Recommendation:** Use the live meter for engagement and feedback only.
Do not use the live score for grading. Score extra credit through a
separate LLM-rubric or manual review at submission time.

## Jev correctness check (experimental)

TypeSafe's Jev (`typesafe/jev-1.13`, via OpenRouter's decisions API) is a
structured decision model. It returns a calibrated probability instead of
free-form text. The interface shows it side by side with the cosine
similarity meter, for comparison. It does not affect hints or scoring.

Test results: correct on all 16 cases in the test bank. Three other
approaches were also tested: cosine similarity, a bigger embedding model,
and an NLI entailment model. Each of the three missed at least one case
that Jev got right. See `scratchpad/README.md` for the individual test
scripts and results.

**Cost:** About $0.0000234 per call, measured directly from the API's own
response. For 40 students over a 10-minute question window, at the
current 500ms update interval, this costs $0.23-$0.76 for the class. The
range depends on how much of that time students spend actively typing. A
submission-time-only check (one call per student) costs about $0.001 for
the same class.

**Latency:** 92ms-1.7s per call, measured under 40-student concurrent
load. Requests can complete out of order because of this. The frontend
sequence-numbers each request and discards any response older than the
one already displayed.

**Status:** Working, side-by-side only. Not yet used for hints, the live
threshold, or grading.

## Not yet built

- Scoring logic for extra credit points. Decision: score the final
  submitted answer, not the live number. The Jev correctness check above
  is the leading candidate for this.
- Per-student session tracking. Current sessions are in-memory. Sessions
  reset on server restart.
- Cloud deployment. Current tests used a local machine. Retest on the
  target host before the exam.
