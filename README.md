# Live Answer Check — midterm extra-credit demo

Students type an answer; a local embedding model computes live cosine
similarity against several valid reference answers (max score wins) and
renders it as a live pixel meter that updates continuously while they type.
Below a threshold, a capped "Get a hint" button calls an LLM (OpenRouter)
for a short nudge — never the answer itself.

**Status:** working demo, presented to Professor Charlie and well received.
Question/reference content is still a placeholder — swap in the real
midterm question before exam day (`QUESTION_TEXT` and `REFERENCE_ANSWERS`
in `backend/main.py`).

## Run it

```bash
cd extra-credit-pixel-display
source .venv/bin/activate   # venv already has fastapi/uvicorn/sentence-transformers installed
uvicorn backend.main:app --reload --port 8001
```

Then open http://localhost:8001

First boot downloads the embedding model (~80MB, one-time, cached after).

## Config

Copy `.env.example` to `.env` and fill in:

- `OPENROUTER_API_KEY` — required for hints to work (similarity meter works without it)
- `OPENROUTER_MODEL` — defaults to `openai/gpt-4o-mini`
- `SIMILARITY_THRESHOLD` — 0-1, default 0.55, where the hint button unlocks
- `MAX_HINTS` — hints allowed per student session, default 3
- `DEMO_MODE` — `true` exposes `/api/reveal-answer` and a "reveal reference
  answer" toggle + "matched reference" label in the UI, for demoing.
  **Must be `false` for any run with real students** — it hands them the
  answer key.

## How it works

- **Similarity engine:** `sentence-transformers` (`all-MiniLM-L6-v2`), fully
  local, no external call. The reference answers are embedded once at
  startup; each student update is embedded and compared via cosine
  similarity against all references, and the **max** score is used.
- **Multiple reference answers:** `REFERENCE_ANSWERS` in `backend/main.py`
  holds several differently-worded, equally-correct phrasings of the answer
  (see "Known limitations" below for why this matters).
- **Live-typing feel:** the frontend throttles to one WebSocket update every
  ~180ms *while actively typing* (not just after a pause), so the meter
  visibly ticks the whole time someone writes, plus a smooth count-up
  animation on the displayed number.
- **Concurrency:** each embedding call runs in a background thread
  (`asyncio.to_thread`), and PyTorch is capped to 1 thread per call
  (`torch.set_num_threads(1)`) so the thread pool can genuinely parallelize
  across students instead of each call hogging all CPU cores. Load-tested:
  40 students continuously typing for 8 sustained seconds → p50 latency
  89ms, p95 237ms, on an 8-core machine. Re-test on the actual deployment
  host before exam day; throughput scales with core count.
- **Hints:** on-demand only (button click, not automatic), capped per
  session, calls OpenRouter with all reference phrasings as context so it
  can nudge based on whichever the student is closest to.

## Known limitations (read before trusting the score for grading)

Cosine similarity measures topical/semantic *proximity*, not correctness.
Tested empirically against this exact model+question:

| Case | Similarity |
|---|---|
| Reference answer itself | 100% |
| Wrong logic, but uses reference vocabulary densely | **84%** |
| Plain keyword list, no actual reasoning | 70% |
| Correct answer, independently-worded (not one of the 3 references) | 62% → **80%** with multi-reference |
| On-topic but doesn't answer the question | 25-28% |

**Multiple reference answers (this feature) fixes false negatives** — a
correct answer phrased differently now scores fairly, because it just needs
to be close to *any* valid phrasing, not one specific one. Tested in
`scratchpad`-style scripts before shipping (see git history / conversation
log for the raw numbers).

**It does not fix false positives.** A wrong answer that reuses reference
vocabulary densely still scores deceptively high — adding more correct
reference phrasings only ever raises scores, it never catches wrong ones.

**Recommendation:** keep the live meter as an engagement/feedback tool, not
the actual grading mechanism. Score real extra credit via a separate
LLM-rubric or TA review at submission (same draft-assist pattern as HW2),
not the live number. This is also a legitimately good teaching moment for
an AI-literacy-themed course — consider surfacing this exact
gameability to students as part of the exercise.

## Not yet built

- Scoring → extra credit points mapping (design call made: score off final
  similarity/answer at submission, not live number — see limitations above).
- Real per-student session/identity tracking — currently in-memory,
  resets on server restart. Needed before running with real students.
- Cloud deployment — runs locally only so far. Small VM (2-4 vCPU) should
  be enough based on load testing; re-test on the actual host before relying
  on it for the exam.
- "JEV" — a second idea from Professor Charlie's feedback, not yet scoped.
