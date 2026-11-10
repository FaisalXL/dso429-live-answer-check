# Live Answer Lab

An assignment with two question types. Jev, a decision model from
TypeSafe, scores each answer while the student works.

- **Live questions.** The student types an answer. Jev scores its
  correctness while the student types. Below a threshold, the student
  can request a hint.
- **Fix-the-AI questions.** An AI gives an answer with planted mistakes.
  The student says what is wrong. The AI fixes only what the student
  points out. Jev scores each new version and checks each planted
  mistake.

After each question, a feedback box earns 1 credit for any non-empty
answer. At the end, a checkbox asks if Claude Code may help grade the
assignment. If the student leaves it unchecked, a person grades it by hand.

**Status:** Working demo. The questions are placeholders about Pedalwise,
a fictional bike-share company. Replace them in `backend/questions.json`.

## Setup

```bash
cd extra-credit-pixel-display
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8001
```

Open http://localhost:8001 in a browser.

## Configuration

Copy `.env.example` to `.env`. Set these values:

- `OPENROUTER_API_KEY`: Required. Jev, hints, and AI revisions all use it.
- `OPENROUTER_MODEL`: The model for hints. Default: `openai/gpt-4o-mini`.
- `REVISION_MODEL`: The model that rewrites the AI's answer in fix-the-AI
  questions. Default: `anthropic/claude-sonnet-5.5`. See "Fix-the-AI
  rounds" before you change it.
- `HINT_THRESHOLD`: The hint button unlocks below this Jev score.
  Default: 0.5.
- `PASS_THRESHOLD`: A fix-the-AI question is solved when every mistake
  is fixed and the Jev score reaches this value. Default: 0.8.
- `MAX_HINTS`: Hints per student per question. Default: 3.
- `DEMO_MODE`: `true` shows a question navigator and an answer-key box.
  Set it to `false` for real students. It exposes the answer key.
- `DEV_COSINE`: `true` shows a cosine-similarity readout for comparison.
  Install `requirements-dev.txt` first. Default: `false`.

## Writing questions

Each entry in `backend/questions.json` has these fields:

- `type`: `live` or `correction`.
- `prompt`: The question the student sees.
- `reference_answer`: The correct answer. Jev uses it. Students never
  see it unless `DEMO_MODE` is on.
- `jev`: The `instructions` and `criteria` that Jev uses to judge
  correctness.
- For `correction` questions only: `seed_answer` (the flawed AI answer),
  `mistakes` (one `error` description per planted mistake), and
  `max_rounds`.

**Write the `jev` instructions for each question by hand.** Name the
ideas that a correct answer must contain. A generic version was tested
and failed. It cut the score of a correct answer from 90% to 45%. See
`scratchpad/jev_generic_instructions_test.py`.

## How it works

**Live scoring.** The browser sends the answer over a WebSocket at most
every 500ms. The server keeps only the newest text per question. It runs
one Jev call at a time. Fast typing therefore never builds a backlog.

**Fix-the-AI rounds.** Each round runs three steps:

1. Jev decides whether the student's feedback names a specific claim.
   If it does not, the answer stays unchanged and the student sees why.
   Prompt rules alone did not stop the model from acting on "fix it."
2. The revision model changes only the claims that the feedback
   addresses. It applies wrong feedback too, so a wrong critique makes
   the answer worse.
3. One Jev call scores the new answer and checks each planted mistake.

A round is solved when every mistake is fixed and the score reaches
`PASS_THRESHOLD`. The server rejects an answer that the model cut off
before the end. A cut-off answer drops the remaining wrong claims, and
Jev would count those mistakes as fixed.

`scratchpad/assignment_validation_test.py` tested three revision models.
`gpt-4o-mini` left broken half-sentences. `claude-haiku-5.5` added ideas
that the student did not give. `claude-sonnet-5.5` kept to the student's
words and passed every check. Run that test again after you change the
model.

**Submission.** The server scores every final answer again on submit.
The stored scores therefore do not depend on the browser. The server
writes each record to `submissions/<student-id>_<time>.json`. The
student can also download a copy.

## Deployment

`render.yaml` defines a free-tier web service on Render. Create a
Blueprint from this repo. Enter `OPENROUTER_API_KEY` in the Render
dashboard; the repo does not store it.

The deployed build does not include PyTorch, so it needs little memory.

## Known limitations

- **Storage.** Render's free tier deletes local files on each restart.
  Submissions saved there are lost. Use a database or the downloaded
  copies before real use.
- **Identity.** Students type their own name and ID. The app does not
  check them.
- **Round limit.** The browser enforces the round limit. The record
  shows every round, so a grader can see extra rounds.
- **Jev instructions.** Each new question needs hand-written Jev
  instructions. See "Writing questions."

## Background

This project started as a live cosine-similarity meter. Tests showed
that cosine similarity measures topic, not correctness. A wrong answer
with the right words scored 84%. A bigger embedding model and an NLI
model did not fix this. Jev scored all 16 test cases correctly. The test
scripts and results are in `scratchpad/README.md`.
