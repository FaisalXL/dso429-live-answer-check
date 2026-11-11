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

**Status:** Ready for an extra-credit run. The four questions are about
Pedalwise, a fictional bike-share company. Edit them in
`backend/questions.json`.

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
- `ADMIN_KEY`: The password for the CSV export. Empty turns the export
  off.
- `CLOSES_AT`: The server refuses submissions after this time. Use the
  form `2026-11-22T23:59:00-08:00`. A time with no offset is Los Angeles
  time. A bad value stops the server at startup.
- `STUDENT_ID_PATTERN`: A regular expression that each student ID must
  match. Default: 10 digits.
- `STUDENT_ID_HINT`: The ID format that error messages show.
- `SUBMISSIONS_DIR`: The folder for submission records. Default:
  `submissions/`.

## Writing questions

Each entry in `backend/questions.json` has these fields:

- `type`: `live` or `correction`.
- `prompt`: The question the student sees.
- `reference_answer`: The correct answer. Jev uses it. Students never
  see it unless `DEMO_MODE` is on.
- `jev`: The `instructions` and `criteria` that Jev uses to judge
  correctness.
- For `correction` questions only: `seed_answer` (the flawed AI answer),
  `mistakes`, and `max_rounds`. Each mistake has an `error` (the wrong
  claim) and a `fix` (the correct idea). Jev counts a mistake as fixed
  only when the correct idea replaces the wrong claim.
- Make the seed answer contain only the planted mistakes. A weak but
  true sentence can stop a fully fixed answer from counting as solved.

**Write the `jev` instructions for each question by hand.** Name the
ideas that a correct answer must contain. A generic version was tested
and failed. It cut the score of a correct answer from 90% to 45%. See
`scratchpad/jev_generic_instructions_test.py`.

## How it works

**Live scoring.** The browser sends the answer over a WebSocket at most
every 500ms. The server keeps only the newest text per question. It runs
one Jev call at a time. Fast typing therefore never builds a backlog.

**Fix-the-AI rounds.** Each round runs four steps:

1. Jev reviews the student's note. The note must say which claim is
   wrong and why. Questions, requests to fix or rewrite, and notes that
   only point at a sentence change nothing. The student sees why.
2. The revision model changes only the claims that the note explains.
   It applies wrong notes too, so a wrong critique gets no credit.
3. One Jev call scores the new answer and checks each planted mistake.
4. The server checks the result. If a mistake became fixed that the
   note did not explain, the server discards the change. This check
   does not depend on the revision model obeying its rules.

The score is progress. Each fixed mistake is worth an equal share. The
last share is Jev's score for the whole answer. A question is solved
when every mistake is fixed and that whole-answer score reaches
`PASS_THRESHOLD`. Students do not see which mistakes are fixed; the
submission record keeps that for graders.

The server rejects an answer that the model cut off before the end. A
cut-off answer drops the remaining wrong claims, and Jev would count
those mistakes as fixed.

`scratchpad/adversarial_correction_test.py` tries to get the AI to solve
the question for the student: direct requests, prompt injection,
questions, and pointing at sentences. Run it after any change.

`scratchpad/assignment_validation_test.py` tested three revision models.
`gpt-4o-mini` left broken half-sentences. `claude-haiku-5.5` added ideas
that the student did not give. `claude-sonnet-5.5` kept to the student's
words and passed every check. Run that test again after you change the
model.

**Submission.** The server scores every final answer again on submit.
The stored scores therefore do not depend on the browser. The server
writes each record to `SUBMISSIONS_DIR/<student-id>_<time>.json`. The
student can also download a copy. The server refuses a submission after
`CLOSES_AT`, or one with an ID that does not match `STUDENT_ID_PATTERN`.

**Export.** Open `/admin/export.csv`. Enter any user name and the
`ADMIN_KEY` password. The CSV has one row per submission, newest first.
A student can submit more than once. The `latest` column marks each
student's newest row. Grade only the rows where `latest` is `yes`.

## Deployment

`render.yaml` defines a web service on Render's Starter plan, with a
1 GB disk at `/var/data`. Submissions stay on that disk through restarts
and deploys. The free plan has no disk, so it would lose submissions.

1. Create a Blueprint from this repo, or set the plan and disk by hand
   on an existing service.
2. In the dashboard, set `OPENROUTER_API_KEY`, `ADMIN_KEY`, and
   `CLOSES_AT`. The repo does not store them.
3. Make sure that `DEMO_MODE` is `false`.
4. Deploy. Then run the validation test against the live site:
   `BASE_URL=https://<your-service>.onrender.com python scratchpad/assignment_validation_test.py`

The deployed build does not include PyTorch, so it needs little memory.
Download the CSV after the close time, then cancel the plan.

## Known limitations

- **Identity.** Students type their own name and ID. The server checks
  only the ID format.
- **Round limit.** The browser enforces the round limit. "Start this
  question over" gives the rounds back. The record counts every reset
  and every full restart, so a grader can see them.
- **Jev instructions.** Each new question needs hand-written Jev
  instructions. See "Writing questions."

## Background

This project started as a live cosine-similarity meter. Tests showed
that cosine similarity measures topic, not correctness. A wrong answer
with the right words scored 84%. A bigger embedding model and an NLI
model did not fix this. Jev scored all 16 test cases correctly. The test
scripts and results are in `scratchpad/README.md`.
