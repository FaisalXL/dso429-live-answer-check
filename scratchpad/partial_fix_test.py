"""Tests that each correct note raises the fix-the-AI score.

A student fixes one claim per round, in short, natural wording. After round
k of 3, about k shares of the score must be earned, and round 3 must solve
the question. A fix that asked for two ideas once gave no credit for a
correct note (50% -> 54%).

Calls the round handler directly, so the overreach check and its retry run
as in production. Run from the project folder:
    .venv/bin/python scratchpad/partial_fix_test.py [q3|q4 ...]
"""

import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import main  # noqa: E402

# Each path fixes one mistake per note. The mistake ids are in brackets.
PATHS = {
    "q4": {
        "order 1-2-3": [
            ("m1", "closed tickets don't always mean happy riders, it's just a proxy"),
            ("m2", "the agent doesn't understand the goal, it just maximizes reward so it "
                   "could close tickets without fixing anything"),
            ("m3", "they should also track how often tickets get reopened, not just the reward"),
        ],
        "order 3-2-1": [
            ("m3", "watching only the reward isn't enough, track reopened tickets too"),
            ("m2", "the agent will close tickets without solving them if that gets reward, "
                   "it doesn't actually understand what Pedalwise wants"),
            ("m1", "more closed tickets doesn't mean happier riders, tickets can be closed "
                   "without the problem being solved"),
        ],
    },
    "q3": {
        "order 1-2-3": [
            ("m1", "mornings vs afternoons isn't fair because different people visit at "
                   "different times, visitors should be randomly assigned to a page"),
            ("m2", "don't stop as soon as the new page is ahead, an early lead could just be luck"),
            ("m3", "30 visitors is way too small, the difference could be random"),
        ],
        "order 3-1-2": [
            ("m3", "30 people per page is not enough, any difference could just be chance"),
            ("m1", "showing pages by time of day is biased, randomly assign each visitor instead"),
            ("m2", "stopping when it's ahead is peeking, set the test length first and wait until the end"),
        ],
    },
}


async def run_path(qid, steps):
    question = main.QUESTIONS[qid]
    answer, prev_fixed, ok = question["seed_answer"], [False] * 3, True
    for k, (target, note) in enumerate(steps, 1):
        d = await main.correction_round({"question_id": qid, "current_answer": answer,
                                         "feedback": note, "previous_fixed": prev_fixed})
        idx = int(target[1]) - 1
        good = d["fixed"][idx] and d["fixed_count"] >= k and not d["rejected"]
        if k == len(steps):
            good = good and d["solved"]
        ok &= good
        flags = "".join("Y" if f else "." for f in d["fixed"])
        print(f"   {k}. fix {target}: progress {d['progress']:6.1%} fixed={flags} "
              f"specific={d['feedback_specific']} rejected={d['rejected']} solved={d['solved']}  "
              f"{'ok' if good else 'FAIL'}")
        if d["changed"]:
            answer, prev_fixed = d["revised_answer"], d["fixed"]
    if not ok:
        print(f"      final answer: {answer}")
    return ok


async def run(which):
    main.http_client = httpx.AsyncClient()
    fails = 0
    try:
        for qid in which:
            for name, steps in PATHS[qid].items():
                print(f"--- {qid} · {name}")
                fails += not await run_path(qid, steps)
    finally:
        await main.http_client.aclose()
    print(f"\npaths failed: {fails}")


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1:] or list(PATHS)))
