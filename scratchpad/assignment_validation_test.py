"""Validates the assignment build against a running local server
(uvicorn on :8001), through the app's own endpoints.

Live questions: Jev should score correct answers high and partial or
wrong answers low.

Fix-the-AI questions: the seeded answer should score low with no
mistakes fixed; vague feedback should change nothing; feedback about one
mistake should fix only that mistake; complete feedback should solve it.
"""

import asyncio
import json

import httpx
import websockets

import os

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8001").rstrip("/")
WS = BASE.replace("https://", "wss://").replace("http://", "ws://") + "/ws/jev"

LIVE_CASES = {
    "q1": [
        ("correct", "high", "Tap rate is only a proxy. Version B might just make the button bigger or more confusing, so people tap more or tap again after a failed unlock, but completed rides didn't change, so B created no value. Pedalwise should decide on the main metric before the test, like completed rides or revenue per user, and judge B on that."),
        ("partial: no replacement metric", "low", "Taps are a proxy. People can tap more without riding more, so a higher tap rate doesn't prove B is better."),
        ("wrong: agrees with team", "low", "B is the winner because 8% more taps means riders are more engaged with the app, and engagement always leads to revenue eventually."),
    ],
    "q2": [
        ("correct", "high", "After epoch 20 the model is overfitting: it keeps fitting the training data better, including its noise, but gets worse on new data, which is why validation loss rises. Pedalwise should keep the version from around epoch 20, where validation loss was lowest, and stop training there."),
        ("partial: no version choice", "low", "After epoch 20 the model is overfitting the training data, so it gets worse on new data."),
        ("wrong: underfitting, keep final", "low", "The model is underfitting after epoch 20, so Pedalwise should keep training and use the final epoch-50 model, since its training loss is lowest."),
    ],
}

CORRECTION_SCRIPTS = [
    ("q3", "step by step", [
        ("vague", "This plan is wrong, fix it."),
        ("one mistake (m1)", "Showing one page in the mornings and the other in the afternoons isn't fair, because morning and afternoon visitors can be different people. Each visitor should be randomly assigned to one of the two pages."),
        ("remaining (m2, m3)", "Stopping as soon as the new page is ahead is wrong: checking every day and stopping at the first lead makes false wins more likely, so decide the number of visitors in advance and check once at the end. And 30 visitors per page is not enough; with so few people the difference is mostly noise, so you need enough visitors and a significance test before picking a winner."),
    ]),
    ("q3", "wrong critique", [
        ("wrong", "The morning and afternoon idea is wrong because the test should only run on weekends, when the most people sign up."),
    ]),
    ("q4", "one round", [
        ("all three", "Closed tickets aren't the same as solved problems, so the reward is only a stand-in for what Pedalwise wants. The agent learns whatever earns reward, so it can learn to close tickets without solving them, which is reward hacking. Pedalwise should also watch a guard metric like how often tickets get reopened, and change the reward, for example by subtracting points for reopened tickets."),
    ]),
]


async def score_live(qid, text):
    async with websockets.connect(WS) as ws:
        await ws.send(json.dumps({"question_id": qid, "text": text, "seq": 1}))
        return json.loads(await ws.recv())


async def main():
    failures = 0
    print("== live questions ==")
    for qid, cases in LIVE_CASES.items():
        for label, expect, text in cases:
            r = await score_live(qid, text)
            if "error" in r:
                print(f"{qid} {label:32s} ERROR {r['error']}")
                failures += 1
                continue
            ok = (r["score"] >= 0.5) == (expect == "high")
            failures += not ok
            print(f"{qid} {label:32s} expect {expect:4s} got {r['score']*100:5.1f}%  {'ok' if ok else 'MISMATCH'}")

    print("\n== fix-the-AI questions ==")
    async with httpx.AsyncClient(base_url=BASE, timeout=120) as client:
        assignment = (await client.get("/api/assignment")).json()
        seeds = {q["id"]: q["seed_answer"] for q in assignment["questions"] if q["type"] == "correction"}
        for qid, name, script in CORRECTION_SCRIPTS:
            answer = seeds[qid]
            r = (await client.post("/api/correction/score", json={"question_id": qid, "answer": answer})).json()
            print(f"--- {qid} · {name}")
            prev = r["fixed"]
            print(f"{qid} seed                     progress {r['progress']*100:5.1f}%  fixed {r['fixed']}  solved {r['solved']}")
            for label, feedback in script:
                r = (await client.post("/api/correction/round", json={"question_id": qid, "current_answer": answer, "feedback": feedback, "previous_fixed": prev})).json()
                if "revised_answer" not in r:
                    print(f"{qid} {label:24s} ERROR {r}")
                    failures += 1
                    break
                unchanged = r["revised_answer"].strip() == answer.strip()
                print(f"{qid} {label:24s} progress {r['progress']*100:5.1f}%  fixed {r['fixed']}  solved {r['solved']}  specific={r['feedback_specific']}  rejected={r['rejected']}  unchanged={unchanged}")
                answer = r["revised_answer"]
                prev = r["fixed"]
            print(f"   final AI answer: {answer}\n")
        print("== vague-feedback gate ==")
        vague = ["This answer is wrong, fix it.", "Try again and be more accurate.",
                 "There are some mistakes in here, please correct them.", "Not quite right."]
        leaks = 0
        for qid in seeds:
            for phrase in vague:
                r = (await client.post("/api/correction/round", json={"question_id": qid, "current_answer": seeds[qid], "feedback": phrase})).json()
                leaked = r["changed"] or r["feedback_specific"]
                leaks += leaked
                print(f"{qid} {phrase:55s} specific={r['feedback_specific']}  changed={r['changed']}  {'LEAK' if leaked else 'ok'}")
        print(f"vague leaks: {leaks} of {len(vague) * len(seeds)}")
    print(f"live mismatches/errors: {failures}")


asyncio.run(main())
