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

BASE = "http://127.0.0.1:8001"
WS = "ws://127.0.0.1:8001/ws/jev"

LIVE_CASES = {
    "q1": [
        ("correct", "high", "Tap rate is only a proxy. Version B might just make the button bigger or more confusing, so people tap more or tap again after a failed unlock, but completed rides didn't change, so B created no value. Pedalwise should decide on the main metric before the test, like completed rides or revenue per user, and judge B on that."),
        ("partial: no replacement metric", "low", "Taps are a proxy. People can tap more without riding more, so a higher tap rate doesn't prove B is better."),
        ("wrong: agrees with team", "low", "B is the winner because 8% more taps means riders are more engaged with the app, and engagement always leads to revenue eventually."),
    ],
    "q2": [
        ("correct", "high", "The coefficients are in utility units, not dollars, so 0.60 isn't 60 cents. To convert to dollars, divide by the size of the price coefficient: 0.60 / 0.03 = $20. Customers are willing to pay about $20 a month for e-bike access."),
        ("wrong: multiplies", "low", "The manager forgot to use the price coefficient. Willingness to pay is 0.60 times 0.03, which is $0.018, so e-bike access is worth almost nothing."),
        ("partial: no number", "low", "The manager is wrong because logit coefficients are measured in utility, not dollars, so you can't read 0.60 as 60 cents."),
    ],
    "q4": [
        ("correct", "high", "This is overfitting: the model memorized the training data, including noise, so it doesn't generalize. The 71% held-out score is the real estimate. They should simplify the model or add regularization, and pick the complexity with a validation set."),
        ("wrong: underfitting", "low", "The model is underfitting, so the team should make it more complex by adding more features and deeper trees until the held-out accuracy catches up."),
        ("partial: no remedy", "low", "The model is overfitting. It does great on data it has seen and badly on new data."),
    ],
}

CORRECTION_SCRIPTS = {
    "q3": [
        ("vague", "This answer is wrong, fix it."),
        ("one mistake (m1)", "Running the fixed test is not free. While it runs, it keeps sending the two weaker notifications to a third of riders each, so Pedalwise loses responses it could have won."),
        ("remaining (m2, m3)", "Thompson sampling does not always send the version with the highest current average; that would be greedy. It keeps a belief about each version's response rate, samples from each belief, and sends the version with the highest sample, so uncertain versions still get tried. Also, a fixed test can still be the better choice, for example when Pedalwise needs a clean, unbiased estimate of how much better one version is for a long-term decision."),
    ],
    "q5": [
        ("all three", "The contract can't see the physical world. It needs an outside data source, an oracle, such as the dock reporting the return. Because of that, it can refund or keep a deposit wrongly if the dock data is faulty or tampered with, even if the code is correct. And deployed contract code generally can't be edited like an app; changing the rules means deploying a new contract or using an upgrade mechanism planned in advance."),
    ],
}


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
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
        assignment = (await client.get("/api/assignment")).json()
        seeds = {q["id"]: q["seed_answer"] for q in assignment["questions"] if q["type"] == "correction"}
        for qid, script in CORRECTION_SCRIPTS.items():
            answer = seeds[qid]
            r = (await client.post("/api/correction/score", json={"question_id": qid, "answer": answer})).json()
            print(f"{qid} seed                       score {r['score']*100:5.1f}%  fixed {r['fixed']}  solved {r['solved']}")
            for label, feedback in script:
                r = (await client.post("/api/correction/round", json={"question_id": qid, "current_answer": answer, "feedback": feedback})).json()
                if "revised_answer" not in r:
                    print(f"{qid} {label:24s} ERROR {r}")
                    failures += 1
                    break
                unchanged = r["revised_answer"].strip() == answer.strip()
                print(f"{qid} {label:24s} score {r['score']*100:5.1f}%  fixed {r['fixed']}  solved {r['solved']}  unchanged={unchanged}")
                answer = r["revised_answer"]
            print(f"   final AI answer: {answer}\n")
    print(f"live mismatches/errors: {failures}")


asyncio.run(main())
