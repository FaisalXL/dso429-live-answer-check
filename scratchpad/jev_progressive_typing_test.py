"""Simulate a student typing progressively (word-by-word snapshots, same
idea as our throttled live updates) and watch how Jev's correctness score
evolves as the answer builds up -- does it ramp smoothly, or sit low then
jump? Also checks whether a wrong-logic answer ever looks deceptively good
mid-typing, before the backwards part is fully typed.
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/alpha/decisions"

REFERENCE = (
    "A smart contract on a blockchain could hold the customer's deposit in "
    "escrow when the pre-order is placed. The contract's code encodes the "
    "pickup deadline as a condition: if the store confirms fulfillment "
    "before the deadline, the deposit is released to the store; if the "
    "deadline passes without confirmation, the contract automatically "
    "refunds the customer, with no manual intervention needed from either "
    "party. A real risk is that the contract depends on an outside system, "
    "an oracle, to report whether the order was actually fulfilled; if that "
    "oracle is wrong or manipulated, the contract executes the wrong "
    "outcome even though the code ran correctly."
)

QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": (
            "Does the student's answer demonstrate correct understanding of "
            "the reference answer's core claims (the escrow/deadline "
            "mechanism AND the oracle risk), even if phrased completely "
            "differently? An answer covering only one of the two required "
            "ideas is incomplete. The student may still be mid-answer; "
            "judge based on what has been written so far."
        ),
        "criteria": {
            "true": (
                "Reasoning so far is correct and, if incomplete, is heading "
                "toward covering both the mechanism and the risk."
            ),
            "false": (
                "Reasoning is wrong or backwards, or is just keywords with "
                "no real explanation."
            ),
        },
    }
}

CORRECT_ANSWER = (
    "basically you'd code the refund to happen on its own if the shop "
    "never checks in by the pickup time, so you don't have to hope they "
    "do the right thing. downside is the contract only knows what some "
    "outside source tells it really happened."
)

WRONG_LOGIC_ANSWER = (
    "The oracle holds the smart contract deposit and the store must "
    "confirm the escrow before the customer refunds the deadline "
    "automatically, which removes risk from the code."
)


def progressive_snapshots(text, step_words=4):
    words = text.split()
    snapshots = []
    for i in range(step_words, len(words) + step_words, step_words):
        snapshots.append(" ".join(words[:i]))
    return snapshots


async def check(text):
    payload = {
        "model": "typesafe/jev-1.13",
        "state": {"reference_answer": REFERENCE, "student_answer": text},
        "questions": QUESTIONS,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    prob = resp.json()["answers"]["correct"]["noul"]
    return prob, elapsed_ms


async def run_sequence(label, text):
    print(f"\n=== {label} ===")
    snapshots = progressive_snapshots(text)
    for snap in snapshots:
        prob, ms = await check(snap)
        pct_typed = round(len(snap) / len(text) * 100)
        bar = "#" * int(prob * 40)
        print(f"{pct_typed:3d}% typed | {prob*100:5.1f}% correct | {ms:5.0f}ms | {bar}")
        print(f'         "{snap}"')


async def main():
    await run_sequence("CORRECT answer, typed progressively", CORRECT_ANSWER)
    await run_sequence("WRONG-LOGIC answer, typed progressively", WRONG_LOGIC_ANSWER)


asyncio.run(main())
