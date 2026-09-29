"""Does a question-agnostic version of the Jev instructions (no hardcoded
mention of 'escrow' or 'oracle risk') still catch the partial-credit case,
or was naming the specific required concepts load-bearing? Tests both
versions against the same case bank so the question swaps cleanly to a
different topic without editing prose by hand.
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

CURRENT_QUESTIONS = {
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
            "true": "Reasoning so far is correct and, if incomplete, is heading toward covering both the mechanism and the risk.",
            "false": "Reasoning is wrong or backwards, covers only one of the two required ideas, or is just keywords with no explanation.",
        },
    }
}

GENERIC_QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": (
            "Does the student's answer demonstrate correct and complete "
            "understanding of everything the reference answer covers, even "
            "if phrased completely differently? If the reference answer "
            "covers multiple distinct ideas, the student must address all "
            "of them, not just one, to be complete. The student may still "
            "be mid-answer; judge based on what has been written so far."
        ),
        "criteria": {
            "true": "Reasoning so far is correct and, if incomplete, is heading toward covering everything the reference covers.",
            "false": "Reasoning is wrong or backwards, covers only some of the reference's distinct ideas, or is just keywords with no explanation.",
        },
    }
}

CASES = {
    "correct_novel_casual": (
        "should_be_high",
        "basically you'd code the refund to happen on its own if the shop "
        "never checks in by the pickup time, so you don't have to hope they "
        "do the right thing. downside is the contract only knows what some "
        "outside source tells it really happened.",
    ),
    "wrong_logic_1": (
        "should_be_low",
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code.",
    ),
    "partial_escrow_only": (
        "should_be_low (missing risk half)",
        "A smart contract could hold the deposit in escrow and release it "
        "automatically based on whether the store confirms the order in "
        "time.",
    ),
    "partial_oracle_risk_only": (
        "should_be_low (missing mechanism half)",
        "A real limitation of smart contracts is that they depend on an "
        "oracle to know what happened in the real world, and that oracle "
        "could be wrong or manipulated.",
    ),
    "keyword_salad_1": (
        "should_be_low",
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly.",
    ),
}


async def check(client, text, questions, retries=3):
    payload = {
        "model": "typesafe/jev-1.13",
        "state": {"reference_answer": REFERENCE, "student_answer": text},
        "questions": questions,
    }
    for attempt in range(retries):
        try:
            resp = await client.post(
                URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["answers"]["correct"]["noul"]
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1)


async def main():
    print(f"{'case':32s} {'expectation':32s} {'current':>9s} {'generic':>9s}")
    print("-" * 85)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for label, (expectation, text) in CASES.items():
            current = await check(client, text, CURRENT_QUESTIONS)
            generic = await check(client, text, GENERIC_QUESTIONS)
            print(f"{label:32s} {expectation:32s} {current*100:8.1f}% {generic*100:8.1f}%")


asyncio.run(main())
