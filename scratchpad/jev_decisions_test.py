"""Test TypeSafe's Jev decisions API (typesafe/jev-1.13 via OpenRouter) as a
correctness classifier, against the exact same hard-case bank used for the
gpt-4o-mini correctness check and the NLI entailment test. Also measures
real latency and cost, since those are the claimed differentiators.
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

CASES = {
    "wrong_logic_1": (
        "false_positive_should_be_FALSE",
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code.",
    ),
    "wrong_logic_3_confident_but_backwards": (
        "false_positive_should_be_FALSE",
        "A smart contract works by having the customer manually confirm the "
        "oracle before the store's escrow deadline releases the risk back to "
        "the deposit, which is what makes it trustworthy.",
    ),
    "keyword_salad_1": (
        "false_positive_should_be_FALSE",
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly.",
    ),
    "correct_novel_casual": (
        "false_negative_should_be_TRUE",
        "basically you'd code the refund to happen on its own if the shop "
        "never checks in by the pickup time, so you don't have to hope they "
        "do the right thing. downside is the contract only knows what some "
        "outside source tells it really happened.",
    ),
    "correct_novel_formal": (
        "false_negative_should_be_TRUE",
        "Funds could be programmatically escrowed upon order placement and "
        "released conditionally based on a time-bound fulfillment "
        "attestation, eliminating reliance on the merchant's discretion. "
        "Correctness is bounded by the fidelity of the external attestation "
        "mechanism.",
    ),
}

QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": (
            "Does the student's answer demonstrate correct understanding of "
            "the reference answer's core claims (the escrow/deadline "
            "mechanism AND the oracle risk), even if phrased completely "
            "differently?"
        ),
        "criteria": {
            "true": (
                "Reasoning is correct and matches the reference's claims, "
                "regardless of wording, register, or structure."
            ),
            "false": (
                "Reasoning is wrong or backwards, or the text is just "
                "keywords/a list with no real explanation."
            ),
        },
    }
}


async def check(label, category, text):
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
    body = resp.json()
    return label, category, body, elapsed_ms


async def main():
    results = await asyncio.gather(*(check(l, c, t) for l, (c, t) in CASES.items()))
    print(f"{'case':38s} {'category':32s} {'noul(true_prob)':>16s} {'latency_ms':>11s}")
    print("-" * 105)
    total_cost = 0
    for label, category, body, elapsed_ms in results:
        answer = body["answers"]["correct"]
        prob = answer["noul"]
        cost = body.get("usage", {}).get("cost", 0)
        total_cost += cost
        print(f"{label:38s} {category:32s} {prob*100:15.1f}% {elapsed_ms:10.0f}ms")
    print(f"\nTotal cost for {len(CASES)} calls: ${total_cost:.6f}")
    print(f"Raw response for first case (for schema inspection):")
    import json
    print(json.dumps(results[0][2], indent=2))


asyncio.run(main())
