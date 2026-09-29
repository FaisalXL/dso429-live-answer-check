"""Run the full case bank (same cases as regression_test.py) through the Jev
decisions correctness check, to see how it holds up beyond the 5 hardest
cases -- including partial-credit and edge cases the LLM/NLI tests never
covered.
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
    "reference_itself": ("should_be_high", REFERENCE),
    "correct_novel_1_casual": (
        "should_be_high",
        "basically you'd code the refund to happen on its own if the shop "
        "never checks in by the pickup time, so you don't have to hope they "
        "do the right thing. downside is the contract only knows what some "
        "outside source tells it really happened.",
    ),
    "correct_novel_2_formal": (
        "should_be_high",
        "Funds could be programmatically escrowed upon order placement and "
        "released conditionally based on a time-bound fulfillment "
        "attestation, eliminating reliance on the merchant's discretion. "
        "Correctness is bounded by the fidelity of the external attestation "
        "mechanism.",
    ),
    "correct_novel_3_example_driven": (
        "should_be_high",
        "Think of it like a vending machine for refunds: the money sits "
        "untouched until a timer runs out, and then it either goes to the "
        "shop if they scanned the order as done, or back to you if they "
        "didn't. It still needs something honest telling it whether the scan "
        "actually happened in real life.",
    ),
    "wrong_logic_1": (
        "should_be_low",
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code.",
    ),
    "wrong_logic_2": (
        "should_be_low",
        "The customer's code deploys an escrow deadline so the smart "
        "contract can confirm the store's refund, and the oracle removes "
        "the manual risk of the deposit outcome.",
    ),
    "wrong_logic_3_confident_but_backwards": (
        "should_be_low",
        "A smart contract works by having the customer manually confirm the "
        "oracle before the store's escrow deadline releases the risk back to "
        "the deposit, which is what makes it trustworthy.",
    ),
    "keyword_salad_1": (
        "should_be_low",
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly.",
    ),
    "keyword_salad_2_reordered": (
        "should_be_low",
        "Oracle risk code correctly manipulated outcome. Customer refund "
        "confirm store deadline. Pickup smart contract deposit escrow.",
    ),
    "off_topic_1": (
        "should_be_low",
        "Cryptocurrency and blockchain technology are changing how "
        "businesses handle payments and contracts in the modern economy.",
    ),
    "off_topic_2_totally_unrelated": (
        "should_be_low",
        "The coffee shop should hire more staff during the holiday season "
        "to reduce wait times for customers standing in line.",
    ),
    "partial_escrow_only": (
        "should_be_low_or_medium (missing risk half)",
        "A smart contract could hold the deposit in escrow and release it "
        "automatically based on whether the store confirms the order in "
        "time.",
    ),
    "partial_oracle_risk_only": (
        "should_be_low_or_medium (missing mechanism half)",
        "A real limitation of smart contracts is that they depend on an "
        "oracle to know what happened in the real world, and that oracle "
        "could be wrong or manipulated.",
    ),
    "empty_string": ("should_be_low", ""),
    "single_word": ("should_be_low", "escrow"),
    "very_long_rambling_but_correct": (
        "should_be_high",
        "So okay basically what you'd want to do here is is is set up like a "
        "system where um the money that the customer pays as a deposit "
        "doesn't go straight to the coffee shop right away, instead it kind "
        "of sits in a holding area, an escrow, and then depending on whether "
        "the shop confirms before the deadline that it actually fulfilled the "
        "order or not, the contract will either let the shop have the money "
        "or send it back to the customer automatically without anyone having "
        "to email anyone or call anyone, and the risk part is that this all "
        "depends on some outside system called an oracle telling the "
        "contract the truth about what happened, and if that thing is wrong "
        "or gets hacked then the contract does the wrong thing anyway.",
    ),
}

QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": (
            "Does the student's answer demonstrate correct understanding of "
            "the reference answer's core claims (the escrow/deadline "
            "mechanism AND the oracle risk), even if phrased completely "
            "differently? An answer covering only one of the two required "
            "ideas is incomplete."
        ),
        "criteria": {
            "true": (
                "Reasoning is correct and covers both the mechanism and the "
                "risk, regardless of wording, register, or structure."
            ),
            "false": (
                "Reasoning is wrong or backwards, covers only one of the two "
                "required ideas, or is just keywords with no explanation."
            ),
        },
    }
}


async def check(label, category, text):
    payload = {
        "model": "typesafe/jev-1.13",
        "state": {"reference_answer": REFERENCE, "student_answer": text or "(empty)"},
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
    print(f"{'case':38s} {'expectation':46s} {'noul':>7s} {'ms':>6s}")
    print("-" * 105)
    total_cost = 0
    latencies = []
    for label, category, body, elapsed_ms in results:
        prob = body["answers"]["correct"]["noul"]
        cost = body.get("usage", {}).get("cost", 0)
        total_cost += cost
        latencies.append(elapsed_ms)
        print(f"{label:38s} {category:46s} {prob*100:6.1f}% {elapsed_ms:5.0f}")
    print(f"\nTotal cost for {len(CASES)} calls: ${total_cost:.6f}")
    print(f"Latency: min={min(latencies):.0f}ms max={max(latencies):.0f}ms avg={sum(latencies)/len(latencies):.0f}ms")


asyncio.run(main())
