"""Test whether a cheap, single LLM call can catch what cosine similarity
can't: confidently-wrong-but-keyword-dense answers, without false-flagging
genuinely correct answers that just happen to score lower on embeddings.
"""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-4o-mini"

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
    # embeddings scored these HIGH (68-84%) despite being wrong/empty of reasoning
    "wrong_logic_1": (
        "false_positive_should_flag",
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code.",
    ),
    "wrong_logic_3_confident_but_backwards": (
        "false_positive_should_flag",
        "A smart contract works by having the customer manually confirm the "
        "oracle before the store's escrow deadline releases the risk back to "
        "the deposit, which is what makes it trustworthy.",
    ),
    "keyword_salad_1": (
        "false_positive_should_flag",
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly.",
    ),
    # embeddings scored these LOWER (61-70%) despite being genuinely correct
    "correct_novel_casual": (
        "false_negative_should_NOT_flag",
        "basically you'd code the refund to happen on its own if the shop "
        "never checks in by the pickup time, so you don't have to hope they "
        "do the right thing. downside is the contract only knows what some "
        "outside source tells it really happened.",
    ),
    "correct_novel_formal": (
        "false_negative_should_NOT_flag",
        "Funds could be programmatically escrowed upon order placement and "
        "released conditionally based on a time-bound fulfillment "
        "attestation, eliminating reliance on the merchant's discretion. "
        "Correctness is bounded by the fidelity of the external attestation "
        "mechanism.",
    ),
}

SYSTEM_PROMPT = (
    "You check whether a student's exam answer demonstrates correct "
    "understanding, compared to a reference answer. The student may use "
    "completely different wording -- that is fine. Flag it as INCORRECT "
    "only if the reasoning is actually wrong, backwards, or absent (e.g. "
    "just a list of keywords with no explanation). Respond with exactly one "
    "line: 'CORRECT' or 'INCORRECT', a dash, then a one-sentence reason."
)


async def check(label, category, text):
    user_prompt = f"Reference answer:\n{REFERENCE}\n\nStudent answer:\n{text}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 60,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip()
        return label, category, verdict


async def main():
    results = await asyncio.gather(*(check(l, c, t) for l, (c, t) in CASES.items()))
    for label, category, verdict in results:
        print(f"{label:32s} {category:32s} -> {verdict}")


asyncio.run(main())
