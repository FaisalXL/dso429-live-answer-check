"""Broader efficacy + correctness regression test against the LIVE server.

Re-checks whether the false-negative fix (multi-reference) still holds and
whether the known false-positive problem still persists, plus basic
functional sanity checks (determinism, partial answers, empty input).
"""

import asyncio
import json

import websockets

URL = "ws://127.0.0.1:8001/ws/similarity"

CASES = {
    # --- sanity: each reference should match itself near-perfectly ---
    "ref_technical_itself": (
        "reference_self",
        "A smart contract on a blockchain could hold the customer's deposit in "
        "escrow when the pre-order is placed. The contract's code encodes the "
        "pickup deadline as a condition: if the store confirms fulfillment "
        "before the deadline, the deposit is released to the store; if the "
        "deadline passes without confirmation, the contract automatically "
        "refunds the customer, with no manual intervention needed from either "
        "party. This removes the need to trust the store to issue refunds "
        "voluntarily, since the rule is enforced by code once deployed. A real "
        "risk is that the contract depends on an outside system, an oracle, to "
        "report whether the order was actually fulfilled in the physical "
        "world; if that oracle is wrong, unavailable, or manipulated, the "
        "contract will execute the wrong outcome even though the code itself "
        "ran correctly.",
    ),
    "ref_plain_itself": (
        "reference_self",
        "You could set up an automated agreement where money is locked up "
        "until either side proves the deal went through by a set time; if "
        "nobody confirms in time, whoever paid gets their money back with "
        "nobody having to manually process it. The catch is you need some "
        "trustworthy way to know in the first place whether the real-world "
        "delivery happened, and that outside check is really the weak link.",
    ),
    # --- correct answers, novel wording not in any reference ---
    "correct_novel_1_casual": (
        "false_negative_check",
        "basically you'd code the refund to happen on its own if the shop "
        "never checks in by the pickup time, so you don't have to hope they "
        "do the right thing. downside is the contract only knows what some "
        "outside source tells it really happened.",
    ),
    "correct_novel_2_formal": (
        "false_negative_check",
        "Funds could be programmatically escrowed upon order placement and "
        "released conditionally based on a time-bound fulfillment attestation, "
        "eliminating reliance on the merchant's discretion. However, "
        "correctness is bounded by the fidelity of the external attestation "
        "mechanism feeding the contract.",
    ),
    "correct_novel_3_example_driven": (
        "false_negative_check",
        "Think of it like a vending machine for refunds: the money sits "
        "untouched until a timer runs out, and then it either goes to the "
        "shop if they scanned the order as done, or back to you if they "
        "didn't. It still needs something honest telling it whether the scan "
        "actually happened in real life.",
    ),
    # --- wrong logic, dense reference vocabulary (false positive check) ---
    "wrong_logic_1": (
        "false_positive_check",
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code.",
    ),
    "wrong_logic_2": (
        "false_positive_check",
        "The customer's code deploys an escrow deadline so the smart contract "
        "can confirm the store's refund, and the oracle removes the manual "
        "risk of the deposit outcome.",
    ),
    "wrong_logic_3_confident_but_backwards": (
        "false_positive_check",
        "A smart contract works by having the customer manually confirm the "
        "oracle before the store's escrow deadline releases the risk back to "
        "the deposit, which is what makes it trustworthy.",
    ),
    # --- keyword salad, no reasoning ---
    "keyword_salad_1": (
        "false_positive_check",
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly.",
    ),
    "keyword_salad_2_reordered": (
        "false_positive_check",
        "Oracle risk code correctly manipulated outcome. Customer refund "
        "confirm store deadline. Pickup smart contract deposit escrow.",
    ),
    # --- off-topic / doesn't answer ---
    "off_topic_1": (
        "true_negative_check",
        "Cryptocurrency and blockchain technology are changing how businesses "
        "handle payments and contracts in the modern economy.",
    ),
    "off_topic_2_totally_unrelated": (
        "true_negative_check",
        "The coffee shop should hire more staff during the holiday season to "
        "reduce wait times for customers standing in line.",
    ),
    # --- partial answers: only one of the two required ideas ---
    "partial_escrow_only": (
        "partial_credit_check",
        "A smart contract could hold the deposit in escrow and release it "
        "automatically based on whether the store confirms the order in "
        "time.",
    ),
    "partial_oracle_risk_only": (
        "partial_credit_check",
        "A real limitation of smart contracts is that they depend on an "
        "oracle to know what happened in the real world, and that oracle "
        "could be wrong or manipulated.",
    ),
    # --- edge cases ---
    "empty_string": ("edge_case", ""),
    "single_word": ("edge_case", "escrow"),
    "very_long_rambling_but_correct": (
        "edge_case",
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


async def check(label, text):
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"text": text}))
        resp = json.loads(await ws.recv())
        return resp["score"], resp.get("matched_reference"), resp["below_threshold"]


async def main():
    print(f"{'case':38s} {'category':22s} {'score':>7s} {'ref':>18s}  below_thr")
    print("-" * 95)
    results = {}
    for label, (category, text) in CASES.items():
        score, ref, below = await check(label, text)
        results[label] = score
        print(f"{label:38s} {category:22s} {score*100:6.1f}% {str(ref):>18s}  {below}")

    # determinism check: re-run one case 3x, confirm identical score
    print("\nDeterminism check (same input, 3 runs):")
    det_text = CASES["correct_novel_1_casual"][1]
    scores = []
    for _ in range(3):
        s, _, _ = await check("determinism", det_text)
        scores.append(round(s, 6))
    print(f"  scores: {scores}  {'CONSISTENT' if len(set(scores)) == 1 else 'INCONSISTENT'}")


asyncio.run(main())
