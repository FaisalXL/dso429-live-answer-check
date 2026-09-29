import asyncio
import json

import websockets

URL = "ws://127.0.0.1:8001/ws/similarity"

CASES = {
    "reference_itself": (
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
        "ran correctly."
    ),
    "keyword_salad_no_reasoning": (
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly."
    ),
    "wrong_logic_dense_keywords": (
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code."
    ),
    "correct_but_totally_different_wording": (
        "You could set up an automated agreement where money is locked up "
        "until either side proves the deal went through by a set time; if "
        "nobody confirms in time, whoever paid gets their money back with "
        "nobody having to manually process it. The catch is you need some "
        "trustworthy way to know in the first place whether the real-world "
        "delivery happened, and that outside check is really the weak link."
    ),
    "topic_adjacent_but_off_target": (
        "Cryptocurrency and blockchain technology are changing how businesses "
        "handle payments and contracts in the modern economy, offering more "
        "transparency and efficiency than traditional banking systems."
    ),
    "four_lines_keyword_matching": (
        "A smart contract uses escrow to hold the deposit. It checks the "
        "deadline and confirms with the store. If not confirmed it refunds "
        "the customer automatically. A risk is the oracle could be wrong."
    ),
}


async def check(label, text):
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"text": text}))
        resp = json.loads(await ws.recv())
        print(f"{label:38s} {resp['score']*100:5.1f}%")


async def main():
    for label, text in CASES.items():
        await check(label, text)


asyncio.run(main())
