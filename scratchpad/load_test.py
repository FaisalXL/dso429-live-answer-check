import asyncio
import time

import websockets

REF = (
    "A smart contract on a blockchain could hold the customer's deposit in "
    "escrow when the pre-order is placed. The contract's code encodes the "
    "pickup deadline as a condition: if the store confirms fulfillment "
    "before the deadline, the deposit is released to the store; if the "
    "deadline passes without confirmation, the contract automatically "
    "refunds the customer, with no manual intervention needed from either "
    "party. This removes the need to trust the store to issue refunds "
    "voluntarily, since the rule is enforced by code once deployed."
)

N = 40
URL = "ws://127.0.0.1:8001/ws/similarity"


async def one_client(idx):
    async with websockets.connect(URL) as ws:
        t0 = time.perf_counter()
        await ws.send(f'{{"text": "{REF} student {idx}"}}')
        await ws.recv()
        return (time.perf_counter() - t0) * 1000


async def main():
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_client(i) for i in range(N)))
    wall = (time.perf_counter() - t0) * 1000
    results.sort()
    avg = sum(results) / len(results)
    print(f"n={N}")
    print(f"wall_clock_ms_for_all={wall:.1f}")
    print(f"min_ms={results[0]:.1f}")
    print(f"p50_ms={results[len(results)//2]:.1f}")
    print(f"p95_ms={results[int(len(results)*0.95)]:.1f}")
    print(f"max_ms={results[-1]:.1f}")
    print(f"avg_ms={avg:.1f}")


asyncio.run(main())
