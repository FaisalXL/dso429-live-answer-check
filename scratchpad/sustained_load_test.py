import asyncio
import time

import websockets

REF = (
    "A smart contract on a blockchain could hold the customer's deposit in "
    "escrow when the pre-order is placed. The contract's code encodes the "
    "pickup deadline as a condition."
)

N_STUDENTS = 40
DURATION_S = 8
THROTTLE_S = 0.18  # matches frontend THROTTLE_MS
URL = "ws://127.0.0.1:8001/ws/similarity"


async def one_student(idx):
    latencies = []
    async with websockets.connect(URL) as ws:
        end_at = time.perf_counter() + DURATION_S
        n = 0
        while time.perf_counter() < end_at:
            t0 = time.perf_counter()
            await ws.send(f'{{"text": "{REF} student {idx} update {n}"}}')
            await ws.recv()
            latencies.append((time.perf_counter() - t0) * 1000)
            n += 1
            await asyncio.sleep(THROTTLE_S)
    return latencies


async def main():
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_student(i) for i in range(N_STUDENTS)))
    wall = time.perf_counter() - t0
    all_latencies = [x for student in results for x in student]
    all_latencies.sort()
    total_requests = len(all_latencies)
    avg = sum(all_latencies) / total_requests

    print(f"students={N_STUDENTS} duration_s={DURATION_S} throttle_s={THROTTLE_S}")
    print(f"total_requests={total_requests}")
    print(f"actual_req_per_sec={total_requests / wall:.1f}")
    print(f"min_ms={all_latencies[0]:.1f}")
    print(f"p50_ms={all_latencies[total_requests // 2]:.1f}")
    print(f"p95_ms={all_latencies[int(total_requests * 0.95)]:.1f}")
    print(f"p99_ms={all_latencies[int(total_requests * 0.99)]:.1f}")
    print(f"max_ms={all_latencies[-1]:.1f}")
    print(f"avg_ms={avg:.1f}")


asyncio.run(main())
