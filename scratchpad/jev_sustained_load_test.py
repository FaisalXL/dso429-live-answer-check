"""Same shape as scratchpad/sustained_load_test.py, but against the real
Jev decisions API instead of the local embedding model: 40 simulated
students, each sending an update roughly every 500ms for 8 seconds
(slower cadence than the 180ms local throttle, since Jev can't keep up
with that -- this tests whether even a relaxed cadence holds up under
real concurrent classroom load).
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
            "the reference answer's core claims, even if phrased "
            "differently? The student may still be mid-answer."
        ),
        "criteria": {
            "true": "Reasoning so far is correct.",
            "false": "Reasoning is wrong, backwards, or just keywords.",
        },
    }
}

N_STUDENTS = 40
DURATION_S = 8
INTERVAL_S = 0.5  # relaxed cadence vs the 0.18s local throttle


async def one_student(idx, client):
    latencies = []
    errors = 0
    end_at = time.perf_counter() + DURATION_S
    n = 0
    while time.perf_counter() < end_at:
        text = f"student {idx} draft update number {n}, explaining the escrow mechanism so far."
        payload = {
            "model": "typesafe/jev-1.13",
            "state": {"reference_answer": REFERENCE, "student_answer": text},
            "questions": QUESTIONS,
        }
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            errors += 1
            latencies.append(None)
        n += 1
        await asyncio.sleep(INTERVAL_S)
    return latencies, errors


async def main():
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one_student(i, client) for i in range(N_STUDENTS)))
        wall = time.perf_counter() - t0

    all_latencies = [x for lats, _ in results for x in lats if x is not None]
    total_errors = sum(errs for _, errs in results)
    total_requests = sum(len(lats) for lats, _ in results)
    all_latencies.sort()

    print(f"students={N_STUDENTS} duration_s={DURATION_S} interval_s={INTERVAL_S}")
    print(f"total_requests={total_requests} errors={total_errors}")
    if all_latencies:
        n = len(all_latencies)
        avg = sum(all_latencies) / n
        print(f"actual_req_per_sec={total_requests / wall:.1f}")
        print(f"min_ms={all_latencies[0]:.0f}")
        print(f"p50_ms={all_latencies[n // 2]:.0f}")
        print(f"p95_ms={all_latencies[int(n * 0.95)]:.0f}")
        print(f"max_ms={all_latencies[-1]:.0f}")
        print(f"avg_ms={avg:.0f}")


asyncio.run(main())
