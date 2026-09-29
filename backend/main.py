import asyncio
import json
import os
import secrets
from pathlib import Path

import httpx
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer, util

# Each encode() call defaults to using all CPU cores for intra-op
# parallelism. With many concurrent small requests (one per student), that
# causes threads to compete with each other instead of running in parallel.
# Capping to 1 thread per call lets the ThreadPoolExecutor in the websocket
# handler parallelize *across* requests instead, which is the win we want.
torch.set_num_threads(1)

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

STATIC_DIR = ROOT_DIR / "static"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.55"))
MAX_HINTS = int(os.environ.get("MAX_HINTS", "3"))
# Exposes /api/reveal-answer so you can see the reference answer while
# demoing. MUST be set to false (or unset) before running this with real
# students, or it hands them the answer.
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"

# Placeholder DSO429-flavored question. Swap in the real midterm question +
# reference answer before the actual exam.
QUESTION_TEXT = (
    "A local coffee retailer wants to let customers pre-order seasonal "
    "drinks and pay a deposit that is automatically refunded if the store "
    "fails to fulfill the order by the promised pickup time. Explain how a "
    "smart contract could be used to implement this, and name one real "
    "risk or limitation of relying on it."
)

# Multiple valid phrasings of the same correct answer. A student's answer is
# scored against ALL of them and the best (max) match wins. This is what
# lets a correct answer in different wording/register/framing score well
# without being penalized for not matching one specific phrasing.
#
# Tested via scratchpad/multi_ref_test.py: this meaningfully closes the
# false-negative gap (a correct 4th independent phrasing went 61.3% ->
# 79.8%). It does NOT fix the false-positive gap -- a wrong answer with
# reference-matching vocabulary scores just as high as with one reference.
# That's expected and is why actual grading should not be based on this
# live number; see README "Known limitations."
REFERENCE_ANSWERS = {
    "technical": (
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
    "plain_language": (
        "You could set up an automated agreement where money is locked up "
        "until either side proves the deal went through by a set time; if "
        "nobody confirms in time, whoever paid gets their money back with "
        "nobody having to manually process it. The catch is you need some "
        "trustworthy way to know in the first place whether the real-world "
        "delivery happened, and that outside check is really the weak link."
    ),
    "alternate_framing": (
        "Instead of relying on the coffee shop's word that it will issue a "
        "refund, you could write the refund rule directly into code that runs "
        "on its own once the deadline is reached, so neither the customer nor "
        "the store has to take any action for the money to move correctly. "
        "The main limitation is that the contract can only see what it's told "
        "by an external data feed about whether the order was fulfilled, so "
        "it's only as trustworthy as that outside data source."
    ),
    "structured_bullets": (
        "Three things happen: the customer's payment goes into escrow at "
        "order time. The contract checks a deadline against store "
        "confirmation. If confirmed in time, funds go to the store; if not, "
        "funds return to the customer automatically. The weak point is the "
        "contract needs external proof that the order was actually "
        "completed, and that proof source could be inaccurate or "
        "compromised."
    ),
    "terse_minimal": (
        "Hold the payment in a smart contract until the deadline. Auto-pay "
        "the store if they confirm, auto-refund the customer if they don't, "
        "no human needed either way. Biggest weakness: it trusts whatever "
        "outside source tells it the order was fulfilled."
    ),
    "rental_deposit_analogy": (
        "It's similar to how a rental security deposit could be handled "
        "automatically: the money sits untouched, and code releases it one "
        "way or the other once a fixed date passes and some outside signal "
        "says whether the tenant met the conditions. The whole system only "
        "works if that outside signal is accurate, since the contract "
        "itself has no way to check reality directly."
    ),
    "risk_first_structure": (
        "The biggest weakness of this approach is that a smart contract "
        "can't independently verify a real-world delivery, so it must trust "
        "some outside oracle to report whether the order happened, and a "
        "compromised or incorrect oracle breaks the whole system. That "
        "said, the core mechanism is simple: deposit funds are held in "
        "code-enforced escrow and released automatically to whichever party "
        "the deadline-and-confirmation logic favors, without needing either "
        "side to manually process a refund."
    ),
    "nonnative_style": (
        "The money from customer is keep in the smart contract like a lock "
        "box when order is place. If store say yes before time is up, money "
        "go to store. If not confirm, money go back to customer by itself, "
        "nobody need to do nothing. Problem is contract only know what "
        "oracle tell it about real world, so if oracle wrong the contract "
        "also wrong even code work fine."
    ),
}

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
reference_labels = list(REFERENCE_ANSWERS.keys())
reference_embeddings = model.encode(
    [REFERENCE_ANSWERS[label] for label in reference_labels],
    convert_to_tensor=True,
    normalize_embeddings=True,
)
print(f"Model ready. {len(reference_labels)} reference answers loaded.")

app = FastAPI()

hint_counts: dict[str, int] = {}


def compute_similarity(text: str) -> tuple[float, str]:
    """Returns (best score, label of the reference answer it matched best)."""
    if not text.strip():
        return 0.0, reference_labels[0]
    embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(embedding, reference_embeddings)[0]
    best_idx = int(sims.argmax())
    score = max(0.0, min(1.0, sims[best_idx].item()))
    return score, reference_labels[best_idx]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/question")
def get_question():
    return {
        "question": QUESTION_TEXT,
        "threshold": SIMILARITY_THRESHOLD,
        "max_hints": MAX_HINTS,
        "session_id": secrets.token_hex(8),
        "demo_mode": DEMO_MODE,
    }


@app.get("/api/reveal-answer")
def reveal_answer():
    if not DEMO_MODE:
        return {"error": "not_available"}
    return {"reference_answers": REFERENCE_ANSWERS}


@app.websocket("/ws/similarity")
async def ws_similarity(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            text = data.get("text", "")
            score, best_ref = await asyncio.to_thread(compute_similarity, text)
            await websocket.send_json(
                {
                    "type": "similarity",
                    "score": score,
                    "below_threshold": score < SIMILARITY_THRESHOLD,
                    "matched_reference": best_ref,
                }
            )
    except WebSocketDisconnect:
        pass


@app.post("/api/hint")
async def get_hint(payload: dict):
    session_id = payload.get("session_id", "anon")
    text = payload.get("text", "")
    used = hint_counts.get(session_id, 0)

    if used >= MAX_HINTS:
        return {"error": "no_hints_left", "hints_used": used, "max_hints": MAX_HINTS}

    if not OPENROUTER_API_KEY:
        return {"error": "no_api_key"}

    system_prompt = (
        "You are a brief exam hint assistant. A student is answering an "
        "exam question. You are given several valid phrasings of the "
        "reference answer (hidden from the student, any one of them counts "
        "as correct) and the student's current draft. Give ONE short hint "
        "(1-2 sentences) pointing to a missing or underdeveloped idea in "
        "their draft. Never reveal the reference answers' wording or give "
        "the answer outright. Be encouraging and specific."
    )
    reference_block = "\n\n".join(
        f"Valid phrasing {i + 1}:\n{answer}"
        for i, answer in enumerate(REFERENCE_ANSWERS.values())
    )
    user_prompt = (
        f"Reference answers (hidden from student):\n\n{reference_block}\n\n"
        f"Student's current draft:\n{text or '(empty so far)'}\n\n"
        "Give one short hint."
    )

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 120,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            hint_text = body["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            return {"error": "llm_failed", "detail": str(exc)}

    hint_counts[session_id] = used + 1
    return {"hint": hint_text, "hints_used": used + 1, "max_hints": MAX_HINTS}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
