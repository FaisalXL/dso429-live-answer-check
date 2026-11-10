import asyncio
import json
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

STATIC_DIR = ROOT_DIR / "static"
QUESTIONS_PATH = Path(os.environ.get("QUESTIONS_PATH", ROOT_DIR / "backend" / "questions.json"))
SUBMISSIONS_DIR = Path(os.environ.get("SUBMISSIONS_DIR", ROOT_DIR / "submissions"))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
# Rewrites in fix-the-AI questions need strict instruction-following. Tested
# (scratchpad/assignment_validation_test.py): gpt-4o-mini left garbled
# half-sentences behind; claude-haiku-5.5 added explanation the student never
# gave; claude-sonnet-5.5 kept to the student's own words.
REVISION_MODEL = os.environ.get("REVISION_MODEL", "anthropic/claude-sonnet-5.5")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
HINT_THRESHOLD = float(os.environ.get("HINT_THRESHOLD", "0.5"))
PASS_THRESHOLD = float(os.environ.get("PASS_THRESHOLD", "0.8"))
MAX_HINTS = int(os.environ.get("MAX_HINTS", "3"))
# Exposes reference answers and a question navigator for demos. MUST be
# false for any run with real students: it hands them the answer key.
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
# Dev-only cosine-similarity readout, kept for comparison with Jev. Off by
# default so the deployed build doesn't need PyTorch.
DEV_COSINE = os.environ.get("DEV_COSINE", "false").lower() == "true"

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

ASSIGNMENT = json.loads(QUESTIONS_PATH.read_text())
QUESTIONS = {q["id"]: q for q in ASSIGNMENT["questions"]}

# One shared client for the app's lifetime. A new client per request caused
# an out-of-memory crash on Render's free tier.
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http_client
    http_client = httpx.AsyncClient()
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

hint_counts: dict[tuple[str, str], int] = {}


def get_question(question_id: str, expected_type: str | None = None) -> dict:
    question = QUESTIONS.get(question_id)
    if question is None or (expected_type and question["type"] != expected_type):
        raise HTTPException(status_code=404, detail="unknown question")
    return question


def describe_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 401:
            return "the scoring service's API key is invalid or expired"
        if code == 402:
            return "the scoring service's API key is out of credit"
        if code == 429:
            return "the scoring service is rate-limiting requests; try again in a moment"
        return f"the scoring service returned an error ({code})"
    if isinstance(exc, httpx.TimeoutException):
        return "the scoring service took too long to answer"
    if isinstance(exc, httpx.HTTPError):
        return "could not reach the scoring service"
    return getattr(exc, "detail", None) or str(exc)


def auth_headers() -> dict:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not set")
    return {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}


async def jev_decide(state: dict, questions: dict) -> tuple[dict, float]:
    """Calls the Jev decisions API. Returns (answers, latency_ms)."""
    t0 = time.perf_counter()
    resp = await http_client.post(
        DECISIONS_URL,
        headers=auth_headers(),
        json={"model": JEV_MODEL, "state": state, "questions": questions},
        timeout=15.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json()["answers"], latency_ms


async def score_live(question: dict, text: str) -> dict:
    if not text.strip():
        return {"score": 0.0, "latency_ms": 0}
    answers, latency_ms = await jev_decide(
        {
            "question": question["prompt"],
            "reference_answer": question["reference_answer"],
            "student_answer": text,
        },
        {"correct": {"type": "noul", **question["jev"]}},
    )
    return {"score": answers["correct"]["noul"], "latency_ms": round(latency_ms)}


def mistake_check(mistake: dict) -> dict:
    # A mistake counts as fixed only when the correct idea replaces it.
    # Merely deleting the wrong sentence used to count, which let students
    # earn progress by pointing at sentences without knowing what was wrong.
    return {
        "type": "noul",
        "instructions": (
            "Check one point in the AI's answer, and ignore every other part of it, "
            "including any other mistakes. Condition A: the answer does not make "
            f"this wrong claim anywhere: {mistake['error']} Mentioning the wrong "
            "idea only to deny it is fine. Condition B: the answer states this "
            f"correct idea, in any wording: {mistake['fix']} The answer does not "
            "need to mention the wrong claim at all. Answer true only if both "
            "conditions hold."
        ),
        "criteria": {
            "true": "The wrong claim is gone and the correct idea is stated.",
            "false": (
                "The wrong claim is still there in any wording, or it was removed "
                "without stating the correct idea, or it was replaced by a "
                "different wrong idea."
            ),
        },
    }


async def score_correction(question: dict, ai_answer: str) -> dict:
    """One Jev call scores overall correctness and checks each seeded mistake."""
    checks = {"correct": {"type": "noul", **question["jev"]}}
    for mistake in question["mistakes"]:
        checks[mistake["id"]] = mistake_check(mistake)
    answers, latency_ms = await jev_decide(
        {
            "question": question["prompt"],
            "reference_answer": question["reference_answer"],
            "ai_answer": ai_answer,
        },
        checks,
    )
    fixed = [answers[m["id"]]["noul"] >= 0.5 for m in question["mistakes"]]
    score = answers["correct"]["noul"]
    # "Fully correct" alone sits near 0 until every mistake is fixed, so
    # students saw no movement for a correct fix. Each fixed mistake is worth
    # an equal share; the last share is Jev's whole-answer correctness.
    progress = (sum(fixed) + score) / (len(fixed) + 1)
    return {
        "score": score,
        "progress": progress,
        "fixed": fixed,
        "fixed_count": sum(fixed),
        "solved": all(fixed) and score >= PASS_THRESHOLD,
        "latency_ms": round(latency_ms),
    }


# The AI must fix only what the student points out. Without this, any LLM
# fixes every mistake as soon as a student types "fix it", and the exercise
# no longer tests whether the student can find the mistakes.
REVISE_SYSTEM_PROMPT = (
    "You wrote an answer to an exam question. A student is reviewing your answer "
    "and gives you feedback. Revise your answer by applying the student's "
    "feedback, following these rules exactly.\n"
    "1. First decide whether the feedback names a specific claim in your answer "
    "or says specifically what is wrong with it. If it does not (for example "
    "'fix it', 'this is wrong', 'try again', or 'be more accurate'), return your "
    "answer exactly unchanged and ignore the other rules.\n"
    "2. Change only the claims that the feedback addresses. Keep every other "
    "sentence exactly as it is, word for word, including any errors the student "
    "did not mention.\n"
    "3. When the feedback says a claim is wrong, rewrite or remove the sentence "
    "that makes that claim, so the answer no longer makes it. Never leave the "
    "wrong claim in place and add a correction next to it. The revised answer "
    "must not contradict itself.\n"
    "4. Build each rewritten sentence from what the student says. If the student "
    "explains the correct idea, state that idea clearly. If the student says a "
    "claim is wrong but not why or what should replace it, leave that claim "
    "exactly as it is.\n"
    "5. Apply the feedback faithfully even if you think it is wrong. Do not "
    "argue, and do not add corrections or ideas of your own.\n"
    "6. Write any sentence you change in clear, natural English that fits the "
    "sentences around it.\n"
    "7. Return only the revised answer text, with no preamble, notes, or "
    "quotation marks."
)


# Prompt rules alone did not stop the rewriting model from acting on vague
# notes like "fix it", questions, or notes that only point at a sentence.
# Jev reviews the note first, and nothing reaches the rewriter unless the
# note says what is wrong and why. Tested in
# scratchpad/adversarial_correction_test.py.
CRITIQUE_CHECK = {
    "type": "noul",
    "instructions": (
        "Is the student's note a critique of the AI's answer that says which claim "
        "is wrong and why it is wrong, or what the correct idea is? Answer false "
        "for: a question to the AI; a request to rewrite, fix, or solve the answer; "
        "a generic note such as 'fix it' or 'this is wrong'; and a note that only "
        "points at a claim or sentence without saying why it is wrong. A critique "
        "counts even if the student's reasoning is itself wrong."
    ),
    "criteria": {
        "true": "Names at least one claim and says why it is wrong or what should replace it.",
        "false": "A question, a request to fix or rewrite, a generic note, or only points at claims without a reason.",
    },
}


def explains_check(mistake: dict) -> dict:
    return {
        "type": "noul",
        "instructions": (
            "Does the student's note correct this point in the AI's answer? The AI's "
            f"answer wrongly claims: {mistake['error']} The correct idea is: "
            f"{mistake['fix']} The note counts only if it says this claim is wrong and "
            "gives the correct idea or a reason that matches it. Only pointing at "
            "the claim does not count."
        ),
        "criteria": {
            "true": "The note identifies this wrong claim and gives the correct idea or a matching reason.",
            "false": "The note does not address this claim, only points at it, or gives a reason that does not match.",
        },
    }


async def review_note(question: dict, current_answer: str, feedback: str) -> dict:
    """One Jev call: is the note a real critique, which mistakes does it
    explain, and which mistakes are already fixed in the current answer."""
    checks = {"critique": CRITIQUE_CHECK}
    for m in question["mistakes"]:
        checks[f"explains_{m['id']}"] = explains_check(m)
        checks[f"before_{m['id']}"] = mistake_check(m)
    answers, _ = await jev_decide(
        {"question": question["prompt"], "ai_answer": current_answer, "student_note": feedback},
        checks,
    )
    return {
        "critique": answers["critique"]["noul"] >= 0.5,
        "explained": [answers[f"explains_{m['id']}"]["noul"] >= 0.5 for m in question["mistakes"]],
        "fixed_before": [answers[f"before_{m['id']}"]["noul"] >= 0.5 for m in question["mistakes"]],
    }


async def revise_answer(question: dict, current_answer: str, feedback: str) -> str:
    resp = await http_client.post(
        CHAT_URL,
        headers=auth_headers(),
        json={
            "model": REVISION_MODEL,
            "messages": [
                {"role": "system", "content": REVISE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Exam question:\n{question['prompt']}\n\n"
                        f"Your current answer:\n{current_answer}\n\n"
                        f"Student feedback:\n{feedback}\n\n"
                        "Revised answer:"
                    ),
                },
            ],
            # Newer models spend part of this budget on hidden reasoning. At 500,
            # answers were cut off mid-sentence.
            "max_tokens": 4000,
            "temperature": 0.2,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    # A cut-off answer silently drops the remaining wrong claims, so Jev would
    # report mistakes as fixed that were only truncated away. Never score one.
    if choice.get("finish_reason") == "length":
        raise HTTPException(status_code=502, detail="the AI's answer was cut off; send the feedback again")
    return choice["message"]["content"].strip().strip('"')


def public_question(question: dict) -> dict:
    view = {k: question[k] for k in ("id", "type", "topic", "title", "prompt")}
    if question["type"] == "correction":
        view["seed_answer"] = question["seed_answer"]
        view["mistake_count"] = len(question["mistakes"])
        view["max_rounds"] = question.get("max_rounds", 5)
    return view


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/assignment")
def get_assignment():
    return {
        "title": ASSIGNMENT["title"],
        "subtitle": ASSIGNMENT.get("subtitle", ""),
        "company_note": ASSIGNMENT.get("company_note", ""),
        "questions": [public_question(q) for q in ASSIGNMENT["questions"]],
        "hint_threshold": HINT_THRESHOLD,
        "pass_threshold": PASS_THRESHOLD,
        "max_hints": MAX_HINTS,
        "session_id": secrets.token_hex(8),
        "demo_mode": DEMO_MODE,
        "dev_cosine": DEV_COSINE,
    }


@app.get("/api/reveal")
def reveal():
    if not DEMO_MODE:
        raise HTTPException(status_code=404, detail="not available")
    return {
        q["id"]: {
            "reference_answer": q["reference_answer"],
            "mistakes": [f"{m['error']} Correct idea: {m['fix']}" for m in q.get("mistakes", [])],
        }
        for q in ASSIGNMENT["questions"]
    }


@app.websocket("/ws/jev")
async def ws_jev(websocket: WebSocket):
    """Scores live answers. Keeps only the newest text per question and runs
    one Jev call at a time, so a student typing faster than Jev answers
    never builds a backlog; intermediate keystrokes are skipped."""
    await websocket.accept()
    pending: dict[str, dict] = {}
    wake = asyncio.Event()

    async def worker():
        while True:
            await wake.wait()
            while pending:
                question_id, msg = pending.popitem()
                reply = {"type": "jev", "question_id": question_id, "seq": msg.get("seq", 0)}
                try:
                    question = get_question(question_id, "live")
                    reply.update(await score_live(question, msg.get("text", "")))
                except Exception as exc:  # noqa: BLE001
                    reply["error"] = describe_error(exc)
                try:
                    await websocket.send_json(reply)
                except Exception:  # noqa: BLE001
                    return
            wake.clear()

    task = asyncio.create_task(worker())
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            pending[str(msg.get("question_id", ""))] = msg
            wake.set()
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()


@app.post("/api/hint")
async def get_hint(payload: dict):
    question = get_question(payload.get("question_id", ""), "live")
    key = (payload.get("session_id", "anon"), question["id"])
    used = hint_counts.get(key, 0)
    if used >= MAX_HINTS:
        return {"error": "no_hints_left", "hints_used": used, "max_hints": MAX_HINTS}

    text = payload.get("text", "")
    system_prompt = (
        "You are a brief exam hint assistant. You are given an exam question, the "
        "reference answer (hidden from the student), and the student's current "
        "draft. Give ONE short hint (1-2 sentences) pointing to a missing or "
        "underdeveloped idea in the draft. Never reveal the reference answer's "
        "wording or give the answer outright. Be encouraging and specific."
    )
    user_prompt = (
        f"Exam question:\n{question['prompt']}\n\n"
        f"Reference answer (hidden from student):\n{question['reference_answer']}\n\n"
        f"Student's current draft:\n{text or '(empty so far)'}\n\n"
        "Give one short hint."
    )
    try:
        resp = await http_client.post(
            CHAT_URL,
            headers=auth_headers(),
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 120,
                "temperature": 0.7,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        hint_text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        return {"error": "llm_failed", "detail": describe_error(exc)}

    hint_counts[key] = used + 1
    return {"hint": hint_text, "hints_used": used + 1, "max_hints": MAX_HINTS}


@app.post("/api/correction/score")
async def correction_score(payload: dict):
    question = get_question(payload.get("question_id", ""), "correction")
    answer = str(payload.get("answer", ""))[:4000]
    try:
        return await score_correction(question, answer)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=describe_error(exc)) from exc


@app.post("/api/correction/round")
async def correction_round(payload: dict):
    question = get_question(payload.get("question_id", ""), "correction")
    current_answer = str(payload.get("current_answer", ""))[:4000]
    feedback = str(payload.get("feedback", "")).strip()[:2000]
    if not feedback:
        raise HTTPException(status_code=400, detail="feedback is empty")
    # Mistakes the browser saw as fixed before this round. Used only to make
    # the overreach check more lenient, never stricter, so a false value
    # can't hurt anyone but the sender.
    previous_fixed = payload.get("previous_fixed") or []
    try:
        review = await review_note(question, current_answer, feedback)
        if not review["critique"]:
            return {
                "revised_answer": current_answer, "feedback_specific": False,
                "changed": False, "rejected": False,
                **await score_correction(question, current_answer),
            }
        revised = await revise_answer(question, current_answer, feedback)
        result = await score_correction(question, revised)
        # Enforce on the output, not by trusting the rewriter: a mistake may
        # only become fixed if it already was, or if this note explained it.
        allowed = [
            before or explained or (i < len(previous_fixed) and previous_fixed[i] is True)
            for i, (before, explained) in enumerate(zip(review["fixed_before"], review["explained"]))
        ]
        if any(fixed and not ok for fixed, ok in zip(result["fixed"], allowed)):
            return {
                "revised_answer": current_answer, "feedback_specific": True,
                "changed": False, "rejected": True,
                **await score_correction(question, current_answer),
            }
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=describe_error(exc)) from exc
    return {
        "revised_answer": revised,
        "feedback_specific": True,
        "changed": revised.strip() != current_answer.strip(),
        "rejected": False,
        **result,
    }


async def rescore(question: dict, answer: dict) -> dict:
    """Re-scores a submitted answer on the server, so the stored score
    doesn't depend on anything the browser reported."""
    try:
        if question["type"] == "live":
            return await score_live(question, str(answer.get("text", "")))
        return await score_correction(question, str(answer.get("ai_answer", "")))
    except Exception as exc:  # noqa: BLE001
        return {"error": describe_error(exc)}


@app.post("/api/submit")
async def submit(payload: dict):
    student = payload.get("student") or {}
    name = str(student.get("name", "")).strip()
    student_id = str(student.get("id", "")).strip()
    if not name or not student_id:
        raise HTTPException(status_code=400, detail="name and student ID are required")

    answers = payload.get("answers") or {}
    feedback = payload.get("feedback") or {}
    server_scores = dict(
        zip(
            QUESTIONS.keys(),
            await asyncio.gather(
                *(rescore(q, answers.get(qid) or {}) for qid, q in QUESTIONS.items())
            ),
        )
    )
    feedback_credits = sum(1 for qid in QUESTIONS if str(feedback.get(qid, "")).strip())

    submitted_at = datetime.now(timezone.utc)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", student_id)[:40] or "anon"
    submission_id = f"{safe_id}_{submitted_at.strftime('%Y%m%dT%H%M%SZ')}"
    record = {
        "submission_id": submission_id,
        "submitted_at": submitted_at.isoformat(),
        "student": {"name": name, "id": student_id},
        "consent_claude_code_grading": bool(payload.get("consent")),
        "client": {
            "session_id": payload.get("session_id"),
            "started_at": payload.get("started_at"),
            "restarts": payload.get("restarts", 0),
        },
        "answers": answers,
        "feedback": feedback,
        "feedback_credits": feedback_credits,
        "server_scores": server_scores,
    }
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (SUBMISSIONS_DIR / f"{submission_id}.json").write_text(json.dumps(record, indent=2))
    return {
        "ok": True,
        "submission_id": submission_id,
        "server_scores": server_scores,
        "feedback_credits": feedback_credits,
        "record": record,
    }


if DEV_COSINE:
    from backend.cosine import register_cosine_routes

    register_cosine_routes(app, QUESTIONS)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
