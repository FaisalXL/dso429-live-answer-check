"""Dev-only cosine-similarity readout, for comparing against Jev.

Imported only when DEV_COSINE=true. Needs requirements-dev.txt
(sentence-transformers, which pulls in PyTorch).
"""

import asyncio
import json

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sentence_transformers import SentenceTransformer, util

# One thread per encode() call lets the thread pool run calls in parallel
# instead of each call competing for every core.
torch.set_num_threads(1)


def register_cosine_routes(app: FastAPI, questions: dict) -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    references = {
        qid: model.encode(q["reference_answer"], convert_to_tensor=True, normalize_embeddings=True)
        for qid, q in questions.items()
        if q["type"] == "live"
    }

    def similarity(question_id: str, text: str) -> float:
        if question_id not in references or not text.strip():
            return 0.0
        embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return max(0.0, min(1.0, util.cos_sim(embedding, references[question_id]).item()))

    @app.websocket("/ws/cosine")
    async def ws_cosine(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                question_id = str(msg.get("question_id", ""))
                score = await asyncio.to_thread(similarity, question_id, msg.get("text", ""))
                await websocket.send_json(
                    {"type": "cosine", "question_id": question_id, "score": score}
                )
        except WebSocketDisconnect:
            pass
