"""A bounded, CPU-only Jet API for Docker Spaces."""
import asyncio
import hmac
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from format import Question
from onnx_model import OnnxJet


@asynccontextmanager
async def lifespan(app):
    with ThreadPoolExecutor(max_workers=1) as worker:
        app.state.worker = worker
        app.state.jet = await asyncio.wrap_future(worker.submit(OnnxJet))
        app.state.busy = False
        yield


app = FastAPI(title="Jet CPU API", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=[x for x in os.getenv("JET_CORS_ORIGINS", "https://jach.me,https://michaljach-jet.hf.space").split(",") if x],
    allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


class Request(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, dict[str, Any]]
    model: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "jet-q8"}


@app.post("/v1/decide")
async def decide(body: Request, authorization: str = Header(default="")):
    key = os.getenv("JET_API_KEY")
    if key and not hmac.compare_digest(authorization.encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "invalid API key")
    if body.model not in (None, "jet-q8", "jet-latest", "jev-latest"):
        raise HTTPException(400, "unknown model")
    if not 1 <= len(body.questions) <= 8:
        raise HTTPException(400, "provide 1–8 questions")
    if len(body.model_dump_json()) > 100_000:
        raise HTTPException(413, "request exceeds 100 KB")
    try:
        questions = {name: Question.from_dict(q) for name, q in body.questions.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if app.state.busy:
        raise HTTPException(503, "Jet is busy; retry shortly", headers={"Retry-After": "5"})
    app.state.busy = True
    future = app.state.worker.submit(app.state.jet.decide, body.state, questions)
    # A disconnected request must not free the slot while its inference is running.
    task = asyncio.wrap_future(future)
    task.add_done_callback(lambda _: setattr(app.state, "busy", False))
    try:
        return await asyncio.shield(task)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
