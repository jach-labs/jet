"""HTTP server exposing POST /v1/decide, request-compatible with Jev (jevtypesafeai.com).

    JET_ADAPTER=adapters/jet jet-serve --port 8000

Set JET_API_KEY to require `Authorization: Bearer <key>`. Over-long states are
middle-truncated to the model's max state tokens rather than rejected.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from jet.evaluate import adapter_base
from jet.format import Question
from jet.model import DEFAULT_BASE_MODEL, Jet

MAX_QUESTIONS = 64


class DecideRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, dict[str, Any]]
    model: str | None = None


def create_app(adapter: str | None, base_model: str, model_name: str) -> FastAPI:
    # MLX streams are per-thread, so the model lives on one worker thread and every
    # request runs there (this also serialises inference).
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jet-mlx")
    jet = worker.submit(Jet, adapter_base(adapter, base_model), adapter).result()
    api_key = os.environ.get("JET_API_KEY")
    app = FastAPI(title="Jet", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.post("/v1/decide")
    async def decide(req: DecideRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid API key")
        if req.model not in (None, "jet-latest", "jev-latest", model_name):
            raise HTTPException(400, f"unknown model {req.model!r}; this server serves {model_name!r}")
        if not req.questions:
            raise HTTPException(400, "questions must not be empty")
        if len(req.questions) > MAX_QUESTIONS:
            raise HTTPException(400, f"at most {MAX_QUESTIONS} questions per request")
        try:
            questions = {name: Question.from_dict(q) for name, q in req.questions.items()}
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid question: {e}") from e

        result = await asyncio.wrap_future(worker.submit(jet.decide, req.state, questions))
        return {"model": model_name, **result}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a Jet model over HTTP.")
    ap.add_argument("--adapter", default=os.environ.get("JET_ADAPTER"))
    ap.add_argument("--base-model", default=os.environ.get("JET_BASE_MODEL", DEFAULT_BASE_MODEL))
    ap.add_argument("--model-name", default="jet-local-0.1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(args.adapter, args.base_model, args.model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
