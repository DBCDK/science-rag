#!/usr/bin/env python3
"""
:mod:`science_rag.service.endpoints` -- FastAPI routes for the RAG service

=========
Endpoints
=========

No ``dbc_pyutils`` imports here — the ``/metrics`` route and the enriched
``/status`` payload are wired up conditionally in ``service.start.create_app``
instead, based on ``service._dbc_optional.DBC_AVAILABLE``.
"""

import json
import logging
import resource
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError

logger = logging.getLogger(__name__)

router = APIRouter()


def _flatten_messages(messages: list[dict]) -> list[dict]:
    """The rag pipeline expects content to be a str, not a list of dicts."""
    return [
        {"role": msg["role"], "content": content["text"]}
        for msg in messages
        for content in (
            msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
        )
        if content.get("type", "") == "text"
    ]


def _frame(
    chat_id: str, created: int, model_name: str, delta: str | None = None, finish_reason: str | None = None
) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = _flatten_messages(body.get("messages", []))
    stream = body.get("stream", False)
    model_name = body.get("model", request.app.state.default_model)

    result = await request.app.state.agentic_graph.graph.ainvoke({"input": messages})

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:

        async def chunk_generator():
            try:
                async for delta in result["output"]:
                    yield f"data: {json.dumps(_frame(chat_id, created, model_name, delta=delta))}\n\n"
                yield f"data: {json.dumps(_frame(chat_id, created, model_name, finish_reason='stop'))}\n\n"
            except APIError as e:
                logger.warning(f"Upstream LLM error mid-stream: {e}")
                error_frame = {"error": {"message": str(e), "type": e.__class__.__name__}}
                yield f"data: {json.dumps(error_frame)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunk_generator(), media_type="text/event-stream")

    try:
        output = "".join([token async for token in result["output"]])
    except APIError as e:
        logger.warning(f"Upstream LLM error: {e}")
        return JSONResponse(
            {"error": {"message": str(e), "type": e.__class__.__name__}},
            status_code=502,
        )

    return JSONResponse(
        {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": "stop",
                }
            ],
        }
    )


@router.get("/status")
async def status(request: Request):
    state = request.app.state
    if not state.dbc_available:
        return {"status": "ok"}

    info = state.build_info
    payload = {
        "ok": True,
        "build": info["build_number"],
        "git": info.get("git_revision"),
        "version": info["version"],
        "instance-id": state.instance_id,
        "ab-id": state.ab_id,
        "mem-usage": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    statistics = list(state.stats.values())
    if statistics:
        payload["statistics"] = [stat.describe() for stat in statistics]
    return payload
