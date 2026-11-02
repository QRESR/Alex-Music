#!/usr/bin/env python3
"""Прокси Codex → mimo: вырезает НЕПОДДЕРЖИВАЕМЫЕ tools (web_search и т.д.),
иначе mimo-шлюз отвечает 400 `responses_feature_not_supported` и чат не работает.

Codex 0.147 на wire_api=responses ВСЕГДА шлёт tool `web_search`, а наш
mimo-шлюз поддерживает ТОЛЬКО `function` tools (проверено: function=200,
web_search/custom/code_interpreter=400). Прокси сидит между Codex и mimo:
принимает POST /v1/responses, фильтрует tools → переадресует в реальный mimo
→ отдаёт ответ (в т.ч. SSE stream) обратно Codex.

Запуск: python3 mimo_proxy.py  (порт 8012)
Ключ: из env MIMO_API_KEY или LLM_API_KEY (тот же, что у music-proxy).
"""
from __future__ import annotations

import json
import os
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MIMO_BASE = os.environ.get(
    "MIMO_BASE_URL", "https://token-plan-ams.xiaomimimo.com/v1").rstrip("/")
MIMO_KEY = (os.environ.get("MIMO_API_KEY")
            or os.environ.get("LLM_API_KEY", "")).strip()
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.6-flash")

# Входящая авторизация прокси (№6 аудита): если задан PROXY_TOKEN — каждый
# запрос к /v1/responses должен нести `Authorization: Bearer <PROXY_TOKEN>`.
# Пусто/не задан → проверка выключена (локальный 127.0.0.1-режим без сети).
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "").strip()

# mimo поддерживает только function (проверено live). Всё остальное режем.
SUPPORTED_TOOL_TYPES = {"function"}

app = FastAPI(title="codex-mimo-proxy")


def _check_auth(request: Request) -> JSONResponse | None:
    """Проверка входящего токена (401 при несовпадении). None = ок."""
    if not PROXY_TOKEN:
        return None
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {PROXY_TOKEN}":
        return None
    return JSONResponse({"error": "unauthorized"}, status_code=401)


def _sanitize_tools(body: dict) -> tuple[dict, list[str]]:
    """Убирает из body tools, которые mimo отвергнет (не function).

    Возвращает (очищенный body, типы удалённых tools).
    """
    tools = body.get("tools")
    if not isinstance(tools, list):
        return body, []
    kept = []
    removed_types: list[str] = []
    for t in tools:
        if isinstance(t, dict):
            ttype = t.get("type")
            # mimo принимает только function; неизвестный/пустой type → режем
            if ttype in SUPPORTED_TOOL_TYPES or (
                    ttype is None and t.get("name") and t.get("parameters")
            ):
                kept.append(t)
                continue
            removed_types.append(str(ttype))
        else:
            removed_types.append(type(t).__name__)
    if removed_types:
        body = dict(body)
        body["tools"] = kept
    return body, removed_types


@app.post("/v1/responses")
async def proxy_responses(request: Request):
    denied = _check_auth(request)
    if denied is not None:
        return denied
    if not MIMO_KEY:
        return JSONResponse(
            {"error": "MIMO_API_KEY/LLM_API_KEY не задан"}, status_code=500)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "bad json"}, status_code=400)
    body, removed_types = _sanitize_tools(body)
    if removed_types:
        print(f"[proxy] вырезано {len(removed_types)} неподдерж. tools -> "
              f"{removed_types}", file=sys.stderr)
        print(f"[proxy] осталось tools={len(body.get('tools') or [])}",
              file=sys.stderr)

    url = f"{MIMO_BASE}/responses"
    headers = {
        "Authorization": f"Bearer {MIMO_KEY}",
        "Content-Type": "application/json",
    }
    stream = bool(body.get("stream"))
    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            # SSE: собираем ВЕСЬ поток в буфер (httpx stream + async-generator
            # через StreamingResponse обрывался: peer closed connection без
            # complete body). mimo-ответы короткие (<120с) — буфер безопасен.
            try:
                chunks: list[bytes] = []
                async with client.stream(
                    "POST", url, json=body, headers=headers,
                ) as upstream:
                    async for chunk in upstream.aiter_bytes():
                        chunks.append(chunk)
                payload = b"".join(chunks)
            except Exception as e:  # noqa: BLE001
                print(f"[proxy] stream error: {type(e).__name__}: {e}",
                      file=sys.stderr)
                return JSONResponse(
                    {"error": f"upstream stream failed: {e}"},
                    status_code=502)
            async def _gen_all():
                # отдаём по кускам, чтобы SSE доезжал до Codex целиком
                step = 4096
                for i in range(0, len(payload), step):
                    yield payload[i:i + step]
            return StreamingResponse(
                _gen_all(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"},
            )
        r = await client.post(url, json=body, headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/health")
async def health():
    return {"ok": True, "base": MIMO_BASE, "model": MIMO_MODEL,
            "key": bool(MIMO_KEY)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8012"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
