"""Общие asyncio-примитивы кандидато-сбора (ТЗ п.1, патч 2).

Семафор ограничивает число ОДНОВРЕМЕННЫХ обращений к источникам музыки
(Яндекс/VK/MA) ВЕЗДЕ (поиск, radio, smart) одним лимитом:
  - защищает от rate-limit при параллельных запросах юзеров;
  - удерживает RAM/сеть в рамках 2 vCPU VPS;
  - медленный источник не утаивает слоты впустую — таймауты отдельные
    у каждого клиента (ym 20с, vk 25-45с, ma 10с).

Использование:
    from . import concurrency as _cc
    async with _cc.source_gate():
        ...
"""
from __future__ import annotations

import asyncio

# Максимум одновременных обращений к источникам музыки.
# 8 = запас при ~3 параллельных юзерах × 3 источника, с зазором на radio/smart.
MAX_SOURCE_CONCURRENCY = 8

_gate: asyncio.Semaphore | None = None


def get_gate() -> asyncio.Semaphore:
    """Лениво создаётся семафор в контексте текущего event loop (uvicorn)."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(MAX_SOURCE_CONCURRENCY)
    return _gate


def source_gate() -> asyncio.Semaphore:
    """`async with source_gate(): ...` — слот на один вызов источника."""
    return get_gate()
