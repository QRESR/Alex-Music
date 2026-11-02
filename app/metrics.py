"""ТЗ п.14 «Метрики качества» — in-process счётчики и тайминги.

Отделяет КАЧЕСТВО ПОНИМАНИЯ КОМАНД (action, fallback, дедуп, таймауты LLM)
от КАЧЕСТВА РЕКОМЕНДАЦИЙ (match@5/uniq — считаются в recommender/bench).

Секретов не содержит. Один пользователь — глобальные счётчики корректны.
Сбрасываются при рестарте сервиса (сознательно: окно = uptime).
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()

_counters: dict[str, int] = {}
# тайминги LLM: кольцевой буфер последних 200 (мс, успех/неуспех)
_llm_times: list[tuple[float, bool]] = []
_LLM_BUF = 200


def event(name: str, n: int = 1) -> None:
    """Инкремент счётчика (smart_action_null, llm_error, dedup_hit...)."""
    with _LOCK:
        _counters[name] = _counters.get(name, 0) + n


def observe_llm(ms: float, ok: bool) -> None:
    """Замер времени ответа LLM (ТЗ: среднее время + p95 + доля ошибок)."""
    with _LOCK:
        _llm_times.append((ms, bool(ok)))
        del _llm_times[:-_LLM_BUF]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, int(round(p / 100.0 * (len(vs) - 1)))))
    return round(vs[idx], 1)


def snapshot() -> dict:
    """Текущее состояние метрик для GET /api/metrics."""
    with _LOCK:
        c = dict(_counters)
        times = list(_llm_times)
    ok_ms = [t for t, ok in times if ok]
    all_ms = [t for t, _ in times]
    smart_total = c.get("smart_total", 0)
    smart_null = c.get("smart_action_null", 0)
    llm_total = c.get("llm_total", 0)
    llm_err = c.get("llm_error", 0)
    return {
        "uptime_events": c,
        # ТЗ п.14: понимание команд
        "smart": {
            "total": smart_total,
            "action_null": smart_null,
            "action_null_pct": round(100.0 * smart_null / smart_total, 1) if smart_total else 0.0,
            "fastpath_hits": c.get("smart_fastpath", 0),
            "dedup_hits": c.get("smart_dedup", 0),
            "quality_reactions": c.get("smart_quality_reaction", 0),
            "text_feedback": c.get("smart_text_feedback", 0),
            "fallback_null": c.get("smart_null_fallback", 0),
            "errors": c.get("smart_error", 0),
        },
        "llm": {
            "calls": llm_total,
            "errors": llm_err,
            "error_pct": round(100.0 * llm_err / llm_total, 1) if llm_total else 0.0,
            "mean_ms": round(sum(all_ms) / len(all_ms), 1) if all_ms else 0.0,
            "p50_ms": _pct(all_ms, 50),
            "p95_ms": _pct(all_ms, 95),
            "intent_parse_failed": c.get("intent_parse_failed", 0),
            "intent_cache_hits": c.get("intent_cache_hit", 0),
        },
        "search": {
            "total": c.get("search_total", 0),
            "empty": c.get("search_empty", 0),
            "recommender_failed": c.get("search_rank_failed", 0),
        },
        # ТЗ п.14: реакция пользователя (реальные события, не разметка)
        "feedback": {
            "like": c.get("fb_like", 0),
            "unlike": c.get("fb_unlike", 0),
            "dislike": c.get("fb_dislike", 0),
            "skip": c.get("fb_skip", 0),
            "completed": c.get("fb_completed", 0),
        },
        "generated_at": int(time.time()),
    }
