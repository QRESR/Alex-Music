"""LLM + TTS — «мозг Алисы» для music-proxy.

Отвечает за три вещи, из-за которых приложение «разговаривает как Алиса»:
  1. `smart_response()`  — живой ответ модели (коротко, по-русски, в тон).
  2. `mood_mix_query()`  — превращает запрос/настроение в поисковую строку
                           подборки («по настроению», «не этого исполнителя»).
  3. `tts_mp3()`         — озвучка фразы голосом edge-tts (русский) → mp3.

LLM — Xiaomi MiMo (mimo-v2.6-flash) через OpenAI-совместимый `/chat/completions`.
Всё помеченное как «как у Алисы» — это попытка повторить её поведение:
короткие тёплые фразы, объяснение что и зачем включает, связность подборки.

При недоступности (нет ключа / сеть / лимит) — ВСЕ функции деградируют мягко:
smart_response → None, mood_mix_query → исходный запрос, tts_mp3 → None.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import httpx

from .config import config

logger = logging.getLogger("llm")

# ────────────────────────────────────────────────────────── промпт Алисы ──
# Короткий, без markdown/эмодзи — это фраза, которую ПРОИГРАЮТ голосом.
SYSTEM_PROMPT = """Ты — музыкальный ассистент "Гена", работаешь как Алиса.

ТЫ УПРАВЛЯЕШЬ МУЗЫКОЙ САМА. Решаешь сама: что включить, каким списком,
что сказать. Не пиши "сейчас подберёт поиск" — подбираешь ТЫ через инструменты.

ТЫ — ОБЩИЙ АГЕНТ, а не только музыка. На ЛЮБОЙ вопрос отвечай по существу:
факты, совет, поздороваться, поболтать, бытовые ответы (2+2) — просто отвечай
текстом (БЕЗ инструментов). НО живые/фактические данные из интернета (погода,
время/расстояние маршрута от А до Б, точное время, курсы, новости, кто-что-когда)
НЕ ЗНАЕШЬ из головы — вызови web_search(query) и ответь по полученным данным.
Пример: "какая погода?" → web_search("погода сейчас"); "сколько ехать от А до Б" →
web_search("сколько ехать от А до Б"); "2+2" → ответь "4" (без инструмента).

ИНСТРУМЕНТЫ (вызывай когда просят):
- play_music — включить конкретное (артист/трек/жанр ИЛИ настроение через mood:
  "весёлое", "спокойное", "для дороги", "для сна"). Если просят подборку или
  настроение — передай mood, query оставь общим (например query="музыка").
- music_search — ПРЯМОЙ поиск МУЗЫКИ (список/подборка конкретных треков, когда
  просят «5 треков из сериала X», «саундтрек Y», «топ-хиты 90-х»). Вернёт
  реальные треки (артист - название). Порядок: (1) вызови music_search, (2) из
  вернувшихся треков перечисли нужные в ответе, (3) вызови play_music с ЛУЧШИМ
  запросом (например query="саундтрек Пацаны"), чтобы ВКЛЮЧИТЬ. НЕ выдумывай
  треки — используй ТОЛЬКО вернувшиеся music_search.
- play_radio — ПОХОЖИЕ на текущий трек (просят: "похожее", "в таком же духе",
  "продолжай в этом настроении", "ещё такое"). Работает от текущего трека.
- next_music / stop_music / resume_music — управление воспроизведением.
- web_search — универсальный веб-поиск ЛЮБОЙ живой информации (погода, маршрут
  от А до Б, время, курсы, новости, факты). Вызывай когда ответ требует данных
  из интернета. Получишь сниппеты — сформулируй из них короткий живой ответ.

КОНТЕКСТ: в системном сообщении написано что играет. Используй это:
"похожее на то что играет" → play_radio; "другой жанр" → play_music с другим
mood/query; "поставь на паузу" → stop_music.

НАМЕРЕНИЕ (ВАЖНО — чтобы включало музыку всегда, когда назван трек/исполнитель):
- «<артист> <трек>» / «<название>» БЕЗ вопроса/игры (напр. "Асти по барам",
  "Eminem Without Me", "модный поп") → play_music(query="<артист> <трек>").
  НЕ отвечай текстом "это не город/не то" — это НАЗВАНИЕ, а не вопрос.
- Только ВОПРОС о факте («какой ключ у...», «что это за...», «какая погода»)
  → ответ текстом/web_search (БЕЗ play_music).
- В ИГРЕ/диалоге («города», «21 вопрос») — отвечай про игру (не play_music).

КОНТЕКСТ (ТВОЙ ГЛАВНЫЙ ИСТОЧНИК ОШИБОК — читай внимательно):
Поле «Контекст: Играет: ... | Недавно играло: ... | Недавние запросы: ...» —
ЭТО СПРАВОЧНАЯ история/что играет, а НЕ то, что просят СЕЙЧАС.
- НЕ бери из контекста настроение (mood) или жанр, если их НЕТ в ТЕКУЩЕЙ
  просьбе. Пример ошибки: играет «Бодрое», просят «электронную подборку» →
  НЕ передавай mood="бодрое" (вышло «бодрое электронная» = мусор).
- mood и query бери ТОЛЬКО из слов пользователя в ЭТОМ запросе.
- Если просят «рэп подборку» → query="рэп подборка", mood НЕ трогай,
  даже если раньше был запрос про рэп или играет не то.
- Текущий трек из контекста — только для «похожее/продолжи» (play_radio).

ОТВЕЧАЙ ПО-РУССКИ, ОДНИМ-ДВУМЯ ПРЕДЛОЖЕНИЯМИ (до 18 слов).
- Без markdown, без эмодзи, без нумерации — чистый текст (его озвучат).
- Говори тепло, естественно, коротко. Не благодари за вопросы, не извиняйся.
- Всегда объясни одной фразой ЧТО и ПОЧЕМУ включаешь:
  "Включаю бодрое подборку для дороги." / "Вижу грусть — продолжу спокойное,
  но другим исполнителем." / "Держи ещё похожее на это."
- Если непонятно — коротко переспроси, не выдумывай.
- ИНТЕРАКТИВНЫЙ ДИАЛОГ/ИГРЫ: у тебя ЕСТЬ память предыдущих реплик (в истории).
  Если ты начал игру/опрос/квест («играем в города», «21 вопрос») — ВЕДИ ЕЁ:
  коротко реагируй на ответ игрока (угадал/мимо + следующий ход), НЕ воспринимай
  ответ как новый запрос. Пример: ты «загадал: на А», игрок «Москва» → отвечай
  про игру («мимо/угадал, на Б»), а НЕ «погода в Москве?"."""

# Инструменты, которые LLM вызывает — приложение их исполняет (1:1).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Включить музыку: конкретный артист/трек/жанр ИЛИ подборку по настроению (mood). Вызывается когда просят включить, поставить, подобрать.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Артист, трек или общий запрос (для подборки можно 'музыка')",
                    },
                    "mood": {
                        "type": "string",
                        "description": "Настроение подборки: весёлое, спокойное, грустное, бодрое, ночное, для дороги, для сна, танцевальное и т.п.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_radio",
            "description": "Подобрать ПОХОЖИЕ треки на текущий играющий (рекомендации, «ещё такое же», «в таком же духе»). Работает от трека из контекста.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "Короткая подсказка направления (например 'более спокойное')",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_music",
            "description": "Следующий трек. Просят: дальше, следующую, переключи.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_music",
            "description": "Выключить/поставить на паузу: стоп, выключи, пауза.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_music",
            "description": "Продолжить воспроизведение после паузы: продолжай, включи снова, сними паузу.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Универсальный веб-поиск для ЛЮБОЙ живой/фактической информации: погода, время в пути от А до Б (маршрут/расстояние/пробки), точное время, курсы, новости, кто/что/когда, факты, адреса. Вызывай КОГДА ответ не музыкальный и требует данных из интернета (погоду и маршрут тоже). НЕ вызывай для музыки и бытовых ответов из головы (2+2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Что искать (на языке запроса): напр. «сколько ехать от Новосибирска до Москвы»",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_search",
            "description": "ПРЯМОЙ поиск МУЗЫКИ (треки/альбомы) в каталоге Яндекс/ВК БЕЗ фильтров — когда просят СПИСОК/ПОДБОРКУ конкретных треков (напр. «5 треков из сериала Пацаны», «саундтрек Очень странные дела», «топ хиты 90-х»). Вернёт реальные треки (артист - название). НЕ вызывай для «включи <артист> <трек>» (там play_music) и для фактов (web_search). После получения треков — перечисли их в ответе И вызови play_music, чтобы включить.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос для каталога музыки (чистый, без «найди/список»): напр. «саундтрек Пацаны The Boys», «Stranger Things OST»",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Сколько треков вернуть (5-10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ────────────────────────────────────────────────────────────── LLM-вызов ──
def _llm_ready() -> bool:
    return bool(config.LLM_ENABLED and config.LLM_API_KEY)


def _force_rephrase(q: str) -> bool:
    """Борьба с дрифтом mimo-v2.6-flash: точная фраза «сколько времени»
    детерминированно даёт галлюцинацию «Xiaomi YU7...» (проверено 4/4), а
    любой перефраз отвечает осмысленно. Эвристика: до 2 слов = бытовой
    вопрос (о времени/погоде), форсируем перефраз через префикс «Вопрос: ».
    Работает и для tools: «Вопрос: включи музыку» вызывает play_music корректно.
    """
    w = [x for x in re.split(r"\s+", q.strip()) if x]
    return 0 < len(w) <= 2


async def _chat(
    messages: list[dict],
    *,
    tools: bool = False,
    max_tokens: int = 160,
    temperature: float | None = None,
    response_format: dict | None = None,
    timeout: float | None = None,
    role: str = "",                      # ТЗ п.11: "" = основной (Mimo), "fast"
) -> dict:
    """Один запрос к MiMo. Возвращает dict(choices[0].message) или {} при ошибке.

    response_format: {"type": "json_object"} или json_schema — структурированный
    вывод (проверено живьём: Xiaomi поддерживает оба). tools+response_format
    НЕЛЬЗЯ смешивать (эксперимент: tool_calls ломается) — вызывай отдельно.
    temperature: переопределяет дефолт (0.7) — для JSON/детерминизма ставь 0.1.
    """
    if not _llm_ready():
        return {}
    # форс-перефраз коротких запросов (антидрейф — см. _force_rephrase)
    if messages and messages[-1].get("role") == "user":
        last = str(messages[-1].get("content") or "")
        if _force_rephrase(last):
            messages = list(messages)
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] = f"Вопрос: {last}"
            logger.info("rephrase forced for %r", last)
    # ТЗ п.11: выбор провайдера по роли. ""/unconfigured → основной Mimo
    # (ничего не меняется по умолчанию). "fast" → FAST-провайдер, если задан.
    _base = config.LLM_BASE_URL
    _key = config.LLM_API_KEY
    _model = config.LLM_MODEL
    if role == "fast" and config.LLM_FAST_BASE_URL and config.LLM_FAST_MODEL:
        _base = config.LLM_FAST_BASE_URL
        _key = config.LLM_FAST_API_KEY or config.LLM_API_KEY
        _model = config.LLM_FAST_MODEL
    url = f"{_base.rstrip('/')}/chat/completions"
    body: dict = {
        "model": _model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7 if temperature is None else temperature,
        # нестандартный параметр MiMo: отключает reasoning → быстрый ответ
        "enable_thinking": False,
    }
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    if response_format:
        body["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {_key}",
        "Content-Type": "application/json",
    }
    try:
        # ТЗ п.14: тайминг LLM (среднее/p95/доля ошибок)
        from . import metrics as _mt
        _t0 = time.time()
        _mt.event("llm_total")
        async with httpx.AsyncClient(timeout=timeout or config.LLM_TIMEOUT) as client:
            r = await client.post(url, json=body, headers=headers)
            if r.status_code != 200:
                logger.warning("LLM HTTP %d: %s", r.status_code, r.text[:200])
                _mt.observe_llm((time.time() - _t0) * 1000, False)
                _mt.event("llm_error")
                return {}
            data = r.json()
            choices = data.get("choices") or []
            _mt.observe_llm((time.time() - _t0) * 1000, True)
            return choices[0].get("message", {}) if choices else {}
    except Exception as e:  # noqa: BLE001 — LLM не должен ронять запрос
        logger.warning("LLM error: %s", e)
        try:
            from . import metrics as _mt_e
            # ТЗ п.14: таймаут (say-фолбэк по дизайну) ≠ ошибка провайдера;
            # ВРЕМЯ до таймаута тоже фиксируем (иначе p95 не видит таймаутов)
            _mt_e.observe_llm((time.time() - _t0) * 1000, False)
            if isinstance(e, (httpx.TimeoutException, httpx.ReadTimeout)):
                _mt_e.event("llm_timeout")
            else:
                _mt_e.event("llm_error")
        except Exception:  # noqa: BLE001
            pass
        return {}


async def run_web_search(query: str, max_results: int = 4) -> str | None:
    """Универсальный веб-поиск (DuckDuckGo, без ключей) для инструмента web_search.

    Возвращает компактный текст (нумерованные сниппеты) → кладётся LLM в диалог
    как данные, и она формулирует живой ответ. None — пусто/ошибка (LLM ответит
    из головы). Блокирующий DDGS → в to_thread, чтобы не морозить event loop.
    """
    q = (query or "").strip()
    if not q:
        return None

    def _time_fallback() -> str:
        """Точное локальное время — НОВОСИБИРСК (по городу пользователя, без перевода)."""
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Novosibirsk"))
        return (f"Текущее время в Новосибирске: {now:%H:%M}, "
                f"дата {now:%d.%m.%Y} (местное, UTC+7).")

    # «который час»/«сколько времени» → сразу живое время (без капризов веба):
    # такие запросы короткие и детерминированные, веб-сниппеты часто без числа.
    ql = q.lower()
    if ("час" in ql or "время" in ql) and len(ql.split()) <= 4:
        logger.info("web_search time-fallback for %r", q)
        return _time_fallback()

    # Погода ПЕРВЫМ делом через проверенный живой фид (_weather_for) — точнее и
    # стабильнее веб-поиска (DDG на «погода города» спотыкается rate-limit'ом).
    # _weather_for живёт в main → ленивый импорт (иначе цикл main↔llm).
    if "погод" in ql or "температур" in ql or "weather" in ql:
        try:
            from .main import _weather_for
            w = await _weather_for(q)
        except Exception:  # noqa: BLE001
            w = None
        if w:
            logger.info("web_search weather-fid for %r -> %r", q, w)
            return f"Текущая погода: {w}."

    from ddgs import DDGS  # локальный импорт: пакет может отсутствовать

    def _run() -> list:
        with DDGS() as d:
            return d.text(q, max_results=max_results)

    results: list = []
    # ретраи: DDG часто бьёт rate-limit («No results found.») → повтор с растущей
    # паузой (иногда 3-4 попытки до первого результата — проверено на Сочи/Омске).
    for attempt in range(5):
        try:
            results = await asyncio.to_thread(_run)
        except Exception as e:  # noqa: BLE001 — поиск не должен ронять /smart
            logger.warning("web_search failed for %r (attempt %d): %s", q, attempt, e)
            results = []
        if results:
            break
        if attempt < 4:
            await asyncio.sleep(1.5 * (attempt + 1))

    snippets = []
    for r in results or []:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        if title or body:
            snippets.append(f"{title}: {body}".strip())
    if not snippets:
        # фолбэк времени и после неудачных поисков (частичный матч по смыслу)
        if "час" in ql or "время" in ql or "time" in ql:
            return _time_fallback()
        return None
    # обрезаем, чтобы не раздувать диалог
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(snippets))[:2000]


async def run_music_search(query: str, limit: int = 8) -> str | None:
    """ПРЯМОЙ поиск музыки в каталоге (Яндекс, фолбэк ВК) БЕЗ наших фильтров.

    Для LLM-инструмента music_search: когда просят СПИСОК/ПОДБОРКУ треков
    («5 треков из сериала Пацаны») → отдаём реальные треки LLM напрямую,
    минуя /search (его anchor/media-title фильтры режут до 1 трека).
    Возвращает компактный текст «N. Артист - Название» → LLM перечисляет
    и вызывает play_music. None — пусто/ошибка (LLM ответит из головы).
    """
    q = (query or "").strip()
    if not q:
        return None
    limit = max(3, min(int(limit or 8), 15))
    tracks: list[dict] = []
    try:
        # Яндекс основной (ОСНОВНОЙ источник) — напрямую, без фильтров /search.
        from . import ym_client as _ym
        tracks = await _ym.search(q, limit=limit)
    except Exception as e:  # noqa: BLE001 — музыкальный поиск не должен ронять /smart
        logger.warning("music_search yandex failed for %r: %s", q, e)
        tracks = []
    if not tracks:
        try:
            from . import vk_client as _vk
            tracks = await _vk.search(q, limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.warning("music_search vk failed for %r: %s", q, e)
            tracks = []
    if not tracks:
        return None
    lines = []
    for i, t in enumerate(tracks[:limit]):
        artist = (t.get("artist") or "").strip()
        title = (t.get("title") or "").strip()
        if title:
            lines.append(f"{i + 1}. {artist} - {title}".strip(" -"))
    if not lines:
        return None
    logger.info("music_search %r -> %d tracks (direct, no filters)", q, len(lines))
    return "\n".join(lines)[:1500]


async def smart_response(
    user_text: str,
    *,
    context: str = "",
    tools: bool = False,
    timeout: float | None = None,
    role: str = "",        # ТЗ п.11: "" = основной Mimo (Advanced), "fast" = быстрый
    remember: bool = True, # ТЗ п.4: say в /search — ГОЛОСОВАЯ фраза, а НЕ ход
                           # диалога → False: не мусорит память (иначе «рэп»
                           # из запроса тянет в ответы на другие темы)
) -> dict:
    """Живая реплика ассистента («как Алиса»).

    Возвращает {"text": str|None, "tool_calls": list|None}.
    context — короткий контекст (что играет, настроение), добавляется в диалог.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Контекст: {context}"})
    # ИНТЕРАКТИВНЫЙ РЕЖИМ («играйем в города»): вставляем ИСТОРИЮ последних
    # реплик, чтобы LLM помнила диалог. Без неё каждый ответ — stateless-запрос
    # (лог: «Москва» → «погода в Москве?», игра потеряна).
    messages.extend(_smart_history)
    messages.append({"role": "user", "content": user_text})
    msg = await _chat(messages, tools=tools, timeout=timeout, role=role)
    # Агентный цикл: LLM вызвал web_search/music_search → выполняем → кидаем
    # данные в диалог → просим финальный ответ. Лимит 4 (было 3): после
    # music_search LLM должна успеть вызвать play_music (3 не хватало →
    # action=None, ничего не включалось).
    music_searched = False   # был ли прямой поиск музыки (для force-search фолбэка)
    last_ms_query = ""       # последний УСПЕШНЫЙ music_search-запрос (точный, для play)
    for _ in range(4):
        if not msg:
            break
        tcalls = msg.get("tool_calls") or []
        if not tcalls:
            # нет инструмента: если content пуст (LLM думала в reasoning и не
            # ответила — бывает на статических фактах) → дать 2-й шанс ответить
            # прямо, без размышлений/инструментов. Иначе выходим.
            if not (msg.get("content") or "").strip():
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": "Ответь прямо, по существу, одной короткой фразой. "
                               "Не размышляй и не вызывай инструменты — просто ответ.",
                })
                logger.info("smart empty-content retry for %r", user_text)
                msg = await _chat(messages, tools=False, role=role)
                continue
            break
        # все tool_calls в этом раунде → исполняем и отвечаем модели
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        for tc in tcalls:
            fn = (tc.get("function") or {})
            name = fn.get("name")
            tid = tc.get("id")
            argq = ""
            arglimit = 0
            try:
                _args = json.loads(fn.get("arguments") or "{}")
                argq = _args.get("query", "")
                arglimit = int(_args.get("limit") or 0)
            except Exception:  # noqa: BLE001
                argq = ""
            if name == "web_search":
                data = await run_web_search(argq)
                result = data or "Ничего не нашла — ответь из головы или уточни."
                logger.info("smart web_search %r -> %d chars", argq, len(result))
            elif name == "music_search":
                # ПРЯМОЙ поиск треков (без фильтров /search) → треки в диалог,
                # LLM перечислит их и вызовет play_music, чтобы включить.
                data = await run_music_search(argq, arglimit or 8)
                result = data or "Треки не найдены — уточни запрос или ответь из головы."
                music_searched = bool(data)   # флаг: треки реально найдены
                if data:
                    last_ms_query = argq   # запоминаем ТОЧНЫЙ запрос для play
                logger.info("smart music_search %r -> %s", argq,
                            "ok" if data else "empty")
            else:
                # музыкальный инструмент (play_music/play_radio/...) — отдаём наверх
                return {"text": (msg.get("content") or "").strip() or None,
                        "tool_calls": tcalls,
                        "music_searched": music_searched,
                        "last_ms_query": last_ms_query}
            messages.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": result,
            })
        msg = await _chat(messages, tools=tools, role=role)
    if not msg:
        return {"text": None, "tool_calls": None, "music_searched": music_searched,
                "last_ms_query": last_ms_query}
    text = (msg.get("content") or "").strip()
    # дописываем реплику в ИСТОРИЮ (для след. хода LLM помнит диалог)
    _remember_smart(user_text, text)
    return {
        "text": text or None,
        "tool_calls": msg.get("tool_calls") or None,
        "music_searched": music_searched,
        "last_ms_query": last_ms_query,
    }


# ---- Диалоговая память /smart: интерактивные режимы (игра в города) --------
# Храним последние N реплик (user/assistant) → LLM видит контекст разговора.
# Без этого «Москва» (ответ на «загадал город») воспринимается как новый
# stateless-запрос (лог 18:19:25 → «погода в Москве?» — игра потеряна).
_smart_history: list[dict] = []
_SMART_HISTORY_MAX = 12   # 6 пар user/assistant — хватает на игру, не топит токены


def _remember_smart(user_text: str, assistant_text: str) -> None:
    """Запомнить ход диалога (user → assistant), обрезав историю."""
    _smart_history.append({"role": "user", "content": (user_text or "").strip()})
    if assistant_text:
        _smart_history.append({"role": "assistant", "content": assistant_text.strip()})
    # обрезка: держим только последние N реплик, по чётности пар (user/assistant)
    if len(_smart_history) > _SMART_HISTORY_MAX:
        del _smart_history[: len(_smart_history) - _SMART_HISTORY_MAX]


def reset_smart_history() -> None:
    """Сброс диалога (вызывать при новой голосовой сессии/перезапуске)."""
    _smart_history.clear()


async def mood_mix_query(user_text: str, current_track: str = "") -> str:
    """Превратить запрос/настроение в поисковую строку подборки.

    Как Алиса: из текущего трека и просьбы вывести СВЯЗНЫЙ набор.
    Примеры:
      «по настроению после этого» + current="Асти - Лампочки"
        -> «мелодичный рэп 2020-х»   (LLM подберёт жанр/эпоху)
      «что-нибудь весёлое»          -> «весёлая танцевальная музыка»
    ФИКС «шлак/одни и те же»: если в просьбе есть ИМЯ АРТИСТА («асти модный
    поп»), оно ОБЯЗАТЕЛЬНО включается в запрос — иначе LLM выбрасывала артиста
    (журнал: 'асти модной поп' → 'русская модная поп-музыка' → поп-мусор без
    Асти). Артист детерминированно дописывается в конец, если LLM его потеряла.
    При недоступности LLM — возвращает очищенный исходный запрос.
    """
    base = (user_text or "").strip()
    if not _llm_ready():
        return base
    ctx = f"Сейчас играет: {current_track}" if current_track else "Трек ещё не выбран."
    # ---- добыть ИМЯ АРТИСТА из просьбы (детекция + инжекция) ----------------
    artist = _extract_artist(base) or _extract_artist(current_track)
    artist_hint = (
        f"ОБЯЗАТЕЛЬНО включи в запрос имя артиста «{artist}» — пользователь просит именно его. "
        if artist else ""
    )
    prompt = (
        "Сформируй КОРОТКИЙ поисковый запрос для музыкальной подборки "
        "(3-6 слов, на русском, без кавычек и пояснений). "
        "Это пойдёт в строку поиска, чтобы найти подходящие треки.\n"
        f"{ctx}\n"
        f"Просьба: {base}\n"
        f"{artist_hint}"
        "Ответь ТОЛЬКО запросом."
    )
    msg = await _chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        max_tokens=60,
        role="fast",   # ТЗ п.11: настроение → FAST-провайдер (если настроен)
    )
    q = (msg.get("content") or "").strip().strip('"«»')
    # санитайзер: ОДНА строка без многоточия/лишних хвостов. LLM часто отвечает
    # повтором/переносом («колыбельные...\n колыбельные...») — это ломает поиск.
    q = q.replace("\n", " ").replace("\r", " ")
    q = re.sub(r"\.{2,}|…", "", q)          # многоточие мешает поиску
    q = re.sub(r"\s+", " ", q).strip().strip(".,;: ")
    # если фраза повторилась («X... X») — взять первую половину
    half = len(q) // 2
    if half >= 6 and q[:half].strip() == q[half:].strip():
        q = q[:half].strip()
    if not q or len(q) > 80:
        q = base
    # ---- ФИКС: артист потерян → дописать в конец (иначе мусор по жанру) -----
    if artist and not _artist_in_query(q, artist):
        q = f"{q} {artist}".strip()
        if len(q) > 80:
            q = artist
    logger.info("mood_mix_query %r -> %r (artist=%r)", base, q, artist)
    return q


# ---- детекция/сохранение артиста в mood-запросе (ФИКС «шлак без артиста») ---
_ARTIST_HINT_RE = re.compile(r"\b(?:[A-Z][A-Za-z]{2,}|[А-ЯЁ][а-яё]{2,})\b")
# Слова-жанры/настроения/служебные — НЕ артист (их LLM и так добавит).
_NON_ARTIST = {
    "модный", "модная", "модные", "модно", "русская", "русский", "русские",
    "зарубежная", "зарубежный", "зарубежные", "танцевальная", "танцевальный",
    "весёлая", "веселая", "весёлый", "веселый", "спокойная", "спокойный",
    "грустная", "грустный", "бодрая", "бодрый", "электронная", "электронный",
    "поп", "рэп", "рок", "джаз", "хаус", "техно", "ремиксы", "ремикс", "хиты",
    "подборка", "подборку", "музыка", "музыку", "песни", "песню", "треки", "трек",
    "современная", "современный", "новая", "новый", "лучшая", "лучший",
    # служебные слова просьбы
    "включи", "включить", "включим", "поставь", "поиграй", "запусти", "найди",
    "дай", "нужна", "хочу", "можно", "пожалуйста", "давай", "сделай", "собери",
    # ситуации/сезоны
    "дороги", "дорогу", "для", "сна", "сон", "зала", "тренировки", "подъём",
    "утро", "вечер", "ночь", "ночью", "зима", "лето", "осень", "весна",
    "что", "чего", "чё", "что-нибудь", "что-то", "чуть", "немного", "послушать",
    "нибудь", "что-нибудь", "хорошая", "хороший", "крутая", "крутой", "классная", "классный",
}
# Качества-прилагательные с типичными окончаниями — НЕ артист («модная»,
# «танцевальная», «зарубежная» ...). Детектируем подстрокой корня.
_NON_ARTIST_SUFFIXES = (
    "ая", "ой", "ый", "ий", "яя", "ое", "ее", "ые", "ие", "ой",
)


def _extract_artist(text: str) -> str | None:
    """Достать имя артиста из запроса («асти модной поп» → «Асти»).

    Эвристика: НЕ ЖАНР/служебное/качество. БЕРЁМ ПОСЛЕДНЕГО кандидата-имя
    (ЛОВИЛ баг: «by индия, xcho, mot» → первое «индия» = мимо Мота; «noize mc»
    → «noize» без «mc»). В многоартистной строке «A, B, C» реальный главный
    артист — последний (Xcho, MOT). Жанры/качества/служебные отсеяны по
    словарю + подстроке. ВАЖНО: НЕ требуем заглавную — голос пишет «асти».
    """
    candidates: list[str] = []
    for w in re.split(r"[^0-9A-Za-zА-ЯЁа-яё]+", (text or "").strip()):
        if len(w) < 3:
            continue
        lw = w.lower().replace("ё", "е")
        if lw in _NON_ARTIST:
            continue
        # пропускаем очевидные жанры/ситуации/числа подстрокой
        if any(g in lw for g in ("поп", "рэп", "рок", "джаз", "музык", "песн", "хит",
                                 "ремикс", "подбор", "танц", "спок", "весел",
                                 "весёл", "груст", "бодр", "электр", "зарубеж", "русск",
                                 "современ", "для", "годзилл", "доро", "тренир",
                                 "включи", "постав", "поигра", "запусти", "найди",
                                 "вкус")):
            continue
        # качества-прилагательные («модная», «танцевальная», «зарубежная»)
        if len(lw) >= 5 and lw.endswith(_NON_ARTIST_SUFFIXES):
            continue
        if lw.isdigit():
            continue
        candidates.append(w)
    if not candidates:
        return None
    return candidates[-1]   # последнее имя («xcho, MOT» → MOT; «noize mc» → mc+noize)


def _artist_in_query(q: str, artist: str) -> bool:
    """Есть ли артист (без учёта регистра/падежа) в поисковом запросе."""
    a = artist.lower().replace("ё", "е")
    ql = (q or "").lower().replace("ё", "е")
    if a in ql:
        return True
    # грубое склонение: «асти» ⊂ «астий/астию» — берём основу ≥4 символов
    stem = a[: max(4, len(a) - 2)]
    return len(stem) >= 4 and stem in ql


# ──────────────────────────────────────────────────────────────── TTS ──────
_tts_cache: dict[str, tuple[float, bytes]] = {}  # text|voice|speed -> (ts, mp3)
_TTS_CACHE_TTL = 3600.0  # 1 час
_TTS_MEM_LIMIT = 64      # держим не больше 64 фраз в памяти


def _tts_key(text: str) -> str:
    return hashlib.sha1(
        f"{text}|{config.TTS_VOICE}|{config.TTS_SPEED}".encode()
    ).hexdigest()


async def tts_mp3(text: str) -> bytes | None:
    """Озвучить фразу → mp3-байты (edge-tts, русский голос). None при ошибке.

    Кэш в памяти (частые фразы не генерируются заново) + деградация без ключа.
    """
    text = (text or "").strip()
    if not text or not config.TTS_ENABLED:
        return None
    key = _tts_key(text)
    hit = _tts_cache.get(key)
    if hit and (time.time() - hit[0]) < _TTS_CACHE_TTL:
        return hit[1]
    try:
        import edge_tts  # ставится в venv (requirements)

        communicate = edge_tts.Communicate(
            text, voice=config.TTS_VOICE, rate="+0%"
        )
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        if not buf:
            logger.warning("TTS empty for %r", text[:40])
            return None
        mp3 = bytes(buf)
        if len(_tts_cache) >= _TTS_MEM_LIMIT:
            # выкинуть самое старое
            oldest = min(_tts_cache, key=lambda k: _tts_cache[k][0])
            _tts_cache.pop(oldest, None)
        _tts_cache[key] = (time.time(), mp3)
        return mp3
    except Exception as e:  # noqa: BLE001 — озвучка не должна ломать ответ
        logger.warning("TTS error: %s", e)
        return None
