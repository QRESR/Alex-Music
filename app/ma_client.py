"""Music Assistant client — search + queue play + stream proxy.

Заменяет YouTube-путь (IP-блок) на Music Assistant (локальный сервер,
порт 18095 API / 18097 stream). Контракт с приложением НЕ меняется:
APK продолжает ходить в /search и /stream/{id}.

Особенности MA:
- stream URL содержит session_id (8 симв.), который MA НЕ отдаёт через
  открытый API — он появляется в логе контейнера при команде PLAY_MEDIA.
  Поэтому после play читаем свежий лог docker и кэшируем URL.
- Поток: FLAC (или mp3) на 127.0.0.1:18097, без пароля, с session в URL.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import subprocess
import time

import httpx

logger = logging.getLogger("ma_client")

MA_API = os.environ.get("MA_API", "http://127.0.0.1:18095")
MA_STREAM = os.environ.get("MA_STREAM", "http://127.0.0.1:18097")
MA_QUEUE = os.environ.get("MA_QUEUE", "demo_0")
MA_CONTAINER = os.environ.get("MA_CONTAINER", "music-assistant")
MA_USER = os.environ.get("MA_USER", "admin")
MA_PASS_FILE = os.environ.get("MA_PASS_FILE", "/opt/music-assistant/.admin_password")

MA_ID_PREFIX = "ma_"  # маркер: id трека из MA (YouTube id всегда ровно 11 симв.)

_token: str = ""
_token_ts: float = 0.0
_stream_cache: dict[str, tuple[str, float]] = {}  # id -> (stream_url, ts)
_lock = asyncio.Lock()


# ---------------------------------------------------------------- кодирование
def encode_id(uri: str) -> str:
    """URI провайдера -> безопасный id для пути /stream/{id}."""
    b64 = base64.urlsafe_b64encode(uri.encode()).decode().rstrip("=")
    return MA_ID_PREFIX + b64


def decode_id(track_id: str) -> str | None:
    """Обратно в URI; None если это не наш id."""
    if not track_id.startswith(MA_ID_PREFIX):
        return None
    b64 = track_id[len(MA_ID_PREFIX):]
    b64 += "=" * (-len(b64) % 4)
    try:
        return base64.urlsafe_b64decode(b64).decode()
    except Exception:
        return None


def is_ma_id(track_id: str) -> bool:
    return track_id.startswith(MA_ID_PREFIX) or len(track_id) != 11


# ---------------------------------------------------------------- API MA
async def _login(client: httpx.AsyncClient) -> str:
    pw_file = MA_PASS_FILE
    password = ""
    if os.path.exists(pw_file):
        with open(pw_file) as f:
            password = f.read().strip()
    resp = await client.post(
        f"{MA_API}/auth/login",
        json={
            "provider_id": "builtin",
            "credentials": {"username": MA_USER, "password": password},
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"MA login failed: {data}")
    return data["token"]


async def _ensure_token(client: httpx.AsyncClient) -> str:
    global _token, _token_ts
    if _token and time.time() - _token_ts < 3600 * 6:
        return _token
    async with _lock:
        if _token and time.time() - _token_ts < 3600 * 6:
            return _token
        _token = await _login(client)
        _token_ts = time.time()
        return _token


async def api(command: str, args: dict | None = None, timeout: float = 30.0) -> object:
    """Вызов JSON-RPC команды MA."""
    async with httpx.AsyncClient() as client:
        token = await _ensure_token(client)
        payload: dict = {"command": command}
        if args:
            payload["args"] = args
        resp = await client.post(
            f"{MA_API}/api",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        if resp.status_code == 401:
            # токен протух — логинимся заново
            global _token, _token_ts
            _token = ""
            _token_ts = 0.0
            token = await _ensure_token(client)
            resp = await client.post(
                f"{MA_API}/api",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"MA API {command}: HTTP {resp.status_code} {resp.text[:200]}")
        return resp.json()


# ---------------------------------------------------------------- поиск
def _item_to_track(it: dict) -> dict:
    """MA MediaItem -> наш TrackMeta-like dict."""
    uri = it.get("uri") or ""
    mappings = it.get("provider_mappings") or []
    thumb = ""
    img = it.get("image") or it.get("metadata", {}).get("image") or {}
    if isinstance(img, dict):
        thumb = img.get("path") or img.get("url") or ""
    if not thumb and mappings:
        thumb = mappings[0].get("url") or ""
    return {
        "id": encode_id(uri),
        "title": it.get("name") or "",
        "artist": _artist_str(it),
        "duration": int(it.get("duration") or 0),
        "thumbnail": thumb,
        "channel": (it.get("provider") or ""),
        "url": uri,
    }


def _artist_str(it: dict) -> str:
    artists = it.get("artists") or []
    if artists:
        return ", ".join(a.get("name", "") for a in artists if a.get("name"))
    return it.get("artist") or it.get("album_artist") or ""


# Мусорные слова, которые ломают точный поиск MA (internet_archive ищет по
# подстроке — лишнее слово «трек»/«включи» даёт 0 результатов). Чистим query.
_QUERY_NOISE = re.compile(
    r"\b(?:трек\w*|песн\w*|композиц\w*|музык\w*|штук\w*|включи\w*|запусти\w*|"
    r"поставь\w*|дай|мне|пожалуйста|ну|поиграй\w*|сыграй\w*|послушай\w*|"
    r"подборк\w*|плейлист\w*|микс|радио|список|сборник|"
    # ловушка «исполнитель эминем» → мусор «ISPOLNITEL» в артистах (патч 2)
    r"исполнител\w*|артист\w*|поет|поёт|играет|играющий)\b",
    re.IGNORECASE,
)


def _clean_query(query: str) -> str:
    """Убрать мусорные слова из запроса к MA («трек баста» → «баста»)."""
    q = _QUERY_NOISE.sub(" ", query or "")
    return re.sub(r"\s+", " ", q).strip()


# Транслит для поиска в MA. Нужны ОБЕ стороны («кривой» поиск):
#   латиница → кириллица: «basta» → «баста» (русский каталог MA — кириллица);
#   кириллица → латиница: «Баста» → «basta» (иногда латинский алиас находит лучше).
# Таблица та же, что в ym_client (единый источник — не плодить расхождений).
try:
    from .ym_client import (
        translit_to_latin as _translit_to_latin,
        translit_to_cyrillic as _translit_to_cyrillic,
    )
except Exception:  # pragma: no cover — если ym_client недоступен, грубый фолбэк
    _translit_to_latin = None
    _translit_to_cyrillic = None


def _translit(query: str) -> str:
    """Транслит в ту сторону, в которой запрос НЕ написан (единая таблица)."""
    q = (query or "").strip()
    if not q:
        return ""
    has_cyr = bool(re.search(r"[а-яё]", q, re.I))
    has_lat = bool(re.search(r"[a-z]", q, re.I))
    if has_cyr and not has_lat and _translit_to_latin:
        return _translit_to_latin(q)
    if has_lat and not has_cyr and _translit_to_cyrillic:
        return _translit_to_cyrillic(q)
    return q


async def search(query: str, limit: int = 30) -> list[dict]:
    """Поиск по MA-провайдерам с умной обработкой запроса.

    MA (internet_archive) чувствителен к лишним словам — «трек ноггано»
    даёт 0, а «ноггано» даёт результат. Стратегия:
      1) очистить от мусорных слов (трек/включи/подборка...) → точный запрос;
      2) если 0 — попробовать КАЖДОЕ слово по отдельности и взять первое,
         что вернуло результат (ключевое слово обычно — артист/название);
      3) латиница → кириллица (транслит), если латинский вариант пуст;
      4) финальный fallback — оригинал запроса как есть.
    Никогда не бросает: пусто = [], не исключение (YouTube-fallback
    отключён — он ведёт в IP-блок).
    """
    attempts: list[str] = []
    cleaned = _clean_query(query)
    # ТРАНСЛИТ В ОБЕ СТОРОНЫ (единая таблица из ym_client):
    #   латиница → кириллица: «basta» → «баста» (русский каталог MA — кириллица,
    #   иначе латиница находит мусор); кириллица → латиница: «Баста» → «basta»
    #   (иногда латинский алиас находит лучше). Пробуем транслит ПЕРВЫМ, потом
    #   оригинал — чтобы лучший вариант шёл в топ.
    if cleaned:
        tr = _translit(cleaned)
        if tr and tr.lower() != cleaned.lower():
            attempts.append(tr)
    if cleaned:
        attempts.append(cleaned)
    # по слову — если многословный запрос не нашёл (тоже с транслитом по слову)
    words = [w for w in cleaned.split() if len(w) >= 3]
    if len(words) > 1:
        for w in words:
            trw = _translit(w)
            if trw and trw.lower() != w.lower():
                attempts.append(trw)
            if w not in attempts:
                attempts.append(w)
    if query and query.strip() not in attempts:
        attempts.append(query.strip())

    for attempt in attempts:
        try:
            data = await api("music/search", {"search_query": attempt})
        except Exception as e:
            logger.warning("MA search error for %r: %s", attempt, e)
            continue
        out: list[dict] = []
        for key in ("tracks", "albums", "artists", "radio"):
            for it in (data.get(key) or []):
                if not it.get("uri"):
                    continue
                out.append(_item_to_track(it))
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        if out:
            if attempt != (query or "").strip():
                logger.info("MA search %r → %r: %d hits", query, attempt, len(out))
            return out
    logger.info("MA search %r: 0 hits after %d attempts", query, len(attempts))
    return []


# ---------------------------------------------------------------- воспроизведение
_URL_RE = re.compile(r"http://[^\s\"]*?:8097/[^\s\"]+")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")  # ANSI-цвета docker logs
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)")  # docker --timestamps: 2026-10-29T13:33:13.733994267Z


def _last_stream_url(since_epoch: float | None = None) -> str | None:
    """Самый свежий stream URL из лога контейнера.

    `since_epoch` (unix-время) — если задан, возвращаем только URL из строк,
    которые MA залогировала ПОЗЖЕ этого момента (ловим гонку: play вернулся,
    а новый PLAY_MEDIA ещё не записан — не берём URL предыдущего play).
    ANSI-коды вычищаем (ловушка httpx InvalidURL).
    """
    try:
        proc = subprocess.run(
            ["docker", "logs", "--since", "120s", "--timestamps", MA_CONTAINER],
            capture_output=True, text=True, timeout=10,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        logger.warning("docker logs failed: %s", e)
        return None
    text = _ANSI_RE.sub("", text)

    import datetime as _dt

    best_url: str | None = None
    best_ts = 0.0
    for ln in text.splitlines():
        if "8097" not in ln or "http://" not in ln:
            continue
        m_ts = _TS_RE.match(ln)
        if not m_ts:
            continue
        # docker --timestamps: ISO-8601 с наносекундами и Z (UTC).
        # Если Z/TZ нет — считаем UTC (naive timestamp иначе даёт локальное
        # время = сдвиг на час при CET и ломает фильтр since_epoch).
        raw_ts = m_ts.group(1)
        if not re.search(r"(Z|[+-]\d{2}:?\d{2})$", raw_ts):
            raw_ts += "Z"
        try:
            ts = _dt.datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if since_epoch is not None and ts < since_epoch:
            continue
        if ts >= best_ts:
            m_url = _URL_RE.search(ln)
            if m_url:
                best_ts = ts
                best_url = m_url.group(0)
    if not best_url:
        return None
    return "".join(c for c in best_url if 32 <= ord(c) < 127)


async def play(uri: str) -> str:
    """Ставит media в очередь MA, возвращает stream URL.

    ВРЕМЕННАЯ механика (проверено замером 2026-10-29):
    - `player_queues/play_media` API сам занимает ~4.3с (MA синхронно
      запускает поток).
    - `PLAY_MEDIA command` (с URL) появляется в логе контейнера через
      ~1с после возврата API.
    Итого от вызова до URL в логе — ~5-6с. Раньше ждал только 2.5с и
    повторял play (ещё +4.3с) → 14-19с на трек = «зависание при смене».

    Теперь: ОДИН play, ждём URL до 7с (опрос каждые 0.3с), без повторного
    play (повтор лишь порождал новую session и ломал предыдущий URL).

    ВАЖНО (фикс «демо песни», 2026-10-29):
    - `option="replace"` — НИЖНИЙ РЕГИСТР (MA-enum парсит только его;
      КАПС "REPLACE" не распознаётся → 0 items → 502 «URL not found»).
      Без option дефолт = ADD → накопилось 197 мусора (Queen/Basta/Кино
      вперемешку) → flow перескакивал на случайный «демо» трек.
    - НЕ делаем clear перед play: бэкенд не знает, новый поиск это или
      автопереход APK (next) — clear обрывал бы каждый переход на 15с.
    """
    t_before = time.time()
    await api(
        "player_queues/play_media",
        {"queue_id": MA_QUEUE, "media": uri, "option": "replace"},
    )
    deadline = time.time() + 7.0
    while time.time() < deadline:
        await asyncio.sleep(0.3)
        url = _last_stream_url(since_epoch=t_before)
        if url:
            return url
    raise RuntimeError("MA: stream URL not found in logs after play (7s)")


async def ensure_stream(track_id: str) -> str:
    """Возвращает рабочий stream URL для id (из кэша или после нового play).

    ВАЖНО (фикс цикла 404, 2026-10-29): НЕ делаем _head_ok — HEAD сразу
    после play отдаёт False (MA логирует URL ДО запуска потока, upstream ещё
    не готов) → pop cache → play заново (15с) → бесконечный цикл «один трек
    и обрыв». Доверяем кэшу по времени; реальную протухшую сессию ловит
    ретрай в _ma_stream_response (один pop+play при 404 от ExoPlayer).
    """
    uri = decode_id(track_id)
    if not uri:
        raise RuntimeError(f"not a MA track id: {track_id}")

    cached = _stream_cache.get(track_id)
    if cached and time.time() - cached[1] < 600:
        return cached[0]

    url = await play(uri)
    _stream_cache[track_id] = (url, time.time())
    return url


async def _head_ok(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.head(url, timeout=2.0)
            return r.status_code in (200, 206)
    except Exception:
        return False
