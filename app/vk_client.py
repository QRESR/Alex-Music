"""VK Music web-клиент — 3-й источник треков.

Два бэкенда:
1. **api.vk.ru** (основной, 2026-10-30+): новый web API VK, методы
   `audio.search`, `audio.getById`, `audio.get`. Требует web-токен
   (refresh через login.vk.ru/?act=web_token с куками p+remixsid).
   Быстрый, чистый JSON, прямые ссылки в ответе.

2. **al_audio.php** (legacy fallback): старый веб-бэкенд. Медленнее
   (парсинг HTML), но работает без токена. Используется если api.vk.ru
   вернул ошибку/блок или web-токен не получить.

Сессия: куки авторизованного аккаунта (p, remixsid, ...) в private/vk-cookies.txt.
Куки выгружены из WebView стенда; при логауте — обновить.

Формат id трека: vk_{owner_id}_{track_id} (стабильный для обоих бэкендов).
"""
from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import re
import time
from pathlib import Path

import httpx

logger = logging.getLogger("vk_client")

VK_ID_PREFIX = "vk_"

_MOBILE_UA = (
    "Mozilla/5.0 (Linux; Android 13; Redmi Note 12 Build/SQ3A; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/120.0 "
    "Mobile Safari/537.36"
)
_WEB_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0"
_AL_AUDIO = "https://vk.ru/al_audio.php"

# Новый web API (iplly/RaisaAI)
_VK_API = "https://api.vk.ru/method"
_VK_API_V = "5.285"
_VK_CLIENT_ID = "6287487"   # vk.com web app
_WEB_TOKEN_URL = "https://login.vk.ru/?act=web_token"

from . import config as _config  # noqa: E402

_COOKIES_FILE = Path(getattr(_config, "VK_COOKIES_FILE",
                             str(Path(__file__).resolve().parent.parent / "private" / "vk-cookies.txt")))

_DATA_AUDIO_RE = re.compile(r'data-audio="((?:[^"\\]|\\.)*)"')

# ————— state —————
_cookies_str: str = ""
_cookies_uid: str = ""
_cookies_loaded_at: float = 0.0

# web-токен (api.vk.ru): access_token, expires_at (epoch sec), user_id
_web_token: str = ""
_web_token_expires: float = 0.0
_web_token_uid: int = 0

# id -> (stream_url, ts)
_stream_cache: dict[str, tuple[str, float]] = {}
_STREAM_TTL = 20 * 3600.0

# legacy: id -> (action_hash, url_hash) для al_audio fallback
_hash_cache: dict[str, tuple[str, str]] = {}
_hash_file = Path(__file__).resolve().parent.parent / "data" / "vk_hashes.json"
_HASH_FILE_TTL = 7 * 24 * 3600.0
_hash_file_loaded = False

# Отдельный лок для web-токена: _resolve_api (стр.~590) держит _lock на весь
# resolve → _api_vk → _get_token. С ОДНИМ общим _lock это DEADLOCK, когда
# токен протух (тест 2026-10-31: audio.getById отвечает за 0.6с, а
# resolve_stream висит ровно до таймаута 60с — рекурсивный захват
# asyncio.Lock). Токен-лок никогда не держится внутри _lock → цепочки нет.
_token_lock = asyncio.Lock()

_lock = asyncio.Lock()
_http: httpx.AsyncClient | None = None


# ——————————————————————————————————————————————
# Cookies / HTTP
# ——————————————————————————————————————————————

def _ensure_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=45.0, follow_redirects=False)
    return _http


def _load_cookies(force: bool = False) -> None:
    global _cookies_str, _cookies_uid, _cookies_loaded_at
    if _cookies_str and not force and (time.time() - _cookies_loaded_at) < 600:
        return
    try:
        raw = _COOKIES_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("VK cookies file missing: %s", _COOKIES_FILE)
        return
    parts: list[str] = []
    uid = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if "uid=" in line:
                uid = line.split("uid=", 1)[1].strip()
            continue
        if "=" not in line:
            continue
        parts.append(line)
        if line.startswith("sui="):
            val = line[4:].split(";", 1)[0]
            try:
                from urllib.parse import unquote
                val = unquote(val)
                uid = val.split(",")[0].strip() or uid
            except Exception:  # noqa: BLE001
                pass
    _cookies_str = "; ".join(parts)
    _cookies_uid = uid
    _cookies_loaded_at = time.time()
    if not uid:
        logger.warning("VK uid not found in cookies file (owner_id will be 0)")


def cookies() -> str:
    _load_cookies()
    return _cookies_str


def uid() -> str:
    _load_cookies()
    return _cookies_uid


def is_vk_id(track_id: str) -> bool:
    return track_id.startswith(VK_ID_PREFIX)


# ——————————————————————————————————————————————
# Web-token (login.vk.ru/?act=web_token)
# ——————————————————————————————————————————————

def _cookie_value(name: str) -> str:
    """Достаёт значение куки из cookie-string."""
    for part in cookies().split("; "):
        if part.startswith(name + "="):
            return part.split("=", 1)[1]
    return ""


async def _refresh_web_token() -> str:
    """Минтит/обновляет web-токен через login.vk.ru (обычно живёт ~4ч, но
    может отдать до 57 лет — зависит от настроек аккаунта).

    Возвращает access_token или "" при ошибке.
    """
    global _web_token, _web_token_expires, _web_token_uid
    ck_p = _cookie_value("p")
    ck_rsid = _cookie_value("remixsid")
    ck_ht = _cookie_value("httoken")
    if not ck_p or not ck_rsid:
        logger.warning("VK cookies missing p/remixsid — cannot mint web token")
        return ""
    cookie_header = f"p={ck_p}; remixsid={ck_rsid}"
    if ck_ht:
        cookie_header += f"; httoken={ck_ht}"
    http = _ensure_http()
    headers = {
        "User-Agent": _WEB_UA,
        "Accept": "*/*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://vk.ru/",
        "Origin": "https://vk.ru",
        "Cookie": cookie_header,
    }
    try:
        r = await http.post(_WEB_TOKEN_URL, content="version=1&app_id=6287487",
                            headers=headers, timeout=30.0)
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("VK web_token refresh failed: %s", e)
        return ""
    if data.get("type") != "okay":
        logger.warning("VK web_token refused: %s", data.get("error_info") or data)
        return ""
    payload = data.get("data") or {}
    tok = payload.get("access_token") or ""
    if not tok:
        logger.warning("VK web_token empty")
        return ""
    _web_token = tok
    _web_token_expires = float(payload.get("expires") or 0)
    _web_token_uid = int(payload.get("user_id") or 0)
    #htt token обновляется в Set-Cookie — не трогаем, при следующем refresh возьмём заново
    logger.info("VK web_token refreshed: uid=%s expires_at=%s", _web_token_uid, _web_token_expires)
    return tok


async def _get_token() -> str:
    """Возвращает валидный web-токен (refresh при необходимости)."""
    global _web_token, _web_token_expires
    now = time.time()
    # обновляем если нет или скоро истечёт (< 5 мин)
    if _web_token and now < _web_token_expires - 300:
        return _web_token
    async with _token_lock:
        # double-check под локом
        if _web_token and time.time() < _web_token_expires - 300:
            return _web_token
        return await _refresh_web_token()


# ——————————————————————————————————————————————
# Legacy: al_audio helpers (fallback)
# ——————————————————————————————————————————————

def _hash_load() -> None:
    global _hash_file_loaded
    if _hash_file_loaded:
        return
    _hash_file_loaded = True
    try:
        import json as _json
        data = _json.loads(_hash_file.read_text(encoding="utf-8"))
        now = time.time()
        for k, v in data.items():
            if isinstance(v, dict) and len(v.get("h", [])) == 2:
                if now - float(v.get("ts", 0)) < _HASH_FILE_TTL:
                    _hash_cache.setdefault(k, (v["h"][0], v["h"][1]))
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.debug("vk hash file load failed: %s", e)


def _hash_save(track_id: str) -> None:
    try:
        import json as _json
        data: dict = {}
        try:
            data = _json.loads(_hash_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
        h = _hash_cache.get(track_id)
        if not h:
            return
        data[track_id] = {"h": [h[0], h[1]], "ts": time.time()}
        if len(data) > 5000:
            cut = sorted(data.items(), key=lambda kv: kv[1].get("ts", 0))[-4000:]
            data = dict(cut)
        _hash_file.parent.mkdir(parents=True, exist_ok=True)
        _hash_file.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.debug("vk hash file save failed: %s", e)


def _parse_data_audio(body_json: dict) -> list[dict]:
    out: list[dict] = []
    payload = body_json.get("payload")
    if not (isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list)):
        return out
    html_str = payload[1][0] if payload[1] else ""
    if not isinstance(html_str, str) or not html_str:
        return out
    for m in _DATA_AUDIO_RE.finditer(html_str):
        try:
            arr = json.loads(_html.unescape(m.group(1)))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(arr, list) or len(arr) < 6:
            continue
        track_id, owner_id = arr[0], arr[1]
        title, artist, dur = arr[3], arr[4], arr[5]
        if not title:
            continue
        stream_url = ""
        thumb = arr[14] if len(arr) > 14 and isinstance(arr[14], str) else ""
        action_hash, url_hash = "", ""
        if len(arr) > 13 and isinstance(arr[13], str) and arr[13]:
            parts = arr[13].split("/")
            action_hash = parts[2] if len(parts) > 2 else ""
            url_hash = parts[5] if len(parts) > 5 else ""
        for cell in arr[12:14]:
            if isinstance(cell, str) and "audio_streams" in cell:
                try:
                    sd = json.loads(cell)
                    streams = sd.get("audio_streams") or []
                    for pref in ("aac", "hls_range", "dash"):
                        for st in streams:
                            if st.get("type") == pref and st.get("url"):
                                stream_url = st["url"].replace("\\/", "/")
                                break
                        if stream_url:
                            break
                except Exception:  # noqa: BLE001
                    pass
                break
        out.append({
            "id": f"{VK_ID_PREFIX}{owner_id}_{track_id}",
            "owner_id": str(owner_id),
            "track_id": str(track_id),
            "title": str(title),
            "artist": str(artist or ""),
            "duration": int(dur or 0),
            "thumbnail": thumb,
            "stream_url": stream_url,
            "action_hash": action_hash,
            "url_hash": url_hash,
        })
    return out


def _parse_reload_payload(data: dict) -> list[dict]:
    from .vk_url_decoder import decode as _decode

    out: list[dict] = []
    uid_int = int(uid() or 0)

    def walk(node, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(node, list):
            if (len(node) >= 6 and isinstance(node[0], int) and isinstance(node[1], int)
                    and isinstance(node[2], str) and isinstance(node[3], str)
                    and isinstance(node[4], str)):
                raw_url = node[2]
                url = ""
                if "audio_api_unavailable" in raw_url and "?" in raw_url:
                    url = _decode(raw_url, uid_int) or ""
                elif raw_url.startswith("http"):
                    url = raw_url
                hashes = node[12] if len(node) > 12 and isinstance(node[12], str) else ""
                hp = hashes.split("/") if hashes else []
                out.append({
                    "id": f"{VK_ID_PREFIX}{node[1]}_{node[0]}",
                    "owner_id": str(node[1]),
                    "track_id": str(node[0]),
                    "title": node[3],
                    "artist": node[4],
                    "duration": int(node[5] or 0),
                    "thumbnail": "",
                    "stream_url": url,
                    "action_hash": hp[2] if len(hp) > 2 else "",
                    "url_hash": hp[5] if len(hp) > 5 else "",
                })
                return
            for x in node:
                walk(x, depth + 1)
        elif isinstance(node, dict):
            for x in node.values():
                walk(x, depth + 1)

    walk(data.get("payload"))
    return out


async def _al_audio(form: dict) -> dict:
    ck = cookies()
    if not ck:
        raise RuntimeError("VK cookies not loaded")
    http = _ensure_http()
    headers = {
        "User-Agent": _MOBILE_UA,
        "Cookie": ck + "; remixaudio_show_alert_today=0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://vk.ru/audios{uid() or '0'}",
        "Origin": "https://vk.ru",
    }
    r = await http.post(_AL_AUDIO, data=form, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"al_audio HTTP {r.status_code}")
    text = r.content.decode("cp1251", errors="replace")
    if '"location"' in text[:400] and "login" in text[:400]:
        raise RuntimeError("VK session not authorized (login redirect)")
    try:
        return json.loads(text)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"al_audio not JSON: {e}")


async def _search_legacy(query: str, limit: int) -> list[dict]:
    """Fallback: поиск через al_audio.php (старый путь)."""
    q = (query or "").strip()
    if not q:
        return []
    form = {
        "al": "1", "act": "section", "claim": "0", "is_layer": "0",
        "owner_id": uid() or "0", "section": "search", "q": q,
    }
    try:
        data = await _al_audio(form)
    except Exception as e:  # noqa: BLE001
        logger.warning("VK legacy search failed for %r: %s", q, e)
        return []
    rows = _parse_data_audio(data)
    out: list[dict] = []
    now = time.time()
    for row in rows:
        if row["stream_url"]:
            _stream_cache[row["id"]] = (row["stream_url"], now)
        if row.get("action_hash") or row.get("url_hash"):
            _hash_cache[row["id"]] = (row["action_hash"], row["url_hash"])
            _hash_save(row["id"])
        out.append({
            "id": row["id"],
            "title": row["title"],
            "artist": row["artist"],
            "duration": row["duration"],
            "thumbnail": row["thumbnail"],
            "channel": "vkmusic",
            "url": f"https://vk.com/audio{row['owner_id']}_{row['track_id']}",
        })
        if len(out) >= limit:
            break
    logger.info("legacy VK search: %d hits for %r", len(out), q)
    return out


async def _resolve_legacy(track_id: str) -> str:
    """Fallback: resolve через reload_audio."""
    hit = _stream_cache.get(track_id)
    if hit and (time.time() - hit[1]) < _STREAM_TTL:
        return hit[0]
    if not is_vk_id(track_id):
        raise RuntimeError(f"not a vk id: {track_id}")
    parts = track_id[len(VK_ID_PREFIX):].split("_", 1)
    if len(parts) != 2:
        raise RuntimeError(f"bad vk id: {track_id}")
    owner_id, tr_id = parts
    _hash_load()

    async with _lock:
        hit = _stream_cache.get(track_id)
        if hit and (time.time() - hit[1]) < _STREAM_TTL:
            return hit[0]
        ah, uh = _hash_cache.get(track_id, ("", ""))
        if not (ah and uh):
            raise RuntimeError(f"no hashes for {track_id} (не был в поиске)")
        ids = f"{owner_id}_{tr_id}_{ah}_{uh}"
        data = await _al_audio({"al": "1", "act": "reload_audio", "ids": ids})
        rows = _parse_reload_payload(data)
        for row in rows:
            if row["stream_url"] and row["track_id"] == tr_id:
                _stream_cache[track_id] = (row["stream_url"], time.time())
                if row.get("action_hash") and row.get("url_hash"):
                    _hash_cache[track_id] = (row["action_hash"], row["url_hash"])
                return row["stream_url"]
        raise RuntimeError(f"vk reload gave no stream for {track_id}")


# ——————————————————————————————————————————————
# New: api.vk.ru (основной)
# ——————————————————————————————————————————————

async def _api_vk(method: str, params: dict) -> dict | None:
    """Вызов метода api.vk.ru. Возвращает response-данные или None при ошибке."""
    token = await _get_token()
    if not token:
        return None
    http = _ensure_http()
    params = {**params, "v": _VK_API_V, "client_id": _VK_CLIENT_ID, "access_token": token}
    headers = {
        "User-Agent": _WEB_UA,
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://vk.ru/",
    }
    try:
        r = await http.post(f"{_VK_API}/{method}", data=params, headers=headers, timeout=25.0)
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("VK api.%s failed: %s", method, e)
        return None
    if "error" in data:
        err = data["error"]
        code = err.get("error_code")
        msg = err.get("error_msg")
        logger.warning("VK api.%s error %s: %s", method, code, msg)
        # авторизационная ошибка → пробуем refresh токена
        if code in (5, 1110, 1114):  # user authorization failed / token expired
            await _refresh_web_token()
        return None
    return data.get("response")


def _track_from_api(item: dict) -> dict | None:
    """audio.* item → TrackMeta-dict."""
    oid = item.get("owner_id")
    tid = item.get("id")
    if not oid or not tid:
        return None
    title = item.get("title") or ""
    if not title:
        return None
    # прямой url (может быть пустым у приватных/забаненных)
    url = item.get("url") or ""
    # audio_streams для выбора качества
    streams = item.get("audio_streams") or []
    stream_url = ""
    for pref in ("aac", "hls_range", "dash"):
        for st in streams:
            if st.get("type") == pref and st.get("url"):
                stream_url = st["url"]
                break
        if stream_url:
            break
    if not stream_url and url:
        stream_url = url
    # thumb
    thumb = ""
    album = item.get("album")
    if isinstance(album, dict):
        thumb_obj = album.get("thumb")
        if isinstance(thumb_obj, dict):
            for k in ("photo_300", "photo_1200", "photo_600", "photo_270", "url"):
                if thumb_obj.get(k):
                    thumb = thumb_obj[k]
                    break
    # год из date (timestamp) — для is_old и pop_score
    year = 0
    date_ts = item.get("date")
    if date_ts and isinstance(date_ts, (int, float)) and date_ts > 0:
        import time as _time
        year = _time.gmtime(date_ts).tm_year
    return {
        "id": f"{VK_ID_PREFIX}{oid}_{tid}",
        "owner_id": str(oid),
        "track_id": str(tid),
        "title": str(title),
        "artist": str(item.get("artist") or ""),
        "duration": int(item.get("duration") or 0),
        "thumbnail": thumb,
        "stream_url": stream_url,
        "access_key": item.get("access_key") or "",
        "year": year,
        # rank — позиция в выдаче ВК (proxy популярности, как в Яндексе)
        "rank": 0,  # заполнится в _search_api
    }


async def _search_api(query: str, limit: int) -> list[dict]:
    """Основной: поиск через api.vk.ru/method/audio.search."""
    q = (query or "").strip()
    if not q:
        return []
    # запрашиваем больше (для ранжирования), потом режем до limit
    resp = await _api_vk("audio.search", {"q": q, "count": max(limit * 2, 20)})
    if not isinstance(resp, dict):
        return []
    items = resp.get("items") or []
    out: list[dict] = []
    now = time.time()
    for idx, item in enumerate(items):
        row = _track_from_api(item)
        if not row:
            continue
        row["rank"] = idx  # позиция ВК = proxy популярности (как в Яндексе)
        if row["stream_url"]:
            _stream_cache[row["id"]] = (row["stream_url"], now)
        out.append({
            "id": row["id"],
            "title": row["title"],
            "artist": row["artist"],
            "duration": row["duration"],
            "thumbnail": row["thumbnail"],
            "channel": "vkmusic",
            "url": f"https://vk.com/audio{row['owner_id']}_{row['track_id']}",
            "year": row.get("year", 0),
            "rank": row.get("rank", 0),
        })
    # РАНЖИРОВАНИЕ КАК В ЯНДЕКСЕ: релевантность артисту → свежесть+популярность
    from .ym_client import _rank_by_artist
    out = _rank_by_artist(out, q)[:limit]
    logger.info("api.vk.ru search: %d hits for %r (ranked)", len(out), q)
    return out


async def _resolve_api(track_id: str) -> str:
    """Основной: resolve через api.vk.ru/method/audio.getById (кэш → API)."""
    hit = _stream_cache.get(track_id)
    if hit and (time.time() - hit[1]) < _STREAM_TTL:
        return hit[0]
    if not is_vk_id(track_id):
        raise RuntimeError(f"not a vk id: {track_id}")
    parts = track_id[len(VK_ID_PREFIX):].split("_", 1)
    if len(parts) != 2:
        raise RuntimeError(f"bad vk id: {track_id}")
    owner_id, tr_id = parts

    need_legacy = False
    async with _lock:
        hit = _stream_cache.get(track_id)
        if hit and (time.time() - hit[1]) < _STREAM_TTL:
            return hit[0]

        audios = f"{owner_id}_{tr_id}"
        resp = await _api_vk("audio.getById", {"audios": audios})
        if isinstance(resp, list) and resp:
            row = _track_from_api(resp[0])
            if row and row["stream_url"]:
                _stream_cache[track_id] = (row["stream_url"], time.time())
                return row["stream_url"]
        # API не дал ссылку → нужен legacy. ВЫХОДИМ из _lock ДО вызова,
        # т.к. _resolve_legacy берёт тот же asyncio.Lock (не reentrant) →
        # иначе DEADLOCK (audit #2, 2026-11-02).
        need_legacy = True

    if need_legacy:
        logger.info("api.vk.ru resolve failed for %s, trying legacy", track_id)
        return await _resolve_legacy(track_id)
    raise RuntimeError(f"cannot resolve vk id: {track_id}")


# ——————————————————————————————————————————————
# Public API (совместимость с main.py)
# ——————————————————————————————————————————————

async def search(query: str, limit: int = 10) -> list[dict]:
    """Поиск треков: api.vk.ru → fallback al_audio."""
    q = (query or "").strip()
    if not q:
        return []
    # основной путь
    out = await _search_api(q, limit)
    if out:
        return out
    # fallback: legacy al_audio
    logger.info("api.vk.ru search empty/failed for %r, trying legacy", q)
    return await _search_legacy(q, limit)


async def resolve_stream(track_id: str) -> str:
    """Прямой CDN-URL: кэш → api.vk.ru → fallback al_audio."""
    hit = _stream_cache.get(track_id)
    if hit and (time.time() - hit[1]) < _STREAM_TTL:
        return hit[0]
    return await _resolve_api(track_id)


def stream_client() -> httpx.AsyncClient:
    return _ensure_http()


async def close() -> None:
    global _http
    if _http is not None and not _http.is_closed:
        await _http.aclose()
    _http = None
