"""ТЗ п.13 «Тестирование» — 16 сценариев.

API-сценарии (1-9, 12, 13, 16) → ЖИВОЙ сервер http://127.0.0.1:8011.
Unit-сценарии (10, 11, 14, 15) → процесс теста с подменой данных в ПАМЯТИ
(настоящие data/файлы НЕ трогаются).

Критерий ТЗ: проверять КОНЕЧНЫЙ результат, а не только JSON LLM.
«Больше треков ≠ лучше»: фиксируем топ-1/match, а не только количество.

Запуск: .venv/bin/python3 tests/test_tz13.py   (или pytest tests/)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE = os.environ.get("TZ13_BASE", "http://127.0.0.1:8011")

import httpx  # noqa: E402


def _search(q: str, mix: int = 0, limit: int = 10) -> dict:
    r = httpx.get(f"{BASE}/search", params={
        "q": q, "limit": limit, "mix": mix, "source": "yandex"}, timeout=90)
    r.raise_for_status()
    return r.json()


def _smart(text: str, ctx: str = "") -> dict:
    r = httpx.post(f"{BASE}/smart", json={"text": text, "ctx": ctx}, timeout=40)
    r.raise_for_status()
    return r.json()


# ── 1 ─────────────────────────────────────────────────────────────────────
def test_p13_01_vkluchit_bastu():
    """«Включи Басту» → релевантный результат, топ-1 = Баста.

    2026-11-01: лимит обычного поиска = 10 (пользователь отменил лимит-4 —
    «пусть 10 там, чтобы выбирать было»; после 1-го трека переход в похожий,
    но список из 10 нужен для выбора).
    """
    d = _search("Баста")
    assert 1 <= len(d["results"]) <= 10, f"лимит поиска нарушен: {len(d['results'])}"
    top = d["results"][0]
    assert "баста" in top["artist"].lower(), f"топ-1 не Баста: {top['artist']}"


# ── 2 ─────────────────────────────────────────────────────────────────────
def test_p13_02_pohozhee_na_trek():
    """«Похожее на этот трек» → /radio similar валиден, есть треки."""
    r = httpx.get(f"{BASE}/radio", params={
        "id": "ym_48592407", "limit": 10, "ctx": "Баста - Выпускной",
        "fresh": 1, "mode": "similar"}, timeout=90)
    r.raise_for_status()
    d = r.json()
    assert len(d["results"]) > 0, "radio пуст"


# ── 3 ─────────────────────────────────────────────────────────────────────
def test_p13_03_spokoinee():
    """«Сделай поспокойнее» → подборка с результатами."""
    d = _search("сделай поспокойнее", mix=1)
    assert len(d["results"]) > 0


# ── 4 ─────────────────────────────────────────────────────────────────────
def test_p13_04_ne_vklyuchay_isclychenie():
    """«не включай больше этого исполнителя» → исключение реально фильтрует."""
    d = _search("баста без слим шейди")
    artists = [t["artist"].lower() for t in d["results"]]
    assert d["results"], "нет результатов"
    assert not any("eminem" in a for a in artists), "Eminem прошёл мимо исключения"


# ── 5 ─────────────────────────────────────────────────────────────────────
def test_p13_05_chto_novoe():
    """«что-нибудь новое» → свежая подборка непустая."""
    d = _search("что-нибудь новое", mix=1)
    assert len(d["results"]) > 0


# ── 6 ─────────────────────────────────────────────────────────────────────
def test_p13_06_prodolzhi_tom_zhe():
    """«продолжи в том же настроении» → action radio (не resume/пустота)."""
    a = _smart("продолжи в том же настроении", "Баста - Выпускной")
    assert a.get("action") and a["action"].get("type") == "radio", a


# ── 7 ─────────────────────────────────────────────────────────────────────
def test_p13_07_odnoobrazno():
    """«слишком однообразно» → radio с разнообразием; mpa=1 в /radio."""
    a = _smart("слишком однообразно", "Баста - Выпускной")
    assert a.get("action", {}).get("type") == "radio", a
    assert "разнообраз" in (a["action"].get("note") or ""), a["action"]
    # и серверный /radio с этим note даёт высокую уникальность
    r = httpx.get(f"{BASE}/radio", params={
        "id": "ym_48592407", "limit": 20,
        "note": "разнообразнее, новые артисты, без повторов",
        "ctx": "Баста - Выпускной", "fresh": 1, "mode": "mix"}, timeout=90)
    d = r.json()
    arts = [t["artist"] for t in d["results"]]
    assert len(set(arts)) >= len(arts) * 0.7, f"мало уникальных: {len(set(arts))}/{len(arts)}"


# ── 8 ─────────────────────────────────────────────────────────────────────
def test_p13_08_verni_preduvshiy_stil():
    """«верни предыдущий стиль» → валидный ответ ассистента."""
    a = _smart("верни предыдущий стиль", "Eminem - Stan")
    assert a.get("say") is not None, a


# ── 9 ─────────────────────────────────────────────────────────────────────
def test_p13_09_vchera_ponravilsya():
    """«вчера мне понравился рэп» → ассистент отвечает без исключений."""
    a = _smart("вчера мне понравился рэп", "")
    assert isinstance(a.get("say"), str) and a["say"], a


# ── 10 ────────────────────────────────────────────────────────────────────
def test_p13_10_nekorektnyj_otvet_llm():
    """Мусор/пустота от LLM → None и fallback, НЕ исключение."""
    from app import llm, music_intent as mi

    async def _bad(*a, **k):
        return {}
    orig = llm._chat
    llm._chat = _bad
    try:
        out = asyncio.run(mi.parse_intent_llm("танцевальная подборка", ""))
    finally:
        llm._chat = orig
    assert out is None, f"ожидался fallback None, получено {out!r}"


# ── 11 ────────────────────────────────────────────────────────────────────
def test_p13_11_api_nedostupen():
    """LLM недоступна → fastpath/offline работают (fallback без сети)."""
    from app import fastpath as fp
    from app.config import config as cfg
    from app.llm import _llm_ready
    orig = cfg.LLM_ENABLED
    cfg.LLM_ENABLED = False
    try:
        assert not _llm_ready(), "флаг LLM_ENABLED=0 должен выключить LLM"
        out = fp.try_fast_path("пауза")
    finally:
        cfg.LLM_ENABLED = orig
    assert out and out["action"]["type"] == "stop", out


# ── 12 ────────────────────────────────────────────────────────────────────
def test_p13_12_dubl_otpravki():
    """Двойная отправка команды → вторая без действия (action=null)."""
    a1 = _smart("следующий")
    a2 = _smart("следующий")
    assert a1.get("action") is not None
    assert a2.get("action") is None, f"дубль не задержан: {a2}"


# ── 13 ────────────────────────────────────────────────────────────────────
def test_p13_13_odnovremennye_komandy():
    """Две разные команды одновременно → обе валидны, сервер не падает."""
    async def _two():
        async with httpx.AsyncClient(timeout=40) as c:
            return await asyncio.gather(
                c.post(f"{BASE}/smart", json={"text": "следующий", "ctx": ""}),
                c.post(f"{BASE}/smart", json={"text": "пауза", "ctx": ""}),
            )
    r1, r2 = asyncio.run(_two())
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json().get("say") and r2.json().get("say")


# ── 14 ────────────────────────────────────────────────────────────────────
def test_p13_14_pustaya_istoriya():
    """Пустая история → функции не падают, /search работает."""
    from app import history as h
    tmp = Path(tempfile.mkdtemp()) / "history.json"
    orig = h._DATA_FILE
    h._DATA_FILE = tmp
    try:
        assert h.recent(10) == []
        assert h.last_searches(5) == []
        assert h.track_meta("ym_x") == {}
    finally:
        h._DATA_FILE = orig
    d = _search("баста", limit=5)
    assert d["results"], "поиск упал с пустой историей"


# ── 15 ────────────────────────────────────────────────────────────────────
def test_p13_15_net_profilya():
    """Нет профиля вкуса → /search без профиля отдаёт результат."""
    from app import taste as t
    tmp = Path(tempfile.mkdtemp())
    orig = t._DATA
    t._DATA = tmp          # никаких taste.json/feedback.json
    try:
        assert t.get_taste() == []
        assert t.artist_weights() == {}
        assert t.banned_track_ids() == set()
    finally:
        t._DATA = orig
    d = _search("эминем", limit=5)
    assert d["results"], "поиск упал без профиля"


# ── 16 ────────────────────────────────────────────────────────────────────
def test_p13_16_nedostatok_kandidatov():
    """Нет кандидатов (мусорный id) → JSON-ответ, не 500."""
    r = httpx.get(f"{BASE}/radio", params={
        "id": "ym_000000000", "limit": 10, "ctx": "", "fresh": 1,
        "mode": "similar"}, timeout=90)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    d = r.json()
    assert isinstance(d.get("results"), list)


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_p13_")]


if __name__ == "__main__":
    # без pytest: последовательный прогон с отчётом (ТЗ п.13 — воспроизводимость)
    ok = 0
    fails = []
    for fn in ALL:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
            fails.append(fn.__name__)
    print(f"\n{ok}/{len(ALL)} passed" + (f", FAILED: {fails}" if fails else ""))
    sys.exit(1 if fails else 0)
