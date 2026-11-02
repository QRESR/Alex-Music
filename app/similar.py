"""Переработка режима «Похожие» (ТЗ similar, 2026-10-31).

Архитектура (ТЗ п.3-7):
  1. ПАРАЛЛЕЛЬНЫЙ сбор кандидатов: Rotor + Яндекс similar + mood-поиск
     (LLM→Яндекс) + CLAP audio-to-audio (кэш-only). Каждый источник под
     общим семафором (concurrency.source_gate), со СВОИМ таймаутом и
     независимой ошибкой (п.3.5): упал один — остальные дают частичные
     результаты (п.10).
  2. Объединение/нормализация/дедуп по (artist,title) — remix/live/cover
     НЕ схлопываются (разные ключи, п.4).
  3. Единый Recommendation Engine: _rec_engine.rank(mode="similar",
     audio_sim=CLAP) — п.5; CLAP только при покрытии кэша ≥50%
     (п.3.2: не подменять отсутствующее сходство нулём).
  4. Diversity Reranking: mmr_rerank — баланс seed-артиста и чужих (п.6).
  5. Добор до 20-30: ≤2 дополнительных запроса Rotor от топ-кандидатов
     (п.3.3: не размножать неограниченно), иначе повторный mood (п.7).

Откат: config.USE_SIMILAR_V2=0 → прежний код в main.api_radio (п.10).
"""
from __future__ import annotations

import asyncio
import logging
import re

from .config import config

logger = logging.getLogger("music_proxy.similar")

# Таймауты источников (п.3.5): отдельные, чтобы медленный не держал пул.
_T_ROTOR = 20.0
_T_SIMILAR = 15.0
_T_MOOD_LLM = 8.0
_T_MOOD_SEARCH = 20.0


async def _src(coro_factory, name: str, timeout: float):
    """Один источник: семафор + таймаут + ловля ошибки → None (п.3.5)."""
    from . import concurrency as _cc
    try:
        async with _cc.source_gate():
            return await asyncio.wait_for(coro_factory(), timeout=timeout)
    except Exception as e:  # noqa: BLE001
        logger.warning("similar: источник %s упал (%s) — пропускаем", name, e)
        return None


def _t_keys(t) -> tuple[str, str]:
    d = t if isinstance(t, dict) else t.model_dump()
    return ((d.get("artist") or "").lower().strip(),
            (d.get("title") or "").lower().strip())


def _to_meta(raw: dict):
    from .models import TrackMeta as _TM
    try:
        return _TM(**raw)
    except Exception:  # noqa: BLE001
        return None


# ——— Фоновый добор CLAP-эмбеддингов кандидатов (2026-10-31) ———————————
# Проблема: кандидаты пула приходят свежими и БЕЗ эмбеддингов → покрытие
# audio_sim садится ниже порога 50% и CLAP-фильтр выключается (бенч:
# покрытие 48% vs 50%, в похожие проскакивал Toby Fox - MEGALOVANIA).
# Разовый бэкфилл не спасает — новые id появляются КАЖДЫЙ вызов.
# Решение: считаем недостающие ym-id в фоне (fire-and-forget, без задержки
# ответа); к следующему вызову покрытие растёт. 2026-11-01: добавили ВК-id —
# resolve_stream починен (отдельный _token_lock от deadlock), стабильно 0.3–0.8с.
_EMB_PENDING: set[str] = set()
_EMB_TASKS: set = set()


def _schedule_emb_backfill(ids: list[str], limit: int = 8) -> None:
    """Планирует фоновый расчёт эмбеддингов для кандидатов без CLAP-вектора."""
    try:
        from . import embeddings as _E
        missing: list[str] = []
        for tid in ids:
            if not (tid.startswith("ym_") or tid.startswith("vk_")):
                continue
            if tid in _EMB_PENDING:
                continue
            if _E.embedding_for(tid, compute=False) is None:
                missing.append(tid)
            if len(missing) >= limit:
                break
        if not missing:
            return
        _EMB_PENDING.update(missing)
        task = asyncio.get_running_loop().create_task(_emb_backfill_worker(missing))
        _EMB_TASKS.add(task)
        task.add_done_callback(_EMB_TASKS.discard)
    except Exception as e:  # noqa: BLE001
        logger.debug("emb-backfill schedule failed: %s", e)


async def _ensure_seed_embedding(seed_id: str, timeout: float = 4.0) -> bool:
    """Считает эмбеддинг seed СИНХРОННО (ждёт ≤ timeout), если его нет.

    ФИКС «CLAP не работает с первого вызова»: раньше seed шёл только в фоне
    → compute=False читал пустой кэш → clap_ref=None → CLAP-блок пропущен
    (20 раз за день). Теперь ждём конкретную задачу seed до timeout сек;
    успел → True (CLAP включится), нет → False (пропускаем, не блокируем).
    """
    from . import embeddings as _E
    if _E.embedding_for(seed_id, compute=False) is not None:
        return True                      # уже в кэше — мгновенно
    if not (seed_id.startswith("ym_") or seed_id.startswith("vk_")):
        return False                     # не аудио-id — нечем считать
    if seed_id in _EMB_PENDING:
        # уже считается фоном — дожидаемся существующей задачи
        task = next((t for t in _EMB_TASKS if not t.done()), None)
        if task is None:
            return _E.embedding_for(seed_id, compute=False) is not None
    else:
        _EMB_PENDING.update({seed_id})
        task = asyncio.get_running_loop().create_task(
            _emb_backfill_worker([seed_id]))
        _EMB_TASKS.add(task)
        task.add_done_callback(_EMB_TASKS.discard)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
        logger.info("similar: seed-эмбеддинг не успел за %.0fс — CLAP пропущен",
                    timeout)
        return False
    return _E.embedding_for(seed_id, compute=False) is not None


async def _emb_backfill_audio(url: str, limit_bytes: int = 6_000_000) -> bytes:
    """Скачивает начало аудио по URL (до limit_bytes — CLAP хватает первых сек)."""
    import httpx
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as hc:
        r = await hc.get(url, headers={"Range": f"bytes=0-{limit_bytes}"})
        r.raise_for_status()
        return r.content


async def _emb_backfill_worker(ids: list[str]) -> None:
    """Скачивает аудио ym/vk-id и считает эмбеддинг в to_thread (не блокирует loop)."""
    from . import embeddings as _E
    try:
        for tid in ids:
            if _E.embedding_for(tid, compute=False) is not None:
                continue
            try:
                if tid.startswith("ym_"):
                    from . import ym_client as _Y
                    url = await _Y.cached_direct_link(tid)
                    audio = await _emb_backfill_audio(url)
                else:  # vk_
                    from . import vk_client as _VK
                    url = await _VK.resolve_stream(tid)
                    if not url:
                        raise RuntimeError("vk resolve дал пустой url")
                    audio = await _emb_backfill_audio(url)
                vec = await asyncio.to_thread(_E.embedding_for, tid, audio, compute=True)
                if vec:
                    logger.info("emb-backfill фон: %s OK (%d b)", tid, len(audio))
            except Exception as e:  # noqa: BLE001
                logger.debug("emb-backfill фон FAIL %s: %s", tid, e)
            await asyncio.sleep(1.2)   # передышка на 2 vCPU
    finally:
        _EMB_PENDING.difference_update(ids)


async def build_similar(
    *, id: str, limit: int, note: str, ctx: str, queue: str, fresh: int
):
    """Новая реализация «Похожих». Возвращает SearchResponse | None.

    None → вызывающий идёт в прежний код (обратная совместимость/фолбэк).
    """
    import time as _time
    t0 = _time.time()

    from . import fastpath as _fp
    from . import llm as _llm
    from . import recommender as _rec
    from . import taste as _taste
    from . import ym_client as _ym
    from .models import TrackMeta as _TM

    # ---- Якоря (как в прежнем коде): артист из ctx/note, жанр из note+ctx ----
    genre_anchor = _ym._query_genre(f"{note} {ctx}".strip())
    artist_anchor = _ym._artist_anchor_from_ctx(ctx) or _ym._artist_anchor_from_ctx(note)

    # ---- 1) ПАРАЛЛЕЛЬНЫЙ СБОР КАНДИДАТОВ (п.3.5) ----------------------------
    tasks: dict[str, asyncio.Task] = {}
    batch_holder: dict = {"batch_id": None}   # batch_id ротора → /radio/feedback

    if _ym.is_ym_id(id):
        station = id
        if fresh and _ym.is_ym_id(queue or ""):
            station = queue

        async def _rotor(sid=station):
            # rotor_tracks -> (list, batch_id): batch нужен петле /radio/feedback
            raw, bid = await _ym.rotor_tracks(sid, limit=min(limit * 2, 40), queue_id=None)
            if bid:
                batch_holder["batch_id"] = bid
            return raw

        def _similar():
            return _ym.similar_tracks(id, limit=min(limit * 2, 40))

        tasks["rotor"] = asyncio.ensure_future(_src(_rotor, "rotor", _T_ROTOR))
        tasks["similar"] = asyncio.ensure_future(_src(_similar, "similar", _T_SIMILAR))

    # mood-поиск (LLM → Яндекс) — как ОДИН источник (п.3.4: расширяет пул,
    # допускает чужих артистов; жанр-якорь фильтруем ниже, артист-жёсткий — нет)
    mood_base = (
        (f"{note} подборка" if (note and genre_anchor) else None)
        or (f"{artist_anchor} подборка" if artist_anchor else None)
        or note
        or (f"похожее на {ctx}" if ctx else "")
    )

    async def _mood():
        if not mood_base:
            return []
        mq = ""
        try:
            mq = await asyncio.wait_for(
                _llm.mood_mix_query(mood_base, current_track=ctx),
                timeout=_T_MOOD_LLM,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("similar: mood LLM упал: %s", e)
        if not mq:
            mq = mood_base  # фолбэк без LLM (как в прежнем коде)
        try:
            return await _ym.search(mq, limit=min(limit * 2, 40))
        except Exception as e:  # noqa: BLE001
            logger.warning("similar: mood search упал: %s", e)
            return []

    tasks["mood"] = asyncio.ensure_future(_src(_mood, "mood", _T_MOOD_LLM + _T_MOOD_SEARCH))

    # CLAP audio-to-audio (п.3.2): эмбеддинг seed ИЗ КЭША; нет эмбеддинга →
    # источник пропускается (не 0!). Без инференса, без сети, мгновенно.
    #
    # ВК-SEED (2026-11-01): у ВК-id эмбеддинга нет (кас ym/vk в фоне идёт на
    # строки 271+ — ПОЗЖЕ, и только по pool, а seed в pool не попадает) →
    # clap_ref был None ВСЕГДА для похожих на ВК-трек. Решение: планируем добор
    # seed ЗДЕСЬ, до чтения.
    # ФИКС «CLAP не работает» (2026-11-01, 20 раз за день «у seed нет
    # CLAP-эмбеддинга → clap=0»): раньше эмбеддинг seed считался ТОЛЬКО в
    # фоне (compute=False читал кэш сразу → пусто → clap_ref=None → весь
    # CLAP-блок пропущен). Теперь: если эмбеддинга seed НЕТ — ждём его
    # синхронно с таймаутом ~4с (resolve 1с + download 6МБ 1-2с + CLAP 1с).
    # Успел → CLAP работает С ПЕРВОГО вызова (по звучанию, как просил);
    # не успел → пропускаем, как раньше (не блокируем ответ). Один seed, а не
    # весь пул — цена терпима и только на первом вызове (потом кэш).
    clap_ref = None
    clap_candidates: list[dict] = []
    try:
        from . import embeddings as _emb
        _emb.init_db()
        _seed_emb = _emb.embedding_for(id, compute=False) if id else None
        if _seed_emb is None and id and (
                id.startswith("vk_") or id.startswith("ym_")):
            # seed без эмбеддинга — СЧИТАЕМ и ждём до 4с (CLAP с 1-го вызова)
            await _ensure_seed_embedding(id, timeout=4.0)
            _seed_emb = _emb.embedding_for(id, compute=False)
        clap_ref = _seed_emb
        if clap_ref:
            clap_candidates = _emb.nearest_tracks(clap_ref, k=min(limit * 2, 40))
        else:
            logger.info("similar: у seed нет CLAP-эмбеддинга — источник пропущен")
    except Exception as e:  # noqa: BLE001
        logger.warning("similar: CLAP источник упал: %s", e)

    # Ждём источники (ошибки уже пойманы внутри _src) — п.3.5 частичные данные
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    got: dict[str, list] = {}
    for name, res in zip(tasks.keys(), results):
        if isinstance(res, BaseException):
            logger.warning("similar: %s: %s", name, res)
            continue
        got[name] = res or []

    # ---- 2) ОБЪЕДИНЕНИЕ/НОРМАЛИЗАЦИЯ/ДЕДУП (п.4) ----------------------------
    pool: list = []
    seen: set[tuple[str, str]] = set()
    src_counts: dict[str, int] = {}
    _absorb_seq = 0                        # общая позиция в пуле (приоритет источника)

    def _absorb(items, source: str):
        nonlocal _absorb_seq
        n = 0
        for it in items or []:
            d = it if isinstance(it, dict) else (it.model_dump() if hasattr(it, "model_dump") else {})
            if not d.get("id") or not d.get("title"):
                continue
            key = ((d.get("artist") or "").lower().strip(),
                   (d.get("title") or "").lower().strip())
            if key in seen:
                continue
            # блок-лист мусора (Macarena и т.п.) — единый для всех источников
            if _ym._is_blocked(d):
                continue
            seen.add(key)
            m = _TM(**d) if not isinstance(it, _TM) else it
            _absorb_seq += 1
            m.rank = _absorb_seq           # позиция в общем пуле = прокси популярности
            pool.append(m)
            n += 1
        src_counts[source] = n

    _absorb(got.get("similar"), "ym_similar")   # Яндекс «похожие» — первый по качеству
    _absorb(got.get("rotor"), "rotor")
    _absorb(got.get("mood"), "mood")
    _absorb(clap_candidates, "clap")
    # Мета для track_meta_cache (2026-11-01): nearest_tracks пропускает id без
    # меты («if tid not in meta») → ВК/Яндекс-кандидаты с эмбеддингом, но без
    # artist/title в кэше, не попадали в clap-источник. Пишем мету из пула.
    try:
        from . import embeddings as _emb_meta
        _emb_meta.put_meta([
            {"id": t.id, "artist": t.artist, "title": t.title,
             "year": getattr(t, "year", 0)}
            for t in pool
        ])
    except Exception as _e_meta:  # noqa: BLE001
        logger.debug("similar put_meta: %s", _e_meta)
    logger.info(
        "similar сбор: пул=%d src=%s seed=%r (лимит=%d)",
        len(pool), src_counts, artist_anchor or ctx[:30], limit,
    )

    # ---- ФИЛЬТР ПО ГОДАМ (ТЗ «не старше 7 лет» → с 2019) --------------------
    # В mode=similar возраст НЕ фильтровался (is_old применялся только к rotor
    # и mood) → в подборку попадали треки 1990/1999/2009. Отсекаем старше 7 лет
    # (2019 и новее) НА ПУЛЕ (до CLAP/ранжирования). Треки БЕЗ года не режем.
    # Guard: если свежих осталось < лимита — не вымирать, оставляем пул как был.
    _now_year = int(__import__("time").gmtime(__import__("time").time()).tm_year)
    _fresh_pool = [
        t for t in pool
        if not (getattr(t, "year", 0) or 0) or (_now_year - (t.year or 0)) <= 7
    ]
    if len(_fresh_pool) >= max(limit, 5):
        if len(_fresh_pool) < len(pool):
            logger.info(
                "similar age: %d -> %d (убрано старше 7 лет / до 2019)",
                len(pool), len(_fresh_pool),
            )
        pool = _fresh_pool

    # Фоновый добор эмбеддингов: сразу после сбора пула, ответ НЕ ждём.
    # К следующему вызову похожих покрытие CLAP вырастет и audio_sim
    # включится даже для свежих кандидатов (порог 50%).
    _schedule_emb_backfill([t.id for t in pool])

    if not pool and not _ym.is_ym_id(id):
        return None  # нечего делать (нет id станции) → прежний код

    # ---- 2.5) ДОБОР (п.7): пул мелкий → ≤2 дополнительных запроса Rotor ----
    rotor_extra_used = 0
    if _ym.is_ym_id(id) and len(pool) < limit * 3 // 2:
        # seed для добра: лучшие кандидаты уже собраны (топ позиций источников)
        extra_seeds = [t.id for t in pool[:2] if _ym.is_ym_id(t.id) and t.id != id]
        if extra_seeds:
            async def _rotor_extra(sid=extra_seeds[0]):
                raw, _bid = await _ym.rotor_tracks(sid, limit=min(limit, 30), queue_id=None)
                return raw

            _futures = [
                _src(lambda sid=sid: _rotor_extra(sid),
                     "rotor_extra", _T_ROTOR)
                for sid in extra_seeds[:2]   # п.3.3: не размножаем неограниченно
            ]
            _extras = await asyncio.gather(*_futures, return_exceptions=True)
            for res in _extras:
                if isinstance(res, BaseException) or not res:
                    continue
                _absorb(res, "rotor_extra")
                rotor_extra_used += 1
            if rotor_extra_used:
                logger.info("similar добор: +%d кандидатов (rotor от топ-seed)",
                            src_counts.get("rotor_extra", 0))

    if not pool:
        return None  # все источники пусты → прежний код (п.10: частичная выдача)

    # ---- 3) CLAP-ПРИЗНАК (п.3.2/п.5): только кэш, покрытие ≥50% ------------
    audio_sim: dict[str, float] | None = None
    if clap_ref:
        try:
            from . import embeddings as _emb2
            audio_sim = _emb2.audio_similarities([t.id for t in pool], clap_ref)
            # nearest_tracks уже дал sim кандидатам из кэша — переиспользуем
            for c in clap_candidates:
                if "sim" in c:
                    audio_sim.setdefault(c["id"], float(c["sim"]))
            cov = len(audio_sim) / len(pool) if pool else 0.0
            if cov < 0.5:
                # п.3.2: не подменять отсутствующее сходство нулём —
                # при малом покрытии часть кандидатов штрафовалась бы
                # искусственно → признак ВЫКЛЮЧАЕМ, работают метаданные.
                logger.info("similar: CLAP покрытие %.0f%% < 50%% — audio_sim выключен", cov * 100)
                audio_sim = None
            else:
                logger.info("similar: CLAP audio_sim покрытие %.0f%% (%d/%d)",
                            cov * 100, len(audio_sim), len(pool))
        except Exception as e:  # noqa: BLE001
            logger.warning("similar: audio_sim failed: %s", e)
            audio_sim = None

    # ---- 4) ЕДИНЫЙ РАНЖЕР (п.5) --------------------------------------------
    seed_for_rank = artist_anchor or (ctx.split(" - ")[0].strip() if ctx else "")
    try:
        ranked = _rec.rank(
            pool,
            query="",
            seed_artist=seed_for_rank,
            mode="similar",
            freshness="old_ok",          # хиты seed не режем (ТЗ 9)
            genres=[genre_anchor] if genre_anchor else [],
            audio_sim=audio_sim,
            liked_artists={a.lower() for a in _taste.get_taste()},
            banned_track_ids=_taste.banned_track_ids(hard_only=True),
            max_per_artist=5,            # верхняя граница до MMR
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("similar: rank упал: %s — берём пул как есть", e)
        ranked = pool

    # ---- 5) DIVERSITY RERANKING (п.6): MMR ----------------------------------
    # rel для MMR = позиция rank (ранжер уже учёл CLAP/вкус/жанр/свежесть)
    n_all = max(len(ranked), 1)
    rel_scores = {
        t.id: (n_all - i) / n_all for i, t in enumerate(ranked)
    }
    ordered = _rec.mmr_rerank(
        ranked, rel_scores,
        seed_artist=seed_for_rank,
        lambda_=0.72,
        # П.1 (2026-11-01, жалоба «подборки по имени, не по звучанию»): было
        # per_artist_seed=5 → до 5 треков исходного артиста наверху. Требование
        # «1–2 трека этого же исполнителя, остальные — другие» → 2. Чужие артисты
        # и CLAP/audio_sim решают порядок (по звучанию), не имя seed.
        per_artist_seed=2,               # «1–2 исходного» (было 5)
        per_artist_other=3,              # п.6: не повторять чужих подряд
    )

    # ---- 6) ФИНАЛЬНЫЕ ФИЛЬТРЫ + ЛИМИТ (п.7) --------------------------------
    ordered = _taste.filter_tracks(ordered)   # дизлайки + скипы (рекомендации)
    seen_ids: set = set()
    dedup = [t for t in ordered if not (t.id in seen_ids or seen_ids.add(t.id))]
    final = dedup[: max(limit, 5)]

    # ---- say (шаблон + LLM 8с, не блокирует) --------------------------------
    say = _fp.say_template("mix", len(final))
    mood_query_out = mood_base or None      # показать клиенту, чем добирали
    try:
        r = await _llm.smart_response(
            "Скажи одной короткой фразой, что продолжаешь подбирать похожие треки",
            context=f"Играет: {ctx}" + (f", подсказка: {note}" if note else ""),
            tools=False,
        )
        from .main import _clean_say  # общий санитайзер say (лениво — без цикла)
        _s = _clean_say((r or {}).get("text"))
        if _s:
            say = _s
    except Exception:  # noqa: BLE001
        pass
    if not say:
        say = "Держи ещё похожее на это." if ctx else "Подбираю похожее."

    elapsed = _time.time() - t0
    logger.info(
        "similar v2: %d треков (пул=%d, src=%s, clap=%s, %.2fс)",
        len(final), len(pool), src_counts,
        "on" if audio_sim else "off", elapsed,
    )

    from .models import SearchResponse
    return SearchResponse(
        query=f"radio:{id or note or ctx}",
        results=final,
        total=len(final),
        intent="playlist",
        mix=True,
        say=say,
        mood_query=mood_query_out,
        batch_id=batch_holder.get("batch_id") or "",
    )
