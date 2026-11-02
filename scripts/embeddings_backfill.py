#!/usr/bin/env python3
"""Batch-предрасчёт аудиоэмбеддингов (ТЗ п.4.3 «пакетно, без повторов»).

Кандидаты: id из плейлистов + feedback + (опционально) --ids из файла.
Треки, у которых уже есть embedding в кэше, пропускаются (LEFT JOIN).

ЗАПУСК (после eval, чтобы не конкурировать за 2 vCPU):
    cd /root/.openclaw/workspace/youtube-music-proxy
    .venv/bin/python3 scripts/embeddings_backfill.py --limit 50

Каждый трек: stream bytes (ограниченные) → CLAP → кэш. ~5.5 с/трек,
поток ОДИН (не параллелить на 2 vCPU). НЕ запускать во время eval/prod-нагрузки.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("emb-backfill")


def candidate_ids(limit: int) -> list[str]:
    """id-кандидаты из локальных данных (плейлисты + feedback), без дублей."""
    data = Path(__file__).resolve().parent.parent / "data"
    ids: list[str] = []
    # плейлисты
    for f in sorted(data.glob("playlist*.json")) + sorted(data.glob("playlists*.json")):
        try:
            d = json.loads(f.read_text())
            items = d if isinstance(d, list) else d.get("tracks") or d.get("items") or []
            for t in items:
                tid = t.get("id") if isinstance(t, dict) else str(t)
                if tid:
                    ids.append(str(tid))
        except Exception:  # noqa: BLE001
            continue
    # feedback
    fb = data / "feedback.json"
    if fb.exists():
        try:
            d = json.loads(fb.read_text())
            for key in ("liked", "completed", "skipped", "disliked"):
                ids.extend((d.get(key) or {}).keys())
        except Exception:  # noqa: BLE001
            pass
    # уникальные, ym_ приоритет (только у Яндекса есть быстрый direct-link)
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    ym = [i for i in out if i.startswith("ym_")]
    rest = [i for i in out if not i.startswith("ym_")]
    return (ym + rest)[:limit]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--ids-file", default="")
    ap.add_argument("--throttle", type=float, default=1.5, help="пауза между треками, сек (нагрузка на VPS)")
    ap.add_argument("--recheck", action="store_true", help="скачивать даже кэшированные, чтобы поймать ИЗМЕНИВШИЕСЯ (по хэшу)")
    args = ap.parse_args()

    # ТЗ п.6: ограничение нагрузки — понижаем приоритет процесса (nice/ionice),
    # чтобы предрасчёт НЕ конкурировал с музыкальным сервисом на 2 vCPU.
    try:
        import os
        os.nice(10)
    except Exception:  # noqa: BLE001
        pass

    from app import embeddings as E
    from app import ym_client

    E.init_db()
    ids = candidate_ids(args.limit)
    if args.ids_file:
        ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()][: args.limit]

    done = skipped = failed = 0
    for tid in ids:
        # resume: кэшированные пропускаем, если НЕ просим перепроверку (ТЗ п.6:
        # продолжение после перезапуска + обработка только новых/изменившихся).
        if not args.recheck and E.embedding_for(tid, compute=False) is not None:
            skipped += 1
            continue
        if not tid.startswith("ym_"):
            logger.info("skip %s (не ym: direct-link только у Яндекса)", tid[:50])
            failed += 1
            continue
        try:
            url = await ym_client.cached_direct_link(tid)
            import httpx
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as hc:
                r = await hc.get(url)
                r.raise_for_status()
                audio = r.content
            # без --recheck: кэшированные мы уже пропустили выше → здесь только
            # новые. с --recheck: сверяем хэш, is_stale фильтрует неизменившиеся.
            if args.recheck and not E.is_stale(tid, audio):
                skipped += 1
                continue
            # считаем в executor, чтобы не блокировать event loop сервиса
            vec = await asyncio.to_thread(E.embedding_for, tid, audio, compute=True)
            if vec:
                done += 1
                logger.info("OK %s (%d/%d)", tid, done + skipped, len(ids))
            else:
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            logger.warning("FAIL %s: %s", tid, e)
        await asyncio.sleep(args.throttle)   # передышка (нагрузка на 2 vCPU)

    logger.info("ИТОГ: computed=%d cached=%d failed=%d | stats=%s",
                done, skipped, failed, E.stats())


if __name__ == "__main__":
    asyncio.run(main())
