#!/usr/bin/env python3
"""Пары «исполнитель - название» радиочартов → id треков Яндекса.

Источники (спарсено вручную): RR superchart, DFM top-30, Европа+ chart.
Матчинг: точечный поиск Яндекса + сверка нормализованных токенов
(артист И название), как в anchor-фильтре. Выход: research/radio_chart_ids.txt
"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ym_client as ym  # noqa: E402
from app import recommender as rec  # noqa: E402


def load_pairs() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for src, path in (
        ("rr", "/tmp/rr_pairs.txt"),
        ("dfm", "/tmp/dfm_pairs.txt"),
        ("ep", "/tmp/ep_pairs.txt"),
    ):
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            a, t = line.split("\t", 1)
            # декод \uXXXX-эскейпы ТОЧЕЧНО (DFM: "Marshmello \u0026 ...");
            # полный unicode_escape сломал бы кириллицу (mojibake)
            a = re.sub(r"\\u([0-9a-fA-F]{4})",
                       lambda m: chr(int(m.group(1), 16)), a)
            t = re.sub(r"\\u([0-9a-fA-F]{4})",
                       lambda m: chr(int(m.group(1), 16)), t)
            a, t = a.strip(), t.strip()
            if a and t:
                rows.append((src, a, t))
    # дедуп по (artist,title)
    seen: set[tuple[str, str]] = set()
    out = []
    for src, a, t in rows:
        key = (a.lower(), t.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((src, a, t))
    return out


def match(a: str, t: str, results: list[dict]) -> dict | None:
    """Лучший кандидат: токены артиста И названия пересекаются."""
    a_toks = {x for x in rec._norm(a).split() if len(x) >= 3}
    t_toks = {x for x in rec._norm(t).split() if len(x) >= 3}
    best, best_score = None, 0.0
    for cand in results:
        ca = set(rec._norm(str(cand.get("artist") or "")).split())
        ct = set(rec._norm(str(cand.get("title") or "")).split())
        a_hit = len(a_toks & ca)
        t_hit = len(t_toks & ct)
        if a_hit and t_hit:
            score = a_hit + t_hit
            if score > best_score:
                best, best_score = cand, score
    return best


async def main() -> None:
    pairs = load_pairs()
    print(f"пар для матчинга: {len(pairs)}")
    ids: list[str] = []
    report: list[str] = []
    miss: list[str] = []
    sem = asyncio.Semaphore(1)  # строго последовательно — против 429

    async def one(src: str, a: str, t: str) -> None:
        async with sem:
            res: list[dict] = []
            for attempt in range(3):
                try:
                    res = await ym.search(f"{a} {t}", limit=10)
                except Exception:  # noqa: BLE001
                    res = []
                if res:
                    break
                # пусто может быть из-за 429 (search возвращает [] без exception)
                await asyncio.sleep(3.0 * (attempt + 1))
            await asyncio.sleep(2.0)  # пауза к Яндексу против 429
        hit = match(a, t, res)
        if hit and hit.get("id"):
            ids.append(str(hit["id"]))
            report.append(f"{src}\t{hit['id']}\t{a} - {t}\t->\t"
                          f"{hit.get('artist')} - {hit.get('title')}")
        else:
            miss.append(f"{src}\t{a} - {t}")

    await asyncio.gather(*(one(*r) for r in pairs))

    # дедуп id + СЛИЯНИЕ с уже найденными (повторные прогоны не теряют)
    out = Path(__file__).resolve().parent / "radio_chart_ids.txt"
    prev = out.read_text(encoding="utf-8").splitlines() if out.exists() else []
    uniq: list[str] = []
    seen: set[str] = set()
    for i in list(prev) + ids:
        i = i.strip()
        if i and i not in seen:
            seen.add(i)
            uniq.append(i)
    out.write_text("\n".join(uniq), encoding="utf-8")
    log = Path(__file__).resolve().parent / "radio_chart_match.tsv"
    log.write_text("\n".join(report) + ("\n--- НЕ НАЙДЕНО ---\n" + "\n".join(miss)
                                        if miss else ""), encoding="utf-8")
    print(f"найдено id: {len(uniq)} (из {len(pairs)} пар); не найдено: {len(miss)}")
    print(f"-> {out}")
    print(f"-> {log}")
    for line in report[:12]:
        print("  ", line.replace("\t", " | ")[:120])
    if miss:
        print("  НЕ НАЙДЕНО:")
        for line in miss[:10]:
            print("   -", line)


if __name__ == "__main__":
    asyncio.run(main())
