"""Fast-path голосовых команд + шаблоны say (P0 ТЗ).

Зачем: простые команды («следующий», «пауза», «громче») обходятся БЕЗ LLM —
мгновенно и бесплатно. Сложные («подборка по настроению», «похожее») → LLM.

Также: шаблоны фраз для озвучки простых случаев — не тратим LLM-вызов.

Возврат: dict(action, say) — НЕ None. Если не fast-path → None (идём в LLM).

ВАЖНО про regex: паттерны используют ПОЛНЫЕ слова русского языка
(«следующий», а не «следующи») + lookahead-границу (?![а-яё]), потому что
\\b в Python работает только на ASCII-границах и не матчит русские слова.
"""
from __future__ import annotations

import re

# ─── Fast-path: команда → (action, say-шаблон) ───────────────────────────
# Паттерны — полные слова/фразы + lookahead (?![а-яё]) для границы.
# Матчатся вначале (после вводных) или после пробела: (?:^|\s)PAT
# Порядок важен: более специфичные раньше.
_FAST_RULES: list[tuple[str, dict]] = [
    # стоп/пауза
    (r"(останови|стоп|пауза|замри|поставь\s+на\s+пауз|stop|pause)(?![а-яё])",
     {"type": "stop", "say": "Останавливаю."}),
    # продолжить
    (r"(продолжи|продолжить|включи\s+обратно|resume|continue)(?![а-яё])",
     {"type": "resume", "say": "Продолжаем."}),
    # следующий/предыдущий
    (r"(следующий|следующ|дальше|next|еще\s+трек|ещё\s+трек)(?![а-яё])",
     {"type": "next", "say": "Слушаем следующий."}),
    (r"(предыдущий|предыдущ|назад|previous|prev)(?![а-яё])",
     {"type": "previous", "say": "Возвращаюсь."}),
    # громкость
    (r"(громче|прибавь|увеличь\s+громк|больше\s+громк|louder)(?![а-яё])",
     {"type": "volume_up", "say": "Громче."}),
    (r"(тише|убавь|уменьши\s+громк|меньше\s+громк|quieter)(?![а-яё])",
     {"type": "volume_down", "say": "Тише."}),
]

# вводные слова в начале команды
_LEADING = re.compile(r"^(а|ну|давай|можно|пожалуйста|ага|ок|короче|просто)\s+")


def try_fast_path(text: str) -> dict | None:
    """Определяет ПРОСТУЮ команду без LLM.

    Возвращает {"action": {...}, "say": str} или None (нужен LLM).
    """
    if not text:
        return None
    t = text.strip().lower().replace("ё", "е")
    # убираем вводные («а», «ну», «давай», «можно») — пока есть
    prev = None
    while prev != t:
        prev = t
        t = _LEADING.sub("", t).strip()
    if not t:
        return None
    # ТЗ п.4/п.12: «продолжи в том же духе» / «в том же настроении» —
    # музыкальное УТОЧНЕНИЕ (нужен LLM → radio/похожие), а НЕ resume плеера.
    # Раньше ловил паттерн resume и превращал подборку в «Продолжаем.»
    if "продолж" in t and re.search(
            r"(дух|настроени|похож|ключ[еи]|тем\s+же|таком\s+же|этом\s+же|еще\s+что|ещё\s+что)", t):
        return None
    for pat, action in _FAST_RULES:
        # паттерн как отдельная команда: вначале (после вводных) или после пробела
        if re.search(r"(?:^|\s)" + pat, t):
            return {"action": action, "say": action["say"]}
    return None


# ─── Шаблоны say (без LLM) ───────────────────────────────────────────────
def _plural(n: int) -> str:
    """Русская форма множественного числа для «трек/трека/треков»."""
    if n % 10 == 1 and n % 100 != 11:
        return "трек"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "трека"
    return "треков"


def say_template(kind: str, n: int = 0, extra: str = "") -> str | None:
    """Готовая фраза для озвучки БЕЗ обращения к LLM.

    kind: found | empty | mix | radio | error | clarify
    n: количество треков (для found/mix/radio)
    extra: подпись запроса (необязательно)
    """
    if kind == "found":
        if n <= 0:
            return "Ничего не нашла, попробуй иначе."
        return f"Нашла {n} {_plural(n)}. Включаю первый."
    if kind == "mix":
        if n <= 0:
            return "Ничего не нашла, попробуй иначе."
        return f"Собрала подборку: {n} {_plural(n)}."
    if kind == "radio":
        return "Подбираю похожее." if not extra else f"Подбираю похожее: {extra}."
    if kind == "empty":
        return "Ничего не нашла, попробуй иначе."
    if kind == "error":
        return "Сервер недоступен, попробуй ещё раз."
    if kind == "clarify":
        return extra or "Что именно включить?"
    return None


# ─── Быстрые музыкальные действия (play из текста без артиста) ────────────
# Эти УЖЕ обрабатываются LLM (tools), но fast-path ускоряет и убирает
# зависимость от LLM-доступности. Совместимо: LLM-путь останется fallback.
_SIMPLE_MUSIC: list[tuple[str, dict]] = [
    (r"(включи|поставь|запусти|поиграй)\s+(музыку|что-нибудь|что нибудь|чтото|что-то)(?![а-яё])",
     {"type": "search", "query": "музыка", "mix": 1, "say": "Включаю музыку."}),
]


def try_simple_music(text: str) -> dict | None:
    """Максимально простые музыкальные команды (без LLM, без угадывания артиста)."""
    if not text:
        return None
    t = text.strip().lower().replace("ё", "е")
    for pat, action in _SIMPLE_MUSIC:
        if re.search(pat, t):
            return {"action": action, "say": action.get("say")}
    return None
