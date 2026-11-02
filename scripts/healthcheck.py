#!/usr/bin/env python3
"""Music proxy healthcheck: verifies search + stream E2E and alerts via Telegram.

Cron: */30 * * * * python3 .../scripts/healthcheck.py >> /tmp/music_health.log 2>&1

Design notes:
- Uses a REAL stream resolve (the part that historically broke: bot blocks,
  cookie rotation). One track, range request, expects 206.
- Transient Google bot blocks self-heal in minutes (see yt_dlp_client
  backoff), so alert only after a SECOND failure 60s later.
- Telegram token is read from /root/.hermes/config.yaml (bot token), chat
  from allowed_users (Alex).
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Отключено по просьбе Алекса (2026-10-29): проверки и лог (/tmp/music_health.log)
# продолжают работать, но в Telegram ничего не отправляем.
TELEGRAM_ALERTS = False

BASE = "http://127.0.0.1:8011"
TG_CONFIG = Path("/root/.hermes/config.yaml")
ALERT_COOLDOWN_FILE = Path("/tmp/music_health.last_alert")
COOLDOWN_S = 2 * 3600  # do not spam more often than once per 2h


def get(url: str, timeout: int = 45, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001
        print(f"  request error: {e}", file=sys.stderr)
        return 0, b""


def check() -> list[str]:
    """Returns list of problems (empty = healthy)."""
    problems: list[str] = []

    status, body = get(f"{BASE}/search?q=kino", timeout=40)
    if status != 200:
        problems.append(f"search HTTP {status}")
        return problems
    try:
        results = json.loads(body).get("results", [])
    except Exception:
        results = []
    if not results:
        problems.append("search returned 0 tracks")
        return problems

    video_id = results[0]["id"]
    status, body = get(
        f"{BASE}/stream/{video_id}",
        timeout=60,
        headers={"Range": "bytes=0-100000"},
    )
    if status != 206:
        problems.append(f"stream/{video_id} HTTP {status} (expected 206)")
    elif len(body) < 50000:
        problems.append(f"stream/{video_id} truncated: {len(body)}B")
    return problems


def send_alert(text: str) -> None:
    try:
        import yaml  # PyYAML available in system python? fall back to regex

        cfg = yaml.safe_load(TG_CONFIG.read_text())
        token = cfg["telegram"]["bot_token"]
        chat_id = str(cfg["telegram"]["allowed_users"][0])
    except Exception:
        raw = TG_CONFIG.read_text()
        import re

        m = re.search(r"bot_token:\s*(\S+)", raw)
        m2 = re.search(r"allowed_users:\s*\[\s*(\d+)", raw)
        if not m or not m2:
            print("cannot parse telegram config", file=sys.stderr)
            return
        token, chat_id = m.group(1), m2.group(1)

    data = json.dumps({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print("alert sent:", json.loads(r.read()).get("ok"))
    except Exception as e:  # noqa: BLE001
        print(f"alert failed: {e}", file=sys.stderr)


def main() -> None:
    problems = check()
    if not problems:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} OK")
        return

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PROBLEMS: {'; '.join(problems)}")
    # transient bot blocks heal in minutes — recheck once after 60s
    time.sleep(60)
    problems2 = check()
    if not problems2:
        print("recovered on recheck (transient bot block) — no alert")
        return

    # cooldown to avoid spam
    now = time.time()
    try:
        last = float(ALERT_COOLDOWN_FILE.read_text().strip())
    except Exception:
        last = 0.0
    if now - last < COOLDOWN_S:
        print("alert suppressed by cooldown")
        return
    ALERT_COOLDOWN_FILE.write_text(str(now))
    if not TELEGRAM_ALERTS:
        print("telegram alerts disabled — problems logged only")
        return
    send_alert(
        "⚠️ Музыкальный сервер: проблемы\n"
        + "\n".join(f"• {p}" for p in problems2)
        + "\n\nПовторная проверка через 60с не помогла."
    )


if __name__ == "__main__":
    main()
