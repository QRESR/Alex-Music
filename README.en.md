# 🎵 Alex Music

**Self-hosted music streaming** with smart playlists («My Wave»-style), a
**voice assistant**, and **machine learning that learns your taste** — no
subscriptions, your own keys.

Two parts:

1. **Server** (this repo) — searches, recommends and streams the audio.
2. **Android app (APK)** — the player **Alex Music** that connects to it.

Sources: **Yandex Music** and **VK Music** (switch between them). The assistant
**understands** requests («play 5 songs from the TV series …») but **never
invents tracks** — everything comes from a real search.

> 🔍 Why this exists: big services push their own catalogue and ads. Alex Music
> is **yours**: self-hosted, no ads, and it gets smarter the more you listen.

---

## Features

- 🔍 **Search** — natural text, up to 10 tracks, exact-match mode, query
  variants run **in parallel** (fast).
- 🌊 **Playlists & similar tracks** — «My Wave», ranked by **sound** (CLAP
  embeddings), not just by titles.
- 🤖 **It learns from you** — likes, dislikes, skips and completions are
  stored and **change what you hear**: unwanted artists/genres are filtered
  out, favorites float up. The more you listen, the better it gets.
- 🎙 **Voice assistant (LLM)** — understands natural requests, searches and
  plays.
- 📱 **Android app** — queue, playlists, dislikes, Yandex/VK source switch.

---

## Quick start (for beginners)

### Step 1. Get the code

Open a terminal and run:

```bash
git clone https://github.com/QRESR/Alex-Music.git
cd Alex-Music
```

### Step 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3. Create your secrets file

Copy the template — this creates the file where your keys will go:

```bash
cp private/.env.example private/.env.music-proxy
```

After this you have **these files** to put keys into:

| File | What goes in there |
|---|---|
| `private/.env.music-proxy` | your assistant (LLM) key |
| `private/yandex-token.txt` | Yandex Music OAuth token (create file, one line = token) |
| `private/vk-cookies.txt` | VK cookies (optional; create file) |

> 💡 Open them with any text editor (Notepad, VS Code, nano).
> `private/` holds only **your** secrets — it is **never** committed to git.

### Step 4. Add your keys (minimum — one)

**Only the LLM key is required** (without it the assistant stays silent, but
search works). Yandex and VK are optional (music sources).

**LLM key:**
1. Open `private/.env.music-proxy` in a text editor.
2. Find the line `LLM_API_KEY=`
3. Paste your key right after `=` (no spaces):
   ```
   LLM_API_KEY=your key here
   ```
4. Optionally set `LLM_BASE_URL` (your OpenAI-compatible endpoint) and
   `LLM_MODEL`.

**Yandex Music token (main source):**
1. Create an empty file `private/yandex-token.txt`.
2. Paste your Yandex Music OAuth token — **one line**, no quotes.

**VK Music (optional):**
- Create `private/vk-cookies.txt` and paste cookies of your logged-in VK
  account. Without it the «VK» source simply doesn't appear — everything else
  works.

> 📌 **Where to get keys:**
> - **LLM** — from your LLM provider (any OpenAI-compatible API key).
> - **Yandex** — OAuth token with access to Yandex Music.
> - **VK** — browser cookies after logging in (`p` and `remixsid`).

### Step 5. Run the server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8011
```

Open `http://127.0.0.1:8011/health` in a browser — you should see `ok`.

### Step 6. Install the Android app

- Build the **APK** or install a release build and point it at your server.
- Done — search, play, and tap 👍/👎 (that's the **learning** part).

---

## How it works

```
Android app (player "Alex Music")
   │   GET /search /radio /stream   ·   POST /smart /radio/feedback
   ▼
Server (FastAPI, port 8011)
   ├── Yandex Music   ← primary source
   ├── VK Music       ← separate source (parallel)
   ├── LLM            ← understands what you ask
   ├── CLAP embeddings← playlists ranked by sound
   └── feedback.json  ← SELF-LEARNING: artist & genre scores
   ▼
Audio to your phone
```

---

## Tests

```bash
python3 tests/test_tz13.py     # expect 16/16
```

## API (short)

| Method | Path | What it does |
|---|---|---|
| GET | `/search?q&limit&mix&source` | search tracks |
| GET | `/radio?id&mode&note` | playlist / similar |
| POST | `/smart {text}` | voice command → action |
| POST | `/radio/feedback {id,type}` | like / dislike / skip → **learning** |
| GET | `/stream/{id}` | audio stream |
| GET | `/api/taste` `/api/metrics` | taste / metrics |
| GET | `/health` | health check |

---

## FAQ

**How does the learning work?**
Like/dislike/skip/completion events are written to `feedback.json` and
immediately affect ranking: artists and genres with a negative score are
filtered out, positive ones float up.

**Why do I need my own keys?**
It's your personal service: you use your own Yandex/VK account and your own
LLM. There are no shared/foreign keys in the code.

**Does the assistant invent tracks?**
No. It only formulates the query; the tracks come from a real Yandex/VK search.

**Is YouTube supported?**
No. The YouTube path is disabled in this build; Yandex and VK are used.

## License

MIT
