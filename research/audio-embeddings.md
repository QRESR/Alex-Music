# Аудиоэмбеддинги для music-proxy: исследование моделей + живой пилот

Дата: 2026-10-31. Окружение: VPS (AMD EPYC-Genoa, `nproc=4`, 8 ГБ RAM, Ubuntu), Python **3.14.9**,
venv проекта `/root/.openclaw/workspace/youtube-music-proxy/.venv`.
Сервис `music-proxy` во время работы **не перезапускался** (проверено: `systemctl is-active music-proxy` → `active`).

---

## 1. TL;DR (выбор)

| | |
|---|---|
| **Выбранная модель** | **`laion/clap-htsat-unfused`** (LAION CLAP, Swin/HTSAT-аудио-энкодер + RoBERTa-текстовый энкодер) |
| **Лицензия** | **Apache-2.0** — коммерческое использование ок, веса публичные, без гейта |
| **Веса** | 615 МБ (`model.safetensors`), RAM в инференсе **650–840 МБ** |
| **Стек** | `torch==2.15.0+cpu` (cp314) + `transformers==5.22.1` + `soundfile` + `soxr` — всё поставилось в .venv проекта под Python 3.14 |
| **Рецепт** | фиксированные окна **10 c** (ровно `max_length_s` экстрактора) → эмбеддинг каждого окна → **mean-pool → L2**. Детерминированно (cos повторного запуска = **1.000**), покрывает весь трек |
| **Скорость** | **5.5 c/трек** (среднее по 4 реальным трекам 134–266 c, `batch=6`, 4 потока); 9.6 c при `batch=1` |
| **Пропускная способность** | **≈460–590 треков/час** (batch=6) / ≈280–365 (batch=1) на этой машине |
| **Пик RAM** | **1304 МБ** (процесс, загружавший веса — скачивание+инференс); в стабильном инференсе 650–840 МБ |
| **text → музыка** | тот же эмбеддинг-пространство: текст-инференс **0.11 c**, косинусы «запрос↔трек» 0.48–0.64 у релевантных пар |
| **Живой пилот** | ✅ 4 реальных CC-трека прогнаны end-to-end, API-прототип `get_embedding(track_bytes) -> list[float]` проверен |

Запасной (второй) вариант: `laion/clap-htsat-fused` — та же лицензия и архитектура, режим fusion
(полный трек за один проход, но 4 мел-спектрограммы на окно → дороже; фиксированное окно 10 c даёт 4 копии — бесполезно).
Мы проверили его прониры (здоровые косинусы, текст-скоры 0.33), но для нашего рецепта он не даёт выигрыша — **не выбран**.

---

## 2. Сравнительная таблица моделей

Все лицензии/размеры проверены по PyPI/HuggingFace API в день исследования; «наш замер» — реальные цифры с этого VPS.

| Модель | Лицензия (коммерция) | Веса / RAM | Python 3.14 | Требования | Наш замер (2–4 ядра CPU) | Качество для «похожесть по звучанию» / «текст→музыка» | Вердикт |
|---|---|---|---|---|---|---|---|
| **CLAP `laion/clap-htsat-unfused`** | **Apache-2.0** ✅ | 615 МБ / 650–840 МБ | ✅ torch cp314 ≥2.9, transformers 5.x | torch-cpu, transformers, soundfile, soxr | **0.49 c/окно (b1), 0.29 c/окно (b6)**; 5.5 c/трек; текст 0.11 c | косинусы между треками **0.14–0.74** (осмысленная структура), текст: 3/4 top-1, 4/4 top-2 | ✅ **ВЫБРАН** |
| CLAP `laion/clap-htsat-fused` (fusion) | Apache-2.0 ✅ | 615 МБ / ~0.7–1.3 ГБ | ✅ | то же | fusion даёт 4 мела на вход (≈×4 к вычислению); замер полного пайплайна не делали | проба: косинусы 0.07–0.50, «acid techno»→techno 0.33 ✅ | ⚠️ запасной |
| **`laion/larger_clap_music`** (music-specific!) | Apache-2.0 ✅ | 776 МБ / ~0.8 ГБ | ⚠️ ставится, **но** | torch+transformers | инференс быстрее (0.71 c/окно), **но выход вырожден**: cos(шум, тишина)=**1.000**, cos между жанрами 0.92–0.995, текст↔аудио ≈0.01 | ❌ непригоден: повторяемо (идентичные цифры) под transformers **4.57.6 и 5.22.1**; веса грузятся без конфликтов (555/555 ключей, shapes совпадают) — битый экспорт/несовместимость архитектуры | ❌ **НЕЛЬЗЯ** (доказано проном) |
| `laion/larger_clap_music_and_speech` | Apache-2.0 ✅ | 776 МБ / ~0.8 ГБ | ✅ | то же | 0.81 c/окно (самый медленный) | эмбеддинги разреженные (mean off-diag 0.126), **text→music слабый** (0.01–0.06, ранги хаотичны) | ⚠️ не выбран |
| MS-CLAP `microsoft/msclap` | ms-pl (коммерция ок) | 690 МБ (2023 ckpt), суммарно репо 4.7 ГБ | ⚠️ не проверяли (своя кодовая база, не transformers; веса под `lasertorch`) | отдельный репозиторий кода | не замеряли | не замеряли | ⚠️ интеграция дороже, не сталкиваемся |
| **MuLan** | — | — | — | — | — | официальных весов в открытом доступе нет (есть только муляжи третьих лиц, NC) | ❌ нет весов |
| **MuQ** `OpenMuQ/MuQ-large-msd-iter` | **CC-BY-NC-4.0** ❌ | 1.33 ГБ / ~1.5+ ГБ | ⚠️ (torch) | torch | не ставили из-за лицензии | (по модель-карточке — сильный музыкальный представитель) | ❌ **не-коммерческая** лицензия |
| **MERT** `audeering/MERT-*` | **gated** (HF API → 401, лицензию не удалось верифицировать без авторизации) ⚠️ | 95M/1.2B | ⚠️ | torch | не ставили без токена | — | ❌ недоступна/неверифицируема для коммерции |
| **audiocraft / MusicGen embeddings** | **CC-BY-NC-4.0** ❌ | 2.4 ГБ (small) | ⚠️ | torch+flashy, старые пинны | не ставили | — | ❌ не-коммерческая |
| **VGGish** | Apache-2.0 (код) ✅ | TF-Hub чекпоинт, тяжёлый TF-рантайм | ❌ **TF только 2.22.0rc0** со wheels для cp314 | TensorFlow | не ставили (нестабильный TF на 3.14) | frame-level эмбеддинги, слабее для музыкальной схожести | ❌ не практично на 3.14 |
| **essentia** (MusiCNN/Discogs-EffNet) | Apache-2.0 (библиотека) ✅ | лёгкая | ⚠️ есть cp314 wheel (2.1b6.dev1438), **но** pretrained-эмбеддинги требуют TensorFlow (rc только) | TF для эмбеддингов | не гоняли (блок TF) | handcrafted-часть работает | ⚠️ только handcrafted-fallback |
| **librosa / handcrafted** (MFCC+chroma+stats) | ISC ✅ | ~300 МБ стек (scipy/numba/sklearn) / ~0.7 ГБ | ✅ numba 0.68.1 cp314 есть, librosa 1.0.0 работает | numba, scipy, soundfile | **0.97 c/трек** (после прогрева JIT), RSS 671 МБ | ❌ **не различает треки**: косинусы 0.84–0.999 (blues↔techno = 0.991) — глобальные средние почти одинаковы у всей музыки | ⚠️ только как дешёвый сигнал (BPM/тембр-хинт), **не замена** |
| onnxruntime-путь (Xenova CLAP ONNX) | Apache-2.0 ✅ | меньше RSS | ✅ ort cp314 wheels есть | onnxruntime | не замеряли | — | 💡 оптимизация на будущее |

**Итог по списку ТЗ**: CLAP (музыкальные варианты) — ✅ проверен; MuLan — ❌ нет весов; VGGish — ❌ TF/3.14;
MuQ — ❌ NC; MERT — ❌ NC/гейт; audiocraft — ❌ NC; essentia — ⚠️ только handcrafted;
librosa — ✅ работает, но как fallback по качеству слаб (замер доказывает).

---

## 3. Реальная установка (что именно ставилось)

```bash
# CPU-сборка torch: на PyPI для linux только CUDA-колёса, у cp314 они тоже есть,
# но ставим явно с CPU-индекса (иначе потянет nvidia-* на ~3 ГБ):
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
                     --extra-index-url https://pypi.org/simple "torch==2.15.0+cpu"
.venv/bin/pip install "transformers==5.22.1" soundfile soxr     # аудио-декод реземпл
# для фолбэка (исследование):
.venv/bin/pip install librosa
```

* **Совместимость с Python 3.14** (проверено по PyPI wheels): `torch` cp314 — только с **2.9.0+** (поставлен 2.15.0+cpu),
  `tokenizers`/`safetensors`/`numba`/`llvmlite`/`scipy`/`soundfile`/`av`/`soxr`/`onnxruntime`/`essentia` — cp314-колёса есть;
  `transformers` — pure-python (любой). **`tensorflow` — только 2.22.0rc0** → VGGish-ветка закрыта.
* **Инвентаризация изменений venv**: `pip freeze` до/после → **+37 пакетов, 0 изменённых, 0 удалённых**
  (numpy 2.5.4, fastapi, uvicorn, requests, av, httpx, pydantic и т.д. — не тронуты). Аддитивно и безопасно для запущенного сервиса.
* Диск: `torch` 817 МБ + `transformers` 126 МБ (ядро ≈1 ГБ); ветка librosa ещё ~300 МБ (`scipy` 110, `numba` 34, `scikit-learn`…);
  venv целиком вырос до 1.9 ГБ. Кэш весов выбранной модели: 615 МБ
  (кэш HF заодно содержит веса всех проверенных моделей ≈4 ГБ — см. §9, рекомендуем удалить лишние).
* Альтернатива «свой venv» тоже проверена: `/tmp/emb-venv4` (torch 2.15.0+cpu + transformers 4.57.6) —
  понадобился для контроля (вывод про `larger_clap_music` битый **в обеих ветках**).

---

## 4. Данные пилота (реальные аудио, открытые лицензии)

Скачивали по 1 файлу жанра из Wikimedia Commons (не чужие альбомы, открытые лицензии):

| Файл | Длит. | Формат | Лицензия |
|---|---|---|---|
| `Chopin - Mazurka in A-flat major, B. 85 (Edward Neeman)` | 134.4 c | flac 44.1k | Public domain |
| `Matt's Blues (ISRC USUAN1100165)` | 166.7 c | mp3 44.1k | CC BY 3.0 |
| `MJ-PIA - Solarity (Acid-Techno)` | 187.7 c | ogg/vorbis 48k | CC BY-SA 4.0 |
| `MokkaMusic - Exotic Jam (Psychedelic 70s Rock)` | 265.6 c | ogg/vorbis 44.1k | CC BY 3.0 |

Файлы лежат в scratch (`.../cache/scratch/audio-emb/`), эмбеддинги — в `.../audio-emb/emb/`.
Декод: `soundfile` (libsndfile читает mp3/ogg/flac и **из `io.BytesIO`** — проверено: mp3 166.7 c за 0.28 c) → мосте-канал → `soxr` ресемпл 48 kHz.

---

## 5. Важная ловушка, найденная вживую: `rand_trunc`

`preprocessor_config.json` у обеих LAION-моделей: `truncation: "rand_trunc"`, `max_length_s: 10`.
То есть **`model.get_audio_features()` на треке любой длины молча берёт СЛУЧАЙНОЕ окно 10 секунд**:

* «full»-инференс трека 187 c занимает **0.6–0.9 c** и не отличается по времени от 30 c-отрывка — обрабатывается 10 c;
* повтор одного и того же файла даёт cos = **0.78…0.91** (каждый раз другое окно) → **нельзя кэшировать и сравнивать**;
* `truncation="fusion"` не спасает — там тоже 3 случайных кропа + даунсэмпл.

Поэтому рабочая схема — **наши собственные фиксированные окна по 10 c** (ровно `max_length_s`, кроп не срабатывает),
эмбеддинг каждого окна, затем mean-pool + L2:

* повторный запуск того же файла: **cos = 1.000** ✅
* `batch=1` vs `batch=6`: **cos = 1.000** ✅ (батчинг не меняет результат)
* наш `get_embedding()` против офлайн-пайплайна: **cos = 1.000** ✅

---

## 6. Замеры (полный протокол: `research/pilot/results.jsonl`, сводка `research/pilot/stats.txt`)

Все числа — с этого VPS, `threads=4` (машина: 4 vCPU без SMT; по данным ТЗ заявлено 2 vCPU —
замеры при `threads=2` и `threads=4` практически совпадают, поэтому в прод-рецепте хватает 2 потоков).

### 6.1 Скорость инференса (chunked, 10 c окна, все 4 трека)

| Конфигурация | Среднее на трек (134–266 c) | Разброс | На 1 окно | Пик RSS |
|---|---|---|---|---|
| **CLAP unfused, batch=1** | **9.59 c** | 5.88–13.60 c | 0.492 c | 646–669 МБ* |
| **CLAP unfused, batch=6** ✅ | **5.54 c** | 3.84–8.50 c | 0.286 c | 799–835 МБ |
| larger_clap_music (брак.), batch=1 | 13.56 c | 9.23–18.44 c | 0.707 c | 811–835 МБ |
| larger_clap_music_and_speech, batch=1 | 15.55 c | 12.03–21.97 c | 0.809 c | 812–835 МБ |
| `full`-режим (учитывается только 10 c!) | 0.55–0.92 c | — | — | 628–647 МБ |
| текстовый запрос (1 промпт) | **0.11 c** | — | — | ~800 МБ |
| загрузка модели (тёплый кэш) | 1.2–1.4 c | — | — | — |
| decode+resample трека | 0.4–1.1 c | — | — | — |

\* один замер 1304 МБ — процесс, в котором **одновременно качались веса** (615 МБ) — это и есть **абсолютный пик RAM пилота: 1304 МБ**.

### 6.2 Прод-API на «дружных к сервису» настройках (`threads=2`, batch=6, модель уже загружена)

```
MJ_PIA_Solarity_AcidTechno.ogg (187.7c)  -> 9.65 c  (включая первую загрузку модели ~1.4 c)
Matt_s_Blues.mp3 (166.7c)                -> 4.16 c
MokkaMusic_Exotic_Jam.ogg (265.6c)       -> 6.40 c
text "psychedelic rock guitar jam"       -> 0.109 c, cos(track)=0.528
peak RSS процесса                       -> 981 МБ
```

### 6.3 Пропускная способность (пересчёт из замеров на окно)

| Рецепт | Трек 180 c | Трек 240 c |
|---|---|---|
| batch=1 | 9.9 c/трек → **≈365 треков/час** | 12.8 c → **≈281/час** |
| **batch=6** ✅ | 6.1 c/трек → **≈586 треков/час** | 7.9 c → **≈458/час** |

С поправкой на параллельно работающий сервис (2 потока, `nice`) закладывать **≈300–450 треков/час** в фоновом режиме;
ночным батчем (сервис простаивает, 4 потока) — до ≈600/час.
100 тыс. треков ≈ 6 суток фонового расчёта при 300/час (реально быстрее: в базе много коротких треков).

### 6.4 Детерминизм и обман `full`-режима (chosen model)

| Пара | cos | Вывод |
|---|---|---|
| chunked повтор того же файла | **1.000** | ✅ стабильно, можно кэшировать |
| chunked batch=1 vs batch=6 | **1.000** | ✅ батчинг безопасен |
| full повтор того же файла | **0.782** | ❌ случайное окно 10 c |
| full vs chunked (тот же трек) | **0.874** | ❌ full-вектор ≈ одно окно, не весь трек |

---

## 7. Качество: правда-матрица

### 7.1 Похожесть по звучанию (chunked, 4 трека)

**`laion/clap-htsat-unfused` (выбранная)** — off-diagonal **0.141 … 0.738, mean 0.387**:

| | chopin | blues | techno | rock |
|---|---|---|---|---|
| chopin | 1.000 | 0.280 | **0.141** | 0.177 |
| blues | 0.280 | 1.000 | 0.391 | **0.738** |
| techno | 0.141 | 0.391 | 1.000 | 0.596 |
| rock | 0.177 | 0.738 | 0.596 | 1.000 |

Структура осмысленная: классика отдельнее всего (0.14–0.28), гитарные blues↔rock ближе всех (0.74),
ритмичные techno↔rock (0.60). Ровно то, что нужно для «похожесть по звучанию».

**`laion/larger_clap_music` (бракованный)** — off-diagonal **0.924 … 0.995**: эмбеддинги всех треков
практически вектор-константа (шум ≡ тишина ≡ падение), текст↔аудио ≈0.01 → отклонён.
**`larger_clap_music_and_speech`** — off-diagonal −0.051…0.321 (разреженно), но text→music слабый (см. 7.2).

### 7.2 Текст → музыка (выбранная модель, все 4 трека)

| Промпт | top-1 | cos | Комментарий |
|---|---|---|---|
| `classical solo piano piece` | **chopin** ✅ | 0.597 (2-й: 0.137) | уверенно |
| `slow blues with electric guitar` | **blues** ✅ | 0.500 (rock 0.496) | топ-1, но rock вплотную |
| `psychedelic rock guitar jam` | blues 0.642 / **rock 0.528 (#2)** ⚠️ | — | 4/4 в топ-2, не топ-1 (blues тоже гитарный) |
| `acid techno with four on the floor beat` | **techno** ✅ | 0.480 (след. 0.146) | уверенно |
| `cheerful pop song with female vocals` | (негативный) | 0.23–0.46 | общий CLAP слабо различает «жанр-не-жанр» — ожидаемо |

**3/4 топ-1, 4/4 топ-2.** Для музыкального теггинга/поиска достаточно; у `music_and_speech` те же промпты дают
скоры 0.01–0.06 и случайные ранги — он проигрывает.

### 7.3 Fallback (librosa, handcrafted 56-d): быстро, но не работает как замена

`0.97 c/трек` (после прогрева numba), RSS 671 МБ — но косинусы между треками **0.84–0.999**
(blues↔techno 0.991, techno↔rock 0.999): глобальные средние MFCC/chroma у всей музыки почти одинаковы.
Вывод: держать как дешёвый признак (BPM/тембр-хинт, блокировка явного мусора), **но не как substitute** —
CLAP и так помещается в RAM и считает 5 c/трек.

---

## 8. План интеграции в music-proxy

### 8.1 Зависимости (добавить в `requirements.txt` / деплой-скрипт)

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.15.0+cpu
transformers==5.22.1
soundfile==0.14.0
soxr==1.1.0
```
(все 4 уже реально стоят в .venv проекта; `numpy` уже есть; `librosa`-стек опционален и только если нужен fallback)

### 8.2 Кэш в SQLite (метаданные уже в проекте)

```sql
CREATE TABLE IF NOT EXISTS track_embeddings (
    track_id     TEXT PRIMARY KEY,      -- внешний ключ на трек
    model        TEXT NOT NULL,         -- 'clap-htsat-unfused@512' — смена модели => пересчёт
    dim          INTEGER NOT NULL,
    embedding    BLOB NOT NULL,         -- float32[512], L2-норма (2 КБ/трек)
    windows      INTEGER,               -- сколько окон 10 c ушло в вектор (для отладки)
    computed_at  TEXT NOT NULL
);
-- кандидаты в очередь предрасчёта:
-- SELECT t.id FROM tracks t LEFT JOIN track_embeddings e ON e.track_id = t.id
--  WHERE e.track_id IS NULL LIMIT 200;
```
Размер: 100 тыс. треков → 200 МБ таблицы. Текст-запросы кэшировать отдельно (TTL):
`text_embeddings(query_hash, model, embedding, computed_at)` — текст инференцируется за 0.11 c, но на частых
запросах всё равно кэш.

### 8.3 Три режима работы

1. **Batch-предрасчёт (основной)** — фоновая задача (asyncio-задача в процессе или отдельный `nice -n 19`
   процесс/cron-воркер): выбирает треки без эмбеддинга, грузит модель **один раз** (singleton), режет по
   10 c, `batch=6`, пишет батчами по 50–100 треков. Потоки: `torch.set_num_threads(2)` днём (не душить uvicorn),
   4 — ночью. Скорость: ≈300–450 треков/час в фоне.
2. **On-demand (ленивый)** — при первом обращении к треку (похожие треки/тег): тот же `get_embedding()`,
   лок в памяти (модель одна), ответ ≈5–7 c — для UI нормально, дальше в кэш. При bulk-запросах
   («похожие для списка») — считать батчем окон разных треков в одном forward (следующий шаг ускорения).
3. **Поиск** — эмбеддинги в памяти: `np.stack` всех векторов (100k×512×4 B = 200 МБ) + один matmul
   (косинусы = dot для L2-векторов), обновление инкрементально после записи в SQLite.
   Для >200–300k треков — перейти на `onnxruntime`-вариант CLAP или FAISS (HNSW), но это уже после метрик.

### 8.4 Код `get_embedding(track_bytes) -> list[float]` (проверен вживую)

Полностью рабочий файл: **`research/pilot/embedding_api.py`** (ядро — ниже; проверено:
`len=512`, `norm=1.0`, повторный вызов cos=1.0, против офлайн-пайплайна cos=1.0, текст-пара 0.48–0.53):

```python
import inspect, io, threading
import numpy as np, soundfile as sf, soxr, torch

MODEL_ID, SR, WINDOW_S, BATCH = "laion/clap-htsat-unfused", 48000, 10.0, 6
_lock = threading.Lock()
_model = _proc = _audio_kw = None

def _load():
    global _model, _proc, _audio_kw
    if _model is None:
        from transformers import ClapModel, ClapProcessor
        _proc = ClapProcessor.from_pretrained(MODEL_ID)
        _model = ClapModel.from_pretrained(MODEL_ID).eval()
        _audio_kw = "audio" if "audio" in inspect.signature(_proc.__call__).parameters else "audios"
    return _model, _proc

def get_embedding(track_bytes: bytes) -> list[float]:
    with _lock:                                   # одна инференция за раз -> RAM ограничен
        model, proc = _load()
        wav, sr = sf.read(io.BytesIO(track_bytes), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != SR:
            wav = soxr.resample(wav, sr, SR)
        win, parts = int(WINDOW_S * SR), []
        segs = [s for s in (wav[i:i+win] for i in range(0, len(wav), win)) if len(s) >= SR]
        for i in range(0, len(segs), BATCH):      # окна по 10 c -> без случайного кропа
            inputs = proc(**{_audio_kw: segs[i:i+BATCH]}, sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                f = model.get_audio_features(**inputs)
            if not torch.is_tensor(f):            # transformers>=5 возвращает объект
                f = f.pooler_output
            parts.append(torch.nn.functional.normalize(f.float(), dim=-1))
        vec = torch.nn.functional.normalize(torch.cat(parts).mean(dim=0), dim=-1)
        return vec.squeeze(0).tolist()            # float32, ||v||=1, dim=512

def get_text_embedding(text: str) -> list[float]:
    with _lock:
        model, proc = _load()
        with torch.inference_mode():
            f = model.get_text_features(**proc(text=[text], return_tensors="pt"))
        if not torch.is_tensor(f):
            f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).squeeze(0).tolist()
```

Похожесть: `cos = float(np.dot(e1, e2))` (векторы уже L2-нормированы) — без всяких библиотек.

### 8.5 Бюджет ресурсов

| Параметр | Значение |
|---|---|
| RAM resident worker | 0.65–0.85 ГБ (пик 1.3 ГБ при холодной загрузке весов) — на 8 ГБ VPS с ~4 ГБ свободных **один** воркер влезает |
| Диск | 615 МБ веса (один раз) + 2 КБ/трек в SQLite |
| CPU | 2 потока во время работы сервиса, 4 — ночью; ≈0.29 c на 10 c аудио (batch=6) |
| Пропускная | 300–450 треков/час фоном, до ≈600/час ночью |
| Первый on-demand запрос | ≈1.4 c (загрузка модели) + ≈5–7 c (трек) |

---

## 9. Что проверено и что осталось

✅ Установка в .venv проекта под Python 3.14 (аддитивная, сервис не тронут) · скачивание весов ·
инференс на 4 реальных треках · замеры времени/RAM · детерминизм · text→music · сравнение с 3 альтернативами
и 2 библиотечными фолбэками · живой API-прототип.

⚠️ **Остаточные риски / рекомендации**
1. `laion/larger_clap_music` — «музыкальный» CLAP формально Apache-2.0, но его публичные веса дают вырожденные
   эмбеддинги (воспроизводится на 4.57.6 и 5.22.1) — не использовать, пока не появится рабочая альтернатива.
   Если очень нужен music-specific — кандидат на проверку: `mtg-upf/clap_*_music` (у MTG обычно **NC**-лицензия — сначала лицензия!).
2. Кэш HF сегодня 8.9 ГБ суммарно (`du -sh /root/.cache/huggingface`): часть — **чужие** модели
   (`tts-hub/XTTS-v2`, `Systran/faster-whisper-*`, их трогать нельзя), часть — 4 проверенные нами CLAP-модели
   (единицы ГБ; нужна только `clap-htsat-unfused`, её веса 615 МБ). Плюс pip-cache 3.8 ГБ.
   Можно освободить несколько ГБ: `pip cache purge` + удалить каталоги
   `~/.cache/huggingface/hub/models--laion--larger_clap_music*`, `--clap-htsat-fused`,
   `--biancataz--*` (оставить `--clap-htsat-unfused`; недостающее перекачивается за ~10 c).
   Мы это **не делали** (не входило в задачу), диск: 19 ГБ свободно.
3. В venv дополнительно стоит librosa-стек (исследование фолбэка, ~300 МБ) — если fallback не нужен:
   `pip uninstall librosa numba llvmlite scipy scikit-learn` (аддитивно ничего не ломает).
4. Скорости чувствительны к фоновой нагрузке VPS (разброс ±30% между прогонами) — финальные числа брать
   с `stats.txt` после установки в прод.
5. Если позже понадобится масштаб >200k треков или <1 c on-demand — переход на ONNX CLAP
   (`Xenova/larger_clap_music` на ONNX уже существует для CLAP-семейства) или `onnxruntime` + int8.

---

## Приложения (артефакты пилота)

* `research/pilot/embed_case.py` — один кейс в отдельном процессе (чистый замер времени и VmHWM)
* `research/pilot/embedding_api.py` — **готовый к интеграции API** (`get_embedding` / `get_text_embedding`)
* `research/pilot/text_case.py` — ранжирование «текст → треки»
* `research/pilot/analyze.py` — косинус-матрицы и проверки детерминизма
* `research/pilot/probe_degenerate.py` — ловушка вырожденных эмбеддингов (шум/тишина/синусы)
* `research/pilot/bench_fallback.py` — бенчмарк librosa-fallback
* `research/pilot/run_matrix.sh` — сценарий прогона; `results.jsonl` (28 замеров), `stats.txt` — сводка
* лицензии тестовых аудио — в §4 (Wikimedia Commons: PD / CC BY / CC BY-SA)
