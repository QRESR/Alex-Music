import json, urllib.parse, urllib.request, sqlite3
from pathlib import Path
B = "http://127.0.0.1:8011"
queries = ["Linkin Park", "Eminem", "Баста", "Кино", "The Cranberries", "Kate Bush",
           "Сплин", "Мумий Тролль", "Radiohead", "Linkin Park Numb", "Eminem Stan",
           "Дайте танк", "Oxxxymiron", "Miyagi", "новогодняя музыка", "саундтрек",
           "рэп подборка", "Linkin Park In the End", "Linkin Park Faint"]
ids = []
for q in queries:
    try:
        url = f"{B}/search?q={urllib.parse.quote(q)}&limit=10&mix=1&source=yandex"
        d = json.loads(urllib.request.urlopen(url, timeout=45).read())
        for t in d.get("results", []):
            tid = t.get("id", "")
            if tid.startswith("ym_"):
                ids.append(tid)
    except Exception:
        pass
uniq = list(dict.fromkeys(ids))
c = sqlite3.connect("data/embeddings.db")
have = {r[0] for r in c.execute("SELECT track_id FROM track_embeddings")}
miss = [i for i in uniq if i not in have]
Path("/tmp/top_ids2.txt").write_text("\n".join(miss))
print(f"collected={len(uniq)} missing={len(miss)}")
print("sample:", miss[:15])
