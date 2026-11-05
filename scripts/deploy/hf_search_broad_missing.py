import urllib.request, json

TERMS = [
    "liver cancer",
    "hepatocellular carcinoma",
    "stomach cancer",
    "gastric cancer",
    "thyroid cancer",
    "thyroid nodule",
]

for term in TERMS:
    try:
        url = f"https://huggingface.co/api/models?search={term.replace(' ', '+')}&limit=15&sort=downloads"
        d = json.load(urllib.request.urlopen(url, timeout=20))
        if d:
            print(f"=== {term} ===")
            for m in d[:8]:
                print(f"  {m['id']} | dl={m.get('downloads',0)} | likes={m.get('likes',0)}")
        else:
            print(f"=== {term} === EMPTY")
    except Exception as e:
        print(f"{term}: ERR {e}")