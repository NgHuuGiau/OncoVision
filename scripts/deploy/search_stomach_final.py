import urllib.request, json

# Search for more specific medical models
TERMS = [
    "stomach cancer 24 channel pytorch",
    "gastric cancer efficientnet",
    "stomach cancer efficientnet-b0",
    "gastric cancer efficientnet-b0",
    "stomach cancer histopathology classification",
    "gastric histopathology pytorch",
    "stomach cancer classification resnet",
    "gastric cancer resnet50",
    "stomach adenocarcinoma pytorch",
    "gastric cancer segmentation pytorch",
]

for term in TERMS:
    try:
        url = f"https://huggingface.co/api/models?search={term.replace(' ', '+')}&limit=10&sort=downloads"
        d = json.load(urllib.request.urlopen(url, timeout=20))
        if d:
            print(f"=== {term} ===")
            for m in d[:5]:
                print(f"  {m['id']} | dl={m.get('downloads',0)} | likes={m.get('likes',0)}")
        else:
            print(f"=== {term} === EMPTY")
    except Exception as e:
        print(f"{term}: ERR {e}")