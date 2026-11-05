import urllib.request, json

TERMS = [
    "liver cancer detection yolo",
    "prostate cancer detection yolo",
    "pancreas cancer detection yolo",
    "cervical cancer detection yolo",
    "kidney cancer detection yolo",
    "thyroid cancer detection yolo",
    "brain tumor detection yolo",
    "breast cancer detection yolo",
    "stomach cancer detection yolo",
    "colorectal cancer detection yolo",
    "prostate cancer segmentation",
    "liver tumor segmentation",
    "pancreas segmentation",
    "breast cancer segmentation",
]

for term in TERMS:
    try:
        url = f"https://huggingface.co/api/models?search={term.replace(' ', '+')}&limit=8&sort=downloads"
        d = json.load(urllib.request.urlopen(url, timeout=20))
        if d:
            print("=== " + term + " ===")
            for m in d[:5]:
                print("  " + m["id"] + " | dl=" + str(m.get("downloads",0)) + " | likes=" + str(m.get("likes",0)))
        else:
            print("=== " + term + " === EMPTY")
    except Exception as e:
        print(term + ": ERR " + str(e))