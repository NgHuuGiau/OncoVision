import urllib.request, json

# Search for more models for missing components
TERMS = [
    # More general cancer detection
    "cancer detection pytorch",
    "tumor detection pytorch",
    "cancer classification pytorch",
    "tumor classification pytorch",
    # Segmentation
    "medical image segmentation pytorch",
    "organ segmentation pytorch",
    "tumor segmentation pytorch",
    # YOLO medical
    "yolo medical",
    "yolo tumor detection",
    "yolo cancer detection",
    # Specific organs
    "liver tumor segmentation",
    "kidney tumor segmentation", 
    "thyroid nodule segmentation",
    "pancreas tumor segmentation",
    "prostate cancer segmentation",
    "liver lesion detection",
    "lung nodule detection",
    "breast tumor detection",
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