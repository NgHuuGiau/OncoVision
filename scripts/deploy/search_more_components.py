import urllib.request, json

# Search for more volume 3D models and detectors
TERMS = [
    "liver 3d segmentation pytorch",
    "prostate mri segmentation pytorch",
    "kidney segmentation pytorch",
    "pancreas 3d segmentation pytorch",
    "liver tumor segmentation 3d",
    "prostate cancer segmentation 3d",
    "kidney tumor segmentation 3d",
    "thyroid nodule segmentation 3d",
    "lung nodule detection 3d",
    "brain tumor segmentation 3d",
    "yolo liver detection",
    "yolo kidney detection",
    "yolo pancreas detection",
    "yolo thyroid detection",
    "yolo brain tumor detection",
    "medical yolo detection",
    "organ detection yolo",
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