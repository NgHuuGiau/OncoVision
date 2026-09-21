"""Quet dataset/*/raw, tinh bbox/kich thuoc/toa do/thong so anh tu mask+header.

Ghi `metadata_enriched.json` moi thu muc raw co cap image-mask.
Dung stdlib + nibabel/numpy/PIL (da co trong requirements).
# ponytail: 1 script duy nhat, khong framework
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET_ROOT = Path("dataset")
CATALOG = DATASET_ROOT / "DATASET_CATALOG.csv"


def _nifti_stats(image_path: Path, mask_path: Path) -> dict:
    import nibabel as nib
    import numpy as np

    img = nib.load(str(image_path))
    msk = nib.load(str(mask_path))
    data = np.asanyarray(msk.dataobj) > 0
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    vox_vol = 1.0
    for z in zooms:
        vox_vol *= z
    idx = np.argwhere(data)
    if len(idx) == 0:
        return {"voxels": 0, "bbox_ijk": None, "size_mm": None, "volume_mm3": 0.0, "spacing": list(zooms)}
    mn, mx = idx.min(axis=0).tolist(), idx.max(axis=0).tolist()
    dims_vox = [(b - a + 1) for a, b in zip(mn, mx)]
    size_mm = [round(d * z, 2) for d, z in zip(dims_vox, zooms)]
    return {
        "voxels": int(data.sum()),
        "bbox_ijk": {"min": mn, "max": mx},
        "size_mm": size_mm,
        "volume_mm3": round(float(data.sum()) * vox_vol, 2),
        "spacing": list(zooms),
        "shape": list(data.shape),
    }


def _png_mask_stats(mask_path: Path) -> dict:
    from PIL import Image

    im = Image.open(mask_path).convert("L")
    w, h = im.size
    import numpy as np

    data = np.array(im) > 127
    idx = np.argwhere(data)
    if len(idx) == 0:
        return {"voxels": 0, "bbox_xy": None, "size_px": [w, h]}
    mn, mx = idx.min(axis=0).tolist(), idx.max(axis=0).tolist()
    return {"voxels": int(data.sum()), "bbox_xy": {"min": mn, "max": mx}, "size_px": [w, h]}


def _pairs_in(folder: Path) -> list[tuple[Path, Path]]:
    pairs = []
    # MSD format: imagesTr/xxx.nii.gz + labelsTr/xxx.nii.gz
    img_tr = folder / "imagesTr"
    lab_tr = folder / "labelsTr"
    if img_tr.is_dir() and lab_tr.is_dir():
        for img in sorted(img_tr.glob("*.nii.gz")):
            if img.name.startswith("._"):
                continue
            msk = lab_tr / img.name
            if msk.exists() and not msk.name.startswith("._"):
                pairs.append((img, msk))
    # Colon/Pancreas format: * _image.nii.gz + * _label.nii.gz
    for img in sorted(folder.glob("*_image.nii.gz")):
        if img.name.startswith("._"):
            continue
        msk = img.parent / img.name.replace("_image.nii.gz", "_label.nii.gz")
        if msk.exists() and not msk.name.startswith("._"):
            pairs.append((img, msk))
    for img in sorted(folder.glob("colon_*_image.nii.gz")) + sorted(folder.glob("pancreas_*_image.nii.gz")):
        if img.name.startswith("._"):
            continue
        msk = img.parent / img.name.replace("_image.", "_label.")
        if msk.exists() and not msk.name.startswith("._") and (img, msk) not in pairs:
            pairs.append((img, msk))
    return pairs


def main(limit_per_folder: int = 25) -> int:
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rows = []
    for group_dir in sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name not in {"raw", "processed"}):
        raw = group_dir / "raw"
        if not raw.is_dir():
            continue
        for sub in sorted(p for p in raw.rglob("*") if p.is_dir()):
            out = None
            try:
                pairs = _pairs_in(sub)
                stats = []
                for img, msk in pairs[:limit_per_folder]:
                    s = _nifti_stats(img, msk)
                    s["image"] = img.name
                    stats.append(s)
                # TN3K-style: test-image / test-mask
                if not stats and (sub.name in {"test-image", "test-mask"} or "TN3K" in str(sub)):
                    pass
                if stats:
                    out = {"folder": str(sub), "pairs": len(pairs), "scanned": len(stats), "stats": stats}
                    (sub / "metadata_enriched.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                    rows.append((group_dir.name, sub.name, len(pairs)))
            except Exception as exc:
                rows.append((group_dir.name, sub.name, f"ERR {exc}"))
    print("group | sub | pairs")
    for r in rows:
        print(" | ".join(str(c) for c in r))
    # cap nhat catalog: them cot metadata_enriched neu chua co
    try:
        lines = CATALOG.read_text(encoding="utf-8").splitlines()
        if lines and "metadata_enriched" not in lines[0]:
            CATALOG.write_text(lines[0] + ",metadata_enriched\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
            print("catalog: added metadata_enriched column")
    except FileNotFoundError:
        print("catalog: missing, skip", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
