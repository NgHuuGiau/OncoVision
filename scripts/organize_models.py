#!/usr/bin/env python3
"""Organize OncoVision model files."""
import os, shutil
from pathlib import Path
BASE = Path(r"D:\Project\OncoVision\models\pretrained\cancers")
def ensure_dir(p): p.mkdir(parents=True, exist_ok=True); return p
def move_file(s, d, desc):
    if s.exists():
        ensure_dir(d.parent)
        if d.exists(): print(f"  [SKIP] {desc}"); return False
        shutil.move(str(s), str(d)); print(f"  [MOVE] {desc}"); return True
    print(f"  [MISSING] {desc}: {s}"); return False
def clean_cache(p):
    for c in p.rglob(".cache"):
        try: shutil.rmtree(c, ignore_errors=True)
        except: pass
def main():
    print("=" * 70); print("ONCOVISION MODEL ORGANIZATION"); print("=" * 70)
    print("\n[1] Creating dirs...")
    for d in ["brain","breast/ultrasound","breast/mammogram","cervical/pap","cervical/colposcopy",
              "colorectal/colonoscopy","kidney","kidney/ct","liver","liver/ct","lung/ct","lung/xray",
              "pancreas","pancreas/ct","prostate/mri","stomach/endoscopy","thyroid/ultrasound"]:
        ensure_dir(BASE / d)
    print("\n[2] Moving files...")
    moves = [
        (BASE/"brain/brats_mri_segmentation/brats_mri_segmentation/models/model.pt", BASE/"brain/volume.pt", "Brain vol"),
        (BASE/"brain/brats_mri_segmentation/brats_mri_segmentation/configs/inference.json", BASE/"brain/volume.json", "Brain cfg"),
        (BASE/"liver/spleen_ct_segmentation/models/model.ts", BASE/"liver/volume.pt", "Liver vol"),
        (BASE/"liver/spleen_ct_segmentation/configs/inference_trt.json", BASE/"liver/volume.json", "Liver cfg"),
        (BASE/"spleen/ct/segmentation.pt", BASE/"spleen/volume.pt", "Spleen vol"),
    ]
    for s,d,desc in moves: move_file(s,d,desc)
    print("\n[3] Stomach check...")
    sd = BASE/"stomach/endoscopy/detector.pt"
    if sd.exists(): print(f"  Stomach: {sd.stat().st_size/1024/1024:.1f}MB (COCO - NEEDS REPLACE)")
    print("\n[4] Clean cache..."); clean_cache(BASE)
    print("\n[5] Remove nested...")
    for n in ["brain/brats_mri_segmentation","liver/spleen_ct_segmentation","kidney/pancreas_ct_dints_segmentation","pancreas/pancreas_ct_dints_segmentation"]:
        p=BASE/n
        if p.exists(): shutil.rmtree(p,ignore_errors=True); print(f"  [REMOVE] {n}")
    for root,dirs,files in os.walk(BASE,topdown=False):
        for d in dirs:
            dp=Path(root)/d
            try:
                if not any(dp.iterdir()): dp.rmdir(); print(f"  [REMOVE EMPTY] {dp}")
            except: pass
    print("\n[6] Verify...")
    expected={"brain":["detector.pt","volume.pt","volume.json"],"breast/ultrasound":["classifier.pt","detector.pt"],
        "breast/mammogram":["detector.pt"],"cervical/pap":["classifier.pt"],"cervical/colposcopy":["detector.pt"],
        "colorectal/colonoscopy":["detector.pt"],"kidney":["volume.pt","volume.json"],"kidney/ct":["detector.pt"],
        "liver":["volume.pt","volume.json"],"liver/ct":["detector.pt","classifier.pt"],"lung/ct":["detector.pt"],
        "lung/xray":["classifier.pt"],"pancreas":["volume.pt","volume.json"],"pancreas/ct":["detector.pt"],
        "prostate/mri":["detector.pt","classifier.pt"],"stomach/endoscopy":["detector.pt","classifier.pt"],
        "thyroid/ultrasound":["classifier.pt"]}
    ready=0; missing=[]
    for folder,files in expected.items():
        fp=BASE/folder; fp.mkdir(parents=True,exist_ok=True)
        for f in files:
            fpp=fp/f
            if fpp.exists(): print(f"  [OK] {folder}/{f} ({fpp.stat().st_size/1024/1024:.1f}MB)"); ready+=1
            else: print(f"  [MISSING] {folder}/{f}"); missing.append(f"{folder}/{f}")
    mp=BASE.parent/"modality_classifier.pt"
    if mp.exists(): print(f"  [OK] modality/modality_classifier.pt ({mp.stat().st_size/1024/1024:.1f}MB)"); ready+=1
    else: print(f"  [MISSING] modality/modality_classifier.pt"); missing.append("modality/modality_classifier.pt")
    print("-"*50); print(f"[SUMMARY] {ready} ready, {len(missing)} missing")
    if missing:
        print("\n[MISSING FILES - need manual download]:"); [print(f"   - {m}") for m in missing]
    print("="*70); print("DONE"); print("="*70)
if __name__=="__main__": main()