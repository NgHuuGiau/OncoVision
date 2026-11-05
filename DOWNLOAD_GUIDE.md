# OncoVision - Missing Models Download Guide

## 📋 Summary
- **9/29 models ready** (auto-processed)
- **18 models need manual download**

---

## 🔴 PRIORITY 1: COLORECTAL DETECTOR
**File:** `models/pretrained/cancers/colorectal/colonoscopy/detector.pt`
**Source:** [Roboflow: goktug14/yolov5_kvasir_polyp](https://universe.roboflow.com/goktug14/yolov5_kvasir_polyp)
- Tab **Model** → **Download Weights** → **YOLOv5 PyTorch** → `best.pt`
- Rename to `detector.pt` → place in `models/pretrained/cancers/colorectal/colonoscopy/`

---

## 🔴 ROBOFLOW YOLO DETECTORS (8 models)

| # | Organ | Target File | Search Keywords |
|-----|-------|-------------|-----------------|
| 1 | Brain Tumor | `brain/detector.pt` | "Brain Tumor Detection YOLOv8" |
| 2 | Lung CT Nodule | `lung/ct/detector.pt` | "Lung Nodule Detection YOLOv8" / "LIDC YOLO" |
| 3 | Liver Tumor | `liver/ct/detector.pt` | "Liver Tumor YOLO" / "LiTS YOLO" |
| 4 | Kidney Tumor | `kidney/ct/detector.pt` | "Kidney Tumor YOLO" / "KiTS23 YOLO" |
| 5 | Pancreas Tumor | `pancreas/ct/detector.pt` | "Pancreas Tumor YOLO" / "MSD Pancreas" |
| 6 | Prostate Lesion | `prostate/mri/detector.pt` | "Prostate Lesion YOLO" / "PICAI YOLO" |
| 7 | Cervical Lesion | `cervical/colposcopy/detector.pt` | "Cervical Lesion YOLO" / "Colposcopy YOLO" |
| 8 | Stomach Polyp | `stomach/endoscopy/detector.pt` | **Reuse colorectal** ✅ |

> **How:** Go to [universe.roboflow.com](https://universe.roboflow.com) → Search keywords → Tab **Model** → **Download Weights** → **YOLOv8 PyTorch** → `best.pt` → rename `detector.pt`

---

## 🔴 6 CLASSIFIERS (Kaggle / GitHub Releases)

| # | Organ | Target File | Download Links |
|---|-------|-------------|----------------|
| 1 | **Brain 4-class** | `brain/classifier.pt` | Kaggle: `navoneel/brain-mri-images-for-brain-tumor-detection` (Models tab) / GitHub: `HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch/releases` / HF: `him-09/brain-tumor-cl` (gated) |
| 2 | **Thyroid US** | `thyroid/ultrasound/classifier.pt` | GitHub: `DIAGNijmegen/TDSC/releases` |
| 3 | **Prostate MRI** | `prostate/mri/classifier.pt` | GitHub: `PICAI/PICAI-grand-challenge/releases` |
| 4 | **Cervical Pap** | `cervical/pap/classifier.pt` | Kaggle: `andrewmvd/sipakmed` (Models tab) / GitHub: `andreaped/SIPaKMeD/releases` |
| 5 | **Stomach Endoscopy** | `stomach/endoscopy/classifier.pt` | GitHub: `josedolz/LiviaNET/releases` |
| 6 | **Liver CT** | `liver/ct/classifier.pt` | GitHub: `yhygao/CBIM-Medical-Image-Segmentation/releases` |

> **How:** GitHub → **Releases** tab → download `.pt`/`.pth` → rename `classifier.pt`  
> **Kaggle:** Dataset page → **Models** tab → download `.pt`/`.pth` → rename `classifier.pt`

---

## 🔵 3D MODELS (MONAI) - Run PowerShell Admin

### Pancreas 3D (~1GB)
```powershell
New-Item -ItemType Directory -Force -Path "models/pretrained/cancers/pancreas"
Invoke-WebRequest -Uri "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/pancreas_ct_dints_segmentation_v0.4.3.zip" -OutFile "pancreas.zip"
Expand-Archive -Path "pancreas.zip" -DestinationPath "temp_pancreas" -Force
Copy-Item "temp_pancreas/pancreas_ct_dints_segmentation/models/model.pt" "models/pretrained/cancers/pancreas/volume.pt"
Copy-Item "temp_pancreas/pancreas_ct_dints_segmentation/configs/inference.json" "models/pretrained/cancers/pancreas/volume.json"
Remove-Item "pancreas.zip", "temp_pancreas" -Recurse -Force
```

### Kidney 3D (~325MB)
```powershell
New-Item -ItemType Directory -Force -Path "models/pretrained/cancers/kidney"
Invoke-WebRequest -Uri "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/renalStructures_UNEST_segmentation_v0.2.2.zip" -OutFile "kidney.zip"
Expand-Archive -Path "kidney.zip" -DestinationPath "temp_kidney" -Force
Copy-Item "temp_kidney/renalStructures_UNEST_segmentation/models/model.pt" "models/pretrained/cancers/kidney/volume.pt"
Copy-Item "temp_kidney/renalStructures_UNEST_segmentation/configs/inference.json" "models/pretrained/cancers/kidney/volume.json"
Remove-Item "kidney.zip", "temp_kidney" -Recurse -Force
```

---

## 🟢 AUTO - RUN ONCE
```cmd
python scripts\train_modality_classifier.py --epochs 10 --batch-size 32
```
> Auto-downloads MedNIST, trains ResNet18 (~10 min GPU) → creates `models/pretrained/modality_classifier.pt`

---

## ✅ VERIFICATION
```cmd
python scripts\verify_full_pipeline.py
```

---

## 📋 CHECKLIST

- [ ] Colorectal detector (Roboflow)
- [ ] Brain detector (Roboflow)
- [ ] Lung CT detector (Roboflow)
- [ ] Liver detector (Roboflow)
- [ ] Kidney detector (Roboflow)
- [ ] Pancreas detector (Roboflow)
- [ ] Prostate detector (Roboflow)
- [ ] Cervical detector (Roboflow)
- [ ] Brain classifier (GitHub/Kaggle)
- [ ] Thyroid classifier (GitHub)
- [ ] Prostate classifier (GitHub)
- [ ] Cervical classifier (Kaggle/GitHub)
- [ ] Stomach classifier (GitHub)
- [ ] Liver classifier (GitHub)
- [ ] Pancreas 3D (PowerShell)
- [ ] Kidney 3D (PowerShell)
- [ ] Modality classifier (Auto-train)

---

## 📊 SUMMARY
| Status | Count |
|--------|-------|
| ✅ Ready | 9 |
| 🔴 Roboflow YOLO | 9 (1 done + 8 manual) |
| 🔴 Classifiers | 6 |
| 🔵 3D MONAI | 2 |
| 🟢 Auto | 1 |
| **Total** | **27** |

**Next:** Run `python scripts/verify_full_pipeline.py` after downloads complete.