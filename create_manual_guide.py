#!/usr/bin/env python3
import os

base = 'models/pretrained/cancers'
folders = [
    'brain', 'lung/ct', 'liver/ct', 'kidney/ct', 'pancreas/ct',
    'prostate/mri', 'cervical/colposcopy', 'breast/mammogram',
    'thyroid/ultrasound', 'stomach/endoscopy'
]
for f in folders:
    os.makedirs(os.path.join(base, f), exist_ok=True)
print('Folders created')

# Create manual download guide
guide = """MANUAL DOWNLOAD GUIDE - 16 MODELS
=====================================

ROBOFLOW YOLO DETECTORS (8 models)
----------------------------------
Go to: https://universe.roboflow.com
Search each keyword -> Model tab -> Download -> YOLOv8 PyTorch -> best.pt -> rename detector.pt

1. brain/detector.pt          <- Search: "Brain Tumor YOLOv8"
2. lung/ct/detector.pt        <- Search: "Lung Nodule YOLOv8 LIDC"
3. liver/ct/detector.pt       <- Search: "Liver Tumor YOLOv8 LiTS"
4. kidney/ct/detector.pt      <- Search: "Kidney Tumor YOLOv8 KiTS23"
5. pancreas/ct/detector.pt    <- Search: "Pancreas Tumor YOLOv8"
6. prostate/mri/detector.pt   <- Search: "Prostate Lesion YOLOv8 PICAI"
7. cervical/colposcopy/detector.pt <- Search: "Cervical Lesion YOLOv8 Colposcopy"
8. breast/mammogram/detector.pt <- Search: "Breast Mammogram YOLOv8"

CLASSIFIERS (6 models)
---------------------
Download .pt/.pth from GitHub Releases or Kaggle -> rename classifier.pt

1. brain/classifier.pt              <- https://github.com/HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch/releases
2. thyroid/ultrasound/classifier.pt <- https://github.com/DIAGNijmegen/TDSC/releases
3. prostate/mri/classifier.pt       <- https://github.com/PICAI/PICAI-grand-challenge/releases
4. cervical/pap/classifier.pt       <- https://www.kaggle.com/datasets/andrewmvd/sipakmed (Models tab)
5. stomach/endoscopy/classifier.pt  <- https://github.com/josedolz/LiviaNET/releases
6. liver/ct/classifier.pt           <- https://github.com/yhygao/CBIM-Medical-Image-Segmentation/releases

AFTER DOWNLOAD:
---------------
Place each file in models/pretrained/cancers/<target_path>
Then run: python check_status.py
"""

with open('MANUAL_DOWNLOAD_GUIDE.txt', 'w', encoding='utf-8') as f:
    f.write(guide)
print('Guide created: MANUAL_DOWNLOAD_GUIDE.txt')