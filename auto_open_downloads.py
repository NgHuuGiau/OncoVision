#!/usr/bin/env python3
"""
Mo trinh duyet tu dong cho tat ca link tai models.
Chay: python auto_open_downloads.py
"""
import webbrowser
import time

# GitHub Releases - Direct links
github_links = [
    "https://github.com/HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch/releases",
    "https://github.com/DIAGNijmegen/TDSC/releases",
    "https://github.com/PICAI/PICAI-grand-challenge/releases",
    "https://github.com/josedolz/LiviaNET/releases",
    "https://github.com/yhygao/CBIM-Medical-Image-Segmentation/releases",
]

# Kaggle - Can login
kaggle_links = [
    "https://www.kaggle.com/datasets/andrewmvd/sipakmed/models",
]

# Roboflow Search - Ban phai search them tren trang
roboflow_searches = [
    "https://universe.roboflow.com/search?q=Brain%20Tumor%20YOLOv8",
    "https://universe.roboflow.com/search?q=Lung%20Nodule%20YOLOv8%20LIDC",
    "https://universe.roboflow.com/search?q=Liver%20Tumor%20YOLOv8%20LiTS",
    "https://universe.roboflow.com/search?q=Kidney%20Tumor%20YOLOv8%20KiTS23",
    "https://universe.roboflow.com/search?q=Pancreas%20Tumor%20YOLOv8",
    "https://universe.roboflow.com/search?q=Prostate%20Lesion%20YOLOv8%20PICAI",
    "https://universe.roboflow.com/search?q=Cervical%20Lesion%20YOLOv8%20Colposcopy",
    "https://universe.roboflow.com/search?q=Breast%20Mammogram%20YOLOv8",
]

def open_tabs(links, label, delay=0.5):
    print(f"\n=== Opening {label} ({len(links)} tabs) ===")
    for i, url in enumerate(links, 1):
        print(f"  [{i}/{len(links)}] {url}")
        webbrowser.open_new_tab(url)
        time.sleep(delay)

if __name__ == "__main__":
    print("=" * 60)
    print("AUTO OPEN DOWNLOAD LINKS - ONCOVISION 16 MODELS")
    print("=" * 60)
    print("Script se mo trinh duyet cac tab download.")
    print("Ban chi viec: Click Download -> Rename -> Paste vao folder")
    print("=" * 60)
    
    open_tabs(github_links, "GitHub Releases (5 classifiers)")
    open_tabs(kaggle_links, "Kaggle (1 classifier - can login)")
    open_tabs(roboflow_searches, "Roboflow Search (8 YOLO detectors)")
    
    print("\n" + "=" * 60)
    print("DA MO XONG TAT CA TAB!")
    print("=" * 60)
    print("\nHUONG DAN:")
    print("1. GitHub Releases: Tab 'Assets' -> Download file .pt/.pth")
    print("2. Kaggle: Login -> Tab 'Models' -> Download")
    print("3. Roboflow: Search -> Tab 'Models' -> Download -> YOLOv8 PyTorch -> best.pt")
    print("\nSAU KHI TAI VE:")
    print("  - Doi ten file .pt -> detector.pt (YOLO) hoac classifier.pt")
    print("  - Copy vao dung folder models/pretrained/cancers/...")
    print("\nKIEM TRA: python check_status.py")