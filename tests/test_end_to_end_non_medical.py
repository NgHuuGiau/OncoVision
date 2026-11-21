#!/usr/bin/env python
"""Test end-to-end: chạy ảnh chó mèo qua full pipeline analyze_image."""
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical.pipeline.analyzer import MedicalImageAnalyzer
from medical.pipeline.config import build_default_medical_analyzer_config

output_dir = project_root / "output" / "test_images"
output_dir.mkdir(parents=True, exist_ok=True)


def create_cat_like_image(width=400, height=400):
    """Tạo ảnh giống chó mèo (nhiều màu, texture phức tạp)."""
    arr = np.random.randint(50, 220, (height, width, 3), dtype=np.uint8)
    for _ in range(30):
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        color = np.random.randint(100, 200, 3, dtype=np.uint8)
        arr[y1:y2, x1:x2, :] = color
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "cat_test.jpg"
    img.save(path, quality=85)
    return path


def create_landscape_image(width=500, height=350):
    """Tạo ảnh phong cảnh."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[0:height//3, :, :] = [135, 206, 235]
    arr[height//3:2*height//3, :, :] = [34, 139, 34]
    arr[2*height//3:, :, :] = [139, 69, 19]
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "landscape_test.jpg"
    img.save(path, quality=85)
    return path


def create_selfie_image(width=300, height=400):
    """Tạo ảnh selfie."""
    arr = np.full((height, width, 3), [200, 180, 170], dtype=np.uint8)
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    mask = ((y - cy) / 80) ** 2 + ((x - cx) / 60) ** 2 <= 1
    arr[mask] = [224, 180, 150]
    noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "selfie_test.jpg"
    img.save(path, quality=85)
    return path


def test_analyze_image(name, path):
    """Test 1 ảnh qua full pipeline."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"File: {path}")
    print(f"{'='*70}")

    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        result = analyzer.analyze_image(
            image_path=str(path),
            patient_code="TEST001",
            target_key=None,  # Tự động xác định
        )
        
        print("\n[Pipeline Result]")
        print(f"  Risk level: {result.risk_level}")
        print(f"  Suspected malignant: {result.suspected_malignant}")
        print(f"  Modality: {result.modality}")
        print(f"  Body region: {result.body_region}")
        print(f"  Average confidence: {result.average_confidence:.4f}")
        print(f"  Model name: {result.model_name}")
        print(f"  Recommendation: {result.recommendation[:200]}")

        stage_conf = result.stage_confidences
        if stage_conf:
            print("\n  [Stage Confidences]")
            for k, v in stage_conf.items():
                print(f"    {k}: {v:.4f}")

        if result.detections:
            print(f"\n  [Detections: {len(result.detections)}]")
            for i, det in enumerate(result.detections[:3]):
                print(f"    #{i+1}: label={det.label}, confidence={det.confidence:.4f}")

        if result.quality_warnings:
            print(f"\n  [Warnings: {len(result.quality_warnings)}]")
            for w in result.quality_warnings[:5]:
                print(f"    - {w}")

        # Đọc report JSON để xem chi tiết
        if result.report_json_path.exists():
            with open(result.report_json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            print("\n  [Report JSON]")
            for k in ("status", "target_key", "modality", "confidence", "error_code", "classification_labels"):
                if k in report:
                    print(f"    {k}: {report[k]}")

        # Đánh giá
        print("\n  [Đánh giá]")
        avg_conf = result.average_confidence
        if avg_conf < 0.3:
            print(f"    ⚠️ Average confidence rất thấp ({avg_conf:.4f}) — có thể cần threshold")
        elif avg_conf < 0.5:
            print(f"    ⚠️ Average confidence thấp ({avg_conf:.4f})")
        else:
            print(f"    ❌ Average confidence cao ({avg_conf:.4f}) — pipeline tin tưởng kết quả!")
        
        return result
        
    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        return None


def main():
    print("=" * 70)
    print("TEST END-TO-END: Full pipeline với ảnh không liên quan")
    print("=" * 70)

    results = []

    # 1. Ảnh chó mèo
    path = create_cat_like_image()
    r = test_analyze_image("Chó mèo (texture)", path)
    if r:
        results.append(r)

    # 2. Ảnh phong cảnh
    path = create_landscape_image()
    r = test_analyze_image("Phong cảnh", path)
    if r:
        results.append(r)

    # 3. Ảnh selfie
    path = create_selfie_image()
    r = test_analyze_image("Selfie", path)
    if r:
        results.append(r)

    # Tổng kết
    print(f"\n\n{'='*70}")
    print("TỔNG KẾT END-TO-END")
    print(f"{'='*70}")
    
    for r in results:
        print(f"\n  modality={r.modality} | body_region={r.body_region} | "
              f"avg_conf={r.average_confidence:.4f} | risk={r.risk_level} | "
              f"detections={len(r.detections)}")


if __name__ == "__main__":
    main()
