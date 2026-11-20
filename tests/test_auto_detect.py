#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test auto-detect: chạy ảnh y khoa thực tế qua pipeline không có target_key."""
import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical.pipeline.analyzer import MedicalImageAnalyzer
from medical.pipeline.config import build_default_medical_analyzer_config
from medical.dataset import infer_medical_upload_context, guess_modality_from_pixels, guess_cancer_from_pixels

def test_infer_context(name, path):
    """Test infer_medical_upload_context với 1 ảnh."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"File: {path}")
    print(f"{'='*70}")
    
    # Pixel modality
    modality = guess_modality_from_pixels(str(path))
    print(f"\n[Pixel Modality]")
    print(f"  Kết quả: {modality}")
    
    # Cancer from pixels
    cancer_key, cancer_modality = guess_cancer_from_pixels(str(path))
    print(f"\n[Cancer From Pixels]")
    print(f"  Cancer key: {cancer_key}")
    print(f"  Cancer modality: {cancer_modality}")
    
    # Full context
    target_key, inferred_modality = infer_medical_upload_context(str(path))
    print(f"\n[Full Context Inference]")
    print(f"  Target key: {target_key}")
    print(f"  Modality: {inferred_modality}")
    
    return target_key, inferred_modality

def test_pipeline_auto_detect(name, path):
    """Test full pipeline với target_key=None (auto-detect)."""
    print(f"\n{'='*70}")
    print(f"Pipeline Auto-Detect: {name}")
    print(f"File: {path}")
    print(f"{'='*70}")
    
    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        result = analyzer.analyze_image(
            image_path=str(path),
            patient_code="AUTO_TEST",
            target_key=None,  # Auto-detect
        )
        
        print(f"\n[Pipeline Result]")
        print(f"  Risk level: {result.risk_level}")
        print(f"  Suspected malignant: {result.suspected_malignant}")
        print(f"  Modality: {result.modality}")
        print(f"  Body region: {result.body_region}")
        print(f"  Average confidence: {result.average_confidence:.4f}")
        print(f"  Model name: {result.model_name}")
        
        if result.detections:
            print(f"\n  [Detections: {len(result.detections)}]")
            for i, det in enumerate(result.detections[:3]):
                print(f"    #{i+1}: label={det.label}, confidence={det.confidence:.4f}")
        
        if result.report_json_path.exists():
            with open(result.report_json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            print(f"\n  [Report JSON]")
            for k in ("status", "target_key", "modality", "confidence", "error_code"):
                if k in report:
                    print(f"    {k}: {report[k]}")
            if report.get('classification_labels'):
                print(f"    labels: {report['classification_labels'][:5]}")
        
        # Đánh giá
        print(f"\n  [Đánh giá]")
        if result.average_confidence > 0.3:
            print(f"    ✅ Auto-detect hoạt động! Confidence={result.average_confidence:.4f}")
        elif result.risk_level == "uncertain" and result.average_confidence == 0:
            print(f"    ⚠️ Auto-detect không xác định được (uncertain, conf=0)")
        else:
            print(f"    ⚠️ Confidence thấp: {result.average_confidence:.4f}")
        
        return result
        
    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("TEST: Auto-Detect với ảnh y khoa thực tế")
    print("=" * 70)
    
    # === Phần 1: Test infer_medical_upload_context ===
    print("\n\n" + "=" * 70)
    print("PHẦN 1: infer_medical_upload_context")
    print("=" * 70)
    
    # Thyroid ultrasound
    thyroid_img = project_root / "data" / "datasets" / "thyroid" / "benign" / "000001.png"
    if thyroid_img.exists():
        test_infer_context("Thyroid Ultrasound", thyroid_img)
    else:
        print(f"\nKhông tìm thấy: {thyroid_img}")
    
    # Lung CT
    lung_img = project_root / "data" / "datasets" / "lung" / "adenocarcinoma" / "000108 (7)_0.png"
    if lung_img.exists():
        test_infer_context("Lung CT", lung_img)
    else:
        print(f"\nKhông tìm thấy: {lung_img}")
    
    # === Phần 2: Test full pipeline auto-detect ===
    print("\n\n" + "=" * 70)
    print("PHẦN 2: Full Pipeline Auto-Detect (target_key=None)")
    print("=" * 70)
    
    if thyroid_img.exists():
        test_pipeline_auto_detect("Thyroid Ultrasound", thyroid_img)
    
    if lung_img.exists():
        test_pipeline_auto_detect("Lung CT", lung_img)
    
    # === Phần 3: Test ảnh không liên quan ===
    print("\n\n" + "=" * 70)
    print("PHẦN 3: Ảnh KHÔNG liên quan (auto-detect nên trả về uncertain)")
    print("=" * 70)
    
    import numpy as np
    from PIL import Image
    
    output_dir = project_root / "output" / "test_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo ảnh noise
    noise_path = output_dir / "noise_test.jpg"
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(noise_path)
    
    test_infer_context("Noise (không y khoa)", noise_path)
    test_pipeline_auto_detect("Noise (không y khoa)", noise_path)
    
    # Tạo ảnh màu đơn
    solid_path = output_dir / "solid_test.jpg"
    arr2 = np.full((224, 224, 3), 128, dtype=np.uint8)
    Image.fromarray(arr2).save(solid_path)
    
    test_infer_context("Màu đơn (không y khoa)", solid_path)
    test_pipeline_auto_detect("Màu đơn (không y khoa)", solid_path)
    
    print("\n\n" + "=" * 70)
    print("HOAN THÀNH")
    print("=" * 70)

if __name__ == "__main__":
    main()
