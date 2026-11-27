#!/usr/bin/env python
"""Test toàn bộ 11 nhóm ung thư với explicit target_key."""
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical.pipeline.analyzer import MedicalImageAnalyzer
from medical.pipeline.config import build_default_medical_analyzer_config


def test_cancer_type(cancer_key, modality_key, dataset_subdir):
    """Test 1 nhóm ung thư với ảnh từ dataset."""
    print(f"\n{'='*70}")
    print(f"Test: {cancer_key} ({modality_key})")
    print(f"{'='*70}")
    
    # Tìm ảnh trong dataset
    dataset_dir = project_root / "data" / "datasets" / dataset_subdir
    if not dataset_dir.exists():
        print(f"  ❌ Dataset không tồn tại: {dataset_dir}")
        return False
    
    # Tìm ảnh đầu tiên
    images = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.jpeg"))
    if not images:
        # Thử tìm trong subdirectories
        for subdir in dataset_dir.iterdir():
            if subdir.is_dir():
                images = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg"))
                if images:
                    break
    
    if not images:
        print(f"  ❌ Không tìm thấy ảnh trong {dataset_dir}")
        return False
    
    image_path = images[0]
    print(f"  Ảnh: {image_path.name}")
    
    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        result = analyzer.analyze_image(
            image_path=str(image_path),
            patient_code="TEST_11CANCER",
            target_key=cancer_key,
        )
        
        print(f"  Risk level: {result.risk_level}")
        print(f"  Modality: {result.modality}")
        print(f"  Average confidence: {result.average_confidence:.4f}")
        print(f"  Model name: {result.model_name}")
        print(f"  Suspected malignant: {result.suspected_malignant}")
        
        if result.detections:
            print(f"  Detections: {len(result.detections)}")
            for i, det in enumerate(result.detections[:2]):
                print(f"    #{i+1}: label={det.label}, confidence={det.confidence:.4f}")
        
        # Đánh giá
        if result.average_confidence > 0:
            print("  ✅ Classifier hoạt động")
            return True
        else:
            print("  ⚠️ Confidence = 0, có thể ảnh không khớp domain model")
            return False
        
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("TEST: Toàn bộ 11 nhóm ung thư với explicit target_key")
    print("=" * 70)
    
    # Danh sách 11 nhóm ung thư + dataset path
    cancer_tests = [
        ("thyroid", "ultrasound", "thyroid/benign"),
        ("lung", "ct", "lung/adenocarcinoma"),
        ("brain", "mri", "brain/mri"),
        ("liver", "ct", "liver/ct"),
        ("breast", "ultrasound", "breast/ultrasound"),
        ("stomach", "endoscopy", "stomach/endoscopy"),
        ("colorectal", "colonoscopy", "colorectal/colonoscopy"),
        ("cervical", "colposcopy", "cervical/colposcopy"),
        ("kidney", "ct", "kidney/ct"),
        ("pancreas", "ct", "pancreas/ct"),
        ("prostate", "mri", "prostate/mri"),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for cancer_key, modality, dataset_path in cancer_tests:
        success = test_cancer_type(cancer_key, modality, dataset_path)
        results[cancer_key] = success
        if success:
            passed += 1
        else:
            failed += 1
    
    # Tổng kết
    print(f"\n\n{'='*70}")
    print("TONG KET")
    print(f"{'='*70}")
    print(f"  Passed: {passed}/{len(cancer_tests)}")
    print(f"  Failed: {failed}/{len(cancer_tests)}")
    
    for cancer_key, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {cancer_key}")
    
    # Lưu kết quả
    output_dir = project_root / "output" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "test_11_cancer.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed,
            "failed": failed,
            "total": len(cancer_tests),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Kết quả lưu tại: {result_file}")

if __name__ == "__main__":
    main()
