#!/usr/bin/env python
"""Test edge cases: ảnh không liên quan, file invalid, target_key sai."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical.pipeline.analyzer import MedicalImageAnalyzer
from medical.pipeline.config import build_default_medical_analyzer_config


def create_test_images():
    """Tạo các ảnh test cho edge cases."""
    output_dir = project_root / "output" / "test_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = {}
    
    # 1. Noise ngẫu nhiên
    noise_path = output_dir / "noise.jpg"
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(noise_path)
    images["noise"] = noise_path
    
    # 2. Màu đơn
    solid_path = output_dir / "solid.jpg"
    arr2 = np.full((224, 224, 3), 128, dtype=np.uint8)
    Image.fromarray(arr2).save(solid_path)
    images["solid_color"] = solid_path
    
    # 3. Gradient
    gradient_path = output_dir / "gradient.jpg"
    arr3 = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        arr3[i, :, :] = i
    Image.fromarray(arr3).save(gradient_path)
    images["gradient"] = gradient_path
    
    # 4. Ảnh đen
    black_path = output_dir / "black.jpg"
    arr4 = np.zeros((224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr4).save(black_path)
    images["black"] = black_path
    
    # 5. Ảnh trắng
    white_path = output_dir / "white.jpg"
    arr5 = np.full((224, 224, 3), 255, dtype=np.uint8)
    Image.fromarray(arr5).save(white_path)
    images["white"] = white_path
    
    # 6. Ảnh JPEG nhỏ (giống chó mèo)
    small_rgb_path = output_dir / "small_rgb.jpg"
    arr6 = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr6).save(small_rgb_path)
    images["small_rgb"] = small_rgb_path
    
    return images

def test_edge_case(name, image_path, target_key=None):
    """Test 1 edge case."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"Target key: {target_key}")
    print(f"File: {image_path}")
    print(f"{'='*70}")
    
    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        result = analyzer.analyze_image(
            image_path=str(image_path),
            patient_code="EDGE_TEST",
            target_key=target_key,
        )
        
        print(f"  Risk level: {result.risk_level}")
        print(f"  Average confidence: {result.average_confidence:.4f}")
        print(f"  Suspected malignant: {result.suspected_malignant}")
        print(f"  Detections: {len(result.detections) if result.detections else 0}")
        
        # Đánh giá
        if target_key is None:
            # Auto-detect: nên là uncertain cho ảnh không y khoa
            if result.risk_level == "uncertain" and result.average_confidence == 0:
                print("  ✅ Đúng: uncertain, confidence=0")
                return True
            else:
                print("  ⚠️ Không uncertain như mong đợi")
                return False
        else:
            # Explicit target: có thể có kết quả, nhưng confidence thấp
            print(f"  ℹ️ Explicit target, confidence={result.average_confidence:.4f}")
            return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {e}")
        return False

def test_invalid_file():
    """Test file không hợp lệ."""
    print(f"\n{'='*70}")
    print("Test: File không hợp lệ (PDF)")
    print(f"{'='*70}")
    
    # Tạo file PDF giả
    output_dir = project_root / "output" / "test_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")
    
    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        analyzer.analyze_image(
            image_path=str(pdf_path),
            patient_code="EDGE_TEST",
            target_key="lung",
        )
        
        print("  ❌ Không nên chạy được với file PDF")
        return False
        
    except Exception as e:
        print(f"  ✅ Đúng: {type(e).__name__}: {str(e)[:100]}")
        return True

def test_invalid_target_key():
    """Test target_key không tồn tại."""
    print(f"\n{'='*70}")
    print("Test: Target key không tồn tại")
    print(f"{'='*70}")
    
    # Dùng ảnh thyroid hợp lệ
    thyroid_img = project_root / "data" / "datasets" / "thyroid" / "benign" / "000001.png"
    if not thyroid_img.exists():
        print("  ⚠️ Không có ảnh thyroid để test")
        return None
    
    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        analyzer.analyze_image(
            image_path=str(thyroid_img),
            patient_code="EDGE_TEST",
            target_key="nonexistent_cancer",
        )
        
        print("  ❌ Không nên chạy được với target_key sai")
        return False
        
    except Exception as e:
        print(f"  ✅ Đúng: {type(e).__name__}: {str(e)[:100]}")
        return True

def test_missing_file():
    """Test file không tồn tại."""
    print(f"\n{'='*70}")
    print("Test: File không tồn tại")
    print(f"{'='*70}")
    
    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        analyzer.analyze_image(
            image_path="D:\\nonexistent\\file.jpg",
            patient_code="EDGE_TEST",
            target_key="lung",
        )
        
        print("  ❌ Không nên chạy được với file không tồn tại")
        return False
        
    except Exception as e:
        print(f"  ✅ Đúng: {type(e).__name__}: {str(e)[:100]}")
        return True

def main():
    print("=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)
    
    results = {}
    
    # Tạo ảnh test
    print("\nTạo ảnh test...")
    test_images = create_test_images()
    
    # Test 1: Ảnh noise auto-detect
    results["noise_auto"] = test_edge_case("Noise auto-detect", test_images["noise"], target_key=None)
    
    # Test 2: Ảnh màu đơn auto-detect
    results["solid_auto"] = test_edge_case("Màu đơn auto-detect", test_images["solid_color"], target_key=None)
    
    # Test 3: Gradient auto-detect
    results["gradient_auto"] = test_edge_case("Gradient auto-detect", test_images["gradient"], target_key=None)
    
    # Test 4: Ảnh đen auto-detect
    results["black_auto"] = test_edge_case("Ảnh đen auto-detect", test_images["black"], target_key=None)
    
    # Test 5: Ảnh trắng auto-detect
    results["white_auto"] = test_edge_case("Ảnh trắng auto-detect", test_images["white"], target_key=None)
    
    # Test 6: Ảnh noise với explicit target
    results["noise_explicit"] = test_edge_case("Noise + explicit target=lung", test_images["noise"], target_key="lung")
    
    # Test 7: File PDF
    results["invalid_pdf"] = test_invalid_file()
    
    # Test 8: Target key sai
    results["invalid_target"] = test_invalid_target_key()
    
    # Test 9: File không tồn tại
    results["missing_file"] = test_missing_file()
    
    # Tổng kết
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"\n\n{'='*70}")
    print("TONG KET EDGE CASES")
    print(f"{'='*70}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    
    for name, success in results.items():
        if success is True:
            status = "✅"
        elif success is False:
            status = "❌"
        else:
            status = "⚠️"
        print(f"  {status} {name}")
    
    # Lưu kết quả
    output_dir = project_root / "output" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "test_edge_cases.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "results": {k: (v is True) for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Kết quả lưu tại: {result_file}")

if __name__ == "__main__":
    main()
