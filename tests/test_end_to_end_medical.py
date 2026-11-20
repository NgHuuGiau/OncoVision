#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test end-to-end với ảnh y khoa thực tế từ data/datasets/."""
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


def test_analyze_image(name, path, expected_cancer=None):
    """Test 1 ảnh y khoa thực tế qua full pipeline."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"File: {path.name}")
    print(f"{'='*70}")

    try:
        config = build_default_medical_analyzer_config()
        analyzer = MedicalImageAnalyzer(config=config)
        
        result = analyzer.analyze_image(
            image_path=str(path),
            patient_code="TEST002",
            target_key=None,
        )
        
        print(f"  Risk level: {result.risk_level}")
        print(f"  Suspected malignant: {result.suspected_malignant}")
        print(f"  Modality: {result.modality}")
        print(f"  Body region: {result.body_region}")
        print(f"  Avg confidence: {result.average_confidence:.4f}")
        print(f"  Model: {result.model_name}")
        print(f"  Detections: {len(result.detections)}")
        
        if result.detections:
            for i, det in enumerate(result.detections[:3]):
                print(f"    #{i+1}: {det.label} ({det.confidence:.4f})")
        
        if result.quality_warnings:
            print(f"  Warnings: {len(result.quality_warnings)}")
            for w in result.quality_warnings[:3]:
                print(f"    - {w[:100]}")
        
        # Đọc report JSON
        if result.report_json_path.exists():
            with open(result.report_json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            target = report.get("target_key", "N/A")
            status = report.get("status", "N/A")
            conf = report.get("confidence", 0)
            labels = report.get("classification_labels", [])
            print(f"  Report: target={target}, status={status}, confidence={conf:.4f}")
            if labels:
                print(f"  Labels: {labels[:5]}")
        
        # Đánh giá
        marker = "✅" if result.risk_level != "uncertain" and result.average_confidence > 0 else "⚠️"
        if expected_cancer:
            report_target = report.get("target_key", "") if result.report_json_path.exists() else ""
            if report_target == expected_cancer:
                marker = "✅"
            else:
                marker = "❌"
        
        print(f"\n  {marker} target={target}, conf={conf:.4f}, risk={result.risk_level}")
        return result
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return None


def main():
    print("=" * 70)
    print("TEST END-TO-END: Ảnh y khoa thực tế từ data/datasets/")
    print("=" * 70)

    data_dir = project_root / "data" / "datasets"
    results = []

    # Tìm ảnh từ các cancer types khác nhau
    cancer_dirs = list(data_dir.iterdir())
    tested = 0
    max_test = 11  # Test tối đa 11 nhóm ung thư
    
    for cancer_dir in sorted(cancer_dirs):
        if tested >= max_test:
            break
        if not cancer_dir.is_dir():
            continue
        
        # Tìm ảnh đầu tiên trong thư mục
        images = list(cancer_dir.rglob("*.png")) + list(cancer_dir.rglob("*.jpg")) + list(cancer_dir.rglob("*.jpeg"))
        if not images:
            continue
        
        # Lấy ảnh đầu tiên
        img = images[0]
        r = test_analyze_image(f"{cancer_dir.name}/{img.parent.name}", img, expected_cancer=cancer_dir.name)
        if r:
            results.append((cancer_dir.name, r))
            tested += 1

    # Tổng kết
    print(f"\n\n{'='*70}")
    print("TỔNG KẾT: Ảnh y khoa thực tế")
    print(f"{'='*70}")
    
    success = 0
    fail = 0
    for cancer_name, r in results:
        report = {}
        if r.report_json_path.exists():
            with open(r.report_json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        target = report.get("target_key", "N/A")
        conf = report.get("confidence", 0)
        risk = r.risk_level
        status = "✅" if target == cancer_name else "❌"
        if target == cancer_name:
            success += 1
        else:
            fail += 1
        print(f"  {status} {cancer_name:15s} → detected={target:15s} conf={conf:.4f} risk={risk}")
    
    print(f"\n  Tổng: {success}/{len(results)} đúng, {fail}/{len(results)} sai")


if __name__ == "__main__":
    main()
