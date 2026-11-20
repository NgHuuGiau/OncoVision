#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test với target_key explicit."""
import io, json, sys
from pathlib import Path
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from medical.pipeline.analyzer import MedicalImageAnalyzer
from medical.pipeline.config import build_default_medical_analyzer_config

tests = [
    ("thyroid", "data/datasets/thyroid/benign/000001.png"),
    ("lung", "data/datasets/lung/adenocarcinoma/000108 (7)_0.png"),
    ("brain", None),  # Không có ảnh mẫu brain trong data/
]

config = build_default_medical_analyzer_config()
for target, img_path in tests:
    if img_path is None:
        print(f"\n{target}: không có ảnh mẫu, skip")
        continue
    p = project_root / img_path
    if not p.exists():
        print(f"\n{target}: file không tồn tại {p}, skip")
        continue
    print(f"\n{'='*50}")
    print(f"Test: {target} (explicit target_key)")
    print(f"{'='*50}")
    analyzer = MedicalImageAnalyzer(config=config)
    try:
        r = analyzer.analyze_image(image_path=str(p), patient_code="TEST", target_key=target)
        print(f"  risk={r.risk_level}, modality={r.modality}, avg_conf={r.average_confidence:.4f}")
        print(f"  detections={len(r.detections)}")
        if r.detections:
            for d in r.detections[:3]:
                print(f"    {d.label} ({d.confidence:.4f})")
        if r.report_json_path.exists():
            rpt = json.loads(r.report_json_path.read_text())
            print(f"  target={rpt.get('target_key')}, status={rpt.get('status')}, conf={rpt.get('confidence',0):.4f}")
            lbls = rpt.get("classification_labels", [])
            if lbls:
                print(f"  labels={lbls[:5]}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Test KHÔNG có target_key (auto-detect)
print(f"\n{'='*50}")
print("Test: Auto-detect (không target_key)")
print(f"{'='*50}")
p = project_root / "data/datasets/thyroid/benign/000001.png"
analyzer = MedicalImageAnalyzer(config=config)
try:
    r = analyzer.analyze_image(image_path=str(p), patient_code="TEST", target_key=None)
    print(f"  risk={r.risk_level}, modality={r.modality}, avg_conf={r.average_confidence:.4f}")
    if r.report_json_path.exists():
        rpt = json.loads(r.report_json_path.read_text())
        print(f"  target={rpt.get('target_key')}, status={rpt.get('status')}")
except Exception as e:
    print(f"  ERROR: {e}")
