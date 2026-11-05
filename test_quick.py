import sys
sys.path.insert(0, '.')

from medical.pipeline import MedicalImageAnalyzer, build_default_medical_analyzer_config

config = build_default_medical_analyzer_config()
analyzer = MedicalImageAnalyzer(config)

# Test key targets
targets = [
    ('brain', 'mri'),
    ('lung', 'xray'),
    ('breast', 'mammogram'),
    ('liver', 'ct'),
]

print("=== QUICK ANALYSIS TEST ===")
for target, modality in targets:
    try:
        result = analyzer.analyze_image('test_img.jpg', patient_code=f'TEST-{target.upper()}', target_key=target, requested_modality=modality)
        print(f"✅ {target:10s} ({modality:8s}) | risk={result.risk_level:8s} | dets={len(result.detections)} | model={result.model_name}")
    except Exception as e:
        print(f"❌ {target:10s} ({modality:8s}) | ERROR: {e}")

print("\n=== QUICK TEST COMPLETE ===")