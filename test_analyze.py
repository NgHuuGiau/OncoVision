import sys
sys.path.insert(0, '.')

from medical.pipeline import MedicalImageAnalyzer, build_default_medical_analyzer_config

config = build_default_medical_analyzer_config()
analyzer = MedicalImageAnalyzer(config)
result = analyzer.analyze_image('test_img.jpg', patient_code='TEST001', target_key='lung', requested_modality='xray')
print('risk_level:', result.risk_level)
print('detections:', len(result.detections))
print('model_name:', result.model_name)
print('quality_warnings:', result.quality_warnings)
for d in result.detections:
    print(f'  - {d.label}: {d.confidence:.2%} {d.bbox}')