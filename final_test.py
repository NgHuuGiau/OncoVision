import requests
import re
import json
import sys
import io
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://127.0.0.1:8000'
s = requests.Session()

print('=== 1. HEALTH CHECK ===')
r = s.get(BASE + '/healthz')
print('Health:', r.status_code, r.json())

print('\n=== 2. LOGIN ===')
r = s.get(BASE + '/login')
csrf = re.search(r'name="csrf_token" value="([^"]+)"', r.text).group(1)
print('GET /login:', r.status_code, 'CSRF:', csrf[:20])

r = s.post(BASE + '/login', data={'username': 'admin', 'password': 'password123456', 'csrf_token': csrf})
print('POST /login:', r.status_code, r.url)

csrf = re.search(r'name="csrf_token" value="([^"]+)"', s.get(BASE + '/').text).group(1)
print('New CSRF:', csrf[:20])

print('\n=== 3. API STATUS ===')
r = s.get(BASE + '/api/status')
d = r.json()
print('Status:', r.status_code)
print('  model_ready:', d.get('model_ready'))
print('  demo_mode:', d.get('demo_mode'))
print('  classification_ready:', d.get('classification_ready'))
print('  localization_ready:', d.get('localization_ready'))
print('  volume_ready:', d.get('volume_ready'))
print('  all_targets_ready:', d.get('all_targets_ready'))
print('  case_count:', d.get('case_count'))
print('  analyzed_cancers:', d.get('analyzed_cancers'))

print('\n=== 4. FILE UPLOAD ===')
img = Image.new('RGB', (512, 512), color='red')
b = io.BytesIO()
img.save(b, 'JPEG')
b.seek(0)
r = s.post(BASE + '/api/upload', files={'file': ('test.jpg', b, 'image/jpeg')}, headers={'x-csrf-token': csrf})
print('Upload:', r.status_code, r.json())
path = r.json()['stored_path']

print('\n=== 5. TEST VIETNAMESE MODALITIES ===')
targets = [
    ('brain', 'MRI não'),
    ('lung', 'X-quang ngực'),
    ('breast', 'Mammogram'),
    ('liver', 'CT gan'),
    ('stomach', 'Nội soi'),
    ('colorectal', 'Nội soi đại tràng'),
    ('prostate', 'MRI tuyến tiền liệt'),
    ('cervical', 'Ảnh soi cổ tử cung'),
    ('kidney', 'CT thận'),
    ('pancreas', 'CT tụy'),
    ('thyroid', 'Siêu âm tuyến giáp'),
]

for target, modality in targets:
    r = s.post(BASE + '/api/analyze',
               data={'image_path': path, 'patient_code': 'TEST-' + target.upper(), 'target_key': target, 'modality': modality},
               headers={'x-csrf-token': csrf})
    d = r.json()
    if d.get('ok'):
        m = json.loads(d['metadata_json'])
        print('  PASS', target.ljust(12), modality.ljust(20), 'risk=' + m.get('risk_level', '').ljust(8), 'dets=' + str(len(m.get('detections', []))), 'model=' + m.get('model_name'))
    else:
        print('  FAIL', target.ljust(12), modality.ljust(20), 'ERROR:', d.get('detail'))

print('\n=== 5B. TEST CANONICAL ENGLISH NAMES ===')
targets_canonical = [
    ('brain', 'mri'),
    ('lung', 'xray'),
    ('breast', 'mammogram'),
    ('liver', 'ct'),
    ('stomach', 'endoscopy'),
    ('colorectal', 'colonoscopy'),
    ('prostate', 'mri'),
    ('cervical', 'colposcopy'),
    ('kidney', 'ct'),
    ('pancreas', 'ct'),
    ('thyroid', 'ultrasound'),
]

for target, modality in targets_canonical:
    r = s.post(BASE + '/api/analyze',
               data={'image_path': path, 'patient_code': 'TEST-' + target.upper(), 'target_key': target, 'modality': modality},
               headers={'x-csrf-token': csrf})
    d = r.json()
    if d.get('ok'):
        m = json.loads(d['metadata_json'])
        print('  PASS', target.ljust(12), modality.ljust(10), 'risk=' + m.get('risk_level', '').ljust(8), 'dets=' + str(len(m.get('detections', []))), 'model=' + m.get('model_name'))
    else:
        print('  FAIL', target.ljust(12), modality.ljust(10), 'ERROR:', d.get('detail'))

print('\n=== ALL TESTS COMPLETE ===')