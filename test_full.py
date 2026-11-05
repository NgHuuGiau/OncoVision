import requests, re, json, sys, io
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://127.0.0.1:8000'
s = requests.Session()

def ok(name, cond, detail=""):
    print(("✓" if cond else "✗") + " " + name + (" " + detail if detail else ""))

print("=== 1. HEALTH ===")
r = s.get(BASE + '/healthz')
ok("Health endpoint", r.status_code == 200 and r.json().get('status') == 'ok', f"status={r.status_code}")

print("\n=== 2. LOGIN ===")
r = s.get(BASE + '/login')
ok("GET /login", r.status_code == 200)
csrf = re.search(r'name="csrf_token" value="([^"]+)"', s.get(BASE + '/login').text).group(1)
ok("CSRF token", csrf is not None, csrf[:20] + "...")
r = s.post(BASE + '/login', data={'username': 'admin', 'password': 'password123456', 'csrf_token': csrf})
ok("POST /login", r.status_code == 200 and r.url == BASE + '/', f"status={r.status_code} url={r.url}")
csrf = re.search(r'name="csrf_token" value="([^"]+)"', s.get(BASE + '/').text).group(1)
ok("CSRF after login", csrf is not None, csrf[:20] + "...")

print("\n=== 3. API STATUS (DEMO MODE) ===")
r = s.get(BASE + '/api/status')
d = r.json()
ok("GET /api/status", r.status_code == 200)
ok("demo_mode=True", d.get('demo_mode') == True)
ok("model_ready=True", d.get('model_ready') == True)
ok("demo_warning present", bool(d.get('demo_warning')), d.get('demo_warning', '')[:50])

print("\n=== 4. FILE UPLOAD ===")
img = Image.new('RGB', (512, 512), 'red')
b = io.BytesIO(); img.save(b, 'JPEG'); b.seek(0)
r = s.post(BASE + '/api/upload', files={'file': ('test.jpg', b, 'image/jpeg')}, headers={'x-csrf-token': csrf})
ok("Upload", r.status_code == 200 and r.json().get('ok'))
path = r.json()['stored_path'] if r.json().get('ok') else ''

print("\n=== 5. ANALYZE 11 TARGETS ===")
targets = ['brain','lung','breast','liver','stomach','colorectal','prostate','cervical','kidney','pancreas','thyroid']
for t in targets:
    r = s.post(BASE + '/api/analyze', data={'image_path': path, 'patient_code': 'TEST-' + t, 'target_key': t, 'modality': ''}, headers={'x-csrf-token': csrf})
    d = r.json()
    success = d.get('ok') == True
    if success:
        m = json.loads(d['metadata_json'])
        dets = len(m.get('detections', []))
        risk = m.get('risk_level', '')
        model = m.get('model_name', '')
        demo_warn = any('DEMO MODE' in w for w in m.get('quality_warnings', []))
        ok(f"  {t}", success, f"dets={dets} risk={risk} model={model} demo_warn={demo_warn}")
    else:
        ok(f"  {t}", False, f"status={r.status_code} error={r.text[:80]}")

print("\n=== 6. CONVERSATIONS & CASES ===")
r = s.get(BASE + '/api/conversations')
ok("Conversations", r.status_code == 200, f"count={len(r.json().get('conversations', []))}")
r = s.get(BASE + '/api/cases')
ok("Cases", r.status_code == 200, f"count={len(r.json().get('cases', []))}")
cases = r.json().get('cases', [])

print("\n=== 7. CASE DETAILS & REPORTS ===")
if cases:
    # Try different possible keys for case ID
    cid = cases[0].get('id') or cases[0].get('case_id') or cases[0].get('caseId')
    ok("Case ID found", cid is not None, f"keys={list(cases[0].keys())}")
    if cid:
        r = s.get(BASE + f'/api/cases/{cid}')
        ok("Case detail", r.status_code == 200)
        r = s.get(BASE + f'/api/cases/{cid}/pdf')
        ok("Case PDF", r.status_code == 200 and 'pdf' in r.headers.get('Content-Type', ''), r.headers.get('Content-Type', ''))
        r = s.get(BASE + f'/api/cases/{cid}/image')
        ok("Case Image", r.status_code == 200 and 'image' in r.headers.get('Content-Type', ''), r.headers.get('Content-Type', ''))

print("\n=== 8. SETTINGS ===")
r = s.get(BASE + '/api/settings')
ok("Settings", r.status_code == 200)

print("\n=== 9. LOGOUT ===")
r = s.post(BASE + '/logout', data={'csrf_token': csrf})
ok("Logout", r.status_code in (200, 302))
r = s.get(BASE + '/api/status')
ok("API blocked after logout", r.status_code == 401)

print("\n=== 10. RE-LOGIN ===")
csrf = re.search(r'name="csrf_token" value="([^"]+)"', s.get(BASE + '/login').text).group(1)
r = s.post(BASE + '/login', data={'username': 'admin', 'password': 'password123456', 'csrf_token': csrf})
ok("Re-login", r.status_code == 200)
r = s.get(BASE + '/api/status')
ok("API works after re-login", r.status_code == 200 and r.json().get('demo_mode') == True)

print("\n=== ALL TESTS COMPLETE ===")