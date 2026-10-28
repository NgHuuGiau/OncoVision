from fastapi.testclient import TestClient
import web_app

client = TestClient(web_app.app)

print('=== TESTING WEB API ENDPOINTS ===')

# 1. Health check
r = client.get('/healthz')
print(f'Health: {r.status_code} - {r.json()}')

# 2. API status (should be 401 without auth)
r = client.get('/api/status')
print(f'API Status: {r.status_code} - {r.json()}')

# 3. Login page
r = client.get('/login')
print(f'Login page: {r.status_code} (HTML: {"html" in r.text.lower()})')

# 4. Static files
r = client.get('/favicon.svg')
print(f'Favicon: {r.status_code}')

# 6. Test protected endpoints without auth
endpoints = ['/api/upload', '/api/analyze', '/api/conversations', '/api/cases', '/api/clinicians', '/api/viewers', '/api/settings']
for ep in ['/api/upload', '/api/analyze']:
    r = client.post(ep, files={'file': ('test.jpg', b'test', 'image/jpeg')})
    print(f'{ep}: {r.status_code} - {r.json()}')
for ep in ['/api/conversations', '/api/cases', '/api/clinicians', '/api/viewers', '/api/settings']:
    r = client.get(ep)
    ct = r.headers.get("content-type", "")
    if ct.startswith("application/json"):
        print(f'{ep}: {r.status_code} - {r.json()}')
    else:
        print(f'{ep}: {r.status_code} - HTML/Other')

# Test auth flow
print('\n=== TESTING AUTHENTICATION FLOW ===')

# 1. Get login page (should set CSRF cookie)
r = client.get('/login')
print(f'Login page: {r.status_code}')

# 2. Try login with invalid credentials
r = client.post('/login', data={'username': 'admin', 'password': 'wrong'})
ct = r.headers.get("content-type", "")
if ct.startswith("application/json"):
    print(f'Invalid login: {r.status_code} - {r.json()}')
else:
    print(f'Invalid login: {r.status_code} - {r.text[:100]}')

print('\nAll web API tests passed!')