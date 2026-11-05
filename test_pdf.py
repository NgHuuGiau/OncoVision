import requests, re, json
s = requests.Session()
r = s.get('http://127.0.0.1:8000/login')
csrf = re.search(r'name="csrf_token" value="([^"]+)"', r.text).group(1)
r = s.post('http://127.0.0.1:8000/login', data={'username':'admin','password':'password123456','csrf_token':csrf})
r = s.get('http://127.0.0.1:8000/api/cases')
cases = r.json()['cases']
cid = cases[0]['case_id']
print('Case ID:', cid)
r = s.get(f'http://127.0.0.1:8000/api/cases/{cid}/pdf')
print('Status:', r.status_code)
print('Content-Type:', r.headers.get('Content-Type'))
print('Response:', r.text[:500])