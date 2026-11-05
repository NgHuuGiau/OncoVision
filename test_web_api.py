import requests
import re
import json
import sys
import io

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

s = requests.Session()
r = s.get('http://127.0.0.1:8000/login')
print('GET login:', r.status_code)

m = re.search(r'name="csrf_token" value="([^"]+)"', r.text)
if m:
    csrf = m.group(1)
    print('CSRF:', csrf)
    r = s.post('http://127.0.0.1:8000/login', data={
        'username': 'admin',
        'password': 'password123456',
        'csrf_token': csrf
    })
    print('POST login:', r.status_code, r.url)
    
    # Get fresh CSRF token after login
    r = s.get('http://127.0.0.1:8000/')
    m = re.search(r'name="csrf_token" value="([^"]+)"', r.text)
    if m:
        csrf = m.group(1)
        print('New CSRF:', csrf)
    
    # Check API status
    r = s.get('http://127.0.0.1:8000/api/status')
    print('API status:', r.status_code)
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    
    # Test upload with a fake image
    # Create a simple valid JPEG
    from PIL import Image
    import io
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    r = s.post('http://127.0.0.1:8000/api/upload', 
               files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
               headers={'x-csrf-token': csrf})
    print('Upload status:', r.status_code)
    print('Upload response:', r.json())
    
    upload_data = r.json()
    if upload_data.get('ok'):
        stored_path = upload_data['stored_path']
        print(f'Stored path: {stored_path}')
        
        # Test analyze with brain target
        r = s.post('http://127.0.0.1:8000/api/analyze',
                   data={
                       'image_path': stored_path,
                       'patient_code': 'TEST001',
                       'user_prompt': 'Test analysis',
                       'target_key': 'brain',
                       'modality': ''
                   },
                   headers={'x-csrf-token': csrf})
        print('Analyze brain status:', r.status_code)
        print('Analyze brain response:', json.dumps(r.json(), indent=2, ensure_ascii=False))
        
        # Test analyze with lung target
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        r = s.post('http://127.0.0.1:8000/api/upload', 
                   files={'file': ('test2.jpg', img_bytes, 'image/jpeg')},
                   headers={'x-csrf-token': csrf})
        upload_data2 = r.json()
        if upload_data2.get('ok'):
            stored_path2 = upload_data2['stored_path']
            r = s.post('http://127.0.0.1:8000/api/analyze',
                       data={
                           'image_path': stored_path2,
                           'patient_code': 'TEST002',
                           'user_prompt': 'Test lung analysis',
                           'target_key': 'lung',
                           'modality': ''
                       },
                       headers={'x-csrf-token': csrf})
            print('Analyze lung status:', r.status_code)
            print('Analyze lung response:', json.dumps(r.json(), indent=2, ensure_ascii=False))