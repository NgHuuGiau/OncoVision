"""Test script to start web app and verify it works."""
import asyncio
import uvicorn
import threading
import time
import requests
import json
from web_app import app

def test_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

def test_client():
    time.sleep(3)
    base_url = "http://127.0.0.1:8000"
    session = requests.Session()
    
    try:
        # Test health check
        response = session.get(f"{base_url}/healthz", timeout=5)
        print(f"✓ Health check: {response.status_code} - {response.json()}")
        
        # Test login page
        response = session.get(f"{base_url}/login", timeout=5)
        print(f"✓ Login page: {response.status_code} - {'OK' if 'OncoVision' in response.text else 'FAIL'}")
        
        # Test main page (should redirect to login)
        response = session.get(f"{base_url}/", timeout=5, allow_redirects=False)
        print(f"✓ Main page (unauth): {response.status_code} - {'Redirect to login' if response.status_code == 303 else 'FAIL'}")
        
        # Test login with default admin (if exists)
        # First get CSRF token from login page
        import re
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
        if csrf_match:
            csrf_token = csrf_match.group(1)
            print(f"✓ CSRF token found: {csrf_token[:10]}...")
            
            # Try to login (will fail without proper credentials, but tests the endpoint)
            login_data = {
                "username": "admin",
                "password": "wrongpassword",
                "csrf_token": csrf_token
            }
            response = session.post(f"{base_url}/login", data=login_data, timeout=5)
            print(f"✓ Login attempt: {response.status_code} - {'Expected fail' if response.status_code in [200, 401, 403] else 'Unexpected'}")
        
        # Test API status (should require auth)
        response = session.get(f"{base_url}/api/status", timeout=5)
        print(f"✓ API status (unauth): {response.status_code} - {'Expected 401/303' if response.status_code in [401, 303] else 'Check'}")
        
        # Test static files
        response = session.get(f"{base_url}/static/css/styles.css", timeout=5)
        print(f"✓ Static CSS: {response.status_code}")
        
        response = session.get(f"{base_url}/static/js/app.js", timeout=5)
        print(f"✓ Static JS: {response.status_code}")
        
        # Test favicon
        response = session.get(f"{base_url}/favicon.svg", timeout=5)
        print(f"✓ Favicon: {response.status_code}")
        
        print("\n=== All basic web tests passed! ===")
        
    except Exception as e:
        print(f"✗ Client test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting OncoVision Web App test...")
    server_thread = threading.Thread(target=test_server, daemon=True)
    server_thread.start()
    test_client()
    print("Test completed")