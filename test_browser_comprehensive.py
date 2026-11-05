#!/usr/bin/env python
"""
Comprehensive browser/web interface test for OncoVision
Tests all API endpoints and functionality without requiring a real browser
"""
import requests
import re
import json
import sys
import io
import time
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = 'http://127.0.0.1:8000'
USERNAME = 'admin'
PASSWORD = 'password123456'

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_test(name, passed, details=""):
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"{status} {name}")
    if details:
        print(f"    {details}")

def print_section(name):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD} {name}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")

def create_test_image(color='red', size=(512, 512)):
    img = Image.new('RGB', size, color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def get_csrf_token(session, url='/'):
    r = session.get(f'{BASE_URL}{url}')
    m = re.search(r'name="csrf_token" value="([^"]+)"', r.text)
    return m.group(1) if m else None