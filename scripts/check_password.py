from app.web_auth import WebAuthDatabase, verify_password

db = WebAuthDatabase('D:\\Project\\OncoVision\\output\\onco.db')

# Check password hash
with db._connect() as conn:
    row = conn.execute("SELECT username, password_hash, role, is_active FROM web_users").fetchone()
    if row:
        print(f'username: {row[0]}')
        print(f'password_hash: {row[1]}')
        print(f'role: {row[2]}')
        print(f'is_active: {row[3]}')
        
        # Test password
        test_passwords = ["AdminPassword123!", "admin", "password123456", "AdminPassword123"]
        for pwd in test_passwords:
            result = verify_password(pwd, row[1])
            print(f'  {pwd}: {result}')