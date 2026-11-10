from app.web_auth import WebAuthDatabase

db = WebAuthDatabase('D:\\Project\\OncoVision\\output\\onco.db')

# List users
users = db.list_users()
print(f'Users: {len(users)}')
for u in users:
    print(f'  {u.username} - {u.role} - {u.is_active} - {u.email}')