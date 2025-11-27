from app import app

with app.test_client() as c:
    resp = c.get('/')
    print('STATUS', resp.status_code)
    data = resp.get_data(as_text=True)
    print(data[:4000])
