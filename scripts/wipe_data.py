import os
import shutil
import sqlite3
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, 'items.db')

if not os.path.exists(DB_PATH):
    print('DB not found at', DB_PATH)
    raise SystemExit(1)

# backup
bak = DB_PATH + '.bak.' + datetime.now().strftime('%Y%m%d%H%M%S')
shutil.copy2(DB_PATH, bak)
print('Backup created at', bak)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

tables = ['message', 'orderitem', 'order', 'item', 'user']
# Some projects use lowercase names; ensure existence
existing = {}
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for r in cur.fetchall():
    existing[r[0]] = True

# helper to safe count and delete
report = []
for t in tables:
    if t in existing:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            before = cur.fetchone()[0]
        except Exception as e:
            before = f'err:{e}'
        try:
            cur.execute(f"DELETE FROM {t}")
            conn.commit()
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            after = cur.fetchone()[0]
        except Exception as e:
            after = f'err:{e}'
        report.append((t, before, after))
    else:
        report.append((t, 'not found', 'not found'))

conn.commit()
conn.close()

print('\nWipe report:')
for t, b, a in report:
    print(f'- {t}: before={b} after={a}')

print('\nIf results are correct you can keep or remove the backup file at:')
print(bak)
