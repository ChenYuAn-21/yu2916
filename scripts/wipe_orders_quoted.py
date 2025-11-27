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
bak = DB_PATH + '.bak.order.' + datetime.now().strftime('%Y%m%d%H%M%S')
shutil.copy2(DB_PATH, bak)
print('Backup created at', bak)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

targets = ['order', 'orderitem']
report = []
for t in targets:
    # check existence (case-sensitive) by exact name
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,))
    found = cur.fetchone()
    if not found:
        # some DB used capitalized or quoted names—try case-insensitive search
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=?", (t.lower(),))
        rf = cur.fetchone()
        if rf:
            table_name = rf[0]
        else:
            report.append((t, 'not found', 'not found'))
            continue
    else:
        table_name = found[0]

    # use quoted table name to be safe
    qname = '"' + table_name + '"'
    try:
        cur.execute(f"SELECT COUNT(*) FROM {qname}")
        before = cur.fetchone()[0]
    except Exception as e:
        report.append((table_name, f'err:{e}', f'err'))
        continue
    try:
        cur.execute(f"DELETE FROM {qname}")
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {qname}")
        after = cur.fetchone()[0]
    except Exception as e:
        report.append((table_name, before, f'err:{e}'))
        continue
    report.append((table_name, before, after))

conn.close()

print('\nWipe quoted-tables report:')
for t, b, a in report:
    print(f'- {t}: before={b} after={a}')

print('\nBackup file:')
print(bak)
