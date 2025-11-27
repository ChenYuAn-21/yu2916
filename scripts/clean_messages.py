import os
import shutil
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'items.db')

if not os.path.exists(DB_PATH):
    print('DB not found at', DB_PATH)
    raise SystemExit(1)

# backup
bak = DB_PATH + '.bak.' + datetime.now().strftime('%Y%m%d%H%M%S')
shutil.copy2(DB_PATH, bak)
print('Backup created at', bak)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# find messages that start with 消費者 (including variants)
cur.execute("SELECT id, msg FROM message WHERE msg LIKE '消費者%';")
rows = cur.fetchall()
print('Found', len(rows), "messages starting with '消費者'")

def strip_consumer_prefix(s):
    # 移除開頭的「消費者」及緊接的冒號/全形冒號/空白
    prefixes = ['消費者:', '消費者：', '消費者 ', '消費者']
    orig = s
    changed = s
    for p in prefixes:
        if changed.startswith(p):
            changed = changed[len(p):]
            # 若開頭還有空白或冒號，再去除
            changed = changed.lstrip(':： \t')
            break
    # 若移除後變成空字串，保留原始內容不修改
    if changed.strip() == '':
        return None
    # 可能移除後造成前後多一個空白，修 trim
    return changed.lstrip()

updated = 0
for r in rows:
    mid, msg = r
    new = strip_consumer_prefix(msg)
    if new and new != msg:
        cur.execute('UPDATE message SET msg = ? WHERE id = ?', (new, mid))
        updated += 1

conn.commit()
conn.close()
print('Updated', updated, 'messages.')
print('If results look good you can delete the backup later.')
