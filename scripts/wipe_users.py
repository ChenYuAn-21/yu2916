#!/usr/bin/env python3
import os
import shutil
import sqlite3
import datetime
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(root, 'items.db')
if not os.path.exists(db_path):
    print('資料庫未找到:', db_path)
    sys.exit(1)

ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
backup_path = db_path + '.bak.users.' + ts
shutil.copy2(db_path, backup_path)
print('已建立備份:', backup_path)

conn = sqlite3.connect(db_path)
c = conn.cursor()

def safe_count(query):
    try:
        c.execute(query)
        row = c.fetchone()
        return row[0] if row is not None else None
    except Exception:
        return None

users_before = safe_count('SELECT COUNT(*) FROM "user"')
print('刪除前會員數:', users_before)

try:
    c.execute('DELETE FROM "user"')
    conn.commit()
    users_after = safe_count('SELECT COUNT(*) FROM "user"')
    print('刪除後會員數:', users_after)
except Exception as e:
    print('刪除 `user` 時發生錯誤:', e)

# 嘗試將 item.seller_id 設為 NULL（如果該欄位存在）
try:
    c.execute("PRAGMA table_info(item)")
    cols = [r[1] for r in c.fetchall()]
    if 'seller_id' in cols:
        c.execute('UPDATE item SET seller_id = NULL')
        conn.commit()
        print('已清空 item.seller_id 欄位（若存在）。')
    else:
        print('item.seller_id 欄位不存在，跳過清空。')
except Exception as e:
    print('嘗試清空 item.seller_id 時發生錯誤:', e)

conn.close()
print('完成。')
