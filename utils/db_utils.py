import sqlite3
from datetime import datetime
import pytz

def init_db():
    """ایجاد جدول دیتابیس اولیه"""
    conn = sqlite3.connect('billet_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS billets
                 (timestamp TEXT, width_mm TEXT, height_mm TEXT, length_mm TEXT)''')
    conn.commit()
    conn.close()

def store_billet(timestamp, width_mm, height_mm, length_mm):
    """ذخیره داده در دیتابیس"""
    conn = sqlite3.connect('billet_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO billets VALUES (?, ?, ?, ?)",
              (timestamp, width_mm, height_mm, length_mm))
    conn.commit()
    conn.close()

def get_last_billet():
    """دریافت آخرین شمش ذخیره‌شده"""
    conn = sqlite3.connect('billet_data.db')
    c = conn.cursor()
    c.execute("SELECT width_mm, height_mm, length_mm FROM billets ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row