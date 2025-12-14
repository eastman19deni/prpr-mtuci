import sqlite3
import csv
from pathlib import Path
import time

class StatsLogger:
    def __init__(self, db_path=None):
        self.db_path = db_path or 'data/output/database/attendance.db'
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS frame_stats (
            ts REAL,
            total INTEGER,
            desk_counts TEXT
        )''')
        self.conn.commit()

    def log_frame(self, info):
        ts = time.time()
        total = int(info.get('total',0))
        desk_counts = ','.join(map(str, info.get('desk_counts',[])))
        cur = self.conn.cursor()
        cur.execute('INSERT INTO frame_stats VALUES (?,?,?)', (ts, total, desk_counts))
        self.conn.commit()

    def export_csv(self, csv_path='data/output/statistics/people_count_log.csv'):
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        cur = self.conn.cursor()
        cur.execute('SELECT ts,total,desk_counts FROM frame_stats')
        rows = cur.fetchall()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','total','desk_counts'])
            for r in rows:
                writer.writerow(r)
        return csv_path
