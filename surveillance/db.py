import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class ActivityDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_conn(self):
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                room_name TEXT NOT NULL,
                camera_number TEXT NOT NULL,
                datetime_from TEXT NOT NULL,
                datetime_to TEXT NOT NULL,
                img_first TEXT,
                img_last TEXT
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_activity_lookup
            ON activity(class_name, room_name, camera_number, datetime_to)
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                filename TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        ''')
        conn.commit()
        self._migrate_activity()

    def _migrate_activity(self):
        conn = self._get_conn()
        for col in ('img_first', 'img_last'):
            try:
                conn.execute(f'ALTER TABLE activity ADD COLUMN {col} TEXT')
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def upsert_activity(self, class_name: str, room_name: str, camera_number: str,
                        dt_from: str, dt_to: str, img_first: str = None,
                        img_last: str = None, merge_window: int = 300):
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT id, datetime_to FROM activity
            WHERE class_name = ? AND room_name = ? AND camera_number = ?
            ORDER BY datetime_to DESC LIMIT 1
        ''', (class_name, room_name, camera_number))
        row = cur.fetchone()

        current_end = datetime.fromisoformat(dt_to)

        if row:
            last_end = datetime.fromisoformat(row['datetime_to'])
            delta = (current_end - last_end).total_seconds()
            if 0 <= delta <= merge_window:
                conn.execute('UPDATE activity SET datetime_to = ?, img_last = ? WHERE id = ?',
                             (dt_to, img_last, row['id']))
                conn.commit()
                return

        conn.execute('''
            INSERT INTO activity (class_name, room_name, camera_number,
                                  datetime_from, datetime_to, img_first, img_last)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (class_name, room_name, camera_number, dt_from, dt_to, img_first, img_last))
        conn.commit()

    def query_activity(self, dt_from: str, dt_to: str, class_filter: Optional[str] = None):
        conn = self._get_conn()
        if class_filter:
            cur = conn.execute('''
                SELECT * FROM activity
                WHERE datetime_from >= ? AND datetime_to <= ? AND class_name = ?
                ORDER BY datetime_from
            ''', (dt_from, dt_to, class_filter))
        else:
            cur = conn.execute('''
                SELECT * FROM activity
                WHERE datetime_from >= ? AND datetime_to <= ?
                ORDER BY datetime_from
            ''', (dt_from, dt_to))
        return cur.fetchall()

    def find_last_seen(self, class_name: str):
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT room_name, datetime_to FROM activity
            WHERE class_name = ?
            ORDER BY datetime_to DESC LIMIT 1
        ''', (class_name,))
        return cur.fetchone()

    def mark_processed(self, filename: str):
        conn = self._get_conn()
        conn.execute('INSERT OR IGNORE INTO processed_files (filename) VALUES (?)',
                     (filename,))
        conn.commit()

    def get_processed_files(self):
        conn = self._get_conn()
        cur = conn.execute('SELECT filename FROM processed_files')
        return {row['filename'] for row in cur.fetchall()}

    def close(self):
        self._conn.close()


class TrainingDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_conn(self):
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relative_path TEXT NOT NULL,
                datetime TEXT NOT NULL,
                class_name TEXT NOT NULL,
                username TEXT NOT NULL,
                annotation TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            )
        ''')
        conn.commit()

    def insert_image(self, relative_path: str, dt: str, class_name: str,
                     username: str, annotation: str, width: int, height: int):
        conn = self._get_conn()
        conn.execute('''
            INSERT INTO images (relative_path, datetime, class_name, username,
                                annotation, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (relative_path, dt, class_name, username, annotation, width, height))
        conn.commit()

    def get_all_images(self):
        conn = self._get_conn()
        cur = conn.execute('SELECT * FROM images ORDER BY id')
        return cur.fetchall()

    def get_usernames(self):
        conn = self._get_conn()
        cur = conn.execute('SELECT DISTINCT username FROM images ORDER BY username')
        return [row['username'] for row in cur.fetchall()]

    def close(self):
        self._conn.close()
