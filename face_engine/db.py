# face_engine/db.py (Enhanced version)

import sqlite3
import os

DB_PATH = os.path.join("data", "faces.sqlite")
os.makedirs("data", exist_ok=True)

class FaceDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                name TEXT PRIMARY KEY,
                display_img_path TEXT,
                total_embeddings INTEGER DEFAULT 0,
                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_face(self, name, img_path=None, increment_embeddings=True):
        """Add or update face record"""
        if img_path:
            # Update with display image path
            self.conn.execute("""
                INSERT OR REPLACE INTO faces (name, display_img_path, total_embeddings) 
                VALUES (?, ?, COALESCE((SELECT total_embeddings FROM faces WHERE name = ?), 0) + ?)
            """, (name, img_path, name, 1 if increment_embeddings else 0))
        else:
            # Just increment embedding count
            self.conn.execute("""
                INSERT OR IGNORE INTO faces (name, total_embeddings) VALUES (?, 1);
                UPDATE faces SET total_embeddings = total_embeddings + 1 WHERE name = ? AND ? = 1;
            """, (name, name, 1 if increment_embeddings else 0))
        self.conn.commit()

    def list_faces(self):
        cursor = self.conn.execute("SELECT name FROM faces ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def get_image_path(self, name):
        cursor = self.conn.execute("SELECT display_img_path FROM faces WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_face_info(self, name):
        """Get complete face information"""
        cursor = self.conn.execute("""
            SELECT name, display_img_path, total_embeddings, enrollment_date 
            FROM faces WHERE name = ?
        """, (name,))
        row = cursor.fetchone()
        if row:
            return {
                'name': row[0],
                'display_img_path': row[1],
                'total_embeddings': row[2],
                'enrollment_date': row[3]
            }
        return None

    def get_all_faces_info(self):
        """Get information about all enrolled faces"""
        cursor = self.conn.execute("""
            SELECT name, display_img_path, total_embeddings, enrollment_date 
            FROM faces ORDER BY name
        """)
        return [
            {
                'name': row[0],
                'display_img_path': row[1], 
                'total_embeddings': row[2],
                'enrollment_date': row[3]
            }
            for row in cursor.fetchall()
        ]

    def delete_face(self, name):
        self.conn.execute("DELETE FROM faces WHERE name = ?", (name,))
        self.conn.commit()