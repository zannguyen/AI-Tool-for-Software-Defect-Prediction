"""
Database Module - SQLite Database Handler for History
"""

import sqlite3
import os
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional


class HistoryDatabase:
    """SQLite database handler for storing analysis history"""

    def __init__(self, db_path: str = "database/history.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database and create tables if not exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                source_type TEXT NOT NULL,
                files_count INTEGER,
                files_list TEXT,
                status TEXT DEFAULT 'completed'
            )
        ''')

        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                LOC INTEGER,
                LOC_BLANK INTEGER,
                LOC_TOTAL INTEGER,
                LOC_COMMENTS INTEGER,
                LOC_CODE INTEGER,
                FUNCTION_COUNT INTEGER,
                CLASS_COUNT INTEGER,
                CYCLOMATIC_COMPLEXITY INTEGER,
                DECISION_COUNT INTEGER,
                COMMENT_RATIO REAL,
                LABEL INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')

        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')

        conn.commit()
        conn.close()

    def save_session(self, source_type: str, files: List[str], df: pd.DataFrame) -> str:
        """
        Save a new analysis session

        Args:
            source_type: Type of source (csv, code_files, sample_data)
            files: List of file names
            df: DataFrame with metrics

        Returns:
            session_id
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert session
        cursor.execute('''
            INSERT INTO sessions (session_id, timestamp, source_type, files_count, files_list)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_type,
            len(files),
            ", ".join(files)
        ))

        # Insert metrics for each file
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO metrics (
                    session_id, file_name, LOC, LOC_BLANK, LOC_TOTAL,
                    LOC_COMMENTS, LOC_CODE, FUNCTION_COUNT, CLASS_COUNT,
                    CYCLOMATIC_COMPLEXITY, DECISION_COUNT, COMMENT_RATIO, LABEL
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                row.get('file_name', row.get('filename', 'unknown')),
                row.get('LOC', 0),
                row.get('LOC_BLANK', 0),
                row.get('LOC_TOTAL', 0),
                row.get('LOC_COMMENTS', 0),
                row.get('LOC_CODE', 0),
                row.get('FUNCTION_COUNT', 0),
                row.get('CLASS_COUNT', 0),
                row.get('CYCLOMATIC_COMPLEXITY', 0),
                row.get('DECISION_COUNT', 0),
                row.get('COMMENT_RATIO', 0),
                row.get('LABEL', 0)
            ))

        conn.commit()
        conn.close()

        return session_id

    def save_predictions(self, session_id: str, results: Dict):
        """
        Save model prediction results

        Args:
            session_id: Session ID
            results: Dictionary with model results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for model_name, metrics in results.items():
            cursor.execute('''
                INSERT INTO predictions (session_id, model_name, accuracy, precision, recall, f1_score, roc_auc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                model_name,
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('roc_auc', 0)
            ))

        conn.commit()
        conn.close()

    def get_all_sessions(self) -> List[Dict]:
        """
        Get all analysis sessions

        Returns:
            List of session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT session_id, timestamp, source_type, files_count, files_list, status
            FROM sessions
            ORDER BY timestamp DESC
        ''')

        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return sessions

    def get_session_details(self, session_id: str) -> Dict:
        """
        Get detailed information of a session

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session details
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get session info
        cursor.execute('''
            SELECT * FROM sessions WHERE session_id = ?
        ''', (session_id,))

        row = cursor.fetchone()
        session = dict(row) if row else {}

        # Get metrics
        cursor.execute('''
            SELECT * FROM metrics WHERE session_id = ?
        ''', (session_id,))

        metrics = [dict(row) for row in cursor.fetchall()]

        # Get predictions
        cursor.execute('''
            SELECT * FROM predictions WHERE session_id = ?
        ''', (session_id,))

        predictions = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            'session': session,
            'metrics': metrics,
            'predictions': predictions
        }

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all related data

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM predictions WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM metrics WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))

            conn.commit()
            conn.close()

            return True
        except:
            return False

    def get_session_count(self) -> int:
        """Get total number of sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM sessions')
        count = cursor.fetchone()[0]

        conn.close()
        return count
