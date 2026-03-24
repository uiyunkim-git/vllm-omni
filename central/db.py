import sqlite3
import json
import os

DB_DIR = "/app/data"
DB_PATH = os.path.join(DB_DIR, "omni.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()
    
    # Workers Table
    # status: pending, active, disconnected
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workers (
            worker_id TEXT PRIMARY KEY,
            custom_name TEXT,
            host TEXT,
            port INTEGER,
            status TEXT,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            gpus_json TEXT
        )
    ''')
    
    # Deployments Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deployments (
            id TEXT PRIMARY KEY,
            name TEXT,
            model TEXT,
            served_model_name TEXT,
            deployment_type TEXT,
            status TEXT,
            gpus_json TEXT,
            nodes_json TEXT
        )
    ''')
    
    # Simple migration for existing DB
    try:
        cursor.execute("ALTER TABLE deployments ADD COLUMN served_model_name TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists

    try:
        cursor.execute("ALTER TABLE deployments ADD COLUMN engine TEXT DEFAULT 'vllm'")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    # Configs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS configs (
            name TEXT PRIMARY KEY,
            config_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()
