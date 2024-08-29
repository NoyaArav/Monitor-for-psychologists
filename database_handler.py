import sqlite3
import numpy as np

def create_tables():
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY,
        name TEXT,
        birthdate TEXT,
        notes TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        session_id TEXT,
        sentence TEXT,
        speaker TEXT,
        embedding BLOB,
        sentiment TEXT,
        sentiment_score REAL,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    ''')

    conn.commit()
    conn.close()

def add_patient(patient_id, name, birthdate, notes):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    cursor.execute('SELECT 1 FROM patients WHERE patient_id = ?', (patient_id,))
    if cursor.fetchone() is None:
        cursor.execute('''
        INSERT INTO patients (patient_id, name, birthdate, notes)
        VALUES (?, ?, ?, ?)
        ''', (patient_id, name, birthdate, notes))

    conn.commit()
    conn.close()

def insert_session_data(patient_id, session_id, sentence, speaker, sentiment, sentiment_score):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO sessions (patient_id, session_id, sentence, speaker, sentiment, sentiment_score)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, session_id, sentence, speaker, sentiment, sentiment_score))

    conn.commit()
    conn.close()

def update_session_embedding(session_id, sentence, embedding):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
    cursor.execute('''
    UPDATE sessions SET embedding = ?
    WHERE session_id = ? AND sentence = ?
    ''', (embedding_blob, session_id, sentence))

    conn.commit()
    conn.close()

def fetch_patient_embeddings(patient_id):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    cursor.execute('SELECT id, sentence, embedding FROM sessions WHERE patient_id = ?', (patient_id,))
    results = cursor.fetchall()
    conn.close()

    embeddings = [(result[0], result[1], np.frombuffer(result[2], dtype=np.float32)) for result in results]
    return embeddings

def fetch_session_data(patient_id, session_id):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT id, sentence, speaker, sentiment, sentiment_score
    FROM sessions
    WHERE patient_id = ? AND session_id = ?
    ORDER BY id
    ''', (patient_id, session_id))

    results = cursor.fetchall()
    conn.close()

    session_data = [
        {
            'id': row[0],
            'sentence': row[1],
            'speaker': row[2],
            'sentiment': row[3],
            'sentiment_score': row[4]
        }
        for row in results
    ]

    return session_data