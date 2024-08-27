# database_handler.py
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
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    ''')

    
def add_patient(patient_id, name, birthdate, notes):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    # Check if the patient already exists
    cursor.execute('SELECT 1 FROM patients WHERE patient_id = ?', (patient_id,))
    if cursor.fetchone() is None:
        cursor.execute('''
        INSERT INTO patients (patient_id, name, birthdate, notes)
        VALUES (?, ?, ?, ?)
        ''', (patient_id, name, birthdate, notes))

    conn.commit()
    conn.close()

def store_embeddings(patient_id, session_id, session_embeddings):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    for entry in session_embeddings:
        # Convert the numpy array embedding to bytes for storage
        embedding_blob = np.array(entry["embedding"], dtype=np.float32).tobytes()
        cursor.execute('''INSERT INTO sessions (patient_id, session_id, sentence, speaker, embedding, sentiment)
                          VALUES (?, ?, ?, ?, ?, ?)''', (patient_id, session_id, entry["sentence"], entry['speaker'], embedding_blob, entry.get('sentiment')))

    conn.commit()
    conn.close()
    

def fetch_patient_embeddings(patient_id):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    # Fetch embeddings for a specific patient
    cursor.execute('SELECT id, sentence, embedding FROM sessions WHERE patient_id = ?', (patient_id,))
    results = cursor.fetchall()
    conn.close()

    # Convert BLOB back to numpy array
    embeddings = [(result[0], result[1], np.frombuffer(result[2], dtype=np.float32)) for result in results]
    return embeddings

    
