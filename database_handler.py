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
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    ''')
    
    # Check if 'speaker' column exists, if not, add it
    cursor.execute("PRAGMA table_info(sessions)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'speaker' not in columns:
        cursor.execute('ALTER TABLE sessions ADD COLUMN speaker TEXT')

    conn.commit()
    conn.close()
    
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
        cursor.execute('''INSERT INTO sessions (patient_id, session_id, sentence, speaker, embedding)
                          VALUES (?, ?, ?, ?, ?)''', (patient_id, session_id, entry["sentence"], entry['speaker'], str(entry["embedding"])))

    conn.commit()
    conn.close()

def search_query(query_embedding, patient_id):
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    cursor.execute("SELECT sentence, embedding FROM sessions WHERE patient_id = ?", (patient_id,))
    results = cursor.fetchall()

    matched_sentences = []
    for sentence, stored_embedding_str in results:
        stored_embedding = np.array(eval(stored_embedding_str))
        similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))

        if similarity > 0.7:
            matched_sentences.append((sentence, similarity))

    conn.close()

    matched_sentences.sort(key=lambda x: x[1], reverse=True)
    return matched_sentences
