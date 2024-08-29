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
    """
    Fetches embeddings for all sentences of a given patient from the database
    and calculates the sentence number within each session.

    Parameters:
    - patient_id: The ID of the patient

    Returns:
    - List of tuples containing session_id, sentence_number, sentence, speaker, and embedding.
    """
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    # Updated SQL query to select sentences ordered by session_id and id
    cursor.execute('SELECT session_id, id, sentence, speaker, embedding FROM sessions WHERE patient_id = ? ORDER BY session_id, id', (patient_id,))
    results = cursor.fetchall()
    conn.close()

    embeddings = []
    current_session_id = None
    sentence_id = 0

    for result in results:
        session_id, _, sentence, speaker, embedding = result
        
        # Check if the session_id has changed and reset sentence_number if so
        if session_id != current_session_id:
            current_session_id = session_id
            sentence_id = 1
        else:
            sentence_id += 1
        
        # Append the tuple with session_id, calculated sentence_id (sentence number in the specific session), sentence, speaker, and embedding
        embeddings.append((session_id, sentence_id, sentence, speaker, np.frombuffer(embedding, dtype=np.float32)))

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


def get_all_patients():
    """Fetches all patients from the database."""
    conn = sqlite3.connect('patient_sessions.db')  # Replace 'patients.db' with your actual database file
    cursor = conn.cursor()
    
    # Fetch all patients
    cursor.execute("SELECT patient_id, name FROM patients")
    patients = cursor.fetchall()
    
    # Convert the result to a list of dictionaries
    patient_list = [{'patient_id': row[0], 'name': row[1]} for row in patients]
    
    conn.close()
    return patient_list


def get_patient_info(patient_id):
    """Fetches detailed information for a specific patient from the database."""
    conn = sqlite3.connect('patient_sessions.db')  # Replace 'patients.db' with your actual database file
    cursor = conn.cursor()
    
    # Fetch patient details
    cursor.execute("SELECT patient_id, name, birthdate, notes FROM patients WHERE patient_id = ?", (patient_id,))
    patient = cursor.fetchone()
    
    # Check if the patient exists
    if not patient:
        return None
    
    # Construct the patient info dictionary
    patient_info = {
        'patient_id': patient[0],
        'name': patient[1],
        'birthdate': patient[2],
        'notes': patient[3],
        'sessions': [],  # Placeholder for session data
    }
    
    # Fetch all session IDs for the patient
    cursor.execute("SELECT DISTINCT session_id FROM sessions WHERE patient_id = ?", (patient_id,))
    session_ids = cursor.fetchall()
    
    # Fetch session data for each session and add to the patient_info dictionary
    for session_id in session_ids:
        session_data = fetch_session_data(patient_id, session_id[0])
        patient_info['sessions'].append({'session_id': session_id[0], 'session_data': session_data})
    
    
    conn.close()
    return patient_info