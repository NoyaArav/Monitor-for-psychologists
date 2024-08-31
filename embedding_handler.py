from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from database_handler import fetch_patient_embeddings

client = OpenAI(
  api_key = "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"
)

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    return embedding

def generate_query_embedding(query):
    embedding = generate_embedding(query)
    return embedding

def search_similar_sentences(patient_id, query, sessions_to_search):
    """
    Searches for similar sentences based on a query and selected sessions.

    Parameters:
    - patient_id: The ID of the patient
    - query: The search query
    - sessions_to_search: A list of session IDs to search within

    Returns:
    - List of similar sentences with their session ID, sentence number, speaker, and sentence text.
    """
    query_embedding = generate_query_embedding(query)
    patient_embeddings = fetch_patient_embeddings(patient_id)

    similarities = []
    query_embedding = np.array(query_embedding).reshape(1, -1)

    for session_id, sentence_id, sentence, speaker, embedding in patient_embeddings:
        if session_id in sessions_to_search:
            words = sentence.split()
            # print(words)
            if len(words) > 4:
                embedding = np.array(embedding).reshape(1, -1)
                similarity_score = cosine_similarity(query_embedding, embedding)[0][0]
                similarities.append({
                    'session_id': session_id,
                    'sentence_number': sentence_id,
                    'sentence': sentence,
                    'speaker': speaker,
                    'similarity_score': similarity_score
                })

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities[:5]  # Return top 5 results