# embedding_handler.py
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

def search_similar_sentences(patient_id, query):
    query_embedding = generate_query_embedding(query)
    patient_embeddings = fetch_patient_embeddings(patient_id)
    
    similarities = []
    query_embedding = np.array(query_embedding).reshape(1, -1)

    for sentence_id, sentence, embedding in patient_embeddings:
        words = sentence.split()
        if len(words) > 4:
            embedding = np.array(embedding).reshape(1, -1)
            similarity_score = cosine_similarity(query_embedding, embedding)[0][0]
            similarities.append((sentence_id, sentence, similarity_score))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities


