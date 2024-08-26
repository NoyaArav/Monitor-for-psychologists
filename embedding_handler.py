# embedding_handler.py
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def generate_session_embeddings(transcript, patient_speaker, patient_data, psychologist_data):
    session_embeddings = []
    for i, utterance in enumerate(transcript.utterances):
        embedding = generate_embedding(utterance.text)
        if utterance.speaker == patient_speaker:
            sentiment = next((item['sentiment'] for item in patient_data if item['index'] == i), None)
        else:
            sentiment = next((item['sentiment'] for item in psychologist_data if item['index'] == i), None)
        
        session_embeddings.append({
            "sentence": utterance.text, 
            "speaker": 'patient' if utterance.speaker == patient_speaker else 'psychologist',
            "embedding": embedding,
            "sentiment": sentiment
        })
    return session_embeddings

def generate_query_embedding(query):
    embedding = generate_embedding(query)
    return embedding

def search_similar_sentences(query_embedding, patient_embeddings):
    similarities = []

    # Ensure the query_embedding is a numpy array
    query_embedding = np.array(query_embedding).reshape(1, -1)

    for sentence_id, sentence, embedding in patient_embeddings:
        words = sentence.split()
        if len(words) > 4:
            # Ensure embedding is a numpy array and compute cosine similarity
            embedding = np.array(embedding).reshape(1, -1)
            similarity_score = cosine_similarity(query_embedding, embedding)[0][0]
            similarities.append((sentence_id, sentence, similarity_score))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities
