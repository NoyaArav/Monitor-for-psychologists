# embedding_handler.py
from openai import OpenAI

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

def generate_session_embeddings(transcript):
    session_embeddings = []
    for utterance in transcript.utterances:
        embedding = generate_embedding(utterance.text)
        session_embeddings.append({
            "sentence": utterance.text, 
            "speaker": 'patient' if utterance.speaker == 'B' else 'psychologist',  # Distinguishing speaker
            "embedding": embedding
        })
    return session_embeddings
