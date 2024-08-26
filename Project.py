import assemblyai as aai
from openai import OpenAI
import json

from embedding_handler import generate_session_embeddings, generate_embedding, generate_query_embedding, search_similar_sentences
from database_handler import create_tables, store_embeddings, add_patient, fetch_patient_embeddings

aai.settings.api_key = "7a43eb14db35446586c8e9938f2b947c"

client = OpenAI(
  api_key = "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"
)
openai_api_model = "gpt-3.5-turbo"


audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/13mins_session_depression.mp4"
# audio_file = "https://www.youtube.com/watch?v=7LD8iC4NqXM" 


# Sentiment score dictionaries
patient_sentiment_scores = {
    "Despair": -5,
    "Anger": -4,
    "Anxiety": -3,
    "Sadness": -2,
    "Discomfort": -1,
    "Natural": 0,
    "Contentment": 1,
    "Hopefulness": 2,
    "Happiness": 3,
    "Excitement": 4,
    "Euphoria": 5
}

psychologist_sentiment_scores = {
    "Overwhelm": -5,
    "Helplessness": -4,
    "Sadness": -3,
    "Frustration": -2,
    "Concern": -1,
    "Natural": 0,
    "Contentment": 1,
    "Encouragement": 2,
    "Empathy": 3,
    "Optimism": 4,
    "Fulfillment": 5
}

def get_sentiment_patient(text):

  sentiments = [
    "Euphoria", "Excitement", "Happiness", "Hopefulness", "Contentment", 
    "Natural", "Discomfort", "Sadness", "Anxiety", "Anger", "Despair"
    ]
  
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0, # Use a larger temperature value to generate more diverse results.
     messages=[
            {"role": "user", "content": f"""Given the following sentence that was said by a patient during a therapy session:"
             "{text}"
             Which of the following sentiments best describes it? Choose one of the following: {", ".join(sentiments)}.
             return only one word - the correct sentiment from the list above."""}
        ],
  )
  sentiment = response.choices[0].message.content
  return sentiment

def get_sentiment_psychologist(text):

    sentiments = [
    "Fulfillment" , "Optimism" , "Empathy", "Encouragement", "Contentment", "Natural", 
    "Concern", "Frustration", "Sadness", "Helplessness", "Overwhelm"
    ]

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0, # Use a larger temperature value to generate more diverse results.
     messages=[
            {"role": "user", "content": f"""Given the following sentence that was said by a psychologist during a therapy session:
             "{text}"
             Which of the following sentiments best describes it? Choose one of the following: {", ".join(sentiments)}.
             return only one word - the correct sentiment from the list above."""}
        ],
  )

    sentiment = response.choices[0].message.content
    return sentiment
  
  
def determine_speaker_roles(transcript):
    # Prepare a sample of the conversation
    sample = "\n".join([f"Speaker {u.speaker}: {u.text}" for u in transcript.utterances[:15]])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an AI trained to analyze conversation transcripts and determine which speaker is the psychologist and which is the patient."},
            {"role": "user", "content": f"""Based on the following transcript sample, determine whether Speaker A is the psychologist and Speaker B is the patient, or vice versa. 
            Return your answer as a JSON object with keys 'psychologist' and 'patient', and values 'A' or 'B'.

            Transcript sample:
            {sample}"""}
        ]
    )
    
    result = response.choices[0].message.content
    # Parse the JSON string to a Python dictionary
    roles = json.loads(result)
    return roles


def search_sentences(patient_id, query):
    # Generate embedding for the query
    query_embedding = generate_query_embedding(query)

    # Fetch all embeddings for the specific patient from the database
    patient_embeddings = fetch_patient_embeddings(patient_id)

    # Find the most similar sentences
    similar_sentences = search_similar_sentences(query_embedding, patient_embeddings)

    return similar_sentences


# Lists to store the data
patient_data = []
psychologist_data = []


config = aai.TranscriptionConfig(
  speaker_labels=True,
  speakers_expected=2
)

# Initialize the database
create_tables()

# Add a new patient 
patient_id = 1  # Replace with the actual ID or fetch dynamically
name = "John Doe"
birthdate = "1980-01-01"
notes = "No specific notes"
add_patient(patient_id, name, birthdate, notes)

# Transcript and sentiment analysis
transcript = aai.Transcriber().transcribe(audio_file, config)

# Determine speaker roles
speaker_roles = determine_speaker_roles(transcript)
psychologist_speaker = speaker_roles['psychologist']
patient_speaker = speaker_roles['patient']

# Check if speaker roles determined successfully
# print(f"Determined roles: Psychologist is Speaker {psychologist_speaker}, Patient is Speaker {patient_speaker}")

session_id = "session_001"  # Example session ID

for i, utterance in enumerate(transcript.utterances):
    words = utterance.text.split()
    if utterance.speaker == patient_speaker:
        if len(words) > 4:
            sentiment = get_sentiment_patient(utterance.text)
            score = patient_sentiment_scores.get(sentiment, 0)
        else:
            sentiment = None
            score = 0
        patient_data.append({"sentence": utterance.text, "sentiment": sentiment, "score": score, "index": i})
    elif utterance.speaker == psychologist_speaker:
        if len(words) > 4:
            sentiment = get_sentiment_psychologist(utterance.text)
            score = psychologist_sentiment_scores.get(sentiment, 0)
        else:
            sentiment = None
            score = 0
        psychologist_data.append({"sentence": utterance.text, "sentiment": sentiment, "score": score, "index": i})
    
    print(f"{'Psychologist' if utterance.speaker == psychologist_speaker else 'Patient'} : {utterance.text}")
    print(f"Sentiment: {sentiment}")
    
  
# Generate and store embeddings
try:
    session_embeddings = generate_session_embeddings(transcript)
    print("Embeddings generated successfully.")
    store_embeddings(patient_id=1, session_id=session_id, session_embeddings=session_embeddings)
    print("Embeddings stored successfully in the database.")
except Exception as e:
    print(f"An error occurred during embedding or database operations: {e}")



patient_id = 1  # Example patient ID
query = "studies"
results = search_sentences(patient_id, query)

# Display the top 5 results
for result in results[:5]:
    print(f"Sentence: {result[1]} (Similarity: {result[2]:.4f})")

