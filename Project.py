import assemblyai as aai
from openai import OpenAI
import json

from embedding_handler import generate_embedding, generate_query_embedding, search_similar_sentences
from sklearn.metrics.pairwise import cosine_similarity

from database_handler import create_tables, insert_session_data, update_session_embedding, add_patient, fetch_patient_embeddings, fetch_session_data


aai.settings.api_key = "7a43eb14db35446586c8e9938f2b947c"

client = OpenAI(
  api_key = "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"
)
openai_api_model = "gpt-3.5-turbo"

# audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/13mins_session_depression.mp4"
audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/session2_social_anxiety_Hannah.mp4"


# Sentiment score dictionaries
patient_sentiment_scores = {
    "Despair": -5, "Anger": -4, "Anxiety": -3, "Sadness": -2, "Discomfort": -1,
    "Natural": 0, "Contentment": 1, "Hopefulness": 2, "Happiness": 3, "Excitement": 4, "Euphoria": 5
}

psychologist_sentiment_scores = {
    "Overwhelm": -5, "Helplessness": -4, "Sadness": -3, "Frustration": -2, "Concern": -1,
    "Natural": 0, "Contentment": 1, "Encouragement": 2, "Empathy": 3, "Optimism": 4, "Fulfillment": 5
}

def get_sentiment(text, is_patient):
    sentiments = list(patient_sentiment_scores.keys()) if is_patient else list(psychologist_sentiment_scores.keys())
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "user", "content": f"""Given the following sentence from a {'patient' if is_patient else 'psychologist'} during a therapy session:
             "{text}"
             Which of the following sentiments best describes it? Choose one of the following: {", ".join(sentiments)}.
             Return only one word - the correct sentiment from the list above."""}
        ],
    )
    return response.choices[0].message.content.strip()

    
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

def detect_drastic_changes(session_data, threshold):
    print(f"Starting drastic change detection with threshold: {threshold}")
    print(f"Number of entries in session_data: {len(session_data)}")
    
    drastic_changes = []
    used_ids = set()

    for i in range(1, len(session_data)):
        current = session_data[i]
        previous = session_data[i-1]
        
        words_current = len(current['sentence'].split())
        words_previous = len(previous['sentence'].split())

        print(f"\nComparing entries {i-1} and {i}:")
        print(f"  Previous: {previous['sentence'][:50]}... (Words: {words_previous}, Score: {previous['sentiment_score']})")
        print(f"  Current:  {current['sentence'][:50]}... (Words: {words_current}, Score: {current['sentiment_score']})")

        if words_current > 5 and words_previous > 5 and current['id'] not in used_ids and previous['id'] not in used_ids:
            score_change = current['sentiment_score'] - previous['sentiment_score']
            print(f"  Score change: {score_change}")

            if abs(score_change) >= threshold:
                drastic_changes.append((previous['id'], current['id'], score_change))
                used_ids.add(current['id'])
                used_ids.add(previous['id'])
                print(f"  Drastic change detected!")
            else:
                print(f"  Change not significant enough. Threshold: {threshold}, Actual change: {abs(score_change)}")
        else:
            if words_current <= 5 or words_previous <= 5:
                print("  Skipped due to insufficient word count")
            elif current['id'] in used_ids or previous['id'] in used_ids:
                print("  Skipped due to previously used IDs")

    print(f"\nTotal drastic changes detected: {len(drastic_changes)}")
    return drastic_changes[:7]
  
def get_context_for_change(transcript, index1, index2):
  # Function to get a 7-sentence context around the change
    
    context = []

    # Ensure indices are within the bounds of the transcript
    context.append(transcript.utterances[max(0, index1 - 3)].text)  # 2 sentences before
    context.append(transcript.utterances[max(0, index1 - 2)].text)  # 1 sentence before
    context.append(transcript.utterances[index1 - 1].text)              # First sentence in the change
    context.append(transcript.utterances[index1].text)          # Sentence in between (response by the other person)
    context.append(transcript.utterances[index2 - 1].text)              # Second sentence in the change
    context.append(transcript.utterances[min(len(transcript.utterances) - 1, index2)].text)  # 1 sentence after
    context.append(transcript.utterances[min(len(transcript.utterances) - 1, index2 + 1)].text)  # 2 sentences after
    
    return context

def identify_topic_of_change(sentences, sentence_1, sentence_2, change, emotion_1, emotion_2, speaker):
    """
    Identifies the topic of conversation that caused a drastic change in the emotion of the patient.

    Parameters:
    - sentences: A list of 7 sentences from the transcript to provide context.
    - sentence_1: The first sentence causing the drastic change.
    - sentence_2: The second sentence causing the drastic change.
    - change: The value of the change (positive or negative).
    - emotion_1: The emotion identified for sentence_1.
    - emotion_2: The emotion identified for sentence_2.
    - openai_client: The OpenAI client object for sending requests to ChatGPT.

    Returns:
    - The topic of the conversation causing the drastic change, or "No drastic emotion change" if no actual change is detected.
    """
    
    # Determine if the change is positive or negative
    change_type = "positive" if change > 0 else "negative"
    
    second_speaker = "psychologist" if speaker == "patient" else "patient"

    # Construct the prompt for ChatGPT
    prompt = f"""
    
Based on the following context, please determine the topic of conversation that caused a drastic {change_type} change in the emotion of the {speaker}:

Context sentences (for reference):
1. {speaker}: {sentences[0]}
2. {second_speaker}: {sentences[1]}
3. {speaker}: {sentences[2]} (First sentence in the drastic change)
4. {second_speaker}: {sentences[3]} (Response by the {second_speaker})
5. {speaker}: {sentences[4]} (Second sentence in the drastic change)
6. {second_speaker}: {sentences[5]}
7. {speaker}: {sentences[6]}

Specific sentences detected to have caused the drastic change:
- Sentence 1: "{sentence_1}" with emotion "{emotion_1}".
- Sentence 2: "{sentence_2}" with emotion "{emotion_2}".

Drastic change value: {change} (indicating a {change_type} change).

Analyze the sentences above and provide the topic of conversation that caused the drastic change. Your answer should include only the topic and it should be 8 words at most. 
If there is no actual drastic change in emotion in the provided context and the detection might be a mistake, please return "No drastic emotion change".
"""

    # Send the prompt to ChatGPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,  # Lower temperature for more deterministic output
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with analyzing transcripts of therapy sessions to identify emotional changes and the topics that cause them."},
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the content of the response
    topic = response.choices[0].message.content.strip()

    return topic


def process_session(audio_file, patient_id, session_id):
    config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=2)
    transcript = aai.Transcriber().transcribe(audio_file, config)

    speaker_roles = determine_speaker_roles(transcript)
    psychologist_speaker = speaker_roles['psychologist']
    patient_speaker = speaker_roles['patient']

    for i, utterance in enumerate(transcript.utterances):
        is_patient = utterance.speaker == patient_speaker
        speaker = 'patient' if is_patient else 'psychologist'
        words = utterance.text.split()

        if len(words) > 4:
            sentiment = get_sentiment(utterance.text, is_patient)
            score = patient_sentiment_scores.get(sentiment, 0) if is_patient else psychologist_sentiment_scores.get(sentiment, 0)
        else:
            sentiment = None
            score = 0

        # Insert session data into the database
        insert_session_data(patient_id, session_id, utterance.text, speaker, sentiment, score)

        print(f"{speaker.capitalize()}: {utterance.text}")
        print(f"Sentiment: {sentiment}, Score: {score}")

        # Generate and update embedding
        embedding = generate_embedding(utterance.text)
        update_session_embedding(session_id, utterance.text, embedding)

    print("Session processing and data storage completed successfully.")


# Initialize the database
create_tables()

# Add a new patient 
patient_id = 2  # Replace with the actual ID or fetch dynamically
name = "Hannah"
birthdate = "2000-06-01"
notes = "Social anxiety"
add_patient(patient_id, name, birthdate, notes)

# Process the session
session_id = "session_001"
process_session(audio_file, patient_id, session_id)

# Fetch session data for analysis
print("Fetching session data...")
session_data = fetch_session_data(patient_id, session_id)
print(f"Fetched {len(session_data)} entries from the database.")

# Separate patient and psychologist data
print("Separating patient and psychologist data...")
patient_data = [entry for entry in session_data if entry['speaker'] == 'patient']
psychologist_data = [entry for entry in session_data if entry['speaker'] == 'psychologist']
print(f"Patient data: {len(patient_data)} entries")
print(f"Psychologist data: {len(psychologist_data)} entries")

# Define thresholds for drastic emotion change
patient_threshold = 3

# Detect drastic changes for patient
print("\nDetecting drastic changes for patient...")
patient_drastic_changes = detect_drastic_changes(patient_data, patient_threshold)

print("\nAnalyzing each drastic change for the patient...")
for change in patient_drastic_changes:
    id1, id2, change_value = change
    print(f"\nAnalyzing change between ids {id1} and {id2} with change value {change_value}")
    
    context = [entry for entry in session_data if id1 - 3 <= entry['id'] <= id2 + 3]
    sentence_1 = next(item['sentence'] for item in patient_data if item['id'] == id1)
    sentence_2 = next(item['sentence'] for item in patient_data if item['id'] == id2)
    emotion_1 = next(item['sentiment'] for item in patient_data if item['id'] == id1)
    emotion_2 = next(item['sentiment'] for item in patient_data if item['id'] == id2)

    print(f"Context size: {len(context)} sentences")
    print(f"Sentence 1: {sentence_1[:50]}...")
    print(f"Sentence 2: {sentence_2[:50]}...")
    print(f"Emotion 1: {emotion_1}")
    print(f"Emotion 2: {emotion_2}")

    topic = identify_topic_of_change([item['sentence'] for item in context], sentence_1, sentence_2, change_value, emotion_1, emotion_2, "patient")
    print(f"Identified Topic for Patient's Drastic Change: {topic}")

print("\nDrastic change analysis complete.")

# Example of embedding-based search
print("\nPerforming embedding-based search...")
query = "studies"
results = search_similar_sentences(patient_id, query)

# Display the top 5 results
print(f"\nTop 5 results for query '{query}':")
for result in results[:5]:
    print(f"Sentence: {result[1][:50]}... (Similarity: {result[2]:.4f})")