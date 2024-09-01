import os
import tempfile
from pathlib import Path
import assemblyai as aai
from openai import OpenAI
import json
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import textwrap

from embedding_handler import generate_embedding, generate_query_embedding, search_similar_sentences
from database_handler import create_tables, insert_session_data, update_session_embedding, add_patient, fetch_patient_embeddings, fetch_session_data, insert_topic

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
    prompt = f"""
    Analyze what sentiment the {"patient" if is_patient else "psychologist"} is experiencing when saying the following text, which is a part of a psychologist-patient conversation. Respond with:
    1. A single word (not the word positive or negative) or short phrase that best describes the emotion.
    2. A number between -10 and 10 representing the intensity and polarity of the emotion, where:
       - -10 represents the most extreme negative emotion (e.g., severe depression, intense hatred)
       - 0 represents a neutral state
       - 10 represents the most extreme positive emotion (e.g., ecstatic joy, intense love)
    Provide your response in the format: "Sentiment: [word/phrase], Score: [number]"
    
    Text: "{text}"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled psychologist with expertise in emotion analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content.strip()
    try:
        if ',' not in result:
            raise ValueError("Unexpected response format")
        
        sentiment, score = result.split(', ')
        sentiment = sentiment.split(': ')[1]
        score = float(score.split(': ')[1])
        return sentiment, score
    except (ValueError, IndexError):
        print(f"Warning: Unexpected response format. Raw response: {result}")
        return None, 0

    
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
    
    # Initialize an empty list to collect session data for topic detection later
    session_data = []

    for i, utterance in enumerate(transcript.utterances):
        is_patient = utterance.speaker == patient_speaker
        speaker = 'patient' if is_patient else 'psychologist'
        words = utterance.text.split()

        if len(words) > 4:
            sentiment, score = get_sentiment(utterance.text, is_patient)  
        else:
            sentiment = None
            score = 0

        # Insert session data into the database
        insert_session_data(patient_id, session_id, utterance.text, speaker, sentiment, score)
        
        # Collect session data in a list for later use in topic detection
        session_data.append({
            'id': i + 1,  # Assuming ID is the index + 1
            'sentence': utterance.text,
            'speaker': speaker,
            'sentiment': sentiment,
            'sentiment_score': score
        })

        print(f"{speaker.capitalize()}: {utterance.text}")
        print(f"Sentiment: {sentiment}, Score: {score}")

        # Generate and update embedding
        embedding = generate_embedding(utterance.text)
        update_session_embedding(session_id, utterance.text, embedding)
    
    # Detect topics causing drastic emotional changes
    detect_and_store_topics(patient_id, session_id, session_data)

    print("Session processing and data storage completed successfully.")
    

def detect_and_store_topics(patient_id, session_id, session_data):
    """
    Detects topics that caused drastic emotional changes and stores them in the database.

    Args:
    - patient_id (int): The ID of the patient.
    - session_id (str): The ID of the session.
    - session_data (list): A list of dictionaries containing session data.
    """
    # Filter patient data from session data
    patient_data = [entry for entry in session_data if entry['speaker'] == 'patient']
    threshold = 3  # Example threshold for detecting drastic changes

    # Detect drastic changes in sentiment
    drastic_changes = detect_drastic_changes(patient_data, threshold)
    topics = []

    for change in drastic_changes:
        id1, id2, change_value = change
        context = [entry for entry in session_data if id1 - 3 <= entry['id'] <= id2 + 3]

        # Check if there are at least 6 sentences in the context
        if len(context) < 7:
            print(f"Skipping change {id1} to {id2} due to insufficient context")
            continue

        sentence_1 = next(item['sentence'] for item in patient_data if item['id'] == id1)
        sentence_2 = next(item['sentence'] for item in patient_data if item['id'] == id2)
        emotion_1 = next(item['sentiment'] for item in patient_data if item['id'] == id1)
        emotion_2 = next(item['sentiment'] for item in patient_data if item['id'] == id2)

        # Identify the topic of change
        topic = identify_topic_of_change([item['sentence'] for item in context], sentence_1, sentence_2, change_value, emotion_1, emotion_2, "patient")
        topics.append(topic)

    # Insert detected topics into the database
    if topics:
        for topic in topics:
            insert_topic(patient_id, session_id, topic)
            print(f"Inserted topic: {topic}")
    else:
        print("No drastic emotional changes detected in this session.")
        
        
def identify_topics_to_revisit(patient_id):
    """
    Identify topics that caused drastic emotional changes in the session before the last 
    but were not discussed in the last session for a specific patient, using ChatGPT 
    to account for similar meanings with different phrasings.

    Args:
    - patient_id (int): The ID of the patient.

    Returns:
    - List[str]: Topics to revisit if more than one session exists; otherwise, an empty list.
    """
    # Establish a connection to the database
    conn = sqlite3.connect('patient_sessions.db')
    cursor = conn.cursor()

    # Fetch the last two session IDs for the patient
    cursor.execute("SELECT DISTINCT session_id FROM sessions WHERE patient_id = ? ORDER BY session_id DESC LIMIT 2", (patient_id,))
    session_ids = cursor.fetchall()

    # Check if there are at least two sessions
    if len(session_ids) < 2:
        conn.close()
        return []

    last_session_id, prev_session_id = session_ids[0][0], session_ids[1][0]

    # Fetch topics for the last session
    cursor.execute("SELECT topic FROM topics WHERE patient_id = ? AND session_id = ?", (patient_id, last_session_id))
    last_session_topics = [row[0] for row in cursor.fetchall()]

    # Fetch topics for the session before the last
    cursor.execute("SELECT topic FROM topics WHERE patient_id = ? AND session_id = ?", (patient_id, prev_session_id))
    prev_session_topics = [row[0] for row in cursor.fetchall()]

    conn.close()

    # Use ChatGPT to determine which topics from the previous session should be revisited
    topics_to_revisit = get_topics_to_revisit(prev_session_topics, last_session_topics)

    return topics_to_revisit
  

def get_topics_to_revisit(prev_session_topics, last_session_topics):
    """
    Use ChatGPT to determine which topics from the previous session should be revisited,
    accounting for different phrasings and similar meanings.

    Args:
    - prev_session_topics (List[str]): Topics from the session before the last.
    - last_session_topics (List[str]): Topics from the last session.

    Returns:
    - List[str]: Topics to revisit.
    """
    # Combine the topics into a single prompt for ChatGPT to analyze
    prompt = f"""
    Compare the following two lists of topics. Identify which topics from the first list were not discussed in the second list, considering that topics may be phrased differently but could have the same or similar meanings.

    Topics from the session before the last: {prev_session_topics}
    
    Topics from the last session: {last_session_topics}
    
    Provide just a list (withot any other heading or prefix like "topic" or more information) of topics from the first list that are not present in the second list, taking into account synonyms or different phrasings.
    Please includee in your response only the topics, without further information. thank you.
    """

    # Send the prompt to ChatGPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,  # Lower temperature for more deterministic output
        messages=[
            {"role": "system", "content": "You are an expert assistant in analyzing conversational topics for therapeutic sessions."},
            {"role": "user", "content": prompt}
        ],
    )

    
    output = response.choices[0].message.content.strip()

    # Remove any empty strings or unnecessary spaces from the list
    topics_to_revisit = output.split('\n')  # Split based on newline to separate topics

    return topics_to_revisit

    
def process_audio_file(patient_id, audio_file):
    """
    Processes the uploaded audio file by calling the process_session function.
    It generates a new session ID and stores all the session data into the database.

    Parameters:
    - patient_id: ID of the patient associated with the session
    - audio_file: Uploaded audio file object
    """
    # Create a temporary directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    # Generate a unique filename
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix, dir=temp_dir)
    temp_file_path = temp_file.name

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Generate a new session ID
        session_id = generate_new_session_id(patient_id)

        # Call the process_session function to handle transcription, sentiment, and storage
        process_session(temp_file_path, patient_id, session_id)

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def generate_new_session_id(patient_id):
    """
    Generates a new session ID for the patient by finding the maximum session ID
    associated with the patient and adding one.

    Parameters:
    - patient_id: ID of the patient

    Returns:
    - A new session ID (integer)
    """
    conn = sqlite3.connect('patient_sessions.db')  
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(CAST(session_id AS INTEGER)) FROM sessions WHERE patient_id = ?", (patient_id,))
    max_session_id = cursor.fetchone()[0]

    conn.close()

    # If no sessions exist yet for this patient, start with 1
    if max_session_id is None:
        return 1
    else:
        return int(max_session_id) + 1
      
      
      

def generate_sentiment_graph(session_data, title, sentiment_scores , speaker):
    """
    Generates a sentiment graph for a given session.
    
    Parameters:
    - session: A dictionary containing session data, including sentences, sentiments, and scores.

    Returns:
    - A Plotly Figure object representing the sentiment graph.
    """
# Filter for specific speaker sentences
    speaker_data = [row for row in session_data if row['speaker'].lower() == speaker.lower() and row['sentiment'] is not None]

    # Extract data, using enumeration starting from 1 for x values
    x = list(range(1, len(speaker_data) + 1))
    y = [row['sentiment_score'] for row in speaker_data]
    text = [row['sentence'] for row in speaker_data]
    sentiments = [row['sentiment'] for row in speaker_data]

    # Create a wider figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Function to wrap text
    def wrap_text(text, width=50):
        return '<br>'.join(textwrap.wrap(text, width=width))

    # Add line and scatter plot for sentiment scores
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(color='rgba(0,100,80,0.5)', width=2),
            marker=dict(
                color=y,
                colorscale='RdYlGn',
                cmin=-10,
                cmax=10,
                size=10,
                colorbar=dict(
                    title="",
                    thickness=15,
                    len=1.05,  
                    yanchor="middle",
                    y=0.5,
                    x=-0.07,
                    tickvals=[-10, -5, 0, 5, 10],
                    ticktext=["", "", "", "", ""]
                )
            ),
            text=[f"<b>Sentiment:</b> {sentiment}<br><b>Score:</b> {int(score)}<br><b>Sentence:</b> {wrap_text(sentence)}" 
                  for sentiment, score, num, sentence in zip(sentiments, y, x, text)],
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
        )
    )

    # Calculate the range for x-axis ticks
    x_max = max(x)
    x_range_max = ((x_max // 10) + 1) * 10  # Round up to nearest 10

    # Generate tick values at intervals of 10
    tick_values = list(range(0, x_range_max + 1, 10))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Sentence Number",
        yaxis_title="",
        yaxis=dict(
            range=[-10.5, 10.5],
            zeroline=False,
            tickvals=[-10, -5, 0, 5, 10],
            ticktext=["-10", "-5", "0", "5", "10"],
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=[str(val) for val in tick_values],
            range=[0, x_range_max],
        ),
        showlegend=False,
        width=1000,
        height=600,
        margin=dict(l=100, r=50, t=50, b=50),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        )
    )

    # Add "Sentiment Score" label to the left of the scale
    fig.add_annotation(
        x=-0.09,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Sentiment Score",
        showarrow=False,
        textangle=-90,
        font=dict(size=12),
        align="center",
    )

    return fig