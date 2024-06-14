import streamlit as st
import assemblyai as aai
from openai import OpenAI
import numpy as np
import plotly.graph_objects as go

aai.settings.api_key = "7a43eb14db35446586c8e9938f2b947c"

client = OpenAI(
    api_key="sk-proj-wQ4taTDDFhbuDrmkqIlOT3BlbkFJlJVo8Zlx0wucOcJ2atou"
)
openai_api_model = "gpt-3.5-turbo"

audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/13mins_session_depression.mp4"

def get_sentiment_patient(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant. Your task is to analyze the sentiment of the following text and describe it with 1 word only (no further explanation) by choosing the most accurate sentiment from following sentiments: Euphoria, Excitement, Happiness, Hopefulness, Contentment, Natural, Discomfort, Sadness, Anxiety, Anger, Despair. Please consider the fact that the text is a sentence said by a patient during a psychologic session. Also consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used."},
            {"role": "user", "content": f"The text for your sentiment analyze: {text}"}
        ],
    )
    sentiment = response.choices[0].message.content
    return sentiment

def get_sentiment_psychologist(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant. Your task is to analyze the sentiment of the following text and describe it with 1 word only (no further explanation) by choosing the most accurate sentiment from following sentiments: Fulfillment, Optimism, Empathy, Encouragement, Contentment, Natural, Concern, Frustration, Sadness, Helplessness, Overwhelm. Please consider the fact that the text is a sentence said by a psychologist during a session with a patient. Also consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used.(for example, if it a question asked for a specific purpose, the emotion represented in the question is not necessarily the emotion of the psychologist)"},
            {"role": "user", "content": f"The text for your sentiment analyze: {text}"}
        ],
    )
    sentiment = response.choices[0].message.content
    return sentiment

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
    "Empathy": 2,
    "Encouragement": 3,
    "Optimism": 4,
    "Fulfillment": 5
}

def is_relevant_change(sentiment_scores, index, window_size=3, threshold=3):
    """
    Determine whether a change in sentiment is relevant based on nearby sentences.
    
    Parameters:
    - sentiment_scores: List of sentiment scores.
    - index: Index of the current sentiment.
    - window_size: Number of sentences to consider before and after the current sentence.
    - threshold: Minimum score change to consider a change relevant.
    
    Returns:
    - True if the change is relevant, False otherwise.
    """
    start = max(0, index - window_size//2)
    end = min(len(sentiment_scores), index + window_size//2 + 1)
    
    # Get the sentiment scores within the window
    window_scores = sentiment_scores[start:end]
    
    # Calculate the average sentiment score in the window
    average_score = np.mean(window_scores)
    
    # Determine if the change is significant
    if abs(sentiment_scores[index] - average_score) >= threshold:
        return True
    return False

# Lists to store the data
patient_data = []
psychologist_data = []
sentiment_scores_patient = []
sentiment_scores_psychologist = []

config = aai.TranscriptionConfig(
    speaker_labels=True,
    speakers_expected=2
)

transcript = aai.Transcriber().transcribe(audio_file, config)

for utterance in transcript.utterances:
    if utterance.speaker == 'B':
        sentiment = get_sentiment_patient(utterance.text)
        score = patient_sentiment_scores.get(sentiment, 0)
        sentiment_scores_patient.append(score)
        if len(utterance.text.split()) >= 5:
            patient_data.append({"sentence": utterance.text, "sentiment": sentiment, "score": score, "index": len(patient_data)})
    elif utterance.speaker == 'A':
        sentiment = get_sentiment_psychologist(utterance.text)
        score = psychologist_sentiment_scores.get(sentiment, 0)
        sentiment_scores_psychologist.append(score)
        if len(utterance.text.split()) >= 5:
            psychologist_data.append({"sentence": utterance.text, "sentiment": sentiment, "score": score, "index": len(psychologist_data)})

# For checking relevant changes
print("Patient Data:", patient_data)

# Streamlit application
st.title("Patient Monitor Application")

st.header("Session Transcript")
for utterance in transcript.utterances:
    if utterance.speaker == 'A':
        st.write(f"**Psychologist**: {utterance.text}")
    elif utterance.speaker == 'B':
        st.write(f"**Patient**: {utterance.text}")

# Plot sentiment analysis
def plot_sentiment(data, title, sentiment_scores):
    fig = go.Figure()

    x = [item['index'] for item in data]
    y = [item['score'] for item in data]
    text = [f"Sentence: {item['sentence']}<br>Sentiment: {item['sentiment']}" for item in data]
    color = ['green' if item['score'] >= 0 else 'red' for item in data]

    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode='lines+markers', 
        marker=dict(color=color),
        text=text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Sentence Index",
        yaxis_title="Sentiment Score",
        yaxis=dict(
            tickmode='array',
            tickvals=list(sentiment_scores.values()),
            ticktext=list(sentiment_scores.keys()),
            range=[-5.5, 5.5]
        ),
        showlegend=False
    )

    return fig

st.header("Sentiment Analysis")

st.subheader("Patient Sentiment")
patient_fig = plot_sentiment(patient_data, "Patient Sentiment Analysis", patient_sentiment_scores)
st.plotly_chart(patient_fig)

st.subheader("Psychologist Sentiment")
psychologist_fig = plot_sentiment(psychologist_data, "Psychologist Sentiment Analysis", psychologist_sentiment_scores)
st.plotly_chart(psychologist_fig)