import assemblyai as aai
from openai import OpenAI

aai.settings.api_key = "7a43eb14db35446586c8e9938f2b947c"

client = OpenAI(
  api_key = "sk-proj-wQ4taTDDFhbuDrmkqIlOT3BlbkFJlJVo8Zlx0wucOcJ2atou"
)
openai_api_model = "gpt-3.5-turbo"


audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/13mins_session_depression.mp4"
# audio_file = "https://www.youtube.com/watch?v=7LD8iC4NqXM" 

def get_sentiment_patient(text):
  
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0, # Use a larger temperature value to generate more diverse results.
      messages=[
          {"role": "system", "content": "You are a sentiment analysis assistant. Your task is to analyze the sentiment of the following text and describe it with 1 word only (no further explanation) by choosing the most accurate sentiment from following sentiments: Euphoria, Excitement, Happiness, Hopefulness, Contentment, Natural, Discomfort, Concern, Sadness, Anxiety, Anger, Despair. Please consider the fact that the text is a sentence said by a patient during a psychologic session. Also consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used."},
          {"role": "user", "content": f"The text for your sentiment analyze: {text}"}
      ],
    
  )
  sentiment = response.choices[0].message.content
  return sentiment

  
def get_sentiment_psychologist(text):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0, # Use a larger temperature value to generate more diverse results.
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
    "Encouragement": 2,
    "Empathy": 3,
    "Optimism": 4,
    "Fulfillment": 5
}

# Lists to store the data
patient_data = []
psychologist_data = []



config = aai.TranscriptionConfig(
  speaker_labels=True,
  speakers_expected=2
)

transcript = aai.Transcriber().transcribe(audio_file, config)

for i, utterance in enumerate(transcript.utterances):
  print(f"Speaker {utterance.speaker}: {utterance.text}")
  if utterance.speaker == 'B':
    sentiment = get_sentiment_patient(utterance.text)
    score = patient_sentiment_scores.get(sentiment, 0)
    patient_data.append({"sentence": utterance.text, "sentiment": sentiment, "score": score, "index": i})
    print(f"Sentiment: {sentiment}")
  elif utterance.speaker == 'A':
    sentiment = get_sentiment_psychologist(utterance.text)
    score = psychologist_sentiment_scores.get(sentiment, 0)
    psychologist_data.append({"sentence": utterance.text, "sentiment": sentiment, "score": score, "index": i})
    print(f"Sentiment: {sentiment}")  
  
  
