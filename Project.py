import assemblyai as aai
from openai import OpenAI

aai.settings.api_key = "7a43eb14db35446586c8e9938f2b947c"

client = OpenAI(
  api_key = "sk-proj-wQ4taTDDFhbuDrmkqIlOT3BlbkFJlJVo8Zlx0wucOcJ2atou"
)
openai_api_model = "gpt-3.5-turbo"


audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/13mins_session_depression.mp4"
# audio_file = "https://www.youtube.com/watch?v=7LD8iC4NqXM" 

def get_sentiment(text, speaker):
  
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0, # Use a larger temperature value to generate more diverse results.
      messages=[
          {"role": "system", "content": "You are a sentiment analysis assistant. Your task is to analyze the sentiment of the following text and describe it with 1 word only (no further explanation) by choosing the most accurate sentiment from following sentiments: Euphoria, Excitement, Happiness, Hopefulness, Contentment, Natural, Discomfort, Concern, Sadness, Anxiety, Anger, Despair. Please consider the speaker's role- whether it's the psychologist, which is speaker A, or the patient, which is speaker B. Also consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used."},
          {"role": "user", "content": f"Speaker: {speaker}, the text for your sentiment analyze: {text}"}
      ],
    
      # max_tokens=60, # Use a larger max_tokens value to generate longer output.
      # top_p=1,
      # frequency_penalty=0, # Use a smaller frequency_penalty value to encourage the model to use more diverse vocabulary.
      # presence_penalty=0, # Use a larger presence_penalty value to discourage the model from repeating itself.
      # stop=["\n"]
  )

  sentiment = response.choices[0].message.content
  return sentiment



config = aai.TranscriptionConfig(
  speaker_labels=True,
  speakers_expected=2
)

transcript = aai.Transcriber().transcribe(audio_file, config)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")
  sentiment = get_sentiment(utterance.text, utterance.speaker)
  print(f"Sentiment: {sentiment}")
  
  
