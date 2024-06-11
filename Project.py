import assemblyai as aai

aai.settings.api_key = "d7e18dd00ed3408dbdd1919c223752f1"

audio_file = "/Users/noyaarav/Desktop/Final-Project-From-Idea-To-Reality/audio_files/test1.mp3"

config = aai.TranscriptionConfig(
  speaker_labels=True,
  speakers_expected=2
)

transcript = aai.Transcriber().transcribe(audio_file, config)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")