from gtts import gTTS
import os

# Sawal jo record karna hai
text = "This is so frustrating. Nothing is working and I am very tired."

# Audio file generate karna
tts = gTTS(text=text, lang='en')
tts.save("Voice Sad_question.mp3")

print("✅ test_question2.mp3 ban gayi hai! Ab isse QA page par upload karein.")