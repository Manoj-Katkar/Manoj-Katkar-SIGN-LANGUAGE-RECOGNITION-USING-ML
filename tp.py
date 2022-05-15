from googletrans import Translator, LANGUAGES
from gtts import gTTS
from playsound import playsound
import os

translator = Translator()
t = translator.translate('My name is sahil',dest='mr')
print("Source:",t.src)
print("Destination:",t.dest)
print(t.origin,"->",t.text)
# speakText()
# print(LANGUAGES)

# def speakText():
language='en'
output = gTTS(text=(t.text.lower()), lang=language, slow=False)
output.save("speak.mp3")
playsound("speak.mp3")