
import os
from gtts import gTTS


def createAudio(classes):
    for class_ in classes:

        tts = gTTS(text=class_, lang='en')
        tts.save(class_+".mp3")
