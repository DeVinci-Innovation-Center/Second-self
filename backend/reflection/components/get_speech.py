import os
from time import sleep

import pyglet
import requests
import speech_recognition as sr
from gtts import gTTS


def send_rasa(message: str) -> str:
    """Sends recognized text to RASA and returns assistant response"""
    response = requests.post(
        "http://localhost:5005/webhooks/rest/webhook",
        json={"sender": "mirror_user", "message": message},
    )
    return response.json()[0]["text"]


def init():
    """Creates Recognizer instance and get microphone source"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    return recognizer, microphone


def listen_to_instruction(recognizer, microphone):
    """Listen to the user and return an instruction"""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.pause_threshold = 1
        print("Speak!")
        audio_data = recognizer.listen(source)
        print("End!")

    speech = recognizer.recognize_google(audio_data, language="fr-FR")
    result = send_rasa(speech)

    print(">", result)
    respond(result)


def respond(sentence):
    """TTS response from rasa"""
    tts = gTTS(sentence, lang="fr", tld="fr")
    filename = "/tmp/temp.mp3"
    tts.save(filename)
    sppech = pyglet.media.load(filename, streaming=False)
    print("Playing")
    sppech.play()
    sleep(sppech.duration)
    os.remove(filename)


r, micro = init()
listen_to_instruction(r, micro)
