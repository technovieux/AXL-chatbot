import fonctions_pause2
import cv2
import threading
import time
from langchain_ollama import OllamaLLM
import pyttsx4
import speech_recognition as sr
from playsound3 import playsound

model = OllamaLLM(model="stablelm-zephyr")

recognizer = sr.Recognizer()


t1 = threading.Thread(target=fonctions_pause2.chatbot)
t2 = threading.Thread(target=fonctions_pause2.objects_recognition)

t1.start()
t2.start()

t1.join()
t2.join()

print("finished")