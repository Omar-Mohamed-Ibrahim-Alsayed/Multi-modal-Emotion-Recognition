# import pyttsx3
# import os
# import pygame
# import threading

# class Speaker:
#     def __init__(self, vc):
#         self.vc = vc
#         self.phrase_to_file = {}
#         self.audio_folder = "QuestionsAudio"
#         os.makedirs(self.audio_folder, exist_ok=True)
#         self.initialize_pygame()

#     def initialize_pygame(self):
#         try:
#             pygame.mixer.pre_init(frequency=44100, size=16, channels=2, buffer=4096)

#             pygame.mixer.init()
#             print("Pygame mixer initialized successfully.")
#         except pygame.error as e:
#             print(f"Failed to initialize pygame mixer: {e}")

#     def prepare_sounds(self, questions):
#         for idx, question in enumerate(questions):
#             self.save_audio(question, idx + 1)

#     def save_audio(self, phrase, index):
#         engine = pyttsx3.init()
#         voices = engine.getProperty("voices")
#         if 0 <= self.vc < len(voices):
#             engine.setProperty('voice', voices[self.vc].id)
#         else:
#             print("Invalid voice index")
#             return

#         filename = os.path.join(self.audio_folder, f"question_{index}.wav")
#         engine.save_to_file(phrase, filename)
#         engine.runAndWait()
#         self.phrase_to_file[phrase] = filename

#     def play_audio(self, filename):
#         if not pygame.mixer.get_init():
#             self.initialize_pygame()

#         try:
#             pygame.mixer.music.load(filename)
#             pygame.mixer.music.play()
#             while pygame.mixer.music.get_busy():
#                 pygame.time.Clock().tick(10)
#         except pygame.error as e:
#             print(f"Failed to play audio: {e}")

#     def speak(self, phrase):
#         engine = pyttsx3.init(driverName='sapi5')
#         voices = engine.getProperty("voices")
#         if 0 <= self.vc < len(voices):
#             engine.setProperty('voice', voices[self.vc].id)
#         else:
#             print("Invalid voice index")
#             return
#         engine.say(phrase)
#         engine.runAndWait()
#         engine.stop()

#     def start_speaking(self, phrase):
#         if phrase not in self.phrase_to_file:
#             print("Phrase not pre-generated")
#             threading.Thread(target=self.speak, args=(phrase,)).start()
#         else:
#             print("Phrase pre-generated")
#             threading.Thread(target=self.play_audio, args=(self.phrase_to_file[phrase],)).start()

#     def stop_speaking(self):
#         if pygame.mixer.get_init():
#             pygame.mixer.music.stop()
#         else:
#             print("Pygame mixer not initialized")

# # Usage example (assuming voice index 1 and a list of questions)
# speaker = Speaker(vc=11)
# #speaker.prepare_sounds(["How are you?", "What is your name?"])
# speaker.start_speaking("How are you?")
import os
import pygame
import threading
from gtts import gTTS

class Speaker:
    def __init__(self):
        self.phrase_to_file = {}
        self.audio_folder = "QuestionsAudio"
        os.makedirs(self.audio_folder, exist_ok=True)
        self.initialize_pygame()

    def initialize_pygame(self):
        try:
            pygame.mixer.pre_init(frequency=44100, size=16, channels=2, buffer=4096)
            pygame.mixer.init()
            print("Pygame mixer initialized successfully.")
        except pygame.error as e:
            print(f"Failed to initialize pygame mixer: {e}")

    def prepare_sounds(self, questions):
        for idx, question in enumerate(questions):
            self.save_audio(question, idx + 1)

    def save_audio(self, phrase, index):
        tts = gTTS(text=phrase, lang='en')
        filename = os.path.join(self.audio_folder, f"question_{index}.mp3")
        tts.save(filename)
        self.phrase_to_file[phrase] = filename

    def play_audio(self, filename):
        if not pygame.mixer.get_init():
            self.initialize_pygame()

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except pygame.error as e:
            print(f"Failed to play audio: {e}")

    def speak(self, phrase):
        tts = gTTS(text=phrase, lang='en')
        temp_filename = "temp.mp3"
        tts.save(temp_filename)
        self.play_audio(temp_filename)
        os.remove(temp_filename)

    def start_speaking(self, phrase):
        if phrase not in self.phrase_to_file:
            print("Phrase not pre-generated")
            threading.Thread(target=self.speak, args=(phrase,)).start()
        else:
            print("Phrase pre-generated")
            threading.Thread(target=self.play_audio, args=(self.phrase_to_file[phrase],)).start()

    def stop_speaking(self):
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        else:
            print("Pygame mixer not initialized")

# # Usage example (assuming a list of questions)
# speaker = Speaker()
# # speaker.prepare_sounds(["How are you?", "What is your name?"])
# speaker.start_speaking("How are you?")
