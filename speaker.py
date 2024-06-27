import pyttsx3
import os
import pygame
import threading

class Speaker:
    def __init__(self, vc):
        self.vc = vc
        self.phrase_to_file = {}
        self.audio_folder = "QuestionsAudio"
        os.makedirs(self.audio_folder, exist_ok=True)
        self.initialize_pygame()

    def initialize_pygame(self):
        try:
            pygame.mixer.init()
            print("Pygame mixer initialized successfully.")
        except pygame.error as e:
            print(f"Failed to initialize pygame mixer: {e}")

    def prepare_sounds(self, questions):
        for idx, question in enumerate(questions):
            self.save_audio(question, idx + 1)

    def save_audio(self, phrase, index):
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if 0 <= self.vc < len(voices):
            engine.setProperty('voice', voices[self.vc].id)
        else:
            print("Invalid voice index")
            return

        filename = os.path.join(self.audio_folder, f"question_{index}.wav")
        engine.save_to_file(phrase, filename)
        engine.runAndWait()
        self.phrase_to_file[phrase] = filename

    def play_audio(self, filename):
        if not pygame.mixer.get_init():
            self.initialize_pygame()

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def speak(self, phrase):
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if 0 <= self.vc < len(voices):
            engine.setProperty('voice', voices[self.vc].id)
        else:
            print("Invalid voice index")
            return
        engine.say(phrase)
        engine.runAndWait()
        engine.stop()

    def start_speaking(self, phrase):
        if phrase not in self.phrase_to_file:
            print("Phrase not pre-generated")
            threading.Thread(target=self.speak, args=(phrase,)).start()
        else:
            print("Phrase pre-generated")
            threading.Thread(target=self.play_audio, args=(self.phrase_to_file[phrase],)).start()

    def stop_speaking(self):
        pygame.mixer.music.stop()
