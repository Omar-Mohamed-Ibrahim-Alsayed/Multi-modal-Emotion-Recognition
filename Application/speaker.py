import multiprocessing
import pyttsx3

class Speaker:
    def __init__(self, vc):
        self.vc = vc
        self.process = None

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
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None

        self.process = multiprocessing.Process(target=self.speak, args=(phrase,))
        self.process.start()

    def stop_speaking(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None

# if __name__ == '__main__':
#     speaker = Speaker(1)
#     speaker.start_speaking("this process is running right now")
#     speaker.stop_speaking()
#     speaker.start_speaking("is running right now")


