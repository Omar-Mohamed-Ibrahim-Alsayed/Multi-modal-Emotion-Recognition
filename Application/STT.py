import threading
import wave
import os
import json
from vosk import Model, KaldiRecognizer
from speaker import Speaker


class SpeechToTextProcessor:
    def __init__(self):
        self.model_path = r'..\models\STT\vosk-model-small-en-us-0.15'  # Path to the Vosk model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        self.model = Model(self.model_path)
        self.speaker = Speaker(1)

    def transcribe_audio_file(self, audio_file_path):
        thread = threading.Thread(target=self.transcribe_async_worker, args=(audio_file_path,))
        thread.start()
        return thread

    def transcribe_async_worker(self, audio_file_path):
        try:
            text = self._transcribe_audio_file(audio_file_path)
            self.on_transcription_complete(audio_file_path, text)
        except Exception as e:
            self.on_transcription_complete(audio_file_path, None, error=e)

    def _transcribe_audio_file(self, audio_file_path):
        try:
            wf = wave.open(audio_file_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise ValueError("Audio file must be WAV format mono PCM.")

            rec = KaldiRecognizer(self.model, wf.getframerate())
            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text += result.get("text", "") + " "
            result = json.loads(rec.FinalResult())
            text += result.get("text", "")
            return text
        except Exception as e:
            print(f"An error occurred while transcribing {audio_file_path}: {e}")

    def on_transcription_complete(self, audio_file_path, text, error=None):
        if error:
            print(f"Error transcribing audio {audio_file_path}: {error}")
        else:
            print(f"Transcribed text from {audio_file_path}: {text}")

# processor = SpeechToTextProcessor()
# text = processor.transcribe_audio_file("audio/03-01-03-02-02-01-11.wav")

