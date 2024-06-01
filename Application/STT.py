import threading
import wave
import os
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import pyaudio

class SpeechToTextProcessor:
    def __init__(self):
        self.answers = {}
        self.threads = []
        arabic = False
        if not arabic:
            self.model_path = r'../models/STT/vosk-model-small-en-us-0.15'  # Path to the Vosk model
        else:
            self.model_path = r'../models\STT\vosk-model-ar-mgb2-0.4'
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        self.model = Model(self.model_path)
        # self.speaker = Speaker(1)

    def record_audio(self, record_seconds, output_file):
        p = pyaudio.PyAudio()
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        rate = 16000  # Record at 16000 samples per second

        print("Recording...")

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []

        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Recording finished.")

        wf = wave.open(output_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.transcribe_audio_file(output_file)

    def transcribe_audio_file(self, audio_file_path):
        thread = threading.Thread(target=self.transcribe_async_worker, args=(audio_file_path,))
        thread.start()
        self.threads.append(thread)

    def transcribe_async_worker(self, audio_file_path):
        try:
            text = self._transcribe_audio_file(audio_file_path)
            self.on_transcription_complete(audio_file_path, text)
        except Exception as e:
            self.on_transcription_complete(audio_file_path, None, error=e)

    def _convert_to_wav_mono_pcm(self, input_file, output_file):
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1)
        audio.export(output_file, format="wav", codec="pcm_s16le")

    def _transcribe_audio_file(self, audio_file_path):
        converted_audio_file_path = audio_file_path
        try:
            wf = wave.open(audio_file_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                converted_audio_file_path = audio_file_path.replace(".wav", "_converted.wav")
                self._convert_to_wav_mono_pcm(audio_file_path, converted_audio_file_path)
                wf.close()
                wf = wave.open(converted_audio_file_path, "rb")

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
        finally:
            if converted_audio_file_path != audio_file_path:
                os.remove(converted_audio_file_path)

    def on_transcription_complete(self, audio_file_path, text, error=None):
        if error:
            print(f"Error transcribing audio {audio_file_path}: {error}")
        else:
            print(f"Transcribed text from {audio_file_path}: {text}")
            audio_file_id = os.path.basename(audio_file_path).split('.')[0]
            self.answers[audio_file_id] = text

    def transcribe_all(self, video_name):
        self.answers = {}
        self.threads = []
        directory = f'../Application/session/{video_name}/'
        for filename in os.listdir(directory):
            if not filename.startswith('f') and filename.endswith('.wav'):
                audio_path = os.path.join(directory, filename)
                self.transcribe_audio_file(audio_path)
        
        for thread in self.threads:
            thread.join()
        
        self.answers = dict(sorted(self.answers.items()))
        return self.answers

# # Example usage:
# processor = SpeechToTextProcessor()
# processor.record_audio(4,'ay_7aga.wav')
# # results = processor.transcribe_all('2024-05-30_13-40-43')
# # print(results)
