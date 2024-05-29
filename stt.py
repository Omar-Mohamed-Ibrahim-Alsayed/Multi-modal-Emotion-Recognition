import deepspeech
import numpy as np
import pyaudio
import wave
import time
import os

# Path to the DeepSpeech model file and scorer file
MODEL_PATH = os.path.join('deepspeech', 'deepspeech-0.9.3-models.pbmm')
SCORER_PATH = os.path.join('deepspeech', 'deepspeech-0.9.3-models.scorer')

# Initialize the DeepSpeech model
try:
    model = deepspeech.Model(MODEL_PATH)
    model.enableExternalScorer(SCORER_PATH)
except Exception as e:
    print(f"Error loading model or scorer: {e}")
    exit(1)

# Audio recording parameters
RATE = 16000
CHUNK = 1024

def record_audio(duration):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error initializing stream: {e}")
        p.terminate()
        return None
    
    print("Recording...")
    frames = []
    try:
        for _ in range(int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")
    
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return b''.join(frames)

def transcribe_audio(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    try:
        text = model.stt(audio_array)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""
    return text

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Recording starts in {i} seconds...")
        time.sleep(1)
    print("Recording starts now!")

if __name__ == "__main__":
    countdown(3)
    audio_data = record_audio(duration=10)
    if audio_data:
        text = transcribe_audio(audio_data)
        print("Transcribed Text: ", text)
    else:
        print("Failed to record audio.")
# import torch
# import torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# import pyaudio
# import numpy as np

# # Load model and processor
# processor = Wav2Vec2Processor.from_pretrained("Zaid/av2Vec2-Large-XLSR-53-Tamil")
# model = Wav2Vec2ForCTC.from_pretrained("Zaid/av2Vec2-Large-XLSR-53-Tamil")
# model.eval()

# def record_audio(seconds=5, sr=16000):
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=1024)
#     print("Recording...")
#     frames = []
#     for _ in range(0, int(sr / 1024 * seconds)):
#         data = stream.read(1024)
#         frames.append(data)
#     print("Finished recording.")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()
#     return np.frombuffer(b''.join(frames), dtype=np.int16)

# def transcribe_audio(audio_input):
#     inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)[0]
#     return transcription

# if __name__ == "__main__":
#     audio_input = record_audio()
#     text_output = transcribe_audio(audio_input)
#     print("Transcription:", text_output)
