import cv2
import numpy as np
import os
import json
import torch
from torch import nn
from PIL import Image
import functools
import numpy as np
import librosa
from torch.autograd import Variable
from opts import parse_opts
from model import generate_model
from models.fusion5 import MultiModalCNN
import transforms
from preprocess_input import extract_faces,extract_audios
import time
import warnings
from transformers import logging as transformers_logging
import moviepy.editor as mp
import soundfile as sf
import numpy as np
import moviepy.editor as mp
import soundfile as sf
import numpy as np
import librosa
import cv2
import pyaudio
import wave
import threading

# Function to record video from the camera
def record_video(video_name, duration, frame_rate=30):
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(f'./Examples/{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to split video into segments
def split_video(video_name, segment_duration=3, frame_rate=30):
    cap = cv2.VideoCapture(f'./Examples/{video_name}.mp4')

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    segment_frames = segment_duration * frame_rate

    segment_number = 0
    while True:
        frames = []
        for _ in range(segment_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) == 0:
            break

        out = cv2.VideoWriter(f'./Examples/{video_name}_segment_{segment_number}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
        for frame in frames:
            out.write(frame)
        out.release()

        segment_number += 1

    cap.release()
    cv2.destroyAllWindows()


def video_loader(video_dir_path):
    if not os.path.exists(video_dir_path):
        raise FileNotFoundError(f"The video file {video_dir_path} does not exist.")
    video = np.load(video_dir_path)
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i, :, :, :]))
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile):
    y, sr = librosa.load(audiofile, sr=22050)
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def input(data_type, video_path, audio_path, spatial_transform=None, audio_transform=None):
    loader = get_default_video_loader()
    path = video_path
    clip = loader(path)

    if data_type == 'video' or data_type == 'audiovisual':
        if spatial_transform is not None:
            spatial_transform.randomize_parameters()
            clip = [spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if data_type == 'video':
            return clip

    if data_type == 'audio' or data_type == 'audiovisual':
        path = audio_path
        y, sr = load_audio(path)

        if audio_transform is not None:
            audio_transform.randomize_parameters()
            y = audio_transform(y)

        mfcc = get_mfccs(y, sr)
        audio_features = mfcc

        if data_type == 'audio':
            return audio_features

    if data_type == 'audiovisual':
        return audio_features, clip

def getinput(video_path, audio_path, modality='both'):
    input_video_path = video_path
    input_audio_path = audio_path

    video_transform = transforms.Compose([
        transforms.ToTensor(255)])

    inputs_audio, inputs_visual = input('audiovisual', input_video_path, input_audio_path, video_transform)
    modality = 'both'
    a = []
    a.append(inputs_visual)
    a = torch.stack(a, dim=0)
    inputs_visual = a
    inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
    inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
    inputs_audio = np.array([inputs_audio])
    inputs_audio = torch.tensor(inputs_audio)

    with torch.no_grad():
        inputs_visual = Variable(inputs_visual)
        inputs_audio = Variable(inputs_audio)

    return inputs_audio, inputs_visual

def print_emotion(output_tensor):
    emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
    probabilities = torch.softmax(output_tensor, dim=1)
    max_index = torch.argmax(probabilities, dim=1).item()
    print(f"Predicted Emotion: {emotions[max_index]}")

def predict(video_name):
    opt = parse_opts()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiModalCNN()
    model.to(device)
    model.load_state_dict(torch.load('./weights/model5 - Copy.pth'))

    video_audio_paths = [
        (f'./Examples/{video_name}.npy', f'./Examples/{video_name}.wav')
    ]

    start_time = time.time()
    extract_faces.extract_faces(video_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Extraction of faces took {elapsed_time:.4f} seconds")


    for video_path, audio_path in video_audio_paths:
        audio, clip = getinput(video_path, audio_path, modality='both')
        model.eval()
        with torch.no_grad():
            output = model(f'./Examples/{video_name}.wav', clip)
        print(output.data)
        print_emotion(output.data)

# Main function to record, split, and predict emotions
def main():
    transformers_logging.set_verbosity_error()

    video_name = 'user_recording'
    record_duration = 9  # Record for 30 seconds
    segment_duration = 3  # Split into 3-second segments

    # Step 1: Record video from the camera
    print("Recording video...")
    record_video('user_recording', record_duration) 
    print("Recording complete.")

    # Step 2: Split the video into 3-second segments
    print("Splitting video into segments...")
    split_video(video_name, segment_duration)
    print("Splitting complete.")

    # Step 3: Predict emotions for each segment
    segment_number = 0
    while os.path.exists(f'./Examples/{video_name}_segment_{segment_number}.mp4'):
        print(f"Predicting emotions for segment {segment_number}...")
        predict(f'{video_name}_segment_{segment_number}')
        segment_number += 1

if __name__ == '__main__':
    main()
