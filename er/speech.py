# # import torch
# # import torch.nn as nn
# # from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
# # import numpy as np
# # from pydub import AudioSegment
# # import os
# #
# # model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# #
# # model.projector = nn.Linear(1024, 1024, bias=True)
# # model.classifier = nn.Linear(1024, 8, bias=True)
# #
# # torch_state_dict = torch.load('pytorch_model.bin', map_location=torch.device('cpu'))
# #
# # model.projector.weight.data = torch_state_dict['classifier.dense.weight']
# # model.projector.bias.data = torch_state_dict['classifier.dense.bias']
# #
# # model.classifier.weight.data = torch_state_dict['classifier.output.weight']
# # model.classifier.bias.data = torch_state_dict['classifier.output.bias']
# #
# # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# #
# #
# # def extract_emotion_label(filename):
# #     # Extract the emotion label from the filename
# #     emotion_code = filename.split('-')[3]
# #     emotion_mapping = {
# #         "01": "neutral",
# #         "02": "calm",
# #         "03": "happy",
# #         "04": "sad",
# #         "05": "angry",
# #         "06": "fearful",
# #         "07": "disgust",
# #         "08": "surprised"
# #     }
# #     return emotion_mapping.get(emotion_code, "unknown")
# #
# #
# # def predict_emotion(audio_file):
# #     sound = AudioSegment.from_file(audio_file)
# #     sound = sound.set_frame_rate(16000)
# #     sound_array = np.array(sound.get_array_of_samples())
# #
# #     input = feature_extractor(
# #         raw_speech=sound_array,
# #         sampling_rate=16000,
# #         padding=True,
# #         return_tensors="pt")
# #
# #     result = model.forward(input.input_values.float())
# #     id2label = {
# #         "0": "angry",
# #         "1": "calm",
# #         "2": "disgust",
# #         "3": "fearful",
# #         "4": "happy",
# #         "5": "neutral",
# #         "6": "sad",
# #         "7": "surprised"
# #     }
# #     interp = dict(zip(id2label.values(), list(round(float(i), 4) for i in result[0][0])))
# #     return interp
# #
# #
# # def get_emotion(audio_file):
# #     predictions_speech = []
# #     labels = []
# #
# #     print("\nPredicting for Speech:")
# #
# #     emotion_result = predict_emotion(audio_file)
# #     predicted_label = max(emotion_result, key=emotion_result.get)
# #     prediction_speech=predicted_label
# #
# #     # Extract the true emotion label from the filename
# #     true_label = extract_emotion_label(audio_file)
# #     labels.append(true_label)
# #
# #     print(f"File: {audio_file}, True Label: {true_label}, Prediction: {predicted_label}")
# #
# #     # Print predictions for speech
# #     print("\nPredictions for Speech:", predictions_speech)
# #
# #     return prediction_speech
# #
# #
# #
# import torch
# import torch.nn as nn
# from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
# import numpy as np
# from pydub import AudioSegment
# import os
#
# # Load the pre-trained model
# model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
#
# # Load the pre-trained weights
# torch_state_dict = torch.load('pytorch_model.bin', map_location=torch.device('cpu'))
#
# # Assign weights to the model's projector and classifier
# model.projector = nn.Linear(1024, 1024, bias=True)
# model.classifier = nn.Linear(1024, 8, bias=True)
# model.projector.weight.data = torch_state_dict['classifier.dense.weight']
# model.projector.bias.data = torch_state_dict['classifier.dense.bias']
# model.classifier.weight.data = torch_state_dict['classifier.output.weight']
# model.classifier.bias.data = torch_state_dict['classifier.output.bias']
#
# # Load the Wav2Vec2 feature extractor
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
#
# def extract_emotion_label(filename):
#     # Extract the emotion label from the filename
#     emotion_code = filename.split('-')[3]
#     emotion_mapping = {
#         "01": "neutral",
#         "02": "calm",
#         "03": "happy",
#         "04": "sad",
#         "05": "angry",
#         "06": "fearful",
#         "07": "disgust",
#         "08": "surprised"
#     }
#     return emotion_mapping.get(emotion_code, "unknown")
#
# def predict_emotion(audio_file):
#     sound = AudioSegment.from_file(audio_file)
#     sound = sound.set_frame_rate(16000)
#     sound_array = np.array(sound.get_array_of_samples())
#
#     # Extract features using Wav2Vec2 feature extractor
#     input = feature_extractor(
#         raw_speech=sound_array,
#         sampling_rate=16000,
#         padding=True,
#         return_tensors="pt")
#
#     # Perform inference using the pre-trained model
#     with torch.no_grad():
#         result = model(input.input_values.float())
#
#     # Mapping from label index to emotion label
#     id2label = {
#         "0": "angry",
#         "1": "calm",
#         "2": "disgust",
#         "3": "fearful",
#         "4": "happy",
#         "5": "neutral",
#         "6": "sad",
#         "7": "surprised"
#     }
#
#     # Interpret the model output into a dictionary of emotions and confidence values
#     interp = {id2label[str(i)]: round(float(result[0][0][i]), 4) for i in range(len(id2label))}
#     print("\nPredictions for Speech:")
#     for emotion, confidence in interp.items():
#         print(f"Emotion: {emotion}, Confidence: {confidence}")
#
#     return interp
#
# def get_emotion(audio_file):
#     print(f"\nPredicting emotion for file: {audio_file}")
#
#     # Predict emotion for the audio file
#     emotion_result = predict_emotion(audio_file)
#
#     # Determine the emotion with the maximum confidence
#     predicted_emotion = max(emotion_result, key=emotion_result.get)
#     max_confidence = emotion_result[predicted_emotion]
#
#     print(f"\nPredicted Emotion: {predicted_emotion}, Confidence: {max_confidence}")
#
#     return predicted_emotion, max_confidence
#
# audio_file = "input/01-01-03-02-01-01-16.wav"
# predicted_emotion, confidence = get_emotion(audio_file)


import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
from pydub import AudioSegment
import os

# Load the pre-trained model
model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Load the pre-trained weights
torch_state_dict = torch.load('../pytorch_model.bin', map_location=torch.device('cpu'))

# Assign weights to the model's projector and classifier
model.projector = nn.Linear(1024, 1024, bias=True)
model.classifier = nn.Linear(1024, 8, bias=True)
model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']
model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']

# Load the Wav2Vec2 feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def extract_emotion_label(filename):
    # Extract the emotion label from the filename
    emotion_code = filename.split('-')[3]
    emotion_mapping = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    return emotion_mapping.get(emotion_code, "unknown")

def predict_emotion(audio_file):
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound_array = np.array(sound.get_array_of_samples())

    # Extract features using Wav2Vec2 feature extractor
    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt")

    # Perform inference using the pre-trained model
    with torch.no_grad():
        result = model(input.input_values.float())

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(result.logits, dim=-1)[0].cpu().numpy()

    # Mapping from label index to emotion label
    id2label = {
        "0": "angry",
        "1": "calm",
        "2": "disgust",
        "3": "fearful",
        "4": "happy",
        "5": "neutral",
        "6": "sad",
        "7": "surprised"
    }

    # Interpret the model output into a dictionary of emotions and confidence values
    interp = {id2label[str(i)]: round(float(probabilities[i]) * 100, 2) for i in range(len(id2label))}
    print("\nPredictions for Speech (in %):")
    for emotion, confidence in interp.items():
        print(f"Emotion: {emotion}, Confidence: {confidence}%")

    return interp,probabilities

def get_emotion(audio_file):
    print(f"\nPredicting emotion for file: {audio_file}")

    # Predict emotion for the audio file
    emotion_result,probabilities = predict_emotion(audio_file)

    # Determine the emotion with the maximum confidence
    predicted_emotion = max(emotion_result, key=emotion_result.get)
    max_confidence = emotion_result[predicted_emotion]

    print(f"\nPredicted Emotion: {predicted_emotion}, Confidence: {max_confidence}%")

    return predicted_emotion, max_confidence,probabilities

# audio_file = "input/01-01-03-02-01-01-16.wav"
# predicted_emotion, confidence = get_emotion(audio_file)
