#from speech import get_emotion
#from extract_audio import extract_audios
from application import run
import numpy as np
np.set_printoptions(suppress=True, precision=2)
video=input("Enter video name: ")
x = run(video)
# print("In video")
# print(np.round(x * 100, 2))
# print("\n\n")
# extract_audios(video)
# audio_file = f"input/{video}.wav"
# predicted_emotion, confidence, y = get_emotion(audio_file)
# # print("In audio")
# print(np.round(y * 100, 2))
#
# id2label = {
#     "0": "angry",
#     "1": "calm",
#     "2": "disgust",
#     "3": "fearful",
#     "4": "happy",
#     "5": "neutral",
#     "6": "sad",
#     "7": "surprised"
# }
#
# affect = {
#     "0": "neutral",
#     "1": "happy",
#     "2": "sad",
#     "3": "surprise",
#     "4": "fear",
#     "5": "disgust",
#     "6": "anger"
# }

