import os
import numpy as np
import pickle
affect = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "surprise",
    "4": "fear",
    "5": "disgust",
    "6": "anger"
}

def load_data(pickle_file_path):
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"No data found for { pickle_file_path}")

    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    return data

# emotions = {}
# directory = '2024-06-24_00-49-12'
# directory = os.path.join('session',directory)
# for filename in os.listdir(directory):
#     if filename.endswith('.pkl'):
#         q, _ = os.path.splitext(filename) 
#         pickle_file_path = os.path.join(directory, f'{ q}.pkl')
#         emotions[q] = ''
#         data = load_data(pickle_file_path)
#         exp_average = data['exp_average']
#         interp = {affect[str(i)]: round(float(exp_average[i]) * 100, 2) for i in range(len(affect))}
#         for emotion, confidence in interp.items():
#             emotions[q] = emotions[q] + f"Emotion: {emotion}, Confidence: {confidence}%" + ' , '

emotions = {}
directory = 'session\\2024-06-24_00-49-12'
#directory = self.get_latest_session_directory()
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        q, _ = os.path.splitext(filename) 
        pickle_file_path = os.path.join(directory, f'{ q}.pkl')
        emotions[q] = ''
        data = load_data(pickle_file_path)
        exp_average = data['exp_average']
        interp = {affect[str(i)]: round(float(exp_average[i]) * 100, 2) for i in range(len(affect))}
        for emotion, confidence in interp.items():
            emotions[q] = emotions[q] + f"Emotion: {emotion} Confidence: {confidence}%" + ' , '
            
        max_emotion = np.argmax(exp_average)
        emotions[q] = affect[str(max_emotion)]
print(emotions)
