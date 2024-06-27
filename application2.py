import cv2
import multimodalcnn
from skimage.transform import resize

import numpy as np


affect = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "surprise",
    "4": "fear",
    "5": "disgust",
    "6": "anger"
}

exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

# from speech import get_emotion
model = multimodalcnn.MultiModalCNN()
print('USING OUR MODEL')
model = model.to('cuda')
model.eval()
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def recognize(img):
    img = resize(img, (224, 224,3))
    img = np.expand_dims(img, axis=0)

    e=model(img)

    e = e.cpu().detach().numpy()
    e = e.reshape(-1)
    # return e
    return exps[np.argmax(e)],e


def write(text, position, frame, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 255, 255)  # Yellow color
    font_thickness = 2

    # Add the text to the video frame
    cv2.putText(frame, text, position, font, scale, font_color, font_thickness, cv2.LINE_4)


def run(video):
    exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
    cap = cv2.VideoCapture(video)
    current_emotion = ''
    running_sum, exp_average = np.zeros(7), np.zeros(7)
    count = 0

    # Get the original frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 30:
        frame_skip_interval = int(fps / 30)
    else:
        frame_skip_interval = 1

    frame_count = 0

    while True:
        ret, img = cap.read()

        if not ret:
            break

        frame_count += 1

        # Process only every nth frame where n is frame_skip_interval
        if frame_count % frame_skip_interval != 0:
            continue

            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            emotion = recognize(face)
            
            if current_emotion != emotion[0]:
                current_emotion = emotion[0]

            running_sum += emotion[1][0]
            count += 1
            exp_average = running_sum / count

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    print(exp_average.shape)
    max_emotion = np.argmax(exp_average)
    interp = {affect[str(i)]: round(float(exp_average[i]) * 100, 2) for i in range(len(affect))}

    emotions = ''
    for emotion, confidence in interp.items():
        emotions = emotions + f"Emotion: {emotion}, Confidence: {confidence}%" + ' , '

    cap.release()
    return emotions

