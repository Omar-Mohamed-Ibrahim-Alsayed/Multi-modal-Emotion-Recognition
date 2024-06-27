import cv2
from test import TestModels

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

# from speech import get_emotion
tester = TestModels(h5_address='AffectNet_6336.h5')
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def recognize(img):
    e=tester.recognize_fer2(img)
    return str(e[0]),e[1]


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
    running_sum,exp_average = np.zeros(7),np.zeros(7)
    count=0
    while True:
        # Read the frame
        ret, img = cap.read()

        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = img[y:y + h, x:x + w]
            emotion = recognize(face)
            if current_emotion != emotion[0]:
                current_emotion = emotion[0]
                # movies = getmovie(emotion, 1)
            write(emotion[0], (x, y), img, 1)
            # write(movies, (x, y+w), img,0.5)
            running_sum += emotion[1][0]
            count += 1
            exp_average = running_sum / count
        # Display
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    print(exp_average)
    max=np.argmax(exp_average)
    print(exps[max])
    print(f"{exp_average[max]*100} %")
    # Release the VideoCapture object
    interp = {affect[str(i)]: round(float(exp_average[i]) * 100, 2) for i in range(len(affect))}
    print("\nPredictions for Video (in %):")
    for emotion, confidence in interp.items():
        print(f"Emotion: {emotion}, Confidence: {confidence}%")

    cap.release()
    return exp_average


run(0)
# audio_file = "path_to_your_audio_file.wav"
# predicted_emotion, confidence = get_emotion(audio_file)
