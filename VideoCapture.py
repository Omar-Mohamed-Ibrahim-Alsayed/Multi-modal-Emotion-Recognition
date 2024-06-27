import sys
import cv2
import imutils
import os
from datetime import datetime  
from PyQt5.QtCore import QThread, pyqtSignal as Signal
from PyQt5.QtGui import QImage
import numpy as np
from test import TestModels
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

tester = TestModels(h5_address='AffectNet_6336.h5')
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def recognize(img):
    e=tester.recognize_fer2(img)
    return str(e[0]),e[1]


def write(text, position, frame, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)  # Yellow color
    font_thickness = 4

    # Add the text to the video frame
    cv2.putText(frame, text, position, font, scale, font_color, font_thickness, cv2.LINE_4)


class VideoCaptureThread(QThread):
    frame_signal = Signal(QImage)
    close_signal = Signal(bool)
    close_received = False
    output_file = "output.mp4"

    def __init__(self, question_no=0):
        super().__init__()
        self.cap = None
        self.out = None
        self.close_signal.connect(self.close_recording)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.emotion = ''
        self.question_no = question_no  # Store the question number


    def set_outfile(self, filename):
        self.output_file = filename

    def get_latest_session_directory(self):
        session_dir = "session"
        if not os.path.exists(session_dir):
            os.mkdir(session_dir)
            return session_dir

        subdirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
        if not subdirs:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_subdir = os.path.join(session_dir, current_time)
            os.mkdir(session_subdir)
            return session_subdir

        latest_subdir = max(subdirs, key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"))
        return os.path.join(session_dir, latest_subdir)

    def save(self, exps, exp_average):
        max = np.argmax(exp_average)
        data = {
            'emotion': exps[max],
            'exp_average': exp_average.tolist()
        }
        data_directory = self.get_latest_session_directory()
        os.makedirs(data_directory, exist_ok=True)
        pickle_file_path = os.path.join(data_directory, f'{self.question_no}.pkl')
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self):
        data_directory = self.get_latest_session_directory()
        pickle_file_path = os.path.join(data_directory, f'{self.question_no}.pkl')

        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError(f"No data found for {self.question_no}")

        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def run(self):
        try:
            latest_session_dir = self.get_latest_session_directory()
            output_file_path = os.path.join(latest_session_dir, self.output_file)

            self.out = cv2.VideoWriter(output_file_path, self.fourcc, 30.0, (640, 480))
            self.cap = cv2.VideoCapture(0)
            current_emotion = ''
            running_sum, exp_average = np.zeros(7), np.zeros(7)
            count = 0

            if not self.cap.isOpened():
                raise ValueError("Failed to open video capture.")

            while not self.close_received:
                exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

                _, frame = self.cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                    face = frame[y:y + h, x:x + w]
                    emotion = recognize(face)
                    if current_emotion != emotion[0]:
                        current_emotion = emotion[0]
                    write(emotion[0], (x, y), frame, 1)
                    self.emotion = emotion[0]
                    running_sum += emotion[1][0]
                    count += 1
                    exp_average = running_sum / count
                self.out.write(frame)
                frame = self.cvimage_to_label(frame)
                self.frame_signal.emit(frame)

            self.save(exps, exp_average)

            try:
                data = self.load_data()
                print("Loaded data:", data)
            except FileNotFoundError as e:
                print(e)

            self.cap.release()
            self.out.release()
        except Exception as e:
            print(f"Error in VideoCaptureThread: {e}")

    def close_recording(self):
        self.close_received = True

    def cvimage_to_label(self, image):
        image = imutils.resize(image, width=640)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(image,
                       image.shape[1],
                       image.shape[0],
                       QImage.Format_RGB888)
        return image
