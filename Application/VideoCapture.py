import sys
import cv2
import imutils
import os
from datetime import datetime  
from PyQt5.QtCore import QThread, pyqtSignal as Signal
from PyQt5.QtGui import QImage


class VideoCaptureThread(QThread):
    frame_signal = Signal(QImage)
    close_signal = Signal(bool)
    close_received = False
    output_file = "output.mp4"

    def __init__(self):
        super().__init__()
        self.cap = None
        self.out = None
        self.close_signal.connect(self.close_recording)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

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

    def run(self):
        try:
            latest_session_dir = self.get_latest_session_directory()
            output_file_path = os.path.join(latest_session_dir, self.output_file)

            self.out = cv2.VideoWriter(output_file_path, self.fourcc, 30.0, (640, 480))
            self.cap = cv2.VideoCapture(0)

            # Check if the video capture is successful
            if not self.cap.isOpened():
                raise ValueError("Failed to open video capture.")

            while not self.close_received:
                _, frame = self.cap.read()
                self.out.write(frame)
                frame = self.cvimage_to_label(frame)
                self.frame_signal.emit(frame)
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
