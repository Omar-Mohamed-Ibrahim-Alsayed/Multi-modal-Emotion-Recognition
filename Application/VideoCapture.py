import sys

import cv2
import imutils
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

    def run(self):
        try:
            self.out = cv2.VideoWriter(self.output_file, self.fourcc, 30.0, (640, 480))
            self.cap = cv2.VideoCapture(0)
            while self.cap.isOpened() and not self.close_received:
                _, frame = self.cap.read()
                self.out.write(frame)
                frame = self.cvimage_to_label(frame)
                self.frame_signal.emit(frame)
            self.cap.release()
            self.out.release()
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

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