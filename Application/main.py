import os.path
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot as Slot, QMutex
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QStackedWidget
from AudioAndVideoMerger import AudioAndVideoMerger
from AudioCapture import AudioCaptureThread
from QuestionsGenerator import QuestionGenerator
from VideoCapture import VideoCaptureThread
from VideoPlayer import VideoPlayer


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.stackedWidget = None
        self.videoPlayer = None
        self.AudioAndVideoMerger = None
        self.audio_thread = None
        self.prev_btn = None
        self.next_btn = None
        self.mutex = QMutex()
        self.camera_thread = None
        self.close_btn = None
        self.open_btn = None
        self.questionLabel = None
        self.VideoLabel = None
        self.QuestionsGenerator = QuestionGenerator()
        self.QuestionsGeneratorIter = iter(self.QuestionsGenerator)
        self.init_ui()
        self.next_question()
        self.show()
        self.signals = 0

    def init_ui(self):
        self.setFixedSize(640, 640)
        self.setWindowTitle("Camera FeedBack")

        widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        self.questionLabel = QtWidgets.QLabel()
        layout.addWidget(self.questionLabel)

        self.VideoLabel = QtWidgets.QLabel()
        self.videoPlayer = VideoPlayer()
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.VideoLabel)
        self.stackedWidget.addWidget(self.videoPlayer)

        layout.addWidget(self.stackedWidget)

        self.open_btn = QtWidgets.QPushButton("Start Recording", clicked=self.open_camera)
        layout.addWidget(self.open_btn)

        self.close_btn = QtWidgets.QPushButton("Stop Recording", clicked=self.close_camera)
        layout.addWidget(self.close_btn)

        self.next_btn = QtWidgets.QPushButton("Next question", clicked=self.next_question)
        layout.addWidget(self.next_btn)

        self.prev_btn = QtWidgets.QPushButton("Prev question", clicked=self.prev_question)
        layout.addWidget(self.prev_btn)
        self.setCentralWidget(widget)

    def open_camera(self):
        self.videoPlayer.clear()
        self.stackedWidget.setCurrentIndex(0)
        self.camera_thread = VideoCaptureThread()
        self.audio_thread = AudioCaptureThread()
        self.camera_thread.frame_signal.connect(self.setImage)
        self.camera_thread.finished.connect(self.VideoLabel.clear)
        self.camera_thread.finished.connect(self.merge_and_save)
        self.audio_thread.finished.connect(self.merge_and_save)
        self.camera_thread.set_outfile(str(self.QuestionsGenerator.get_index()) + ".mp4")
        self.camera_thread.start()
        while not self.camera_thread.isRunning():
            continue
        self.audio_thread.set_outfile(str(self.QuestionsGenerator.get_index()) + ".wav")
        self.audio_thread.start()

    def next_question(self):
        try:
            self.questionLabel.setText(next(self.QuestionsGeneratorIter))
            if os.path.exists("./f" + str(self.QuestionsGenerator.get_index()) + ".mp4"):
                self.videoPlayer.set_mediafile("./f" + str(self.QuestionsGenerator.get_index()) + ".mp4")
                self.stackedWidget.setCurrentIndex(1)
            else:
                self.videoPlayer.clear()
                self.stackedWidget.setCurrentIndex(0)
        except StopIteration:
            msg = QtWidgets.QMessageBox()
            msg.setText("This is the last question!")
            msg.exec_()

    def prev_question(self):
        try:
            self.questionLabel.setText(self.QuestionsGenerator.prev())
            if os.path.exists("./f"+str(self.QuestionsGenerator.get_index())+".mp4"):
                self.videoPlayer.set_mediafile("./f"+str(self.QuestionsGenerator.get_index())+".mp4")
                self.stackedWidget.setCurrentIndex(1)
            else:
                self.videoPlayer.clear()
                self.stackedWidget.setCurrentIndex(0)

        except StopIteration:
            msg = QtWidgets.QMessageBox()
            msg.setText("This is the first Question!")
            msg.exec_()

    def close_camera(self):
        self.camera_thread.close_signal.emit(True)
        self.audio_thread.close_signal.emit(True)

    @Slot(QImage)
    def setImage(self, image):
        self.VideoLabel.setPixmap(QPixmap.fromImage(image))

    def merge_and_save(self):
        self.mutex.lock()
        self.signals += 1
        if self.signals == 2:
            self.signals = 0
            self.AudioAndVideoMerger = AudioAndVideoMerger()
            self.AudioAndVideoMerger.set_audiofile(str(self.QuestionsGenerator.get_index()) + ".wav")
            self.AudioAndVideoMerger.set_videofile(str(self.QuestionsGenerator.get_index()) + ".mp4")
            self.AudioAndVideoMerger.set_outfile("f" + str(self.QuestionsGenerator.get_index()) + ".mp4")
            self.AudioAndVideoMerger.start()
            self.AudioAndVideoMerger.finished.connect(self.show_player)
        self.mutex.unlock()

    def show_player(self):
        self.videoPlayer.set_mediafile("./f" + str(self.QuestionsGenerator.get_index()) + ".mp4")
        self.stackedWidget.setCurrentIndex(1)
        self.videoPlayer.play()

    def clear_player(self):
        self.videoPlayer.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainApp()
    sys.exit(app.exec())

"""
Task 1: add audio recording with the video Done
Task 2: display the recording for revision on GUI Done
Task 3: beautify GUI
Task 4: Generate Questions Dynamically
"""
