import os.path
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtCore import pyqtSlot as Slot, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QStackedWidget, QLayout
from AudioAndVideoMerger import AudioAndVideoMerger
from AudioCapture import AudioCaptureThread
from QuestionsGenerator import QuestionGenerator
from VideoCapture import VideoCaptureThread
from VideoPlayer import VideoPlayer

class QuestionsUI(QtWidgets.QWidget):
    question_ui_requested_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.finish_btn = None
        self.home_btn = None
        self.QuestionsGeneratorIter = None
        self.QuestionsGenerator = None
        self.font5 = None
        self.grid_page2 = None
        self.grid_page1 = None
        self.grid_child_layout1 = None
        self.page2 = None
        self.page1 = None
        self.hlayout2 = None
        self.hlayout1 = None
        self.horizontalLayout_btns = None
        self.btn_style_1 = None
        self.vlayout = None
        self.grid_layout = None
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

        self.init_ui()
        self.signals = 0

    def init_ui(self):
        # self.setFixedSize(640, 640)
        # self.setWindowTitle("Camera FeedBack")
        # self.widget = QtWidgets.QWidget(self)
        self.question_ui_requested_signal.connect(self.init_question_generators)

        self.grid_layout = QtWidgets.QGridLayout(self)

        self.vlayout = QtWidgets.QVBoxLayout()
        self.hlayout1 = QtWidgets.QHBoxLayout()
        self.hlayout2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_btns = QtWidgets.QHBoxLayout()

        self.page1 = QtWidgets.QWidget()
        self.page2 = QtWidgets.QWidget()

        self.grid_page1 = QtWidgets.QGridLayout(self.page1)
        self.grid_page2 = QtWidgets.QGridLayout(self.page2)

        self.grid_child_layout1 = QtWidgets.QHBoxLayout()

        # self.grid_child_layout2 = QtWidgets.QHBoxLayout()
        # self.grid_layout.addWidget(self.vlayout)

        self.questionLabel = QtWidgets.QLabel()
        self.hlayout1.addWidget(self.questionLabel)

        self.VideoLabel = QtWidgets.QLabel()
        self.grid_child_layout1.addWidget(self.VideoLabel)

        self.videoPlayer = VideoPlayer()

        self.grid_page1.addLayout(self.grid_child_layout1, 0, 0, 1, 1)
        self.grid_page2.addWidget(self.videoPlayer, 0, 0, 1, 1)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.page1)
        self.stackedWidget.addWidget(self.page2)

        self.hlayout2.addWidget(self.stackedWidget)

        self.open_btn = QtWidgets.QPushButton("Start Recording", clicked=self.open_camera)
        self.close_btn = QtWidgets.QPushButton("Stop Recording", clicked=self.close_camera)
        self.next_btn = QtWidgets.QPushButton("Next", clicked=self.next_question)
        self.prev_btn = QtWidgets.QPushButton("Back", clicked=self.prev_question)
        self.home_btn = QtWidgets.QPushButton("Home")
        self.finish_btn = QtWidgets.QPushButton("Finish")

        self.home_btn.setObjectName("home_btn")
        self.finish_btn.setObjectName("finish_btn")

        self.vlayout.setSpacing(1)
        self.vlayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.vlayout.setContentsMargins(10, 10, 10, 10)

        self.hlayout1.setSpacing(1)
        self.hlayout1.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.hlayout1.setContentsMargins(10, 10, 10, 10)

        self.hlayout2.setSpacing(1)
        self.hlayout2.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.hlayout2.setContentsMargins(10, 10, 10, 10)

        self.horizontalLayout_btns.setSpacing(1)
        self.hlayout2.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout_btns.setContentsMargins(10, 10, 10, 10)

        self.vlayout.addLayout(self.hlayout1)
        self.vlayout.addLayout(self.hlayout2)
        self.vlayout.addLayout(self.horizontalLayout_btns)

        self.grid_layout.addLayout(self.vlayout, 0, 0, 1, 1)

        font = QFont()
        font.setFamily(u"Franklin Gothic Medium")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)
        self.questionLabel.setFont(font)
        self.questionLabel.setStyleSheet(u"color: rgb(255, 255, 255)")
        self.questionLabel.setTextFormat(Qt.RichText)
        self.questionLabel.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
        self.questionLabel.setMargin(25)
        self.questionLabel.setWordWrap(True)

        self.VideoLabel.setAlignment(Qt.AlignLeading | Qt.AlignCenter)

        self.open_btn.setMinimumSize(QSize(150, 0))
        self.close_btn.setMinimumSize(QSize(150, 0))
        self.next_btn.setMinimumSize(QSize(150, 0))
        self.prev_btn.setMinimumSize(QSize(150, 0))
        self.home_btn.setMinimumSize(QSize(150, 0))
        self.finish_btn.setMinimumSize(QSize(150, 0))

        self.font5 = QFont()
        self.font5.setFamily(u"Franklin Gothic Demi")
        self.font5.setPointSize(12)
        self.open_btn.setFont(self.font5)
        self.close_btn.setFont(self.font5)
        self.next_btn.setFont(self.font5)
        self.prev_btn.setFont(self.font5)
        self.finish_btn.setFont(self.font5)
        self.home_btn.setFont(self.font5)

        self.btn_style_1 = """
            :hover {
                background: white;
                color: #495464;
            }
    
            QWidget {
                border-style: solid;
                border-color: black;
                border-radius: 20;
                background-color: #495464;
                color: white;
                height: 40;
                margin: 5px;
            }
        """

        self.open_btn.setStyleSheet(self.btn_style_1)
        self.close_btn.setStyleSheet(self.btn_style_1)
        self.next_btn.setStyleSheet(self.btn_style_1)
        self.prev_btn.setStyleSheet(self.btn_style_1)
        self.finish_btn.setStyleSheet(self.btn_style_1)
        self.home_btn.setStyleSheet(self.btn_style_1)

        self.stackedWidget1 = QtWidgets.QStackedWidget()
        self.stackedWidget2 = QtWidgets.QStackedWidget()
        self.stackedWidget3 = QtWidgets.QStackedWidget()
        self.stackedWidget4 = QtWidgets.QStackedWidget()

        self.stackedWidget1.setFixedHeight(50)
        self.stackedWidget2.setFixedHeight(50)
        self.stackedWidget3.setFixedHeight(50)
        self.stackedWidget4.setFixedHeight(50)


        self.stackedWidget1.addWidget(self.home_btn)
        self.stackedWidget1.addWidget(self.prev_btn)

        self.stackedWidget2.addWidget(self.next_btn)
        self.stackedWidget2.addWidget(self.finish_btn)

        self.stackedWidget3.addWidget(self.close_btn)
        self.stackedWidget4.addWidget(self.open_btn)

        self.horizontalLayout_btns.addWidget(self.stackedWidget4)
        self.horizontalLayout_btns.addWidget(self.stackedWidget3)
        self.horizontalLayout_btns.addWidget(self.stackedWidget1)
        self.horizontalLayout_btns.addWidget(self.stackedWidget2)
        # self.horizontalLayout_btns.setStretch(0, 1)
        # self.horizontalLayout_btns.setStretch(1, 1)

    def init_question_generators(self):
        self.QuestionsGenerator = QuestionGenerator()
        self.QuestionsGeneratorIter = iter(self.QuestionsGenerator)
        self.stackedWidget1.setCurrentIndex(0)
        self.stackedWidget2.setCurrentIndex(0)
        self.next_question()

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
            if self.QuestionsGenerator.get_index() == self.QuestionsGenerator.get_size()-1:
                self.stackedWidget2.setCurrentIndex(1)
            if self.QuestionsGenerator.get_index() == 1:
                self.stackedWidget1.setCurrentIndex(1)

        except StopIteration:
            msg = QtWidgets.QMessageBox()
            msg.setText("This is the last question!")
            msg.exec_()

    def prev_question(self):
        try:
            self.questionLabel.setText(self.QuestionsGenerator.prev())
            if os.path.exists("./f" + str(self.QuestionsGenerator.get_index()) + ".mp4"):
                self.videoPlayer.set_mediafile("./f" + str(self.QuestionsGenerator.get_index()) + ".mp4")
                self.stackedWidget.setCurrentIndex(1)
            else:
                self.videoPlayer.clear()
                self.stackedWidget.setCurrentIndex(0)

            if self.QuestionsGenerator.get_index() == self.QuestionsGenerator.get_size()-2:
                self.stackedWidget2.setCurrentIndex(0)
            if self.QuestionsGenerator.get_index() == 0:
                self.stackedWidget1.setCurrentIndex(0)


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
    main_window = QuestionsUI()
    sys.exit(app.exec())

"""
Task 1: add audio recording with the video Done
Task 2: display the recording for revision on GUI Done
Task 3: beautify GUI
Task 4: Generate Questions Dynamically
"""
