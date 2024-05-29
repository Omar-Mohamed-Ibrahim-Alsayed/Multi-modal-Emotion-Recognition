import os.path
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
        self.record_btn = None
        self.questionLabel = None
        self.VideoLabel = None
        self.questionNumberLabel = None  # Label for question number

        self.init_ui()
        self.signals = 0
        self.is_recording = False

    def init_ui(self):
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

        self.questionNumberLabel = QtWidgets.QLabel()  # Initialize the label for question number
        self.questionNumberLabel.setAlignment(Qt.AlignCenter)
        self.questionNumberLabel.setFont(QFont("Franklin Gothic Medium", 16, QFont.Bold))
        self.questionNumberLabel.setStyleSheet("color: rgb(255, 255, 255)")
        
        self.questionLabel = QtWidgets.QLabel()
        self.questionLabel.setAlignment(Qt.AlignCenter)  # Set alignment to center
        self.hlayout1.addWidget(self.questionLabel)
    
        self.VideoLabel = QtWidgets.QLabel()
        self.VideoLabel.setFixedSize(640, 480)
        self.grid_child_layout1.addWidget(self.VideoLabel)
        self.videoPlayer = VideoPlayer()

        self.grid_page1.addLayout(self.grid_child_layout1, 0, 0, 1, 1)
        self.grid_page2.addWidget(self.videoPlayer, 0, 0, 1, 1)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.page1)
        self.stackedWidget.addWidget(self.page2)

        self.hlayout2.addWidget(self.stackedWidget)

        self.record_btn = QtWidgets.QPushButton("Start Recording", clicked=self.toggle_recording)
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

        self.vlayout.addWidget(self.questionNumberLabel)  # Add question number label to the layout
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
        self.questionLabel.setAlignment(Qt.AlignCenter)  # Set alignment to center
        self.questionLabel.setMargin(25)
        self.questionLabel.setWordWrap(True)

        self.VideoLabel.setAlignment(Qt.AlignLeading | Qt.AlignCenter)

        self.record_btn.setMinimumSize(QSize(150, 0))
        self.next_btn.setMinimumSize(QSize(150, 0))
        self.prev_btn.setMinimumSize(QSize(150, 0))
        self.home_btn.setMinimumSize(QSize(150, 0))
        self.finish_btn.setMinimumSize(QSize(150, 0))

        self.font5 = QFont()
        self.font5.setFamily(u"Franklin Gothic Demi")
        self.font5.setPointSize(12)
        self.record_btn.setFont(self.font5)
        self.next_btn.setFont(self.font5)
        self.prev_btn.setFont(self.font5)
        self.finish_btn.setFont(self.font5)
        self.home_btn.setFont(self.font5)

        self.btn_style_1 = """
            :hover {
                background: white;
                color: #191970;
            }
    
            QWidget {
                border-style: solid;
                border-color: black;
                border-radius: 20;
                background-color: #535C91;
                color: white;
                height: 40;
                margin: 5px;
            }
        """

        self.record_btn.setStyleSheet(self.btn_style_1)
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

        self.stackedWidget3.addWidget(self.record_btn)

        self.horizontalLayout_btns.addWidget(self.stackedWidget3)
        self.horizontalLayout_btns.addWidget(self.stackedWidget1)
        self.horizontalLayout_btns.addWidget(self.stackedWidget2)

    def init_question_generators(self):
        self.QuestionsGenerator = QuestionGenerator()
        self.QuestionsGeneratorIter = iter(self.QuestionsGenerator)
        self.stackedWidget1.setCurrentIndex(0)
        self.stackedWidget2.setCurrentIndex(0)
        self.next_question()

    def toggle_recording(self):
        if self.is_recording:
            self.close_camera()
        else:
            self.open_camera()
        self.is_recording = not self.is_recording
        self.record_btn.setText("Stop Recording" if self.is_recording else "Start Recording")

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

    def get_latest_session_directory(self):
        session_dir = "session"
        if not os.path.exists(session_dir):
            return None

        subdirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
        if not subdirs:
            return None

        latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(session_dir, d)))
        return os.path.join(session_dir, latest_subdir)

    def update_question_number_label(self):
        self.questionNumberLabel.setText(f"Question {self.QuestionsGenerator.get_index() + 1}")

    def next_question(self):
        try:
            self.questionLabel.setText(next(self.QuestionsGeneratorIter))
            self.update_question_number_label()  # Update question number label
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
            self.update_question_number_label()  # Update question number label
            
            latest_directory = self.get_latest_session_directory()
            media_file = os.path.join(latest_directory, "f" + str(self.QuestionsGenerator.get_index()) + ".mp4")
            
            if latest_directory and os.path.exists(media_file):
                self.videoPlayer.set_mediafile(media_file)
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
        latest_directory = self.get_latest_session_directory()
        if latest_directory:
            media_file = os.path.join(latest_directory, "f" + str(self.QuestionsGenerator.get_index()) + ".mp4")
            self.videoPlayer.set_mediafile(media_file)
            self.stackedWidget.setCurrentIndex(1)
            self.videoPlayer.play()
            
    def clear_player(self):
        self.videoPlayer.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = QuestionsUI()
    main_window.show()
    sys.exit(app.exec_())
