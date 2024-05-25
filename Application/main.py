import sys

from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from PyQt5 import uic
from questions_ui import QuestionsUI
from QuestionsGenerator import *

pages = {
    "homepage": 0,
    "analysis": 1,
    "questions": 2,
    "loading": 3
}


class MainApp(QMainWindow):

    processing_signal = Signal(bool)

    def __init__(self):
        super(MainApp, self).__init__()
        self.generate_questions_thread = None
        uic.loadUi("firstScreen.ui", self)

        font = QFont()
        font.setFamily(u"Franklin Gothic Medium")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)

        self.loadingScreen = QWidget()

        self.loading_grid_layout = QGridLayout(self.loadingScreen)
        self.vlayout = QVBoxLayout()

        self.loading_label = QLabel()
        self.loading_label.setText("Processing...")
        self.loading_label.setAlignment(Qt.AlignLeading | Qt.AlignCenter)
        self.loading_label.setFont(font)
        self.loading_label.setStyleSheet(u"color: rgb(255, 255, 255)")
        self.loading_label.setTextFormat(Qt.RichText)
        self.loading_label.setMargin(25)

        self.vlayout.addWidget(self.loading_label)
        self.loading_grid_layout.addLayout(self.vlayout, 0, 0, 1, 1)

        self.stackedPages = self.findChild(QStackedWidget, "stackedWidget")
        self.start_btn = self.findChild(QPushButton, "pushButton_2")
        self.disorder_input = self.findChild(QLineEdit, "lineEdit_2")
        self.new_test_btn = self.findChild(QPushButton, "newTest_btn")
        self.back_final_btn = self.findChild(QPushButton, "backbtn")

        self.new_test_btn.clicked.connect(self.go_to_home)
        self.back_final_btn.clicked.connect(self.go_to_question)


        self.disorder_input.setTextMargins(2, 2, 2, 2)

        self.start_btn.clicked.connect(self.generate_questions)
        self.questionScreen = QuestionsUI()
        self.stackedPages.addWidget(self.questionScreen)
        self.stackedPages.addWidget(self.loadingScreen)

        self.processing_signal.connect(self.go_to_loading)

        self.finish_btn_question = self.questionScreen.finish_btn
        self.home_btn_question = self.questionScreen.home_btn

        self.finish_btn_question.clicked.connect(self.go_to_analysis)
        self.home_btn_question.clicked.connect(self.go_to_home)

        self.show()

    def generate_questions(self):
        self.generate_questions_thread = QuestionGeneratorThread()
        if self.disorder_input.text() == "":
            msg = QMessageBox()
            msg.setText("Disorder input field cannot be empty!!")
            msg.exec_()
            return
        self.generate_questions_thread.set_topic(self.disorder_input.text())
        self.processing_signal.emit(True)
        self.generate_questions_thread.start()
        self.generate_questions_thread.finished.connect(self.go_to_question)


    def perform_analysis(self):
        pass

    def go_to_question(self):
        self.questionScreen.question_ui_requested_signal.emit(True)
        self.stackedPages.setCurrentIndex(pages["questions"])

    def go_to_loading(self):
        self.stackedPages.setCurrentIndex(pages["loading"])

    def go_to_home(self):
        self.stackedPages.setCurrentIndex(pages["homepage"])

    def go_to_analysis(self):
        self.stackedPages.setCurrentIndex(pages["analysis"])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    app.exec_()
