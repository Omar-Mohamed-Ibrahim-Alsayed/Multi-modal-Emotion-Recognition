# import os.path
# from PyQt5 import QtWidgets
# from PyQt5.QtCore import Qt, QSize, pyqtSignal
# from PyQt5.QtCore import pyqtSlot as Slot, QMutex
# from PyQt5.QtGui import QImage, QPixmap, QFont
# from PyQt5.QtWidgets import QStackedWidget, QLayout
# from AudioAndVideoMerger import AudioAndVideoMerger
# from AudioCapture import AudioCaptureThread
# from QuestionsGenerator import QuestionGenerator
# from VideoPlayer import VideoPlayer
# from emotions2 import ERmodel
# import sys
# import sys
# import cv2
# import imutils
# import os
# from datetime import datetime  
# from PyQt5.QtCore import QThread, pyqtSignal as Signal
# from PyQt5.QtGui import QImage
# import sys
# import wave
# import pyaudio
# import os
# from datetime import datetime
# from PyQt5.QtCore import QThread, pyqtSignal as Signal



# class AudioCaptureThread(QThread):
#     close_signal = Signal(bool)
#     close_received = False
#     chunk = 1024
#     format = pyaudio.paInt16
#     channels = 2
#     rate = 22050
#     filename = "output.wav"

#     def set_outfile(self, filename):
#         self.filename = filename

#     def __init__(self):
#         super().__init__()
#         self.close_signal.connect(self.close_audio)

#     def get_latest_session_directory(self):
#         session_dir = "session"
#         if not os.path.exists(session_dir):
#             os.mkdir(session_dir)
#             return session_dir

#         subdirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
#         if not subdirs:
#             current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             session_subdir = os.path.join(session_dir, current_time)
#             os.mkdir(session_subdir)
#             return session_subdir

#         latest_subdir = max(subdirs, key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"))
#         return os.path.join(session_dir, latest_subdir)

#     def run(self):
#         try:
#             p = pyaudio.PyAudio()

#             # Open a stream for recording
#             stream = p.open(format=self.format,
#                             channels=self.channels,
#                             rate=self.rate,
#                             input=True,
#                             frames_per_buffer=self.chunk)
#             frames = []
#             while not self.close_received:
#                 data = stream.read(self.chunk)
#                 frames.append(data)
#                 if len(frames) == 30:
                     
#                     model.rec_aud(frames)


#             stream.stop_stream()
#             stream.close()
#             p.terminate()

#             latest_session_dir = self.get_latest_session_directory()
#             output_file_path = os.path.join(latest_session_dir, self.filename)
            
#             wf = wave.open(output_file_path, 'wb')
#             wf.setnchannels(self.channels)
#             wf.setsampwidth(p.get_sample_size(self.format))
#             wf.setframerate(self.rate)
#             wf.writeframes(b''.join(frames))
#             wf.close()
#         except:
#             (type, value, traceback) = sys.exc_info()
#             sys.excepthook(type, value, traceback)

#     def close_audio(self):
#         self.close_received = True


# class VideoCaptureThread(QThread):
#     frame_signal = Signal(QImage)
#     close_signal = Signal(bool)
#     close_received = False
#     output_file = "output.mp4"

#     def __init__(self):
#         super().__init__()
#         self.cap = None
#         self.out = None
#         self.close_signal.connect(self.close_recording)
#         self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     def set_outfile(self, filename):
#         self.output_file = filename

#     def get_latest_session_directory(self):
#         session_dir = "session"
#         if not os.path.exists(session_dir):
#             os.mkdir(session_dir)
#             return session_dir

#         subdirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
#         if not subdirs:
#             current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             session_subdir = os.path.join(session_dir, current_time)
#             os.mkdir(session_subdir)
#             return session_subdir

#         latest_subdir = max(subdirs, key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"))
#         return os.path.join(session_dir, latest_subdir)

#     def run(self):
#         try:
#             latest_session_dir = self.get_latest_session_directory()
#             output_file_path = os.path.join(latest_session_dir, self.output_file)

#             self.out = cv2.VideoWriter(output_file_path, self.fourcc, 30.0, (640, 480))
#             self.cap = cv2.VideoCapture(0)

#             # Check if the video capture is successful
#             if not self.cap.isOpened():
#                 raise ValueError("Failed to open video capture.")
            
#             frames = []
#             num_frames = 0
#             while not self.close_received:
#                 if num_frames == 30:
#                     num_frames = 0
#                     model.rec_vid(frames)
#                     frames = []
#                 num_frames +=1
#                 ret, frame = self.cap.read()
#                 frames.append(frame)
#                 self.out.write(frame)
#                 frame = self.cvimage_to_label(frame)
#                 self.frame_signal.emit(frame)
#             self.cap.release()
#             self.out.release()
#         except Exception as e:
#             print(f"Error in VideoCaptureThread: {e}")

#     def close_recording(self):
#         self.close_received = True

#     def cvimage_to_label(self, image):
#         image = imutils.resize(image, width=640)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = QImage(image,
#                        image.shape[1],
#                        image.shape[0],
#                        QImage.Format_RGB888)
#         return image


# class QuestionsUI(QtWidgets.QWidget):
#     question_ui_requested_signal = pyqtSignal(bool)

#     def __init__(self):
#         super().__init__()
#         self.finish_btn = None
#         self.home_btn = None
#         self.QuestionsGeneratorIter = None
#         self.QuestionsGenerator = None
#         self.font5 = None
#         self.grid_page2 = None
#         self.grid_page1 = None
#         self.grid_child_layout1 = None
#         self.page2 = None
#         self.page1 = None
#         self.hlayout2 = None
#         self.hlayout1 = None
#         self.horizontalLayout_btns = None
#         self.btn_style_1 = None
#         self.vlayout = None
#         self.grid_layout = None
#         self.stackedWidget = None
#         self.videoPlayer = None
#         self.AudioAndVideoMerger = None
#         self.audio_thread = None
#         self.prev_btn = None
#         self.next_btn = None
#         self.mutex = QMutex()
#         self.camera_thread = None
#         self.record_btn = None
#         self.questionLabel = None
#         self.VideoLabel = None
#         self.questionNumberLabel = None  # Label for question number
        
#         self.init_ui()
#         self.signals = 0
#         self.is_recording = False

#     def init_ui(self):
#         self.question_ui_requested_signal.connect(self.init_question_generators)

#         self.grid_layout = QtWidgets.QGridLayout(self)

#         self.vlayout = QtWidgets.QVBoxLayout()
#         self.hlayout1 = QtWidgets.QHBoxLayout()
#         self.hlayout2 = QtWidgets.QHBoxLayout()
#         self.horizontalLayout_btns = QtWidgets.QHBoxLayout()

#         self.page1 = QtWidgets.QWidget()
#         self.page2 = QtWidgets.QWidget()

#         self.grid_page1 = QtWidgets.QGridLayout(self.page1)
#         self.grid_page2 = QtWidgets.QGridLayout(self.page2)

#         self.grid_child_layout1 = QtWidgets.QHBoxLayout()

#         self.questionNumberLabel = QtWidgets.QLabel()  # Initialize the label for question number
#         self.questionNumberLabel.setAlignment(Qt.AlignCenter)
#         self.questionNumberLabel.setFont(QFont("Franklin Gothic Medium", 16, QFont.Bold))
#         self.questionNumberLabel.setStyleSheet("color: rgb(255, 255, 255)")
        
#         self.questionLabel = QtWidgets.QLabel()
#         self.questionLabel.setAlignment(Qt.AlignCenter)  # Set alignment to center
#         self.hlayout1.addWidget(self.questionLabel)
    
#         self.VideoLabel = QtWidgets.QLabel()
#         self.VideoLabel.setFixedSize(640, 480)
#         self.grid_child_layout1.addWidget(self.VideoLabel)
#         self.videoPlayer = VideoPlayer()

#         self.grid_page1.addLayout(self.grid_child_layout1, 0, 0, 1, 1)
#         self.grid_page2.addWidget(self.videoPlayer, 0, 0, 1, 1)

#         self.stackedWidget = QStackedWidget()
#         self.stackedWidget.addWidget(self.page1)
#         self.stackedWidget.addWidget(self.page2)

#         self.hlayout2.addWidget(self.stackedWidget)

#         self.record_btn = QtWidgets.QPushButton("Start Recording", clicked=self.toggle_recording)
#         self.next_btn = QtWidgets.QPushButton("Next", clicked=self.next_question)
#         self.prev_btn = QtWidgets.QPushButton("Back", clicked=self.prev_question)
#         self.home_btn = QtWidgets.QPushButton("Home")
#         self.finish_btn = QtWidgets.QPushButton("Finish")

#         self.home_btn.setObjectName("home_btn")
#         self.finish_btn.setObjectName("finish_btn")

#         self.vlayout.setSpacing(1)
#         self.vlayout.setSizeConstraint(QLayout.SetDefaultConstraint)
#         self.vlayout.setContentsMargins(10, 10, 10, 10)

#         self.hlayout1.setSpacing(1)
#         self.hlayout1.setSizeConstraint(QLayout.SetDefaultConstraint)
#         self.hlayout1.setContentsMargins(10, 10, 10, 10)

#         self.hlayout2.setSpacing(1)
#         self.hlayout2.setSizeConstraint(QLayout.SetDefaultConstraint)
#         self.hlayout2.setContentsMargins(10, 10, 10, 10)

#         self.horizontalLayout_btns.setSpacing(1)
#         self.hlayout2.setSizeConstraint(QLayout.SetDefaultConstraint)
#         self.horizontalLayout_btns.setContentsMargins(10, 10, 10, 10)

#         self.vlayout.addWidget(self.questionNumberLabel)  # Add question number label to the layout
#         self.vlayout.addLayout(self.hlayout1)
#         self.vlayout.addLayout(self.hlayout2)
#         self.vlayout.addLayout(self.horizontalLayout_btns)

#         self.grid_layout.addLayout(self.vlayout, 0, 0, 1, 1)

#         font = QFont()
#         font.setFamily(u"Franklin Gothic Medium")
#         font.setPointSize(14)
#         font.setBold(True)
#         font.setItalic(False)
#         font.setUnderline(False)
#         font.setWeight(75)
#         font.setKerning(True)
#         self.questionLabel.setFont(font)
#         self.questionLabel.setStyleSheet(u"color: rgb(255, 255, 255)")
#         self.questionLabel.setTextFormat(Qt.RichText)
#         self.questionLabel.setAlignment(Qt.AlignCenter)  # Set alignment to center
#         self.questionLabel.setMargin(25)
#         self.questionLabel.setWordWrap(True)

#         self.VideoLabel.setAlignment(Qt.AlignLeading | Qt.AlignCenter)

#         self.record_btn.setMinimumSize(QSize(150, 0))
#         self.next_btn.setMinimumSize(QSize(150, 0))
#         self.prev_btn.setMinimumSize(QSize(150, 0))
#         self.home_btn.setMinimumSize(QSize(150, 0))
#         self.finish_btn.setMinimumSize(QSize(150, 0))

#         self.font5 = QFont()
#         self.font5.setFamily(u"Franklin Gothic Demi")
#         self.font5.setPointSize(12)
#         self.record_btn.setFont(self.font5)
#         self.next_btn.setFont(self.font5)
#         self.prev_btn.setFont(self.font5)
#         self.finish_btn.setFont(self.font5)
#         self.home_btn.setFont(self.font5)

#         self.btn_style_1 = """
#             :hover {
#                 background: white;
#                 color: #191970;
#             }
    
#             QWidget {
#                 border-style: solid;
#                 border-color: black;
#                 border-radius: 20;
#                 background-color: #535C91;
#                 color: white;
#                 height: 40;
#                 margin: 5px;
#             }
#         """

#         self.record_btn.setStyleSheet(self.btn_style_1)
#         self.next_btn.setStyleSheet(self.btn_style_1)
#         self.prev_btn.setStyleSheet(self.btn_style_1)
#         self.finish_btn.setStyleSheet(self.btn_style_1)
#         self.home_btn.setStyleSheet(self.btn_style_1)

#         self.stackedWidget1 = QtWidgets.QStackedWidget()
#         self.stackedWidget2 = QtWidgets.QStackedWidget()
#         self.stackedWidget3 = QtWidgets.QStackedWidget()
#         self.stackedWidget4 = QtWidgets.QStackedWidget()

#         self.stackedWidget1.setFixedHeight(50)
#         self.stackedWidget2.setFixedHeight(50)
#         self.stackedWidget3.setFixedHeight(50)
#         self.stackedWidget4.setFixedHeight(50)

#         self.stackedWidget1.addWidget(self.home_btn)
#         self.stackedWidget1.addWidget(self.prev_btn)

#         self.stackedWidget2.addWidget(self.next_btn)
#         self.stackedWidget2.addWidget(self.finish_btn)

#         self.stackedWidget3.addWidget(self.record_btn)

#         self.horizontalLayout_btns.addWidget(self.stackedWidget3)
#         self.horizontalLayout_btns.addWidget(self.stackedWidget1)
#         self.horizontalLayout_btns.addWidget(self.stackedWidget2)

#     def init_question_generators(self):
#         self.QuestionsGenerator = QuestionGenerator()
#         self.QuestionsGeneratorIter = iter(self.QuestionsGenerator)
#         self.stackedWidget1.setCurrentIndex(0)
#         self.stackedWidget2.setCurrentIndex(0)
#         self.next_question()

#     def toggle_recording(self):
#         if self.is_recording:
#             self.close_camera()
#         else:
#             self.open_camera()
#         self.is_recording = not self.is_recording
#         self.record_btn.setText("Stop Recording" if self.is_recording else "Start Recording")

#     def open_camera(self):
#         self.videoPlayer.clear()
#         self.stackedWidget.setCurrentIndex(0)
#         self.camera_thread = VideoCaptureThread()
#         self.audio_thread = AudioCaptureThread()
#         self.camera_thread.frame_signal.connect(self.setImage)
#         self.camera_thread.finished.connect(self.VideoLabel.clear)
#         # self.camera_thread.finished.connect(self.merge_and_save)
#         # self.audio_thread.finished.connect(self.merge_and_save)
#         self.camera_thread.set_outfile(str('1') + ".mp4")
#         self.camera_thread.start()
#         while not self.camera_thread.isRunning():
#             continue
#         self.audio_thread.set_outfile(str('1') + ".wav")
#         self.audio_thread.start()

#     def get_latest_session_directory(self):
#         session_dir = "session"
#         if not os.path.exists(session_dir):
#             return None

#         subdirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
#         if not subdirs:
#             return None

#         latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(session_dir, d)))
#         return os.path.join(session_dir, latest_subdir)

#     def update_question_number_label(self):
#         self.questionNumberLabel.setText(f"Question {'1' + 1}")

#     def next_question(self):
#         try:
#             self.questionLabel.setText(next(self.QuestionsGeneratorIter))
#             self.update_question_number_label()  # Update question number label
#             if os.path.exists("./f" + str('1') + ".mp4"):
#                 self.videoPlayer.set_mediafile("./f" + str('1') + ".mp4")
#                 self.stackedWidget.setCurrentIndex(1)
#             else:
#                 self.videoPlayer.clear()
#                 self.stackedWidget.setCurrentIndex(0)
#             if '1' == self.QuestionsGenerator.get_size()-1:
#                 self.stackedWidget2.setCurrentIndex(1)
#             if '1' == 1:
#                 self.stackedWidget1.setCurrentIndex(1)

#         except StopIteration:
#             msg = QtWidgets.QMessageBox()
#             msg.setText("This is the last question!")
#             msg.exec_()

#     def prev_question(self):
#         try:
#             self.questionLabel.setText(self.QuestionsGenerator.prev())
#             self.update_question_number_label()  # Update question number label
            
#             latest_directory = self.get_latest_session_directory()
#             media_file = os.path.join(latest_directory, "f" + str('1') + ".mp4")
            
#             if latest_directory and os.path.exists(media_file):
#                 self.videoPlayer.set_mediafile(media_file)
#                 self.stackedWidget.setCurrentIndex(1)
#             else:
#                 self.videoPlayer.clear()
#                 self.stackedWidget.setCurrentIndex(0)

#             if '1' == self.QuestionsGenerator.get_size()-2:
#                 self.stackedWidget2.setCurrentIndex(0)
#             if '1' == 0:
#                 self.stackedWidget1.setCurrentIndex(0)

#         except StopIteration:
#             msg = QtWidgets.QMessageBox()
#             msg.setText("This is the first Question!")
#             msg.exec_()

#     def close_camera(self):
#         self.camera_thread.close_signal.emit(True)
#         self.audio_thread.close_signal.emit(True)

#     @Slot(QImage)
#     def setImage(self, image):
#         self.VideoLabel.setPixmap(QPixmap.fromImage(image))

#     def merge_and_save(self):
#         self.mutex.lock()
#         self.signals += 1
#         if self.signals == 2:
#             self.signals = 0
#             self.AudioAndVideoMerger = AudioAndVideoMerger()
#             self.AudioAndVideoMerger.set_audiofile(str('1') + ".wav")
#             self.AudioAndVideoMerger.set_videofile(str('1') + ".mp4")
#             self.AudioAndVideoMerger.set_outfile("f" + str('1') + ".mp4")
#             self.AudioAndVideoMerger.start()
#             self.AudioAndVideoMerger.finished.connect(self.show_player)
#         self.mutex.unlock()

#     def show_player(self):
#         latest_directory = self.get_latest_session_directory()
#         if latest_directory:
#             media_file = os.path.join(latest_directory, "f" + str('1') + ".mp4")
#             self.videoPlayer.set_mediafile(media_file)
#             self.stackedWidget.setCurrentIndex(1)
#             self.videoPlayer.play()
            
#     def clear_player(self):
#         self.videoPlayer.clear()


# model = None

# if __name__ == "__main__":
#     model = ERmodel()
     
#     app = QtWidgets.QApplication([])
#     main_window = QuestionsUI()
#     main_window.show()
#     sys.exit(app.exec_())

import json




# Open and read the JSON file
# with open('questions.json', 'r') as file:
#     json_data = file.read()

# # Parse the JSON data into a Python dictionary
# original_response = json.loads(json_data)

# Questions = []
# for key, value in original_response.items():
#     Questions.append(value["Question"])

# new_response = {}
# for i, (key, value) in enumerate(original_response.items(), start=1):
#     new_response[str(i-1)] = value["Question"]


with open('questions.json', 'r') as file:
    json_data = file.read()

# Parse the JSON data into a Python dictionary
original_response = json.loads(json_data)

Questions = []
for key, value in original_response.items():
    Questions.append(value["Question"])

new_response= {}
for i, (key, value) in enumerate(original_response.items(), start=1):
    new_response[str(i-1)] = value["Question"]
print(new_response)

# json_string = """
# {
#     "Question1": {
#         "Question": "Describe a situation where you felt completely overwhelmed, and how did you cope with it?",
#         "Answer": "During my final year of college, I had multiple projects and exams piling up, and I felt completely overwhelmed. I took a step back, prioritized my tasks, and delegated some of the workload to my teammates. I also made sure to take breaks and practice self-care to avoid burnout.",
#         "Emotion": "Calm"
#     },
#     "Question2": {
#         "Question": "What is the most spontaneous thing you have ever done, and would you do it again?",
#         "Answer": "I once decided to take a road trip with friends to a nearby city on a whim. It was amazing, and I would love to do it again. The freedom and excitement of not planning anything and just going with the flow was exhilarating.",
#         "Emotion": "Happy"
#     },
#     "Question3": {
#         "Question": "Think of a person you admire, what qualities do they possess that you wish you had, and how can you work on developing those qualities?",
#         "Answer": "I admire my grandmother's kindness and empathy towards others. I wish I had her ability to connect with people on a deeper level. I can work on developing this quality by actively listening to others and being more present in my interactions.",
#         "Emotion": "Sad"
#     },
#     "Question4": {
#         "Question": "Tell me about a time when you had to make a difficult decision, what was the outcome, and would you make the same choice again?",
#         "Answer": "I had to choose between two job offers, one with a higher salary and one with better work-life balance. I chose the latter, and it was the best decision I ever made. I'd make the same choice again because my mental health and happiness are more valuable to me than the extra money.",
#         "Emotion": "Calm"
#     },
#     "Question5": {
#         "Question": "How do you handle criticism or negative feedback, and can you give me an example from your past?",
#         "Answer": "I try to separate my self-worth from the criticism and focus on the constructive aspects. In a previous project, I received negative feedback on my presentation skills, which initially made me defensive. However, I took the feedback to heart, worked on improving, and saw significant growth in my abilities.",
#         "Emotion": "Neutral"
#     },
#     "Question6": {
#         "Question": "Describe a moment when you felt a strong sense of belonging, where was it, and what made it so special?",
#         "Answer": "During a volunteer trip to a rural village, I felt a strong sense of belonging with the community and my fellow volunteers. We worked together, shared stories, and supported each other, creating an unforgettable bond.",
#         "Emotion": "Happy"
#     },
#     "Question7": {
#         "Question": "What is something you used to believe in strongly when you were younger, but no longer believe in, and what caused you to change your mind?",
#         "Answer": "I used to believe that success was solely about achieving a high-paying job. However, as I grew older, I realized that success is more about finding fulfillment and happiness in what I do. This change in perspective was influenced by my experiences and seeing the unhappiness of others who were stuck in unfulfilling careers.",
#         "Emotion": "Surprised"
#     },
#     "Question8": {
#         "Question": "What do you value more, being liked by others or being true to yourself, and can you explain why?",
#         "Answer": "I value being true to myself more. I've learned that trying to appease others can lead to internal conflict and unhappiness. Being true to myself allows me to live authentically and find self-acceptance.",
#         "Emotion": "Calm"
#     }
# }
# """

# # Parse the JSON string
# data = json.loads(json_string)

# # Update the emotions using the sorted dictionary
# for i, key in enumerate(new_response):
#     question_key = f"Question{i+1}"
#     if question_key in data:
#         data[question_key]["Question"] = new_response[key]

# print(data)