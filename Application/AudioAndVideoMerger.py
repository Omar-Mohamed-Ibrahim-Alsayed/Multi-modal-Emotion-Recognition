import contextlib
import subprocess
import sys
import wave
import os
from datetime import datetime
import cv2
from PyQt5.QtCore import QThread

class AudioAndVideoMerger(QThread):

    video_file = "output.mp4"
    audio_file = "output.wav"  # Or any supported audio format
    output_file = "merged_video.mp4"  # You can change the output format

    def set_outfile(self, filename):
        self.output_file = filename

    def set_videofile(self, filename):
        self.video_file = filename

    def set_audiofile(self, filename):
        self.audio_file = filename

    def get_audio_duration(self, fname):
        with contextlib.closing(wave.open(fname, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    def get_video_duration(self, fname):
        data = cv2.VideoCapture(fname)
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = data.get(cv2.CAP_PROP_FPS)

        # Check if fps is zero or negative
        if fps <= 0:
            # Handle the case where fps is invalid
            print("Invalid FPS value:", fps)
            return None

        duration = frames / fps
        return duration

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
        latest_session_dir = self.get_latest_session_directory()
        audio_file_path = os.path.join(latest_session_dir, self.audio_file)
        video_file_path = os.path.join(latest_session_dir, self.video_file)
        
        self.video_duration = self.get_video_duration(video_file_path)
        self.audio_duration = self.get_audio_duration(audio_file_path)

        self.temp = ""

        if self.video_duration > self.audio_duration:
            self.long_file = video_file_path
            self.short_file = audio_file_path
        else:
            self.long_file = audio_file_path
            self.short_file = video_file_path

        self.temp = "t" + os.path.basename(self.long_file)

        self.min_duration = min(self.video_duration, self.audio_duration)
        self.command_1 = [
            "ffmpeg",
            "-sseof", str(-1.0 * self.min_duration),  # Input video file
            "-i", self.long_file,  # Input audio file
            "-t", str(self.min_duration),  # Copy video stream without re-encoding
            "-y", self.temp,
        ]

        output_file_path = os.path.join(latest_session_dir, self.output_file)

        self.command_2 = [
            "ffmpeg",
            "-i", self.short_file,  # Input video file
            "-i", self.temp,  # Input audio file
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-y", output_file_path,  # Output file path
        ]

        try:
            p1 = subprocess.run(self.command_1, cwd="./")
        except Exception as e:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

        try:
            p2 = subprocess.run(self.command_2, cwd="./")
        except Exception as e:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)
