import contextlib
import subprocess
import sys
import wave

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
        duration = frames / fps
        return duration

    def run(self):
        self.video_duration = self.get_video_duration(self.video_file)
        self.audio_duration = self.get_audio_duration(self.audio_file)

        self.temp = ""

        if self.video_duration > self.audio_duration:
            self.long_file = self.video_file
            self.short_file = self.audio_file
        else:
            self.long_file = self.audio_file
            self.short_file = self.video_file

        self.temp = "t" + self.long_file

        self.min_duration = min(self.video_duration, self.audio_duration)
        self.command_1 = [
            "./ffmpeg/bin/ffmpeg",
            "-sseof", str(-1.0 * self.min_duration),  # Input video file
            "-i", self.long_file,  # Input audio file
            "-t", str(self.min_duration),  # Copy video stream without re-encoding
            "-y", self.temp,
        ]

        self.command_2 = [
            "./ffmpeg/bin/ffmpeg",
            "-i", self.short_file,  # Input video file
            "-i", self.temp,  # Input audio file
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-y", self.output_file,  # Output file path
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