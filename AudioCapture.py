import sys
import wave
import pyaudio
import os
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal as Signal


class AudioCaptureThread(QThread):
    close_signal = Signal(bool)
    close_received = False
    chunk = 1024
    format = pyaudio.paInt16
    channels = 2
    rate = 44100
    filename = "output.wav"

    def set_outfile(self, filename):
        self.filename = filename

    def __init__(self):
        super().__init__()
        self.close_signal.connect(self.close_audio)

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
            p = pyaudio.PyAudio()

            # Open a stream for recording
            stream = p.open(format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk)
            frames = []
            while not self.close_received:
                data = stream.read(self.chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            latest_session_dir = self.get_latest_session_directory()
            output_file_path = os.path.join(latest_session_dir, self.filename)
            
            wf = wave.open(output_file_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def close_audio(self):
        self.close_received = True
