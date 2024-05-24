import sys
import wave

import pyaudio
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
            wf = wave.open(self.filename, 'wb')
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