import sys

from PyQt5.QtCore import QThread
from llm import *
import json

Questions = []


class QuestionGeneratorThread(QThread):
    api_key = "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"
    psychologicalReportGenerator = PsychologicalReportGenerator(api_key)
    topic = ""
    global Questions
    def run(self):
        try:
            Questions.clear()
            jsonAnswer = json.loads(self.psychologicalReportGenerator.generate_questions(questions_type=self.topic))
            for key, value in jsonAnswer.items():
                Questions.append(value["Question"])
        except Exception:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def set_topic(self, topic):
        self.topic = topic


class QuestionGenerator:
    indx = 0

    def __iter__(self):
        self.indx = -1
        return self

    def __next__(self):
        if self.indx >= len(Questions) - 1:
            raise StopIteration
        self.indx += 1
        result = Questions[self.indx]
        return result

    def prev(self):
        self.indx -= 1
        if self.indx < 0:
            self.indx += 1
            raise StopIteration
        return Questions[self.indx]

    def get_size(self):
        return len(Questions)

    def get_index(self):
        return self.indx

    def get_questions(self):
        return Questions
