import requests


class GeminiAPIRequester:
    query = """
    Consider yourself as a psychiatric with comprehensive knowledge about psychological disease or disorders so 
    I'll provide you with the name of psychological disease or disorder and I want you to provide a list of question 
    to be answered by patients  to determine whether the person has that disease or not, then I'll provide you with 
    the patient answers and his emotion when he said that answer. please provide the numbered list of questions 
    separated by new lines without any extra characters. The name of the disease is 
    """
    apiKey = "AIzaSyAXdsMLYA8_g95nBiPuQ35-wX4TSXbGea4"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + apiKey
    jsonReq = "{\"contents\":[{\"parts\":[{\"text\":\"" + query.replace("\n", "") + "\"}]}]}"
    payload = {
        "contents": [
            {"parts":
                [
                    {
                        "text": ""
                    }
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json"
    }

    def generateQuestionList(self, disease_name):
        new_query = self.query + disease_name
        jsonReq = "{\"contents\":[{\"parts\":[{\"text\":\"" + new_query.replace("\n", "") + "\"}]}]}"
        answer = requests.post(url=self.url, headers=self.headers, json=jsonReq)
        Questions = answer.text.splitlines()
        print(Questions)




class QuestionGenerator:
    Questions = ["How have you been feeling lately?",
                 "Have you been experiencing changes in your appetite, either eating more or less than usual?",
                 "Are you finding it difficult to concentrate or make decisions?",
                 "Have you been feeling more tired or fatigued than usual?",
                 "Have you lost interest in activities that you used to enjoy?",
                 "Do you find yourself feeling hopeless or worthless?"]
    indx = 0

    def __iter__(self):
        self.indx = -1
        return self

    def __next__(self):
        if self.indx >= len(self.Questions) - 1:
            raise StopIteration
        self.indx += 1
        result = self.Questions[self.indx]
        return result

    def prev(self):
        self.indx -= 1
        if self.indx < 0:
            self.indx += 1
            raise StopIteration
        return self.Questions[self.indx]

    def get_index(self):
        return self.indx


if __name__ == '__main__':
    GeminiAPIRequester = GeminiAPIRequester()
    GeminiAPIRequester.generateQuestionList("phobia")