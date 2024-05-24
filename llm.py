from groq import Groq
import json
import os

class PsychologicalReportGenerator:
    def __init__(self, api_key, model="llama3-70b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.script_dir = os.path.dirname(os.path.realpath(__file__))

    def generate_questions(self, number_of_questions=8, questions_type="psychological"):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""provide me with a set of {number_of_questions} questions as a {questions_type} test to be given they are open questions where the respondent will answer by talking
                    Provide the response in JSON format only and nothing outside the JSON format.
                    Use this as an example:
                    {{
                        "Question1": {{
                            "Question": "Your generated Question 1"
                        }},
                        "Question2": {{
                            "Question": "Your generated Question 2"
                        }}
                    }}"""
                }
            ],
            model=self.model
        )

        response_content = response.choices[0].message.content
        print("Response:", response_content)
        self._save_response(response_content, "questions.json")
        return response_content
    
    def _generate_answers(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """answer these questions as if you were the respondent and label each answer with one of the following emotions
                    emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
                    Provide the response in JSON format only and nothing outside the JSON format.
                    Use this as an example:
                    {
                        "Question1": {
                            "Question": "Provided Question",
                            "Answer": "Your Answer",
                            "Emotion": "Random Emotion"
                        },
                        "Question2": {
                            "Question": "Provided Question",
                            "Answer": "Your Answer",
                            "Emotion": "Random Emotion"
                        }
                    }"""
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model
        )

        response_content = response.choices[0].message.content
        print("Response:", response_content)
        self._save_response(response_content, "answers.json")
        return response_content
    

    def _generate_report(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You will be provided with a series of questions and answers with the respondent's emotion as a label.
                    Your job is to generate a report on the respondent's psychological health.
                    Provide the response in JSON format only and nothing outside the JSON format.
                    Use this as an example:
                    {
                        "Question1": {
                            "Question": "Provided Question",
                            "Answer": "Provided Answer",
                            "Emotion": "Provided Emotion",
                            "Comment": "Your Report"
                        },
                        "Conclusion": {
                            "Diagnoses": "Your Conclusion and Remarks"
                        }
                    }"""
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model
        )

        response_content = response.choices[0].message.content
        print("Response:", response_content)
        self._save_response(response_content, "report.json")
        return response_content
    




# Saving Response
    def _save_response(self, response_content, filename):
        filepath = os.path.join(self.script_dir, filename)
        try:
            response_json = json.loads(response_content)
            with open(filepath, "w") as json_file:
                json.dump(response_json, json_file, indent=4)
            print(f"Response saved to {filepath}")
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            filepath = os.path.join(self.script_dir, filename.replace(".json", ".txt"))
            with open(filepath, "w") as text_file:
                text_file.write(response_content)
            print(f"Response saved to {filepath} as plain text")

# Generate Answers
    def generate_answers(self, questions):
        prompt = json.dumps(questions)
        return self._generate_answers(prompt)

# Generate Report
    def generate_report(self, questions_and_answers):
        prompt = json.dumps(questions_and_answers)
        return self._generate_report(prompt)



def main():
    # Test
    api_key="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"
    report_generator = PsychologicalReportGenerator(api_key)
    # List of psychological tests
    psychological_tests = [
        "Psychological Test",
        "Depression Test",
        "Anxiety Test",
        "Stress Test",
        "Personality Test",
        "Cognitive Function Test",
        "Emotional Intelligence Test",
        "Self-Esteem Test",
        "Social Skills Test",
        "Mood Disorder Test",
        "ADHD Test",
        "Bipolar Disorder Test",
        "PTSD Test",
        "Eating Disorder Test",
        "Autism Spectrum Disorder Test",
        "Obsessive-Compulsive Disorder (OCD) Test",
        "Phobia Test",
        "Sleep Disorder Test",
        "Substance Abuse Test",
        "General Mental Health Assessment"
    ]

    while(True):
        # Display the list of tests
        print("Please choose a test by entering the corresponding number:")
        for i, test in enumerate(psychological_tests, start=1):
            print(f"{i}. {test}")

        # Get user input
        choice = int(input("Enter the number of the test you want to take: "))

        # Validate the input
        if 1 <= choice <= len(psychological_tests):
            selected_test = psychological_tests[choice - 1]
            print(f"You have selected: {selected_test}")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and", len(psychological_tests))

    questions = report_generator.generate_questions(selected_test)
    print(f"Questions generated:{questions}")

    answers = report_generator.generate_answers(questions)
    print(f"answers generated:{answers}")

    report = report_generator.generate_report(answers)
    print(f"Report generated:{report}")


if __name__ == "__main__":
    main()
