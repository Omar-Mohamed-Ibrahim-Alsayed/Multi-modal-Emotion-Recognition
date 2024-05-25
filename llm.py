from groq import Groq
import json
import os
import time


class PsychologicalReportGenerator:
    def __init__(self, api_key="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", model="llama3-70b-8192"):
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
        response_content = self._save_response(response_content, "questions.json")
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

    def fix_output(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """ The following response should be in JSON format however it has an error which will be provided. Fix it and return the json format only no extra messages.
                                        Provide the response in JSON format only and nothing outside the JSON format.

                    """
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model
        )
        return response.choices[0].message.content

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
        json_response = self._save_response(response_content, "report.json")
        return json_response

    # Saving Response
    def _save_response(self, response_content, filename):
        filepath = os.path.join(self.script_dir, filename)
        while (True):
            try:
                response_json = json.loads(response_content)
                with open(filepath, "w") as json_file:
                    json.dump(response_json, json_file, indent=4)
                print(f"Response saved to {filepath}")
                return response_content
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                response_content = self.fix_output(response_content)

    # Generate Answers
    def generate_answers(self, questions):
        while (True):
            try:
                prompt = json.dumps(questions)
                break
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                questions = self.fix_output(questions)
        return self._generate_answers(prompt)

    # Generate Report
    def generate_report(self, questions_and_answers):
        while (True):
            try:
                prompt = json.dumps(questions_and_answers)
                break
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                questions_and_answers = self.fix_output(questions_and_answers)
        return self._generate_report(prompt)


def main():
    # Test
    report_generator = PsychologicalReportGenerator()
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
    for j in range(1, 21):
        while (True):
            # Display the list of tests
            print("Please choose a test by entering the corresponding number:")
            for i, test in enumerate(psychological_tests, start=1):
                print(f"{i}. {test}")

            # Get user input
            choice = j
            print(f"iteration number {j} out of 20")
            # Validate the input
            if 1 <= choice <= len(psychological_tests):
                selected_test = psychological_tests[choice - 1]
                print(f"You have selected: {selected_test}")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and", len(psychological_tests))

        questions = report_generator.generate_questions()
        print(f"Questions generated:{questions}")

        answers = report_generator.generate_answers(questions)
        print(f"answers generated:{answers}")

        report = report_generator.generate_report(answers)
        print(f"Report generated:{report}")
        time.sleep(10)
    print("7amdellah 3al salama w adr w latf")


if __name__ == "__main__":
    main()
