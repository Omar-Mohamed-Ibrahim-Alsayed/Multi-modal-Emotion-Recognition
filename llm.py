from groq import Groq
import json
import os

client = Groq(
    api_key="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"
)

# Get the directory of the script file
script_dir = os.path.dirname(os.path.realpath(__file__))

while True:
    # Take user input for the prompt
    prompt = input("Enter your prompt (or type 'quit' to exit): ")

    # Check if the user wants to quit
    if prompt.lower() == "quit":
        print("Exiting...")
        break

    # Example usage: make a completion request
    response = client.chat.completions.create(
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": """You will be provided with a series of questions and answers with the respondent's emotion as a label.
                your job is to generate a report on the respondent's psychological health.
                provide me the response in json format only and nothing outside the json format.
                use this as an example
                {
                "Question1"{
                    "Question": "Provided Question"
                    "Answer": "Provided Answer"
                    "Emotion": "Provided Emotion"
                    "Comment": "Your Report"
                },
                                "Conclusion"{
                    "Diagnoses": "Your Conclusion and Remarks"
                }
                }"""
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192"
    )

    # Get the response content
    response_content = response.choices[0].message.content

    # Print the response content
    print("Response:", response_content)

    # Save the response content to a JSON file
    filename = os.path.join(script_dir, "response2.json")
    try:
        # Convert response content to a JSON object
        response_json = json.loads(response_content)

        # Save as JSON
        with open(filename, "w") as json_file:
            json.dump(response_json, json_file, indent=4)
        print(f"Response saved to {filename}")

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

        # Save as text if JSON decoding fails
        filename = os.path.join(script_dir, "response.txt")
        with open(filename, "w") as text_file:
            text_file.write(response_content)
        print(f"Response saved to {filename} as plain text")
