from groq import Groq

client = Groq(
    api_key="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"
)

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
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192"
    )

    # Print the response content
    print("Response:", response.choices[0].message.content)

# import os
# from groq import Groq

# client = Groq(api_key="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")
# filename = os.path.dirname(__file__) + "/tmp.wav"

# with open(filename, "rb") as file:
#     transcription = client.audio.transcriptions.create(
#       file=(filename, file.read()),
#       model="whisper-large-v3",
#     )
#     print(transcription.text)