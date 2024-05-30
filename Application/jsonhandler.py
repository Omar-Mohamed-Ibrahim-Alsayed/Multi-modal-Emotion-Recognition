import json

# Original dictionary
emotion_dict = {'3': 'Happy', '6': 'Sad', '1': 'Sad', '2': 'Sad', '0': 'Happy', '7': 'Sad', '5': 'Sad', '4': 'Happy'}

# Sort the dictionary by keys
sorted_emotion_dict = dict(sorted(emotion_dict.items()))

# JSON string
json_string = """
{
    "Question1": {
        "Question": "Describe a situation where you felt completely overwhelmed, and how did you cope with it?",
        "Answer": "During my final year of college, I had multiple projects and exams piling up, and I felt completely overwhelmed. I took a step back, prioritized my tasks, and delegated some of the workload to my teammates. I also made sure to take breaks and practice self-care to avoid burnout.",
        "Emotion": "Calm"
    },
    "Question2": {
        "Question": "What is the most spontaneous thing you have ever done, and would you do it again?",
        "Answer": "I once decided to take a road trip with friends to a nearby city on a whim. It was amazing, and I would love to do it again. The freedom and excitement of not planning anything and just going with the flow was exhilarating.",
        "Emotion": "Happy"
    },
    "Question3": {
        "Question": "Think of a person you admire, what qualities do they possess that you wish you had, and how can you work on developing those qualities?",
        "Answer": "I admire my grandmother's kindness and empathy towards others. I wish I had her ability to connect with people on a deeper level. I can work on developing this quality by actively listening to others and being more present in my interactions.",
        "Emotion": "Sad"
    },
    "Question4": {
        "Question": "Tell me about a time when you had to make a difficult decision, what was the outcome, and would you make the same choice again?",
        "Answer": "I had to choose between two job offers, one with a higher salary and one with better work-life balance. I chose the latter, and it was the best decision I ever made. I'd make the same choice again because my mental health and happiness are more valuable to me than the extra money.",
        "Emotion": "Calm"
    },
    "Question5": {
        "Question": "How do you handle criticism or negative feedback, and can you give me an example from your past?",
        "Answer": "I try to separate my self-worth from the criticism and focus on the constructive aspects. In a previous project, I received negative feedback on my presentation skills, which initially made me defensive. However, I took the feedback to heart, worked on improving, and saw significant growth in my abilities.",
        "Emotion": "Neutral"
    },
    "Question6": {
        "Question": "Describe a moment when you felt a strong sense of belonging, where was it, and what made it so special?",
        "Answer": "During a volunteer trip to a rural village, I felt a strong sense of belonging with the community and my fellow volunteers. We worked together, shared stories, and supported each other, creating an unforgettable bond.",
        "Emotion": "Happy"
    },
    "Question7": {
        "Question": "What is something you used to believe in strongly when you were younger, but no longer believe in, and what caused you to change your mind?",
        "Answer": "I used to believe that success was solely about achieving a high-paying job. However, as I grew older, I realized that success is more about finding fulfillment and happiness in what I do. This change in perspective was influenced by my experiences and seeing the unhappiness of others who were stuck in unfulfilling careers.",
        "Emotion": "Surprised"
    },
    "Question8": {
        "Question": "What do you value more, being liked by others or being true to yourself, and can you explain why?",
        "Answer": "I value being true to myself more. I've learned that trying to appease others can lead to internal conflict and unhappiness. Being true to myself allows me to live authentically and find self-acceptance.",
        "Emotion": "Calm"
    }
}
"""

# Parse the JSON string
data = json.loads(json_string)

# Update the emotions using the sorted dictionary
for i, key in enumerate(sorted_emotion_dict):
    question_key = f"Question{i+1}"
    if question_key in data:
        data[question_key]["Emotion"] = sorted_emotion_dict[key]

# Convert the updated JSON back to a string
updated_json_string = json.dumps(data, indent=4)

# Generate the markdown text (assuming report_generator is defined)
markdown_text = report_generator.generate_report(updated_json_string)

print(updated_json_string)  # To see the updated JSON structure
