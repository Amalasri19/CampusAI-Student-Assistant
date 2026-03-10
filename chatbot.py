import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

# Prepare training data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train machine learning model
model = LogisticRegression()
model.fit(X, tags)

print("AI College Chatbot Started!")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    # Convert user text to vector
    input_vector = vectorizer.transform([user_input])

    # Predict intent
    predicted_tag = model.predict(input_vector)[0]

    # Get response
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            print("Bot:", random.choice(intent["responses"]))