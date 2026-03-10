import json
import random
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load intents
with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Train ML model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

model = LogisticRegression()
model.fit(X, tags)

def get_response(user_input):

    input_vector = vectorizer.transform([user_input])
    predicted_tag = model.predict(input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():

    user_message = request.json["message"]

    response = get_response(user_message)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)