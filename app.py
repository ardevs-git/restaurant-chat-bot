from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load Data
df_qa = pd.read_csv("Shivneri_Menu_QA.csv")

# Extract questions and responses
questions = df_qa['questions'].fillna('')
responses = df_qa['responses'].fillna('')

# Convert text data into numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chatbot Function using Cosine Similarity
def chatbot_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_input_vectorized, X)
    best_match_idx = np.argmax(similarity)
    return responses.iloc[best_match_idx]

# Flask App
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    bot_reply = chatbot_response(user_input)
    return jsonify({"reply": bot_reply})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
