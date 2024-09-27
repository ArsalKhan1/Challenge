from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)

# Load the custom fine-tuned DistilBERT model and tokenizer
MODEL_PATH = './distilbert-finetuned-model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Store conversation history as a global variable
conversation_history = []

# Define the labels for each classification level
level_3_labels = ['Obscene', 'Threat', 'Insult', 'Identity Hate']

# Function to make predictions using the custom model
def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()[0]  # Apply sigmoid for binary/multilabel classification

    # Map probabilities to classifications and define thresholds for each label
    level_1_classification = "Clean" if probs[0] <= 0.5 else "Toxic"
    level_2_classification = "Severe Toxic" if level_1_classification == "Toxic" and probs[1] > 0.5 else "Toxic"
    level_3_classification = {
        level_3_labels[i]: "true" if probs[i + 2] > 0.5 else "false" for i in range(len(level_3_labels))
    }

    # Results dictionary to store probabilities for each label
    results = {
        'toxicity': probs[0],
        'severe_toxicity': probs[1],
        'obscene': probs[2],
        'threat': probs[3],
        'insult': probs[4],
        'identity_attack': probs[5]
    }

    return level_1_classification, level_2_classification, level_3_classification, results

@app.route('/', methods=['GET', 'POST'])
def home():
    global conversation_history  # Access global variable
    text = None
    level_1_classification = None
    level_2_classification = None
    level_3_classification = {}
    results = None

    if request.method == 'POST':
        text = request.form['text']

        # Use the custom model to classify the input text
        level_1_classification, level_2_classification, level_3_classification, results = classify_text(text)

        # Append user input and response to conversation history
        conversation_history.append({
            'text': text,
            'level_1_classification': level_1_classification,
            'level_2_classification': level_2_classification,
            'level_3_classification': level_3_classification,
            'results': results
        })

    # Pass the conversation history to the template
    return render_template('indexv2.html', conversation_history=conversation_history)

if __name__ == "__main__":
    app.run(debug=True)
