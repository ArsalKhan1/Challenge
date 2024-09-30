from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from collections import Counter
import os

# Set the Matplotlib backend to 'Agg' to prevent GUI errors on macOS
plt.switch_backend('Agg')

# Initialize Flask application
app = Flask(__name__)

# In-memory storage for message history and phrase counter (For production, consider using a database)
message_history = []
phrase_counter = Counter()

# Load pre-trained model and tokenizer (adjust the model path if necessary)
model_path = "../train/bert-base-uncased-challenge-1"  # This should be the directory where your model is saved
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Helper function to classify input text
def classify_text(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    # Map probabilities to labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result = {label: prob for label, prob in zip(labels, probabilities)}

    return result

# Route for the "Detector" page (default page)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        user_input = request.form.get("message")
        
        # Classify the input message
        classification_result = classify_text(user_input)
        
        # Save the input message and its classification result to history
        message_history.append({
            'message': user_input,
            'result': classification_result
        })

        # Track common phrases
        phrase_counter[user_input] += 1

        # Redirect to home page to clear POST request and update results
        return redirect(url_for('index'))

    # Render page with message history
    return render_template("index.html", message_history=message_history)

# Route for the "Results" page
@app.route("/results")
def results():
    # Debugging: Print the contents of phrase_counter
    print(f"Phrase Counter: {phrase_counter}")

    # Get the most common phrases and their counts
    top_phrases = phrase_counter.most_common(10)
    if not top_phrases:
        print("No data found for generating graph.")
        return render_template("results.html", graph_url=None)

    phrases, counts = zip(*top_phrases)

    # Enhanced graph visualization
    plt.figure(figsize=(14, 8))  # Increased figure size for better visibility
    bars = plt.barh(phrases, counts, color='#5FA8D3', edgecolor='#1d6e8d', height=0.6)

    # Set title and labels with improved styling
    plt.title("AI Abuse Detector's Most Commonly Searched Phrases", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel("Number of Searches", fontsize=16)
    plt.ylabel("Phrases", fontsize=16)

    # Remove values inside and around the bars, only keep values on the x-axis
    plt.xticks(fontsize=14)  # Increase font size of x-axis values
    plt.yticks(fontsize=14)  # Increase font size of y-axis values

    # Additional styling options
    plt.gca().invert_yaxis()  # Display highest number of searches at the top
    plt.grid(False)  # Remove gridlines for a cleaner look

    # Set axis background and frame
    plt.gca().set_facecolor('#f9f9f9')  # Light gray background for axes
    plt.gca().spines['top'].set_visible(False)  # Hide the top frame
    plt.gca().spines['right'].set_visible(False)  # Hide the right frame
    plt.gca().spines['left'].set_visible(False)  # Hide the left frame

    # Save the graph as an image
    static_folder = 'static'
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    graph_path = os.path.join(static_folder, 'common_phrases.png')
    plt.savefig(graph_path, bbox_inches='tight', dpi=300)  # Save with higher DPI for better quality
    plt.close()


    # Debugging: Check if the image is saved correctly
    if os.path.exists(graph_path):
        print(f"Graph saved at: {graph_path}")
    else:
        print("Failed to save graph.")

    # Render results page with the graph
    return render_template("results.html", graph_url=url_for('static', filename='common_phrases.png'))

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
