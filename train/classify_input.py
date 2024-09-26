import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model and tokenizer
model_path = './toxic-classifier'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set device to MPS if available, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Function to preprocess the input text for BERT
def preprocess_text(text, tokenizer, max_len=128):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    return inputs['input_ids'].to(device), inputs['attention_mask'].to(device)

# Function to classify input text at all three levels
def classify_message(text):
    # Preprocess the text
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    
    # Get the model's prediction (logits)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()[0]  # Get probabilities using sigmoid

    # Output the percentage for each label
    print(f"Message: '{text}'")

    # Level 1: Toxic
    toxic_percent = predictions[0] * 100
    print(f"Level 1 - Toxic: {toxic_percent:.2f}%")

    # Level 2: Severe Toxic
    severe_toxic_percent = predictions[1] * 100
    print(f"Level 2 - Severe Toxic: {severe_toxic_percent:.2f}%")

    # Level 3: Obscene, Threat, Insult, Identity Hate
    level_3_categories = ['Obscene', 'Threat', 'Insult', 'Identity Hate']
    level_3_predictions = predictions[2:6]  # Get probabilities for the four categories

    for i, category in enumerate(level_3_categories):
        category_percent = level_3_predictions[i] * 100
        print(f"Level 3 - {category}: {category_percent:.2f}%")

# Test the classification on a sample input message
input_message = "Fuck you bitch"
classify_message(input_message)
