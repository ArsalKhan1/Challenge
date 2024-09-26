import torch
from transformers import BertTokenizer, BertForSequenceClassification
import ast

# Load the trained model and tokenizer
model_path = './toxic-classifier'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)

# Set device to MPS if available, otherwise CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Preprocess the text into BERT input format
def preprocess_input_text(text, tokenizer, max_len=128):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device), encoding['token_type_ids'].to(device)

# Classify input message into toxic categories
def classify_message(text):
    input_ids, attention_mask, token_type_ids = preprocess_input_text(text, tokenizer)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]  # Convert to probabilities

    categories = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Attack']
    
    # Print results for each category with probability
    for i, category in enumerate(categories):
        print(f"{category}: {probabilities[i] * 100:.2f}%")

# Example usage
input_message = "You are an idiot!"
classify_message(input_message)
