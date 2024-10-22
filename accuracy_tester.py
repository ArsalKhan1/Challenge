import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('path_to_your_model')
tokenizer = BertTokenizer.from_pretrained('path_to_your_model')

# Set the model to evaluation mode
model.eval()

# Load the data.csv file
data = pd.read_csv('data.csv')

# Function to classify text using the model
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        # Get the probability of the 'toxic' label
        toxic_prob = probs[0][0].item()
        # Return 1 if toxic_prob >= 0.5, else return 0
        return 1 if toxic_prob >= 0.5 else 0

# Apply the classify_text function to each row in the 'text' column
data['bert_prod_result'] = data['text'].apply(classify_text)

# Save the updated data.csv file
data.to_csv('data_with_bert_prod_result.csv', index=False)
