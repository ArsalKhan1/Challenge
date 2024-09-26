import pandas as pd
import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import ast

# Step 1: Load and Preprocess Data (with subset sampling option)
def load_data(sample_size=None):
    # Load the full dataset
    train_data = pd.read_csv('../jigsaw-multilingual-toxic-comment-classification/new/jigsaw-unintended-bias-train-processed-seqlen128.csv')
    validation_data = pd.read_csv('../jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv')

    
    # If a sample size is provided, take a random sample of the data
    if sample_size:
        train_data = train_data.sample(n=sample_size)
        validation_data = validation_data.sample(n=sample_size)
    
    return train_data, validation_data

# def preprocess_data(df):
#     input_ids = torch.tensor([ast.literal_eval(x) for x in df['input_word_ids']])
#     attention_masks = torch.tensor([ast.literal_eval(x) for x in df['input_mask']])
#     segment_ids = torch.tensor([ast.literal_eval(x) for x in df['all_segment_id']])
#     labels = torch.tensor(df[['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].values)
    
#     return input_ids, attention_masks, segment_ids, labels
# Preprocess the labels based on the actual column names from the CSV
def preprocess_data(df):
    input_ids = torch.tensor([ast.literal_eval(x) for x in df['input_word_ids']])
    attention_masks = torch.tensor([ast.literal_eval(x) for x in df['input_mask']])
    segment_ids = torch.tensor([ast.literal_eval(x) for x in df['all_segment_id']])
    
    # Adjust column names to match what is in your CSV
    labels = torch.tensor(df[['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].values)
    
    return input_ids, attention_masks, segment_ids, labels



# Step 2: Create DataLoader objects
def create_dataloaders(input_ids, attention_masks, segment_ids, labels, batch_size=16):
    dataset = TensorDataset(input_ids, attention_masks, segment_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Step 3: Load BERT model with multiple output labels
def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

# Step 4: Train the model with attention masks and logging
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.to(device)
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits for multi-label classification
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids, attention_mask, segment_ids, labels = [t.to(device) for t in batch]
            
            # Forward pass through the model
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
            logits = outputs.logits

            # Calculate the loss using BCEWithLogitsLoss
            loss = loss_fn(logits, labels.float())
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
        
        print(f"Loss after epoch {epoch+1}: {total_loss/len(train_loader)}")

# Step 5: Evaluate the model
def evaluate_model(model, validation_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, segment_ids, labels = [t.to(device) for t in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Here, we'll implement evaluation logic based on your criteria
            # Example: If probability >= 0.5, classify as that label
            correct_predictions = (predictions >= 0.5) == labels.cpu().numpy()
            correct += correct_predictions.sum()
            total += labels.numel()
    
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy}')
    return accuracy

# Step 6: Save the model
def save_model(model, path='./toxic-classifier-preprocessed-bert-small'):
    model.save_pretrained(path)


def main():
    # Set device to MPS if available, otherwise CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else print("no mps")
    
    # Load a subset of the data (e.g., 100 samples for quick testing)
    print("Loading data...")
    sample_size = 100  # Adjust sample size for testing purposes
    train_data, validation_data = load_data(sample_size=sample_size)
    
    # Preprocess data
    print("Preprocessing data...")
    train_input_ids, train_attention_masks, train_segment_ids, train_labels = preprocess_data(train_data)
    validation_input_ids, validation_attention_masks, validation_segment_ids, validation_labels = preprocess_data(validation_data)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloaders(train_input_ids, train_attention_masks, train_segment_ids, train_labels, batch_size=16)
    validation_loader = create_dataloaders(validation_input_ids, validation_attention_masks, validation_segment_ids, validation_labels, batch_size=16)
    
    # Load BERT model and set optimizer
    print("Loading BERT model...")
    model = load_model(num_labels=6)  # 6 labels for the different categories
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Train the model
    print("Training model...")
    train_model(model, train_loader, optimizer, device)
    
    # Evaluate the model
    print("Evaluating model...")
    accuracy = evaluate_model(model, validation_loader, device)
    
    # Save the model
    print("Saving model...")
    save_model(model)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
