import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import time

# Step 1: Load and Preprocess Data
def load_data():
    # Load the training data
    train_toxic = pd.read_csv('../jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
    train_bias = pd.read_csv('../jigsaw-multilingual-toxic-comment-classification/new/jigsaw-unintended-bias-train.csv')
    validation_data = pd.read_csv('../jigsaw-multilingual-toxic-comment-classification/validation.csv')
 
    
    return train_toxic, train_bias, validation_data

def preprocess_data(train_toxic, train_bias, validation_data, tokenizer, max_len=128):
    # Replace NaN values with an empty string and ensure all values are strings
    train_toxic['comment_text'] = train_toxic['comment_text'].fillna('').astype(str)
    train_bias['comment_text'] = train_bias['comment_text'].fillna('').astype(str)
    validation_data['comment_text'] = validation_data['comment_text'].fillna('').astype(str)
    
    # Tokenize the comment_text for BERT input, include attention masks
    train_encodings_toxic = tokenizer(list(train_toxic['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    train_encodings_bias = tokenizer(list(train_bias['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    validation_encodings = tokenizer(list(validation_data['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    
    # Convert 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate' to tensors (multi-label classification)
    train_labels_toxic = torch.tensor(train_toxic[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values)
    train_labels_bias = torch.tensor(train_bias[['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].values)
    validation_labels = torch.tensor(validation_data['toxic'].values)
    
    return train_encodings_toxic, train_labels_toxic, train_encodings_bias, train_labels_bias, validation_encodings, validation_labels

# Step 2: Create DataLoader objects
def create_dataloaders(train_encodings_toxic, train_labels_toxic, train_encodings_bias, train_labels_bias, validation_encodings, validation_labels, batch_size=16):
    train_dataset_toxic = TensorDataset(train_encodings_toxic['input_ids'], train_encodings_toxic['attention_mask'], train_labels_toxic)
    train_dataset_bias = TensorDataset(train_encodings_bias['input_ids'], train_encodings_bias['attention_mask'], train_labels_bias)
    validation_dataset = TensorDataset(validation_encodings['input_ids'], validation_encodings['attention_mask'], validation_labels)
    
    train_loader_toxic = DataLoader(train_dataset_toxic, batch_size=batch_size, shuffle=True)
    train_loader_bias = DataLoader(train_dataset_bias, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    
    return train_loader_toxic, train_loader_bias, validation_loader

# Step 3: Load BERT model with multiple output labels
def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Step 4: Train the model with attention masks and logging
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.to(device)
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits for multi-label classification
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f'Epoch {epoch+1}/{epochs}')
        total_loss = 0
        
        # Iterate over the batches in the DataLoader
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids, attention_mask, labels = batch  # Get input data, attention mask, and labels
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).float()  # Convert labels to float for BCEWithLogitsLoss
            
            # Forward pass through the model
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate the loss using BCEWithLogitsLoss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item()}')
        
        print(f"Loss after epoch {epoch+1}: {total_loss/len(train_loader)}")
        end_time = time.time()
        print(f'Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds')

# Step 5: Evaluate the model
def evaluate_model(model, validation_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).float()
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy}')
    return accuracy

# Step 6: Save the model
def save_model(model, tokenizer, path='./toxic-classifier'):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# Main function to run the whole process
def main():
    # Set device to MPS if available, otherwise CPU
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
        print("mps selected!")

    
    # Load data
    print("Loading data...")
    train_toxic, train_bias, validation_data = load_data()
    
    # Load BERT model and tokenizer
    print("Loading BERT model...")
    model, tokenizer = load_model(num_labels=6)  # 6 labels for the different categories
    
    # Preprocess data
    print("Preprocessing data...")
    train_encodings_toxic, train_labels_toxic, train_encodings_bias, train_labels_bias, validation_encodings, validation_labels = preprocess_data(train_toxic, train_bias, validation_data, tokenizer)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader_toxic, train_loader_bias, validation_loader = create_dataloaders(train_encodings_toxic, train_labels_toxic, train_encodings_bias, train_labels_bias, validation_encodings, validation_labels, batch_size=16)
    
    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Train the model
    print("Training model...")
    train_model(model, train_loader_toxic, optimizer, device)
    
    # Evaluate the model
    print("Evaluating model...")
    accuracy = evaluate_model(model, validation_loader, device)
    
    # Save the model
    print("Saving model...")
    save_model(model, tokenizer)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
