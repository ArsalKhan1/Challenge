import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import time

# Step 1: Load and Preprocess Data (only train_bias, with optional sampling)
def load_data(sample_size=None):
    # Load the full training data
    train_bias = pd.read_csv('../jigsaw-multilingual-toxic-comment-classification/new/jigsaw-unintended-bias-train-modified.csv')
    
    # If a sample size is provided, take a random sample of the data
    if sample_size:
        train_bias = train_bias.sample(n=sample_size, random_state=42)  # Random seed for consistency
    
    return train_bias

def preprocess_data(train_bias, tokenizer, max_len=128):
    # Replace NaN values with an empty string and ensure all values are strings
    train_bias['comment_text'] = train_bias['comment_text'].fillna('').astype(str)
    
    # Tokenize the comment_text for BERT input, include attention masks
    train_encodings_bias = tokenizer(list(train_bias['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    
    # Convert 'toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack' to tensors (multi-label classification)
    train_labels_bias = torch.tensor(train_bias[['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].values)
    
    return train_encodings_bias, train_labels_bias

# Step 2: Create DataLoader objects (only for train_bias)
def create_dataloader(train_encodings_bias, train_labels_bias, batch_size=16):
    train_dataset_bias = TensorDataset(train_encodings_bias['input_ids'], train_encodings_bias['attention_mask'], train_labels_bias)
    train_loader_bias = DataLoader(train_dataset_bias, batch_size=batch_size, shuffle=True)
    return train_loader_bias

# Step 3: Load BERT model with multiple output labels
def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Step 4: Train the model with attention masks and logging (only train_bias)
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.to(device)
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits for multi-label classification
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f'Epoch {epoch+1}/{epochs}')
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids, attention_mask, labels = batch  # Get input data, attention mask, and labels
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(torch.float32).to(device)  # Convert labels to float32 for MPS compatibility
            
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


# Step 5: Save the model
def save_model(model, tokenizer, path='./toxic-classifier-test'):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# Main function to run the whole process (focus only on train_bias, with sampling option)
def main():
    # Set device to MPS if available, otherwise CPU
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print("mps selected!")

    # Load data with a small sample size (for testing)
    print("Loading data...")
    sample_size = 100  # Set the sample size for testing, e.g., 100 rows
    train_bias = load_data(sample_size=sample_size)
    
    # Load BERT model and tokenizer
    print("Loading BERT model...")
    model, tokenizer = load_model(num_labels=6)  # 6 labels for the different categories
    
    # Preprocess data (only train_bias)
    print("Preprocessing data...")
    train_encodings_bias, train_labels_bias = preprocess_data(train_bias, tokenizer)
    
    # Create data loader (only train_bias)
    print("Creating data loader...")
    train_loader_bias = create_dataloader(train_encodings_bias, train_labels_bias, batch_size=16)
    
    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Train the model (only on train_bias)
    print("Training model...")
    train_model(model, train_loader_bias, optimizer, device)
    
    # Save the model
    print("Saving model...")
    save_model(model, tokenizer)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
