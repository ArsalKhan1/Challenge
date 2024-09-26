import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import time

# Step 1: Load and Preprocess Data (train and validation)
def load_data(train_file, validation_file):
    # Load the training data and validation data
    train_bias = pd.read_csv(train_file)
    validation_data = pd.read_csv(validation_file)
    
    return train_bias, validation_data

def preprocess_data(data, tokenizer, max_len=128, is_validation=False):
    # Replace NaN values with an empty string and ensure all values are strings
    data['comment_text'] = data['comment_text'].fillna('').astype(str)

    # Tokenize the comment_text for BERT input, include attention masks
    encodings = tokenizer(list(data['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    if is_validation:
        # For validation data, we only have the 'toxic' column
        labels = torch.tensor(data['toxic'].values)
    else:
        # For training data, we have multiple labels
        labels = torch.tensor(data[['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].values)
    
    return encodings, labels

# Step 2: Create DataLoader objects (for train and validation data)
def create_dataloader(encodings, labels, batch_size=16):
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Load BERT model with multiple output labels
def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Step 4: Train the model with attention masks and logging (train and validation)
def train_model(model, train_loader, validation_loader, optimizer, device, epochs=3):
    model.to(device)
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits for multi-label classification
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f'Epoch {epoch+1}/{epochs}')
        total_loss = 0
        
        # Training loop
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids, attention_mask, labels = batch  # Get input data, attention mask, and labels
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Convert labels to float32 before moving them to the device
            labels = labels.to(torch.float32).to(device)  # Ensure labels are float32 for MPS compatibility
            
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
        
        # Validation after each epoch
        validate_model(model, validation_loader, device)
        
        end_time = time.time()
        print(f'Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds')

# Step 5: Validation function to evaluate the model
def validate_model(model, validation_loader, device):
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    correct = 0
    total = 0
    
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    
    with torch.no_grad():  # No gradients needed for validation
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(torch.float32).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate validation loss
            val_loss = loss_fn(logits, labels)
            total_val_loss += val_loss.item()
            
            # Calculate validation accuracy (binary classification for toxic/non-toxic)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.numel()  # Total number of elements
    
    avg_val_loss = total_val_loss / len(validation_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Step 6: Save the model
def save_model(model, tokenizer, path='./toxic-classifier'):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# Main function to run the whole process (training and validation files provided)
def main():
    # Set device to MPS if available, otherwise CPU
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        # device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print("mps selected!")

    # Load training and validation data
    print("Loading data...")
    train_file = '../jigsaw-multilingual-toxic-comment-classification/new/jigsaw-unintended-bias-train-modified.csv'
    validation_file = '../jigsaw-multilingual-toxic-comment-classification/validation.csv'
    train_bias, validation_data = load_data(train_file, validation_file)
    
    # Load BERT model and tokenizer
    print("Loading BERT model...")
    model, tokenizer = load_model(num_labels=6)  # 6 labels for training categories
    
    # Preprocess training data
    print("Preprocessing training data...")
    train_encodings_bias, train_labels_bias = preprocess_data(train_bias, tokenizer)
    
    # Preprocess validation data (only 'toxic' label)
    print("Preprocessing validation data...")
    validation_encodings, validation_labels = preprocess_data(validation_data, tokenizer, is_validation=True)
    
    # Create data loaders (for both train and validation data)
    print("Creating data loaders...")
    train_loader_bias = create_dataloader(train_encodings_bias, train_labels_bias, batch_size=16)
    validation_loader = create_dataloader(validation_encodings, validation_labels, batch_size=16)
    
    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Train the model with validation
    print("Training model...")
    train_model(model, train_loader_bias, validation_loader, optimizer, device)
    
    # Save the model
    print("Saving model...")
    save_model(model, tokenizer)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
