import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import time

# Step 1: Load and Preprocess Data (train and validation)
def load_data(train_file, validation_file, nrows=None):
    # Load only 10,000 rows from the training data
    train_bias = pd.read_csv(train_file, nrows=nrows)  # Add nrows parameter to limit rows
    validation_data = pd.read_csv(validation_file)
    
    return train_bias, validation_data

def preprocess_data(data, tokenizer, max_len=128, is_validation=False):
    data['comment_text'] = data['comment_text'].fillna('').astype(str)

    encodings = tokenizer(list(data['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    if is_validation:
        labels = torch.tensor(data['toxic'].values)
    else:
        labels = torch.tensor(data[['toxic', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values)
    
    return encodings, labels

# Step 2: Create DataLoader objects
def create_dataloader(encodings, labels, batch_size=16):
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Load DistilBERT model with multiple output labels
def load_model(num_labels):
    # Use DistilBERT instead of BERT
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

# Step 4: Train the model
def train_model(model, train_loader, validation_loader, optimizer, device, epochs=3):
    model.to(device)
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        start_time_epoch = time.time()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            iteration_start_time = time.time()
            
            optimizer.zero_grad()
            
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(torch.float32).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                    print(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item()}')
                    
        total_time_for_epoch = time.time() - start_time_epoch
        it_per_sec = len(train_loader) / total_time_for_epoch
        print(f'Epoch {epoch+1}/{epochs} completed in {total_time_for_epoch:.2f} seconds')
        print(f"Iterations per second (IT/s): {it_per_sec:.2f}")
        
        validate_model(model, validation_loader, device)

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
            
            # Select only the first logit (toxic) for validation since the validation data only has the 'toxic' label
            logits_toxic = logits[:, 0]  # First logit corresponds to 'toxic'
            
            # Calculate validation loss (for toxic only)
            val_loss = loss_fn(logits_toxic, labels)
            total_val_loss += val_loss.item()
            
            # Calculate validation accuracy (binary classification for toxic/non-toxic)
            predictions = (torch.sigmoid(logits_toxic) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.numel()  # Total number of elements
    
    avg_val_loss = total_val_loss / len(validation_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Step 6: Save the model
def save_model(model, tokenizer, path='./toxic-classifier'):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# Main function
def main():
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

    train_file = '../jigsaw-multilingual-toxic-comment-classification/new/jigsaw-unintended-bias-train-modified.csv'
    validation_file = '../jigsaw-multilingual-toxic-comment-classification/validation.csv'
    
    # Load data with only 10,000 rows
    train_bias, validation_data = load_data(train_file, validation_file, nrows=10000)

    print("Loading DistilBERT model...")
    model, tokenizer = load_model(num_labels=6)

    print("Preprocessing training data...")
    train_encodings_bias, train_labels_bias = preprocess_data(train_bias, tokenizer)
    
    print("Preprocessing validation data...")
    validation_encodings, validation_labels = preprocess_data(validation_data, tokenizer, is_validation=True)
    
    train_loader_bias = create_dataloader(train_encodings_bias, train_labels_bias, batch_size=16)
    validation_loader = create_dataloader(validation_encodings, validation_labels, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("Training model...")
    train_model(model, train_loader_bias, validation_loader, optimizer, device)

if __name__ == "__main__":
    main()