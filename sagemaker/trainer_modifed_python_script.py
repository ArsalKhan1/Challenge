import argparse
import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Step 1: Load and Preprocess Data from S3
def load_data(train_file, validation_file):
    # Load the training and validation data from S3 or local storage
    train_data = pd.read_csv(train_file)
    validation_data = pd.read_csv(validation_file)
    
    return train_data, validation_data

def preprocess_data(data, tokenizer, max_len=128, is_validation=False):
    data['comment_text'] = data['comment_text'].fillna('').astype(str)

    encodings = tokenizer(list(data['comment_text']), truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    if is_validation:
        labels = torch.tensor(data['toxic'].values)
    else:
        for col in ['toxic', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
        
        labels = torch.tensor(data[['toxic', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values)
    
    return encodings, labels

def create_dataloader(encodings, labels, batch_size=16):
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_model(num_labels):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

def train_model(model, train_loader, validation_loader, optimizer, device, epochs=3):
    model.to(device)
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
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
            
        print(f'Epoch {epoch+1}/{epochs} completed with total loss: {total_loss}')
        validate_model(model, validation_loader, device)

def validate_model(model, validation_loader, device):
    model.eval()
    total_val_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(torch.float32).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            val_loss = loss_fn(logits[:, 0], labels)  # Assuming binary classification for validation
            total_val_loss += val_loss.item()

    print(f"Validation Loss: {total_val_loss}")

def save_model(model, tokenizer, model_dir):
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--train-file', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation-file', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    
    args = parser.parse_args()

    # Load data from S3
    print("Loading data from S3...")
    train_data, validation_data = load_data(args.train_file, args.validation_file)

    # Load model and tokenizer
    print("Loading DistilBERT model...")
    model, tokenizer = load_model(num_labels=6)

    # Preprocess data
    print("Preprocessing data...")
    train_encodings, train_labels = preprocess_data(train_data, tokenizer)
    validation_encodings, validation_labels = preprocess_data(validation_data, tokenizer, is_validation=True)

    # Create DataLoader objects
    train_loader = create_dataloader(train_encodings, train_labels, batch_size=args.batch_size)
    validation_loader = create_dataloader(validation_encodings, validation_labels, batch_size=args.batch_size)

    # Set up training
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    print("Training the model...")
    train_model(model, train_loader, validation_loader, optimizer, device, epochs=args.epochs)

    # Save the model to the output directory
    print(f"Saving model to {args.output_dir}...")
    save_model(model, tokenizer, args.output_dir)
