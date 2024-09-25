import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

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
    train_dataset_toxic = TensorDataset(torch.tensor(train_encodings_toxic['input_ids']), train_labels_toxic)
    train_dataset_bias = TensorDataset(torch.tensor(train_encodings_bias['input_ids']), train_labels_bias)
    validation_dataset = TensorDataset(torch.tensor(validation_encodings['input_ids']), validation_labels)
    
    train_loader_toxic = DataLoader(train_dataset_toxic, batch_size=batch_size, shuffle=True)
    train_loader_bias = DataLoader(train_dataset_bias, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    
    return train_loader_toxic, train_loader_bias, validation_loader

# Step 3: Load BERT model with multiple output labels
def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# only added fix 1, not fix 2 yet
# Step 4: Train the model
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.to(device)  # Move the model to the device (CPU or GPU)
    model.train()     # Set the model in training mode

    # Define the appropriate loss function for multi-label classification
    loss_fn = nn.BCEWithLogitsLoss()  # Use binary cross-entropy with logits for multi-label classification
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')  # Print the current epoch number
        total_loss = 0  # Initialize total loss for the epoch
        
        # Iterate over the batches in the DataLoader
        for batch in train_loader:
            optimizer.zero_grad()  # Clear any previous gradients
            
            input_ids, labels = batch  # Get input data (tokenized text) and labels
            input_ids = input_ids.to(device)  # Move inputs to the correct device (CPU/GPU)
            labels = labels.to(device).float()  # Move labels to the correct device (CPU/GPU) and convert to float
            
            # Forward pass through the model
            outputs = model(input_ids)
            logits = outputs.logits  # Get the logits from the model output

            # Calculate the loss using BCEWithLogitsLoss
            loss = loss_fn(logits, labels)  # Calculate multi-label binary cross-entropy loss
            total_loss += loss.item()  # Accumulate loss for reporting
            
            loss.backward()  # Backpropagate the loss (calculate gradients)
            optimizer.step()  # Update model parameters (weights)
        
        print(f"Loss after epoch {epoch+1}: {total_loss/len(train_loader)}")

# Step 5: Evaluate the model
def evaluate_model(model, validation_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids)
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
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
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
    train_loader_toxic, train_loader_bias, validation_loader = create_dataloaders(train_encodings_toxic, train_labels_toxic, train_encodings_bias, train_labels_bias, validation_encodings, validation_labels)
    
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
