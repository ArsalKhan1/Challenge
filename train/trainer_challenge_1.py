import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn.functional as F

# Configuration
device = torch.device("mps") if torch.backends.mps.is_available() else print("mps not found")
model_name = "bert-base-uncased"
batch_size = 16
epochs = 3
learning_rate = 2e-5
max_length = 128

# Load data
data = pd.read_csv("../jigsaw-toxic-comment-classification-challenge/challenge_1_train.csv")
data = data.head(100)

# Prepare Dataset class
class ToxicCommentsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = dataframe['comment_text'].values
        self.labels = dataframe[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        labels = torch.tensor(self.labels[index], dtype=torch.float)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained(model_name)
train_texts, val_texts, train_labels, val_labels = train_test_split(data, data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=42)

train_dataset = ToxicCommentsDataset(train_texts, tokenizer, max_length)
val_dataset = ToxicCommentsDataset(val_texts, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds.append(logits.sigmoid().cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    # Flatten results
    preds = [item for sublist in preds for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    return total_loss / len(data_loader), preds, true_labels

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f'Training loss: {train_loss:.4f}')
    
    val_loss, preds, true_labels = eval_model(model, val_loader, device)
    print(f'Validation loss: {val_loss:.4f}')
    
    # Calculate metrics
    preds_binary = (torch.tensor(preds) > 0.5).int()
    true_labels_binary = torch.tensor(true_labels).int()
    accuracy = accuracy_score(true_labels_binary, preds_binary)
    f1 = f1_score(true_labels_binary, preds_binary, average='macro')
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation F1 Score: {f1:.4f}')
    print(classification_report(true_labels_binary, preds_binary, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))

# Save the model
model.save_pretrained("toxic-comment-model")
tokenizer.save_pretrained("toxic-comment-model")

print("Model and tokenizer saved to 'toxic-comment-model'.")
