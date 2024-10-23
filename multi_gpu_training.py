# Install necessary packages
!pip install torch datasets

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
from datetime import datetime
from tqdm import tqdm

# -----
CONFIG = {
    "model_type": "bert",  
    "max_len": 128,
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "data_file": "data/jigsaw-toxic-comment-train.csv",
    "output_model_dir": "trained_models/jigsaw-toxic-comment-bert",
    "num_workers": 2,
    "pin_memory": True,
    "device": "cuda"
}

# Initialize Distributed Process Group
dist.init_process_group(backend='nccl')

# Dataset class remains the same
class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# -----
# Function to load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The DataFrame is empty. Please check the CSV file content.")
        print(data.shape)
        data['label'] = data[['toxic', 'severe_toxic', 'obscene', 'identity_hate', 'insult', 'threat']].values.tolist()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['comment_text', 'label'])

# Function to get the model and tokenizer
def get_model_and_tokenizer(model_type):
    if model_type == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
    elif model_type == "albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=6)
    else:  # Default to BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    return model, tokenizer

# -----
# Check device
device = torch.device(CONFIG['device'])

# Load data
print("Loading Data...")
data = load_data(CONFIG['data_file'])
print("Splitting into train & val datasets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['comment_text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Get model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Move model to device and wrap with DistributedDataParallel
model.to(device)
model = DDP(model, device_ids=[torch.cuda.current_device()])

# -----
# Create datasets and loaders with DistributedSampler
train_dataset = ToxicCommentDataset(train_texts, train_labels, tokenizer, max_len=CONFIG['max_len'])
val_dataset = ToxicCommentDataset(val_texts, val_labels, tokenizer, max_len=CONFIG['max_len'])

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                          num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'], sampler=train_sampler)

val_sampler = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], num_workers=CONFIG['num_workers'],
                        pin_memory=CONFIG['pin_memory'], sampler=val_sampler)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
scaler = GradScaler()

# -----
# Training loop
print("start_time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
start_time = time.time()

for epoch in range(CONFIG['num_epochs']):
    print(f"Epoch {epoch + 1}")
    model.train()
    train_loss = 0
    train_sampler.set_epoch(epoch)  # Ensure data is shuffled differently in each epoch

    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Training Loss: {avg_train_loss:.4f}")

    # Validation code remains the same...

# -----
print("Finished")

end_time = time.time()
print("end_time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
time_taken = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"\nTotal time: {time_taken // (1000*60*60*24):d}d {(time_taken // (1000*60*60)) % 24:.0f}h "
      f"{(time_taken // (1000*60)) % 60:.0f}m {(time_taken // 1000) % 60:.0f}s")

# -----
# Save the model (only by the main process)
if dist.get_rank() == 0:
    if not os.path.exists(CONFIG['output_model_dir']):
        os.makedirs(CONFIG['output_model_dir'])
    model.module.save_pretrained(CONFIG['output_model_dir'])  # Save only the base model
    tokenizer.save_pretrained(CONFIG['output_model_dir'])

    print("Model saved to disk.")
