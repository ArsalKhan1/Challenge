# Install necessary packages
!pip install torch datasets

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np
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
    "device": "cuda",
    "shared_file": "/tmp/sharedfile"  # Shared file for rendezvous
}

# Initialize Distributed Process Group with file-based rendezvous
if 'WORLD_SIZE' in os.environ:
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{CONFIG["shared_file"]}',
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )
else:
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{CONFIG["shared_file"]}',
        world_size=torch.cuda.device_count(),
        rank=0  # Assuming the notebook is running the main process
    )

# -----
class ToxicCommentDataset(torch.utils.data.Dataset):
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
# Check device
device = torch.device(CONFIG['device'])

# Load dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The DataFrame is empty. Please check the CSV file content")
        print(data.shape)
        data['label'] = data[['toxic', 'severe_toxic', 'obscene', 'identity_hate', 'insult', 'threat']].values.tolist()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['comment_text', 'label'])

print("Loading Data...")
data = load_data(CONFIG['data_file'])
print("Splitting into train & val datasets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['comment_text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# -----
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

# Define optimizer and GradScaler
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
scaler = GradScaler()

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

# Save the model (only by the main process)
if dist.get_rank() == 0:
    if not os.path.exists(CONFIG['output_model_dir']):
        os.makedirs(CONFIG['output_model_dir'])
    model.module.save_pretrained(CONFIG['output_model_dir'])
    tokenizer.save_pretrained(CONFIG['output_model_dir'])

    print("Model saved to disk.")
