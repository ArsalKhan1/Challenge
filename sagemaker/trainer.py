!pip install transformers datasets torch scikit-learn boto3 sagemaker


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)  # 6 labels for the output


# 5. Preprocessing the data
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load training data from your SageMaker notebook environment
train_df = pd.read_csv('./jigsaw-uninted-bias-train-modified.csv')

# 1. Check for and handle invalid entries in the 'toxic' column

# Convert the 'toxic' column to numeric, coercing errors
train_df['toxic'] = pd.to_numeric(train_df['toxic'], errors='coerce')

# Drop rows where conversion failed (i.e., NaN values in the 'toxic' column)
train_df.dropna(subset=['toxic'], inplace=True)

# Optional: You may also want to ensure other columns are correctly converted
# Example: Handling the 'severe_toxicity', 'obscene', etc. columns similarly
columns_to_check = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
for col in columns_to_check:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    train_df.dropna(subset=[col], inplace=True)

# 2. Remove any unnecessary columns if required (e.g., unnamed or empty columns)
train_df.dropna(how='all', inplace=True)

# Convert the training data into a dataset for easier tokenization
train_dataset = Dataset.from_pandas(train_df)

# 3. Tokenize the training data
def preprocess_function(examples):
    return tokenizer(examples['comment_text'], truncation=True, padding=True)

tokenized_train = train_dataset.map(preprocess_function, batched=True)

# Load and preprocess validation data similarly
val_df = pd.read_csv('./validation.csv')
val_df.dropna(how='any', inplace=True)  # Ensure there are no missing values in validation data

# Use only 'comment_text' and 'toxic' columns for validation
val_dataset = Dataset.from_pandas(val_df[['comment_text', 'toxic']])

# Tokenize the validation data
tokenized_val = val_dataset.map(preprocess_function, batched=True)




# 6. Training the model
# You will now fine-tune the model using the preprocessed 
# training data. Use Trainer from the transformers library 
# to handle the training process.
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    per_device_train_batch_size=16,  # batch size
    per_device_eval_batch_size=16,   # batch size
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# Train the model
trainer.train()


# Evaluating the model
# Once training is complete, evaluate the model on the validation dataset.

# Evaluate model on validation data
eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")

# Saving the model
model_save_path = './distilbert-finetuned-model'
trainer.save_model(model_save_path)



# 9. Inference
# For inference, 
# you can load the model and tokenizer, then classify new messages:
# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('s3://your-bucket/distilbert-finetuned-model')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize input text
inputs = tokenizer("Your input message here", return_tensors="pt", truncation=True, padding=True)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted class
predictions = torch.argmax(outputs.logits, dim=-1)
print(f"Prediction: {predictions}")

