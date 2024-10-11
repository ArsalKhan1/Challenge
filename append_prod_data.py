import pandas as pd

# Load data.csv
data_df = pd.read_csv('data.csv')

# Define function to format data according to the given rules
def format_data(row):
    comment_text = f'"{row["masked_data"]}"'  # Adding quotes around text
    toxic = 0
    severe_toxic = 0
    obscene = 0
    threat = 0
    insult = 0
    identity_hate = 0

    # Apply the rules to set the values
    if row['result'] == 1 and row['result_manual'] == 1:
        toxic = 1
        severe_toxic = 1
    elif row['result'] == 0 and row['result_manual'] == 1:
        toxic = 1
        severe_toxic = 1

    # Create a new row with the specified columns in training.csv
    return pd.Series([comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate])

# Apply the format_data function to each row in data.csv
formatted_df = data_df.apply(format_data, axis=1)

# Set the columns for formatted_df
formatted_df.columns = ['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load training.csv if it exists, otherwise create an empty DataFrame with the same columns
try:
    training_df = pd.read_csv('training.csv')
except FileNotFoundError:
    training_df = pd.DataFrame(columns=['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

# Append the formatted data to training.csv
training_df = pd.concat([training_df, formatted_df], ignore_index=True)

# Save the updated training.csv
training_df.to_csv('training.csv', index=False)

print("Data formatted and appended to training.csv successfully.")
