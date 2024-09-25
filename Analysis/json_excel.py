import pandas as pd
import json

def flatten_json(y, parent_key='', sep='.'):
    """
    Recursively flattens a nested JSON dictionary.
    :param y: JSON object (dictionary or list)
    :param parent_key: key for the parent node
    :param sep: separator between levels
    :return: a flat dictionary with dotted paths as keys
    """
    items = []
    if isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    elif isinstance(y, list):
        for i, v in enumerate(y):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, y))
    return dict(items)

def json_to_vertical_excel(json_file, excel_file):
    # Load JSON data from file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Flatten the JSON structure to make each nested field a fully qualified path
    if isinstance(data, list):
        # If it's a list, we flatten each element individually
        all_rows = []
        for entry in data:
            flattened_entry = flatten_json(entry)
            all_rows.extend(flattened_entry.items())  # Add the flattened key-value pairs to the list
    elif isinstance(data, dict):
        # If it's a single dictionary, just flatten it directly
        all_rows = list(flatten_json(data).items())
    else:
        raise ValueError("Unsupported JSON format")

    # Convert the flattened key-value pairs into a DataFrame with two columns
    df = pd.DataFrame(all_rows, columns=["Field Name", "Value"])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_file, index=False)

# Example usage
json_to_vertical_excel('your_json_file.json', 'output_flattened.xlsx')
