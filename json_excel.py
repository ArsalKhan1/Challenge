import pandas as pd
import json

def json_to_vertical_excel(json_file, excel_file):
    # Load JSON data from file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create a list of key-value pairs to store in DataFrame
    if isinstance(data, list):
        # If JSON is a list of dictionaries, we will process each dictionary
        all_rows = []
        for entry in data:
            entry_pairs = list(entry.items())  # Get key-value pairs for each entry
            all_rows.extend(entry_pairs)
    elif isinstance(data, dict):
        # If it's a single dictionary, convert key-value pairs to rows
        all_rows = list(data.items())
    else:
        raise ValueError("Unsupported JSON format")

    # Convert the key-value pairs into a DataFrame with two columns
    df = pd.DataFrame(all_rows, columns=["Field Name", "Value"])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_file, index=False)

# Example usage
json_to_vertical_excel('your_json_file.json', 'output_field_names_values.xlsx')
