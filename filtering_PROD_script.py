import pandas as pd

# Load the Excel file
file_path = "your_excel_file.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Filter rows: Keep rows where result_manual == 0
filtered_df = df[df['result_manual'] == 0].copy()

# Convert 'masked_data' column to string values
filtered_df['masked_data'] = filtered_df['masked_data'].astype(str)

# Save the filtered and modified data back to an Excel file
output_file = "filtered_excel_data.xlsx"  # Specify your desired output file name
filtered_df.to_excel(output_file, index=False)

print(f"Filtered data saved to {output_file}")
