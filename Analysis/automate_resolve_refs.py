import yaml
import pandas as pd

# Function to recursively resolve $ref references
def resolve_references(schema, components):
    if isinstance(schema, dict):
        if '$ref' in schema:
            # Resolve the reference
            ref_path = schema['$ref'].split('/')
            ref_schema = components
            for part in ref_path[1:]:  # Skip the initial '#' part
                ref_schema = ref_schema.get(part, {})
            return resolve_references(ref_schema, components)
        else:
            # Recursively resolve nested objects
            return {k: resolve_references(v, components) for k, v in schema.items()}
    elif isinstance(schema, list):
        # Resolve each item in a list
        return [resolve_references(item, components) for item in schema]
    else:
        # Return primitive types as-is
        return schema

# Load the .yaml file
with open('visa_oct_schema.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Extract components/schemas for reference resolution
components = data.get('components', {}).get('schemas', {})

# Extract and resolve the VisaOCTRequestWrapper schema
visa_oct_request_schema = data['components']['schemas']['VisaOCTRequestWrapper']
resolved_schema = resolve_references(visa_oct_request_schema, components)

# Convert the resolved schema properties to a DataFrame
resolved_schema_properties = resolved_schema.get('properties', {})
schema_df = pd.DataFrame.from_dict(resolved_schema_properties, orient='index').reset_index()
schema_df.columns = ['Property', 'Details']

# Load the Excel file for comparison
excel_df = pd.read_excel('excel_data.xlsx')

# Merge or compare the two DataFrames based on Property name
comparison_df = pd.merge(schema_df, excel_df, left_on='Property', right_on='Field Name', how='left')

# Display the comparison
import ace_tools as tools; tools.display_dataframe_to_user(name="Resolved Schema Comparison", dataframe=comparison_df)
