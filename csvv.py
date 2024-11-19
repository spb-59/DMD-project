import pandas as pd
import ast

# Read the CSV data into a dataframe
df = pd.read_csv('features/AFIB.csv')

# Function to replace spaces with commas and convert to list
def parse_array_string(array_string):
    try:
        # Replace spaces with commas and ensure proper formatting
        array_string = array_string.replace(' ', ',').replace('[,', '[').replace(',]', ']')
        # Add brackets if missing
        if not array_string.startswith('['):
            array_string = '[' + array_string
        if not array_string.endswith(']'):
            array_string = array_string + ']'
        # Convert to list
        return ast.literal_eval(array_string)
    except Exception as e:
        print(f"Error parsing array: {array_string}")
        print(f"Exception: {e}")
        return []

# Apply the function to the relevant columns
df['M_s'] = df['M_s'].apply(parse_array_string)
df['P_s'] = df['P_s'].apply(parse_array_string)
df['M_u'] = df['M_u'].apply(parse_array_string)
df['P_u'] = df['P_u'].apply(parse_array_string)

# Function to expand a column of lists into separate columns
def expand_column(df, column_name, new_names):
    df_expanded = pd.DataFrame(df[column_name].tolist(), columns=new_names)
    df = pd.concat([df.drop(columns=[column_name]), df_expanded], axis=1)
    return df

# Expand columns
df = expand_column(df, 'M_s', [f'M_s{i}' for i in range(len(df['M_s'][0]))])
df = expand_column(df, 'P_s', [f'P_s{i}' for i in range(len(df['P_s'][0]))])
df = expand_column(df, 'M_u', [f'M_u{i}' for i in range(len(df['M_u'][0]))])
df = expand_column(df, 'P_u', [f'P_u{i}' for i in range(len(df['P_u'][0]))])

# Save the modified dataframe to a new CSV file
df.to_csv('features/AFIB.csv', index=False)

print("Transformation complete. Check 'modified_data.csv' for results.")
