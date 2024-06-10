import pandas as pd

# Load the Excel file
file_path = '/feature_extractor/MasterlistAug30-2017.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataframe to understand its structure
df.head()

# Define the classification rules
def classify_embryo(expansion, icm, te):
    if expansion < 3:
        return 1  # fair
    elif expansion >= 3 and (icm in ['A', 'B', 1, 2] or te in ['A', 'B', 1, 2]):
        return 2  # good
    elif expansion >= 3 and (icm == 'C' or te == 'C' or icm == 3 or te == 3):
        return 1  # fair
    elif expansion >= 3 and (icm == 'C' and te == 'C' or icm == 3 and te == 3):
        return 0  # poor
    else:
        return None

# Apply the classification rules to the dataframe
df['annotation'] = df.apply(lambda row: classify_embryo(row['EXP'], row['ICM'], row['TE']), axis=1)

# Filter rows where annotation is not None
df_filtered = df[df['annotation'].notna()]

# Create the text file content
file_content = ""
for index, row in df_filtered.iterrows():
    file_content += f"{row['filename']}\t{row['annotation']}\n"

# Save to a text file
output_file_path = ''
with open(output_file_path, 'w') as file:
    file.write(file_content)