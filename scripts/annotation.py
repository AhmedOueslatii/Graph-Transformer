import pandas as pd

# Load the Excel file
file_path = "MasterlistAug30-2017.xlsx"
data = pd.read_excel(file_path)

data_clean = data.loc[1:].copy()

# Rename the column for convenience
data_clean.rename(columns={'File Name': 'filename', 'Outcome': 'label'}, inplace=True)

# Create the text file content
data_clean['formatted'] = data_clean['filename'].astype(str) + "\t" + data_clean['label'].astype(str)

txt_file_path = '../data/filename_label_list.txt'
with open(txt_file_path, 'w') as file:
    file.write("\n".join(data_clean['formatted']))