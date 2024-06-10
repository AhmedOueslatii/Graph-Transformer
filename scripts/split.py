import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from the .txt file
file_path = '../scripts/augmented_labels.txt'  # Make sure to provide the correct path
data = pd.read_csv(file_path, sep="\t", header=None, names=['filename', 'label'])

# Perform the train-test split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)  # 20% of the data is used for testing

# Save the training and testing sets to separate .txt files
train_set.to_csv('train_set.txt', sep='\t', index=False, header=False)
test_set.to_csv('test_set.txt', sep='\t', index=False, header=False)

# If you want to verify the contents of the files, you can load and print the first few rows
print("Training Set:")
print(train_set.head())
print("\nTesting Set:")
print(test_set.head())
