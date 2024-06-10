import os

# Path to the directory containing folders
folder_path = 'data/dataset'

# Path to the text file
text_file_path = 'filtered_text_file.txt'

# Read folder names into a set

folders_set = set(os.listdir(folder_path))

# Read the text file and check for names that don't exist in the folder
folders_in_file_set = set()
with open(text_file_path, 'r') as file:
    for line in file:
        folder_name = line.strip().split('\t')[0]
        folders_in_file_set.add(folder_name)

# Print the names of the folders that are in the folder but not in the file
print("Folders in the folder but not in the file:")
for folder_name in folders_set:
    if folder_name not in folders_in_file_set:
        print(folder_name)
