import os

# Function to generate augmented samples
def generate_augmented_samples(sample, num_augmentations=5):
    augmented_samples = [f"{sample}_aug_{i + 1}" for i in range(num_augmentations)]
    return augmented_samples

# Read the original text file
input_file = '../scripts/filename_label_list.txt'
output_file = '../scripts/filename_label_list_aug.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        sample, label = line.split('\t')
        augmented_samples = generate_augmented_samples(sample)
        for aug_sample in augmented_samples:
            outfile.write(f"{aug_sample}\t{label}\n")

print(f"Augmented annotations saved to {output_file}")
