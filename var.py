import os
import numpy as np
from PIL import Image

def calculate_variance(image_path):
    """Calculate and return the variance of the image at the given path."""
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_array = np.array(image)
        return np.var(image_array)
    except IOError:
        print(f"Error opening or reading image file {image_path}")
        return None

def process_directory(input_dir):
    """Process all image files in the given directory and print their variance."""
    # Check if the directory exists
    if not os.path.exists(input_dir):
        print(f"The directory {input_dir} does not exist.")
        return

    # Iterate through all files in the directory
    for entry in os.listdir(input_dir):
        file_path = os.path.join(input_dir, entry)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            variance = calculate_variance(file_path)
            if variance is not None:
                print(f"Variance of {entry}: {variance}")

def main():
    import argparse
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate and print the variance of each image in a directory.")
    parser.add_argument("input_directory", help="Directory containing images")

    # Parse arguments
    args = parser.parse_args()

    # Process the directory of images
    process_directory(args.input_directory)

if __name__ == "__main__":
    main()
