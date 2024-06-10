import os
import argparse
from PIL import Image
import numpy as np

def tile_image(input_image_path, output_dir, tile_size, variance_threshold):
    # Load the image
    image = Image.open(input_image_path).convert('L')  # Convert to grayscale to simplify variance calculation
    width, height = image.size

    # Calculate the number of tiles in each dimension
    x_tiles = width // tile_size
    y_tiles = height // tile_size

    # Create an output subdirectory based on the image name
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    image_output_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    # Create tiles
    for i in range(y_tiles):
        for j in range(x_tiles):
            # Define the bounding box for each tile
            left = j * tile_size
            upper = i * tile_size
            right = left + tile_size
            lower = upper + tile_size

            # Crop the image to create the tile
            tile = image.crop((left, upper, right, lower))

            # Convert tile to numpy array and calculate variance
            tile_array = np.array(tile)
            if np.var(tile_array) > variance_threshold:
                # Save the tile
                tile.save(os.path.join(image_output_dir, f"tile_{i}_{j}.jpg"))

def process_directory(input_dir, output_dir, tile_size, variance_threshold):
    # Iterate through all image files in the directory
    for entry in os.listdir(input_dir):
        file_path = os.path.join(input_dir, entry)
        if os.path.isfile(file_path) and entry.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            tile_image(file_path, output_dir, tile_size, variance_threshold)

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate tiles from images in a directory, dropping tiles with low variance.")
    parser.add_argument("input_directory", help="Directory containing images")
    parser.add_argument("output_directory", help="Directory where image tiles will be saved")
    parser.add_argument("--tile_size", type=int, default=64, help="Size of each tile (default is 256 pixels)")
    parser.add_argument("--variance_threshold", type=float, default=200.0, help="Variance threshold below which tiles will not be saved (default is 50.0)")

    # Parse arguments
    args = parser.parse_args()

    # Process the directory of images
    process_directory(args.input_directory, args.output_directory, args.tile_size, args.variance_threshold)

if __name__ == "__main__":
    main()
