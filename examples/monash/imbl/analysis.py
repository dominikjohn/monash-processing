import os
import numpy as np
from skimage.measure import block_reduce
import tifffile
from pathlib import Path
import glob


# Define the folder path
folder_path = '/data/imbl/23081/output-during-beamtime/Day3/Dominik_KI_salts_0p75m_30keV_0p16s/recon_phase'

# Create the binned8 subfolder if it doesn't exist
output_folder = os.path.join(folder_path, 'binned16')
os.makedirs(output_folder, exist_ok=True)

print("Loading all TIFF files into memory...")

# Create a list to store all images
all_images = []

# Load all the TIFF files from index 463 to 1400
for idx in range(463, 1401):
    # Construct the filename
    filename = f"recon_cs01000_idx_{idx:04d}.tiff"
    file_path = os.path.join(folder_path, filename)

    # Check if file exists
    if os.path.exists(file_path):
        # Load the image
        print(f"Loading {filename}...")
        image = tifffile.imread(file_path)
        all_images.append(image)
    else:
        print(f"Warning: File {file_path} not found. Skipping.")

# Convert the list of images to a 3D numpy array
if all_images:
    print(f"Loaded {len(all_images)} TIFF files. Creating 3D volume...")
    volume = np.stack(all_images, axis=0)

    # Check the shape of the volume
    print(f"Volume shape before binning: {volume.shape}")

    # Apply 3D block_reduce with median
    print("Performing 3D binning...")
    binned_volume = block_reduce(volume, block_size=(16, 16, 16), func=np.median)

    print(f"Volume shape after binning: {binned_volume.shape}")

    # Save each slice of the binned volume
    print("Saving binned images...")
    for i, binned_slice in enumerate(binned_volume):
        # Calculate the original index this corresponds to
        original_idx = 463 + (i * 16)
        output_filename = f"recon_cs00885_idx_{original_idx:04d}_binned.tiff"
        output_path = os.path.join(output_folder, output_filename)

        # Save the binned image
        tifffile.imwrite(output_path, binned_slice)
        print(f"Saved binned slice to {output_path}")

    print("Processing complete!")