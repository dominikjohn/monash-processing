import os
import numpy as np
import tifffile
from tqdm import tqdm
# Path to your projections
base_path = '/asap3/petra3/gpfs/p07/2024/data/11020408/processed/016_basel5_a_stitched_dpc/corrected_phi'

# Loop through all projections
for i in tqdm(range(4501)):
    filename = f'projection_{i:04d}.tif'
    filepath = os.path.join(base_path, filename)

    try:
        # Read the image
        img = tifffile.imread(filepath)

        # Check for NaNs
        if np.isnan(img).any():
            # Delete file if it contains NaN values
            os.remove(filepath)
            print(f"Removed {filename} - contains NaN values")

    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")