import cupy as cp
from cupyx.scipy import ndimage
import tifffile
import os
from pathlib import Path


def process_tif(input_path, output_dir, sigma=1.0):
    """
    Process a TIF file by multiplying with 1e9 and applying Gaussian filter using CuPy.

    Args:
        input_path (str): Path to input TIF file
        output_dir (str): Directory to save processed file
        sigma (float): Standard deviation for Gaussian kernel
    """
    try:
        # Read the TIF file
        img = tifffile.imread(input_path)

        # Transfer to GPU
        img_gpu = cp.array(img, dtype=cp.float32)

        # Multiply by 1e9
        img_gpu *= 1e9

        # Apply Gaussian filter using cupyx.scipy.ndimage
        img_filtered = ndimage.gaussian_filter(img_gpu, sigma=sigma)

        # Transfer back to CPU
        img_processed = cp.asnumpy(img_filtered)

        # Create output filename
        input_filename = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{input_filename}_processed.tif")

        # Save processed image
        tifffile.imwrite(output_path, img_processed)

        print(f"Processed {input_path} -> {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")


def process_directory(input_dir, output_dir, pattern="*.tif", sigma=1.0):
    """
    Process all TIF files in a directory.

    Args:
        input_dir (str): Input directory containing TIF files
        output_dir (str): Output directory for processed files
        pattern (str): File pattern to match
        sigma (float): Standard deviation for Gaussian kernel
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all TIF files in the directory
    input_path = Path(input_dir)
    tif_files = list(input_path.glob(pattern))

    print(f"Found {len(tif_files)} TIF files to process")

    # Process each file
    for tif_file in tif_files:
        process_tif(str(tif_file), output_dir, sigma)


if __name__ == "__main__":
    # Example usage
    input_dir = "/data/mct/22203/results/K3_3H_Manual/umpa_window3/df_positive_stitched"
    output_dir = "/data/mct/22203/results/K3_3H_Manual/umpa_window3/df_positive_stitched_processed"

    # Process all TIF files in the directory
    process_directory(input_dir, output_dir, sigma=10.0)