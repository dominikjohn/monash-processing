import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm

class Binner:
    def __init__(self, input_path):
        """
        Initialize the Binner with a path to TIFF files.

        Args:
            input_path (str or Path): Path to directory containing TIFF files
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path {input_path} does not exist")

    def _get_tiff_files(self):
        """Get sorted list of TIFF files from input directory."""
        tiff_files = sorted(list(self.input_path.glob('*.tif*')))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {self.input_path}")
        return tiff_files

    def _bin_2d(self, image, factor):
        """
        Bin a 2D image by the specified factor.

        Args:
            image (np.ndarray): 2D input image
            factor (int): Binning factor

        Returns:
            np.ndarray: Binned image
        """
        if factor == 1:
            return image

        # Calculate new dimensions that are divisible by factor
        new_height = (image.shape[0] // factor) * factor
        new_width = (image.shape[1] // factor) * factor

        # Crop image to be divisible by factor
        cropped = image[:new_height, :new_width]

        # Reshape and mean
        shape = (new_height // factor, factor,
                 new_width // factor, factor)
        return cropped.reshape(shape).mean(axis=(1, 3))

    def process_stack(self, binning_factor, output_suffix=None):
        """
        Load, bin, and save the TIFF stack.

        Args:
            binning_factor (int): Factor by which to bin the images
            output_suffix (str, optional): Custom suffix for output files.
                         If None, uses 'binned{binning_factor}'

        Returns:
            Path: Path to the output directory
        """
        if not isinstance(binning_factor, int) or binning_factor < 1:
            raise ValueError("Binning factor must be a positive integer")

        if binning_factor == 1:
            print("Binning factor is 1, no processing needed")
            return self.input_path

        # Setup output path and suffix
        if output_suffix is None:
            output_suffix = f'binned{binning_factor}'

        output_dir = self.input_path / output_suffix
        output_dir.mkdir(exist_ok=True)

        # Process files
        tiff_files = self._get_tiff_files()
        print(f"\nProcessing {len(tiff_files)} files with binning factor {binning_factor}...")

        for tiff_file in tqdm(tiff_files):
            # Load image
            img = tifffile.imread(tiff_file)

            # Bin image
            binned_img = self._bin_2d(img, binning_factor)

            # Save binned image
            output_path = output_dir / f"{tiff_file.stem}_{output_suffix}{tiff_file.suffix}"
            tifffile.imwrite(output_path, binned_img.astype(img.dtype))

        print(f"\nProcessed files saved to {output_dir}")
        return output_dir

    def process_multiple_factors(self, factors):
        """
        Process the stack with multiple binning factors.

        Args:
            factors (list of int): List of binning factors to apply

        Returns:
            dict: Dictionary mapping factors to output directories
        """
        results = {}
        for factor in factors:
            print(f"\nProcessing binning factor: {factor}")
            output_dir = self.process_stack(factor)
            results[factor] = output_dir
        return results