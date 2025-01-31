import cupy as cp
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import tifffile

def create_paganin_filter(shape, pixel_size, dist, wavelength, delta_beta_ratio):
    """Precompute the Paganin filter on GPU."""
    ny, nx = shape
    y, x = cp.ogrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
    y = cp.fft.fftshift(y)
    x = cp.fft.fftshift(x)

    kx = 2 * cp.pi * x / (nx * pixel_size)
    ky = 2 * cp.pi * y / (ny * pixel_size)
    k = cp.sqrt(kx ** 2 + ky ** 2)

    return 1 / (1 + wavelength * dist * delta_beta_ratio * k ** 2)

def process_batch(batch, filter_gpu):
    """Process a batch of images on GPU."""
    batch_gpu = cp.asarray(batch)
    batch_fft = cp.fft.fft2(batch_gpu)
    filtered_fft = batch_fft * filter_gpu
    result = cp.real(cp.fft.ifft2(filtered_fft))
    return cp.asnumpy(result)

def batch_paganin_filter(image_paths, output_dir, pixel_size, dist, wavelength,
                         delta_beta_ratio, batch_size=4):
    """
    Process multiple images with Paganin filter in batches.

    Parameters:
    -----------
    image_paths : list
        List of paths to input images
    output_dir : str or Path
        Directory to save processed images
    pixel_size : float
        Detector pixel size in meters
    dist : float
        Sample-to-detector distance in meters
    wavelength : float
        X-ray wavelength in meters
    delta_beta_ratio : float
        Ratio of refractive index decrement to absorption index
    batch_size : int
        Number of images to process simultaneously
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load first image to get dimensions and precompute filter
    test_img = np.array(tifffile.imread(image_paths[0]))  # or use appropriate loading function
    filter_gpu = create_paganin_filter(test_img.shape, pixel_size, dist,
                                       wavelength, delta_beta_ratio)

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]

        # Load batch
        batch = np.stack([np.array(tifffile.imread(path)) for path in batch_paths])

        # Process batch
        processed_batch = process_batch(batch, filter_gpu)

        # Save results
        for j, path in enumerate(batch_paths):
            output_path = output_dir / f"processed_{Path(path).name}.tif"
            tifffile.imwrite(
                output_path,
                processed_batch[j],
                compression=None,  # No compression for compatibility
                planarconfig='contig'  # Standard configuration
            )

        # Explicit cleanup after each batch
        del batch
        cp.get_default_memory_pool().free_all_blocks()

image_paths = list(Path('/data/mct/22203/results/P6_Manual/recon_att').glob('*.tif*'))
output_dir = Path('/data/mct/22203/results/P6_Manual/recon_att_paganin_ptfe-ethanol')
output_dir.mkdir(exist_ok=True)

energy = 25000
energy_keV = energy / 1000
wavelength = 1.24e-9 / energy_keV
#delta_beta_ratio = 147.79
delta_beta_ratio = 1079.29

# Set parameters
params = {
    'pixel_size': 1.444e-6,
    'dist': 0.155,
    'wavelength': wavelength,
    'delta_beta_ratio': delta_beta_ratio,
    'batch_size': 16
}

batch_paganin_filter(image_paths, output_dir, **params)