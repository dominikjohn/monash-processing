import ray
import gc
from tqdm import tqdm
import os
from pathlib import Path
import time
from monash_processing.core.data_loader_desy import DataLoaderDesy
import scipy
import numpy as np
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask
from monash_processing.algorithms.phase_integration import PhaseIntegrator
import scipy.fft as fft
# Initialize Ray with memory management configuration
ray.init(object_store_memory=150 * 1024 * 1024 * 1024)  # Set internal memory limit

# Configuration remains the same
scan_base = '/asap3/petra3/gpfs/p07/2024/data/11020408/'
stitched_name = "processed/016_basel5_a_stitched_dpc/"
pixel_size = 1.28e-6
energy = 40.555
prop_distance = 0.3999
sigma = 0.035

# Modify the ProjectionProcessor to include resource requirements
@ray.remote
class ProjectionProcessor:
    def __init__(self, scan_base, stitched_name, pixel_size, energy, prop_distance, sigma):
        self.loader = DataLoaderDesy(scan_base, stitched_name)
        self.pixel_size = pixel_size
        self.energy = energy
        self.prop_distance = prop_distance
        self.sigma = sigma
        self.wavevec = 2 * np.pi * energy / (
                scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
        self.delta_mu = 3600
        self.conversion_factor = (self.wavevec / self.prop_distance) * (self.pixel_size ** 2)

    def process_projection(self, projection_i):
        try:
            result = self._process_projection_internal(projection_i)
            gc.collect()  # Force garbage collection
            return projection_i, True
        except Exception as e:
            return projection_i, False, str(e)

    def _process_projection_internal(self, projection_i):
        # Load data
        T = self.loader.load_processed_projection(projection_i, 'T_stitched', format='tif', simple_format=True)
        dx = self.loader.load_processed_projection(projection_i, 'dx_stitched', simple_format=True)
        dy = self.loader.load_processed_projection(projection_i, 'dy_stitched', simple_format=True)

        # Process bad pixels
        dx = BadPixelMask.correct_bad_pixels(dx)[0]
        dy = BadPixelMask.correct_bad_pixels(dy)[0]

        # Mirror images
        mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
        mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

        # Clean up intermediate results
        del dx, dy
        gc.collect()

        # Calculate frequency grids
        k = fft.fftfreq(mdx.shape[1])
        l = fft.fftfreq(mdy.shape[0])
        k[k == 0] = 1e-10
        l[l == 0] = 1e-10
        k, l = np.meshgrid(k, l)

        # Calculate phi_umpa
        ft = fft.fft2(mdx + 1j * mdy, workers=2)
        phi_raw = fft.ifft2(ft / ((2 * np.pi * 1j) * (k + 1j * l)), workers=2)
        phi_umpa = np.real(phi_raw)[mdx.shape[0] // 2:, mdy.shape[1] // 2:] * self.conversion_factor

        # Clean up large intermediates
        del mdx, mdy, ft, phi_raw
        gc.collect()

        if np.any(T <= 0):
            print(f"Warning: Projection {projection_i} contains invalid transmission values <= 0")
            T = np.clip(T, 1e-10, None)  # Clip to small positive values

        # Calculate phi_coarse
        phi_coarse = -np.log(T) * self.wavevec * self.delta_mu * self.pixel_size ** 2

        # Create and apply mask
        mask = np.zeros_like(phi_umpa, dtype=bool)
        mask[:, 0:2000] = True
        mask[:, -2000:-5] = True

        # Corrections
        phi_umpa_cor = phi_umpa - PhaseIntegrator.img_poly_fit(phi_umpa, order=1, mask=mask)
        phi_coarse_cor = phi_coarse - np.average(phi_coarse[mask])

        # Clean up intermediates
        del phi_umpa, phi_coarse, mask
        gc.collect()

        # Mirror corrections
        m_umpa = PhaseIntegrator.sym_mirror_im(phi_umpa_cor, 'reflect')
        m_coarse = PhaseIntegrator.sym_mirror_im(phi_coarse_cor, 'reflect')

        # Calculate Fourier transforms
        phi_umpa_ft = fft.fft2(m_umpa)
        phi_coarse_ft = fft.fft2(m_coarse)

        # Calculate weighted phi
        weighted_phi_ft = phi_umpa_ft * (1 - np.exp(-np.sqrt(k ** 2 + l ** 2) / (2 * self.sigma ** 2))) + \
                          phi_coarse_ft * np.exp(-np.sqrt(k ** 2 + l ** 2) / (2 * self.sigma ** 2))

        # Clean up more intermediates
        del phi_umpa_ft, phi_coarse_ft, m_umpa, m_coarse
        gc.collect()

        final_phi = np.real(fft.ifft2(weighted_phi_ft))[phi_umpa_cor.shape[0]:, phi_umpa_cor.shape[1]:]

        # Save results
        self.loader.save_tiff('corrected_phi', projection_i, final_phi)
        self.loader.save_tiff('original_phi', projection_i, phi_umpa_cor)

        # Final cleanup
        del final_phi, weighted_phi_ft, phi_umpa_cor, phi_coarse_cor
        gc.collect()

        return True


# Reduce number of parallel processors
num_processors = 15  # Reduced from 50 to 15

# Create processor actors
processors = [ProjectionProcessor.remote(
    scan_base, stitched_name, pixel_size, energy, prop_distance, sigma
) for _ in range(num_processors)]

# Process in smaller batches
total_projections = 4501
batch_size = 50  # Reduced batch size


def process_batch(start_idx, end_idx):
    futures = []
    for i in range(start_idx, min(end_idx, total_projections)):
        processor_idx = i % num_processors
        futures.append(processors[processor_idx].process_projection.remote(i))

    failed_projections = []
    with tqdm(total=len(futures), desc=f"Batch {start_idx}-{end_idx}") as pbar:
        while futures:
            try:
                done_id, futures = ray.wait(futures, timeout=120.0)
                if done_id:
                    for done in done_id:
                        try:
                            result = ray.get(done)
                            if isinstance(result, tuple):
                                proj_idx, success, *error = result
                                if not success:
                                    failed_projections.append((proj_idx, error[0] if error else "Unknown error"))
                        except Exception as e:
                            print(f"Error processing result: {str(e)}")
                        pbar.update(1)
                else:
                    print("Waiting for results...")
            except Exception as e:
                print(f"Batch processing error: {str(e)}")

    return failed_projections


def get_remaining_projections(scan_base, stitched_name, total_projections):
    # Path to check for completed files
    output_path = Path(scan_base) / stitched_name / 'corrected_phi'

    # List to store indices that need processing
    remaining = []

    # Check each projection
    for i in range(total_projections):
        output_file = output_path / f'projection_{i:04d}.tif'

        # Add to remaining list if output doesn't exist
        if not output_file.exists():
            remaining.append(i)

    print(f"Found {len(remaining)} projections that need processing out of {total_projections}")
    return remaining

# Add this right before creating the processors:
remaining_projections = get_remaining_projections(scan_base, stitched_name, total_projections)
if not remaining_projections:
    print("All projections have been processed. Nothing to do.")
    ray.shutdown()
    exit()

# Process all projections in batches with error handling
all_failed_projections = []
try:
    for batch_start in range(0, len(remaining_projections), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_projections))
        current_batch = remaining_projections[batch_start:batch_end]
        print(f"\nProcessing batch {batch_start}-{batch_end} of remaining projections")

        try:
            # Modify process_batch to take a list of indices
            failed = process_batch(current_batch[0], current_batch[-1] + 1)
            all_failed_projections.extend(failed)

            gc.collect()
        except Exception as e:
            print(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")

        time.sleep(2)
finally:
    # Print summary of failures
    if all_failed_projections:
        print("\nFailed projections:")
        for proj_idx, error in all_failed_projections:
            print(f"Projection {proj_idx}: {error}")

    # Shutdown Ray
    ray.shutdown()