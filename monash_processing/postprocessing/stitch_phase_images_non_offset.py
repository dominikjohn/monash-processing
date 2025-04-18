import numpy as np
from typing import Optional, Tuple
from dask.distributed import Client, LocalCluster
import dask.bag as db
from scipy.ndimage import shift
from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher

# This class is used to stitch projections that are not acquired using offset scans
class ProjectionStitcherNonOffset(ProjectionStitcher):

    def __init__(self, data_loader_left, data_loader_right, data_loader_stitched, angle_spacing, center_shift, slices=None, suffix=None, format='tif'):
        self.data_loader_left = data_loader_left
        self.data_loader_right = data_loader_right
        self.data_loader_stitched = data_loader_stitched
        self.offset = 2 * center_shift
        self.angle_spacing = angle_spacing
        self.ceil_offset = np.ceil(self.offset).astype(int)
        self.residual = self.ceil_offset - self.offset
        self.slices = slices
        self.suffix = suffix
        self.format = format

        if center_shift < 0:
            raise ValueError("Negative center shift not supported")

    def load_projection_pair(self, idx: int, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        proj_left = self.data_loader_left.load_processed_projection(idx, channel, format=self.format)
        proj_right = self.data_loader_right.load_processed_projection(idx, channel, format=self.format)

        if self.slices is not None:
            proj_left = proj_left[self.slices[0]:self.slices[1], :]
            proj_right = proj_right[self.slices[0]:self.slices[1], :]

        return proj_left, proj_right

    def normalize_projection(self, projection, part, channel):
        if part == 'right':
            mean = np.mean(projection[:, -800:-5])
        elif part == 'left':
            mean = np.mean(projection[:, 5:800])
        else:
            raise ValueError("Invalid part")

        return projection - mean

    def stitch_projection_pair(self, proj_index: int, channel: str) -> Tuple[np.ndarray, dict]:

        proj_left, proj_right = self.load_projection_pair(proj_index, channel)
        if channel != 'T':
            proj_left = self.normalize_projection(proj_left, 'right', channel)
            proj_right = self.normalize_projection(proj_right, 'left', channel)

        y_shape = proj_left.shape[0]
        single_x_shape = proj_left.shape[1]

        full_width = 2 * single_x_shape - self.ceil_offset

        p1 = np.full((y_shape, full_width), np.nan)
        p2 = np.full((y_shape, full_width), np.nan)

        p1[:, :single_x_shape] = proj_left
        p2[:, single_x_shape-self.ceil_offset:] = proj_right
        #p1 = shift(p1, shift=[0, -self.residual], mode='reflect', order=1)

        # Find overlap region
        overlap = ~(np.isnan(p1) | np.isnan(p2))

        # Initialize composite with p1 values
        composite = p1.copy()

        # In overlap region, take the average
        composite[overlap] = (p1[overlap] + p2[overlap]) / 2

        # Fill remaining areas from p2
        composite = np.where(np.isnan(composite), p2, composite)

        return composite

    def process_and_save_range(self, start_idx: int, end_idx: int, channel: str) -> list:
        """
        Process and save a range of projection pairs using Dask for parallel processing.

        Args:
            start_idx: Starting projection index
            end_idx: Ending projection index (inclusive)
            channel: Name of the input channel directory

        Returns:
            list: List of stitching statistics for each processed pair
        """
        # Set up local Dask cluster with simplified config
        cluster = LocalCluster(
            n_workers=50,
            threads_per_worker=1,
            memory_limit='4GB',
            lifetime_stagger='2 seconds',
            lifetime_restart=True,
        )
        client = Client(cluster)

        try:
            # Create processing function that returns None if file exists
            def process_single_projection(idx):
                # Check if output already exists
                if self.suffix is not None:
                    folder_name = f'{channel}_stitched_{self.suffix}'
                else:
                    folder_name = f'{channel}_stitched'

                output_path = (self.data_loader_stitched.results_dir /
                               folder_name /
                               f'projection_{idx:04d}.tif')

                if output_path.exists():
                    return None

                try:
                    stitched = self.stitch_projection_pair(idx, channel)
                    if self.suffix is not None:
                        save_channel = channel + f'_stitched_{self.suffix}'
                    else:
                        save_channel = channel + '_stitched'

                    self.data_loader_stitched.save_tiff(
                        channel=save_channel,
                        angle_i=idx,
                        data=stitched
                    )
                except Exception as e:
                    print(f"Failed to process {channel} projection {idx}: {str(e)}")
                    raise

            # Create dask bag of indices and process
            indices = list(range(start_idx, end_idx + 1))
            bag = db.from_sequence(indices, npartitions=50)

            # Process and collect results
            futures = bag.map(process_single_projection).compute()

            # Filter out None results (already processed files)
            stats_list = [s for s in futures if s is not None]

            return stats_list

        finally:
            # Clean up
            client.close()
            cluster.close()

    @staticmethod
    def fourier_shift(img, shift):
        """
        Shift image by a sub-pixel offset using Fourier phase shifting,
        with zero-padding to prevent wrapping
        """
        rows, cols = img.shape
        # Pad the image to prevent wrapping
        padded = np.pad(img, ((0, 0), (cols, cols)), mode='constant')

        F = np.fft.fft2(padded)
        phase_shift = np.exp(-2j * np.pi * (shift[1] * np.fft.fftfreq(padded.shape[1])))
        F_shifted = F * phase_shift[np.newaxis, :]
        shifted = np.real(np.fft.ifft2(F_shifted))

        # Return only the valid region
        return shifted[:, cols:2 * cols]

    def calculate_cross_correlation(self, proj1, proj2, shifts):
        results = []
        for shift in shifts:
            # Use Fourier phase shifting instead of ndimage.shift
            shifted_proj1 = self.fourier_shift(proj1, (0, shift))

            valid_region = min(proj2.shape[1], shifted_proj1.shape[1])
            p1_valid = shifted_proj1[:, :valid_region].flatten()
            p2_valid = proj2[:, :valid_region].flatten()

            corr = np.corrcoef(p1_valid, p2_valid)[0, 1]
            results.append((shift, corr))
            print(f"Shift: {shift}, Correlation: {corr}")

        return results

    def test_correlation(self, range=np.linspace(400, 900, 30)):
        import matplotlib
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt

        proj1_raw, proj2_raw = self.load_opposing_projections(500, 'dx')
        proj1 = self.normalize_projection(proj1_raw, 'right', 'dx')
        proj2_flipped = self.normalize_projection(proj2_raw, 'left', 'dx')

        test = self.calculate_cross_correlation(proj1, proj2_flipped, range)

        shifts = [x[0] for x in test]  # First element of each tuple
        corrs = [x[1] for x in test]  # Second element of each tuple

        plt.figure(figsize=(10, 6))
        plt.plot(shifts, corrs)
        plt.xlabel('Shift')
        plt.ylabel('Correlation')
        plt.title('Cross-correlation vs Shift')
        plt.grid(True)
        plt.show()