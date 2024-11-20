import numpy as np
from tqdm import tqdm
import tifffile
import astra


class VolumeBuilder:
    def __init__(self, pixel_size, max_angle, channel, data_loader, method='FBP', num_iterations=150, debug=False):
        """
        Args:
            pixel_size: Size of each pixel in physical units
            max_angle: Maximum angle in degrees
            channel: 'phase' or 'attenuation'
            data_loader: DataLoader instance
            method: 'FBP' or 'SIRT'
            num_iterations: Number of iterations for SIRT (ignored for FBP)
            debug: If True, only loads first projection and fills rest with zeros
        """
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.max_angle_rad = np.deg2rad(max_angle)
        self.method = method.upper()
        self.num_iterations = num_iterations
        self.debug = debug

        if self.method not in ['FBP', 'SIRT']:
            raise ValueError("Method must be either 'FBP' or 'SIRT'")

    def load_projections(self):
        """Load all projections for the specified channel."""
        input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'att')
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))

        if self.debug:
            print("DEBUG MODE: Loading only first projection")
            first_proj = tifffile.imread(tiff_files[0])
            projections = np.zeros((len(tiff_files), *first_proj.shape), dtype=first_proj.dtype)
            projections[0] = first_proj
            print(f"Debug array shape: {projections.shape}")
            return projections

        projections = []
        for tiff_file in tqdm(tiff_files, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_file)
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_file}: {str(e)}")

        return np.array(projections)

    def reconstruct(self):
        try:
            # Load projections
            projections = self.load_projections()
            print(f"Loaded projections shape: {projections.shape}")

            # Create angles array
            angles = np.linspace(0, self.max_angle_rad, projections.shape[0])

            # Get dimensions from projections
            n_proj, detector_rows, detector_cols = projections.shape
            print(f"Dimensions - Projections: {n_proj}, Detector rows: {detector_rows}, Detector cols: {detector_cols}")

            if self.debug:
                print("DEBUG INFO:")
                print(f"Angle range: {np.rad2deg(angles[0])} to {np.rad2deg(angles[-1])} degrees")
                print(f"Memory usage of projections: {projections.nbytes / 1e9:.2f} GB")

            # Create volume geometry (cubic volume)
            vol_geom = astra.create_vol_geom(detector_cols, detector_cols, detector_rows)

            # Create projection geometry with swapped dimensions to match ASTRA's expectations
            proj_geom = astra.create_proj_geom('parallel3d',
                                               1.0, 1.0,
                                               detector_cols, detector_rows,  # Note the swap here
                                               angles)

            if self.debug:
                print("\nGeometry Info:")
                print(f"Volume Geometry: {vol_geom}")
                print(f"Projection Geometry: {proj_geom}")
                print(f"Projection data shape: {projections.shape}")

            # Transpose projections to match ASTRA's expected format
            projections_astra = np.transpose(projections, (0, 2, 1))

            if self.debug:
                print(f"Transposed projection shape: {projections_astra.shape}")

            # Create sinogram data
            sino_id = astra.data3d.create('-sino', proj_geom, projections_astra)

            # Create reconstruction volume
            rec_id = astra.data3d.create('-vol', vol_geom)

            if self.method == 'FBP':
                cfg = astra.astra_dict('FBP3D_CUDA')
                cfg['ProjectionDataId'] = sino_id
                cfg['ReconstructionDataId'] = rec_id
                cfg['option'] = {'FilterType': 'Ram-Lak'}
                print("Using FBP reconstruction...")

            else:  # SIRT
                cfg = astra.astra_dict('SIRT3D_CUDA')
                cfg['ProjectionDataId'] = sino_id
                cfg['ReconstructionDataId'] = rec_id
                print(f"Using SIRT reconstruction with {self.num_iterations} iterations...")

            if self.debug:
                print("\nAlgorithm Configuration:")
                print(cfg)

            # Create and run the algorithm
            alg_id = astra.algorithm.create(cfg)

            if self.method == 'SIRT':
                astra.algorithm.run(alg_id, self.num_iterations)
            else:
                astra.algorithm.run(alg_id)

            # Get the result
            print("Retrieving result...")
            result = astra.data3d.get(rec_id)

            if self.debug:
                print("\nResult Info:")
                print(f"Result shape: {result.shape}")
                print(f"Result range: [{result.min():.2e}, {result.max():.2e}]")
                print(f"Number of non-zero elements: {np.count_nonzero(result)}")

            # Clean up ASTRA objects
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(rec_id)
            astra.data3d.delete(sino_id)

            if not self.debug:
                print('Saving tomographic slices...')
                for i in tqdm(range(result.shape[0])):
                    self.data_loader.save_projection(
                        channel=self.channel,
                        angle_i=i,
                        data=result[i])
            else:
                print("Debug mode: Skipping file saving")

            return result

        except Exception as e:
            if self.debug:
                print("\nError occurred during reconstruction:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                raise
            else:
                raise

    @staticmethod
    def get_available_fbp_filters():
        """Returns list of available FBP filters"""
        return ['Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann', 'None']