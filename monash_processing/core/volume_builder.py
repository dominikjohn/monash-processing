import numpy as np
from tqdm import tqdm
import tifffile
import astra


class VolumeBuilder:
    def __init__(self, pixel_size, max_angle, channel, data_loader, method='FBP', num_iterations=150, debug=False):
        """
        Args:
            pixel_size: Size of each pixel in physical units
            max_angle: Maximum angle in degrees (will be cropped to 180°)
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

        # Generate angles and create mask for <= 180°
        angles = np.linspace(0, self.max_angle_rad, len(tiff_files))
        valid_angles_mask = angles <= np.pi  # π radians = 180°
        valid_indices = np.where(valid_angles_mask)[0]

        if self.debug:
            print(f"Total projections: {len(tiff_files)}")
            print(f"Projections up to 180°: {len(valid_indices)}")
            print(f"Angle range used: 0 to {np.rad2deg(angles[valid_indices[-1]]):.1f}°")

        if self.debug and len(tiff_files) > 0:
            print("DEBUG MODE: Loading only first valid projection")
            first_proj = tifffile.imread(tiff_files[0])
            projections = np.zeros((len(valid_indices), *first_proj.shape), dtype=first_proj.dtype)
            projections[0] = first_proj
            print(f"Debug array shape: {projections.shape}")
            return projections, angles[valid_angles_mask]

        projections = []
        for idx in tqdm(valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_files[idx])
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_files[idx]}: {str(e)}")

        return np.array(projections), angles[valid_angles_mask]

    def reconstruct(self):
        try:
            # Load projections and get valid angles
            projections, angles = self.load_projections()
            print(f"Loaded projections shape: {projections.shape}")

            # Get dimensions from projections
            n_proj, detector_rows, detector_cols = projections.shape
            print(f"Dimensions - Projections: {n_proj}, Detector rows: {detector_rows}, Detector cols: {detector_cols}")

            if self.debug:
                print("DEBUG INFO:")
                print(f"Angle range: {np.rad2deg(angles[0]):.1f}° to {np.rad2deg(angles[-1]):.1f}°")
                print(f"Memory usage of projections: {projections.nbytes / 1e9:.2f} GB")

            # First reshape array to match ASTRA's expected format
            projections_astra = np.transpose(projections, (1, 0, 2))

            if self.debug:
                print("\nArray shapes:")
                print(f"Original projections: {projections.shape}")
                print(f"Reshaped for ASTRA: {projections_astra.shape}")

            # Create volume geometry
            vol_geom = astra.create_vol_geom(detector_cols, detector_cols, detector_rows)

            # Create projection geometry
            proj_geom = astra.create_proj_geom('parallel3d',
                                               1.0, 1.0,
                                               detector_cols, detector_rows,
                                               angles)

            if self.debug:
                print("\nGeometry Info:")
                print(f"Volume Geometry: {vol_geom}")
                print(f"Projection Geometry: {proj_geom}")

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