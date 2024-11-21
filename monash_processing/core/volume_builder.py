import numpy as np
from tqdm import tqdm
import tifffile
import astra
from scipy.ndimage import shift as scipy_shift

class VolumeBuilder:
    def __init__(self, pixel_size, max_angle, channel, data_loader, center_shift=0, method='FBP', num_iterations=150):
        """
        Args:
            pixel_size: Size of each pixel in physical units
            max_angle: Maximum angle in degrees (will be cropped to 180°)
            channel: 'phase' or 'attenuation'
            data_loader: DataLoader instance
            method: 'FBP' or 'SIRT'
            num_iterations: Number of iterations for SIRT (ignored for FBP)
        """
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.max_angle_rad = np.deg2rad(max_angle)
        self.method = method.upper()
        self.num_iterations = num_iterations
        self.center_shift = center_shift

        if self.method not in ['FBP', 'SIRT']:
            raise ValueError("Method must be either 'FBP' or 'SIRT'")

    def load_projections(self):
        """
        :return: np.ndarray, np.ndarray
        """
        input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'att')
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))

        # Generate angles and create mask for <= 180°
        angles = np.linspace(0, self.max_angle_rad, len(tiff_files))
        valid_angles_mask = angles <= np.pi  # π radians = 180°
        valid_indices = np.where(valid_angles_mask)[0]

        projections = []
        for projection_i in tqdm(valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_files[projection_i])
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")

        return np.array(projections), angles[valid_indices]

    @staticmethod
    def apply_centershift(projections, center_shift, cuda=True):
        """
        Apply center shift to projections.
        :param projections: 2D or 3D numpy array
        :param center_shift: float, shift in pixels
        :param cuda: bool, whether to use GPU
        :return: shifted projections array
        """
        print(f"Applying center shift of {center_shift} pixels to projection data of shape {projections.shape}")

        # Get number of dimensions and create appropriate shift vector
        ndim = projections.ndim
        shift_vector = (0, center_shift) if ndim == 2 else (0, 0, center_shift)

        if cuda:
            try:
                import cupy as cp
                from cupyx.scipy import ndimage
                projections_gpu = cp.asarray(projections)
                shifted = ndimage.shift(projections_gpu,
                                        shift=shift_vector,  # Use correct shift vector
                                        mode='nearest',
                                        order=0)
                return cp.asnumpy(shifted)
            except Exception as e:
                print(f"GPU shift failed: {str(e)}, falling back to CPU")
                cuda = False

        if not cuda:
            return scipy_shift(projections, shift_vector,  # Use same shift vector
                               mode='nearest', order=0)

    @staticmethod
    def reconstruct_slice(projections, angles, pixel_size):
        """
        Reconstruct a single slice using FBP algorithm with ASTRA Toolbox.
        Args:
            projections: 2D numpy array of projection data (projections, rows, cols)
            angles: Array of projection angles in radians
            pixel_size: Size of detector pixels in mm
            detector_cols: Number of detector columns

        Returns:
            2D numpy array of reconstructed slice
        """

        # Make sure this is a 2D object instead of (angles, 1, cols)
        projection_slices = np.squeeze(projections)
        detector_cols = projection_slices.shape[1]

        vol_geom = astra.create_vol_geom(detector_cols, detector_cols)

        # Create projection geometry with center shift
        proj_geom = astra.create_proj_geom('parallel', 1., detector_cols, angles)

        # Create sinogram
        sino_id = astra.data2d.create('-sino', proj_geom, projection_slices)

        # Create reconstruction volume
        rec_id = astra.data2d.create('-vol', vol_geom)

        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        # Create FBP configuration
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        # Create and run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Get the result
        result = astra.data2d.get(rec_id)

        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)

        return result  # Return the single slice

    def reconstruct(self):
        """
        Efficient slice-by-slice FBP reconstruction using CUDA
        Args:
            projections: shape (angles, rows, cols)
            angles: projection angles in radians
        """

        projections, angles = self.load_projections()

        n_slices = projections.shape[1]
        detector_cols = projections.shape[2]

        vol_geom = astra.create_vol_geom(detector_cols, detector_cols)
        proj_geom = astra.create_proj_geom('parallel', 1.0, detector_cols, angles)
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        # Pre-create configuration (reuse for all slices)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        # Preallocate output array
        result = np.zeros((n_slices, detector_cols, detector_cols))

        try:
            # Process all slices
            for i in tqdm(range(n_slices), desc="Reconstructing slices"):
                shifted_projection = VolumeBuilder.apply_centershift(projections[:, i, :], self.center_shift)

                # Create sinogram for this slice
                sino_id = astra.data2d.create('-sino', proj_geom, shifted_projection)
                rec_id = astra.data2d.create('-vol', vol_geom)

                # Update config with new data IDs
                cfg['ProjectionDataId'] = sino_id
                cfg['ReconstructionDataId'] = rec_id

                # Create and run algorithm
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)

                # Get result for this slice
                slice_result = astra.data2d.get(rec_id)
                result[i] = slice_result

                reco_channel = 'phase_reco' if self.channel == 'phase' else 'att_reco'
                self.data_loader.save_tiff(
                    channel=reco_channel,
                    angle_i=i,
                    data=slice_result,
                    prefix='slice'
                )

                # Clean up slice-specific objects
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(rec_id)
                astra.data2d.delete(sino_id)

        finally:
            # Clean up shared objects
            astra.projector.delete(proj_id)

        return result