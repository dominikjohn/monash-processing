import numpy as np
from setuptools.command.easy_install import is_sh
from tqdm import tqdm
import tifffile
import astra

from monash_processing.core.vector_reconstructor import VectorReconstructor
from monash_processing.core.chunk_manager import ChunkManager
from monash_processing.core.base_reconstructor import BaseReconstructor
from monash_processing.utils.utils import Utils

class VolumeBuilder:
    def __init__(self, pixel_size, max_angle, channel, data_loader, center_shift=0, method='FBP', num_iterations=150, limit_max_angle=True):
        """
        Args:
            pixel_size: Size of each pixel in physical units
            max_angle: Maximum angle in degrees (will be cropped to 180°)
            channel: 'phase' or 'attenuation'
            data_loader: DataLoader instance
            method: 'FBP' or 'SIRT'
            num_iterations: Number of iterations for SIRT (ignored for FBP)
            center_shift: Center shift in pixels
            limit_max_angle: Whether to limit the max angle to 180°
        """
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.max_angle_rad = np.deg2rad(max_angle)
        self.method = method.upper()
        self.num_iterations = num_iterations
        self.center_shift = center_shift
        self.limit_max_angle = limit_max_angle

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

        if self.limit_max_angle:
            valid_angles_mask = angles <= np.pi  # π radians = 180°
            valid_indices = np.where(valid_angles_mask)[0]
        else:
            valid_indices = np.arange(len(tiff_files))

        projections = []
        for projection_i in tqdm(valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_files[projection_i])
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")

        return np.array(projections), angles[valid_indices]

    @staticmethod
    def reconstruct_slice(projections, angles, pixel_size, is_stitched=False):
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

        if is_stitched:
            vol_geom = astra.create_vol_geom(int(detector_cols * 1.8), int(detector_cols))
        else:
            vol_geom = astra.create_vol_geom(detector_cols, detector_cols)

        scaling_factor = 1e6
        source_distance = 21.5 * scaling_factor
        detector_distance = 0.158 * scaling_factor
        pixel_size = 1.444e-6 * scaling_factor

        # Create projection geometry with center shift
        proj_geom = astra.create_proj_geom('fanflat', pixel_size, detector_cols, angles, source_distance, detector_distance)

        # Create sinogram
        sino_id = astra.data2d.create('-sino', proj_geom, projection_slices)
        print('Sinogram shape:', projection_slices.shape)
        print(angles.shape)
        print(angles)
        # Create reconstruction volume
        rec_id = astra.data2d.create('-vol', vol_geom)

        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        # Create FBP configuration
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id
        print('Short scan is', is_short_scan)
        cfg['option'] = {'ShortScan': is_short_scan, 'FilterType': 'Ram-Lak'}

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
                shifted_projection = Utils.apply_centershift(projections[:, i, :], self.center_shift)

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

    def reconstruct_3d(self, enable_short_scan=True, debug=False, chunk_size=128, vector_mode=True):
        '''
        Reconstruct a 3D volume using ASTRA Toolbox with FDK but quasi-parallel beam geometry (large source distance)
        :param enable_short_scan: bool, True means 180° scan is sufficient
        :param debug: bool, if True only loads first projection and skips saving
        :param chunk_size: int, number of slices to process at once
        :return: reconstruction result array
        '''
        input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'att')
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))
        angles = np.linspace(0, self.max_angle_rad, len(tiff_files))

        if self.limit_max_angle:
            valid_angles_mask = angles <= np.pi
            valid_indices = np.where(valid_angles_mask)[0]
        else:
            valid_indices = np.arange(len(tiff_files))

        if debug:
            data = tifffile.imread(tiff_files[valid_indices[0]])
            template = np.zeros((len(valid_indices), *data.shape), dtype=data.dtype)
            template[0] = data
            projections = template
        else:
            projections = []
            for projection_i in tqdm(valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
                try:
                    data = tifffile.imread(tiff_files[projection_i])
                    projections.append(data)
                except Exception as e:
                    raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")
            projections = np.array(projections)

        angles = angles[valid_indices]

        chunk_manager = ChunkManager(
            projections=projections,
            chunk_size=chunk_size,
            angles=angles,
            center_shift=self.center_shift,
            channel=self.channel,
            debug=debug,
            vector_mode=vector_mode
        )

        print(f"Processing volume in {chunk_manager.n_chunks} chunks of {chunk_size} slices")
        if vector_mode:
            reconstructor = VectorReconstructor(enable_short_scan=enable_short_scan, center_shift=self.center_shift)
        else:
            reconstructor = BaseReconstructor(enable_short_scan=enable_short_scan, center_shift=self.center_shift)

        # Process chunks
        for chunk_idx in tqdm(range(chunk_manager.n_chunks), desc="Processing volume chunks"):
            # Get chunk data
            chunk_info = chunk_manager.get_chunk_data(chunk_idx)

            print('Starting reconstruction...')
            # Reconstruct chunk
            chunk_result = reconstructor.reconstruct_chunk(
                chunk_info['chunk_data'],
                chunk_info,
                angles
            )
            print(f'Chunk {chunk_idx} reconstruction finished')

            # Save results
            chunk_manager.save_chunk_result(chunk_result, chunk_info, self.data_loader)

            # Force GPU memory cleanup
            import gc
            gc.collect()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except ImportError:
                pass

            print('Chunk processing finished successfully!')

        if debug:
            print('Debug reconstruction completed - results not saved')
        else:
            print('Full reconstruction completed successfully!')