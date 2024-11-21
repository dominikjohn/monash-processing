import numpy as np
from tqdm import tqdm
import tifffile
import astra
from scipy.ndimage import shift as scipy_shift

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
    def apply_centershift(projections, center_shift, cuda=True, batch_size=10):
        """
        Apply center shift to projections in batches to avoid GPU memory exhaustion.
        :param projections: 2D or 3D numpy array
        :param center_shift: float, shift in pixels
        :param cuda: bool, whether to use GPU
        :param batch_size: int, number of projections to process at once
        :return: shifted projections array
        """

        if center_shift == 0:
            print("Center shift is 0, skipping shift")
            return projections

        print(f"Applying center shift of {center_shift} pixels to projection data of shape {projections.shape}")

        # Get number of dimensions and create appropriate shift vector
        ndim = projections.ndim
        shift_vector = (0, center_shift) if ndim == 2 else (0, 0, center_shift)

        # If 2D or small enough, process normally
        if ndim == 2 or (not cuda):
            return scipy_shift(projections, shift_vector, mode='nearest', order=0)

        try:
            import cupy as cp
            from cupyx.scipy import ndimage

            # Process in batches
            result = np.zeros_like(projections)
            for i in tqdm(range(0, len(projections), batch_size), desc="Applying center shift"):
                batch = projections[i:i + batch_size]
                batch_gpu = cp.asarray(batch)

                # Process batch
                shifted = ndimage.shift(batch_gpu,
                                        shift=(0, 0, center_shift),
                                        mode='nearest',
                                        order=0)

                # Store result and free GPU memory
                result[i:i + batch_size] = cp.asnumpy(shifted)
                del batch_gpu
                del shifted
                print('Freeing GPU memory')
                cp.get_default_memory_pool().free_all_blocks()
                print('Memory freed')
            return result

        except Exception as e:
            print(f"GPU shift failed: {str(e)}, falling back to CPU")
            return scipy_shift(projections, shift_vector, mode='nearest', order=0)

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

    def reconstruct_3d(self, enable_short_scan=True, debug=False, chunk_size=128):
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

        detector_rows = projections.shape[1]
        detector_cols = projections.shape[2]

        # Calculate number of chunks needed
        n_chunks = (detector_rows + chunk_size - 1) // chunk_size

        print(f"Processing volume in {n_chunks} chunks of {chunk_size} slices (quasi-parallel beam using FDK)")

        # Use very large source distance to approximate parallel beam
        source_distance = 1e8  # 100 million units
        detector_distance = 1e6  # 1 million units

        # Initialize result array
        full_result = np.zeros((detector_cols, detector_cols, detector_rows), dtype=np.float32)

        for chunk_idx in tqdm(range(n_chunks), desc="Processing volume chunks"):
            # Calculate chunk boundaries (no overlap needed for quasi-parallel beam)
            start_row = chunk_idx * chunk_size
            end_row = min(detector_rows, (chunk_idx + 1) * chunk_size)
            chunk_rows = end_row - start_row

            # Create geometry for this chunk
            chunk_vol_geom = astra.create_vol_geom(detector_cols, detector_cols, chunk_rows)
            chunk_proj_geom = astra.create_proj_geom('cone', 1.0, 1.0,
                                                     chunk_rows, detector_cols,
                                                     angles, source_distance, detector_distance)

            # Extract relevant portion of projections
            chunk_projs = projections[:, start_row:end_row, :]

            # Apply center shift to chunk
            shifted_chunk = VolumeBuilder.apply_centershift(chunk_projs, self.center_shift).transpose(1,0,2)

            print('Centershift applied, starting reconstruction...')

            # Create ASTRA objects for this chunk
            proj_id = astra.create_projector('cuda3d', chunk_proj_geom, chunk_vol_geom)
            sino_id = astra.data3d.create('-proj3d', chunk_proj_geom, shifted_chunk)
            recon_id = astra.data3d.create('-vol', chunk_vol_geom)

            # Configure reconstruction
            cfg = astra.astra_dict('FDK_CUDA')
            cfg['ProjectorId'] = proj_id
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = recon_id
            cfg['option'] = {'ShortScan': enable_short_scan}

            # Run reconstruction for this chunk
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            # Get chunk result
            chunk_result = astra.data3d.get(recon_id)

            # Clean up ASTRA objects
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(sino_id)
            astra.data3d.delete(recon_id)
            astra.projector.delete(proj_id)

            print(f'Chunk {chunk_idx} reconstruction finished')
            # Insert chunk into full result (no need to handle overlap)
            #full_result[:, :, start_row:end_row] = chunk_result

            print('Chunk result shape:', chunk_result.shape)

            if not debug:
                for slice_idx in tqdm(range(chunk_rows), desc="Saving slices"):
                    # Calculate the absolute slice index in the full volume
                    abs_slice_idx = start_row + slice_idx
                    reco_channel = 'phase_reco_3d' if self.channel == 'phase' else 'att_reco_3d'
                    self.data_loader.save_tiff(
                        channel=reco_channel,
                        angle_i=abs_slice_idx,
                        data=chunk_result[:, :, slice_idx],  # Use chunk_result directly
                        prefix='slice'
                    )

            # Force GPU memory cleanup
            import gc
            gc.collect()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except ImportError:
                pass

            print('Reconstruction finished successfully!')
        else:
            print('Debug reconstruction completed - results not saved')

        return full_result