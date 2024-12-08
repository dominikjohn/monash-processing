import numpy as np
from tqdm import tqdm
import tifffile
import astra

from monash_processing.postprocessing.filters import RingFilter
from monash_processing.core.vector_reconstructor import VectorReconstructor
from monash_processing.core.chunk_manager import ChunkManager
from monash_processing.core.base_reconstructor import BaseReconstructor
from monash_processing.utils.utils import Utils
import scipy.constants

class CILVolumeBuilder():
    def __init__(self, pixel_size, max_angle, channel, data_loader, energy, center_shift=0, is_stitched=False):
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.max_angle_rad = np.deg2rad(max_angle)
        self.center_shift = center_shift
        self.energy = energy
        self.is_stitched = is_stitched

    def load_projections(self, format='tif'):
        """
        :return: np.ndarray, np.ndarray
        """
        if self.is_stitched:
            input_dir = self.data_loader.results_dir / ('phi_stitched' if self.channel == 'phase' else 'T_stitched')
        else:
            input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'T')

        tiff_files = sorted(input_dir.glob(f'projection_*.{format}*'))

        # Generate angles and create mask for <= 180°
        angles = np.linspace(0, self.max_angle_rad, len(tiff_files))

        valid_indices = np.arange(len(tiff_files))

        projections = []
        for projection_i in tqdm(valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_files[projection_i])
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")

        return np.array(projections), angles[valid_indices]

    def reconstruct(self, ring_filter=True):
        """
        Efficient slice-by-slice FBP reconstruction using CUDA
        Args:
            projections: shape (angles, rows, cols)
            angles: projection angles in radians
        """

        projections, angles = self.load_projections()

        if self.channel == 'att':
            epsilon = 1e-8
            projections = np.clip(projections, epsilon, 1.0)
            projections = -np.log(projections)
            print('MIN: ', projections.min())
            print('MAX: ', projections.max())

        n_slices = projections.shape[1]
        detector_cols = projections.shape[2]

        scaling_factor = 1e6
        pixel_size = 1.444e-6 * scaling_factor

        vol_geom = astra.create_vol_geom(detector_cols, detector_cols)
        proj_geom = astra.create_proj_geom('parallel', pixel_size, detector_cols, angles)
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

                # Convert to physical units
                if self.channel == 'phase':
                    wavevec = 2 * np.pi * self.energy / (
                            scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
                    r0 = scipy.constants.physical_constants['classical electron radius'][0]
                    conv = wavevec / (2 * np.pi) / r0 / 1E27
                    slice_result *= conv * 1E9 # convert to nm^3
                    reco_channel = 'phase_reco'
                    rf = RingFilter()
                else:
                    # Attenuation just needs to be divided by 100
                    reco_channel = 'att_reco'
                    slice_result *= scaling_factor # corrects for the scaling factor
                    slice_result /= 100 # converts to cm^-1
                    rf = RingFilter(rwidth=15)

                print('appl. ring filter')
                slice_result = rf.filter_slice(slice_result)

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