from tqdm import tqdm
from monash_processing.utils.utils import Utils
from monash_processing.postprocessing.filters import RingFilter

class ChunkManager:
    def __init__(self, projections, chunk_size, angles, center_shift, channel='phase', debug=False, vector_mode=True, use_ring_filter=False):
        """
        Initialize the chunk manager
        Helper class to manage the processing of a large volume in chunks

        Parameters:
        -----------
        projections : ndarray
            3D array of projection data (n_projections, detector_rows, detector_cols)
        chunk_size : int
            Number of slices to process in each chunk
        angles : ndarray
            Array of projection angles in radians
        center_shift : float
            Center of rotation shift value
        channel : str
            'phase' or 'att' for phase or attenuation channel
        debug : bool
            If True, only process first chunk
        """
        self.projections = projections
        self.chunk_size = chunk_size
        self.angles = angles
        self.center_shift = center_shift
        self.channel = channel
        self.debug = debug
        self.vector_mode = vector_mode
        self.use_ring_filter = use_ring_filter

        self.detector_rows = projections.shape[1]
        self.detector_cols = projections.shape[2]
        self.n_chunks = (self.detector_rows + chunk_size - 1) // chunk_size

    def get_chunk_bounds(self, chunk_idx):
        """Calculate the start and end rows for a given chunk"""
        start_row = chunk_idx * self.chunk_size
        end_row = min(self.detector_rows, (chunk_idx + 1) * self.chunk_size)
        chunk_rows = end_row - start_row
        return start_row, end_row, chunk_rows

    def get_chunk_data(self, chunk_idx):
        """Extract and prepare data for a specific chunk"""
        start_row, end_row, chunk_rows = self.get_chunk_bounds(chunk_idx)

        # Extract chunk projections
        chunk_projs = self.projections[:, start_row:end_row, :]

        # Apply ring filter if enabled
        if self.use_ring_filter:
            ring_filter = RingFilter()
            print('Applying ring filter...')
            chunk_projs = ring_filter.filter_projections(chunk_projs)

        # In vector mode, the center shift is applied using an astra function
        # In a normal geometry, the projections need to be shifted "by hand" before reconstruction
        if not self.vector_mode:
            print('Applying center shift manually...')
            # Apply center shift
            shifted_chunk = Utils.apply_centershift(chunk_projs, self.center_shift).transpose(1, 0, 2)
        else:
            shifted_chunk = chunk_projs.transpose(1, 0, 2)

        return {
            'chunk_data': shifted_chunk,
            'start_row': start_row,
            'end_row': end_row,
            'chunk_rows': chunk_rows,
            'detector_cols': self.detector_cols
        }

    def save_chunk_result(self, chunk_result, chunk_info, data_loader):
        """Save the reconstruction results for a chunk"""
        if not self.debug:
            for slice_idx in tqdm(range(chunk_info['chunk_rows']), desc="Saving slices"):
                abs_slice_idx = chunk_info['start_row'] + slice_idx
                reco_channel = 'phase_reco_3d' if self.channel == 'phase' else 'att_reco_3d'
                data_loader.save_tiff(
                    channel=reco_channel,
                    angle_i=abs_slice_idx,
                    data=chunk_result[slice_idx, :, :],
                    prefix='slice'
                )