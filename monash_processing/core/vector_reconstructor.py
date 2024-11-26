import astra

from monash_processing.core.base_reconstructor import BaseReconstructor
import numpy as np

class VectorReconstructor(BaseReconstructor):

    def setup_geometry(self, chunk_info, angles, volume_scaling=1.8):

        detector_width = chunk_info['detector_cols']
        offset = int(self.center_shift)
        print('Using offset ', offset)
        n_cols = int(detector_width + (2 * np.abs(offset)))

        chunk_vol_geom = astra.create_vol_geom(
            n_cols,
            n_cols,
            chunk_info['chunk_rows'],
            -offset, # min_x
            detector_width + offset, # max_x
            -n_cols // 2, # min_y
            n_cols // 2, # max_y
            -chunk_info['chunk_rows'] // 2, # min_z
            chunk_info['chunk_rows'] // 2 # max_z
        )

        print("Chunk volume geometry:", chunk_vol_geom)

        scaling_factor = 1e6
        source_distance = 21.5 * scaling_factor
        detector_distance = 0.158 * scaling_factor
        pixel_size = 1.444e-6 * scaling_factor

        print("Source distance [m]:", source_distance / scaling_factor)
        print("Detector distance [m]:", detector_distance / scaling_factor)
        print("Detector pixel size [m]:", pixel_size / scaling_factor)

        chunk_proj_geom = astra.create_proj_geom(
            'cone', pixel_size, pixel_size,
            chunk_info['chunk_rows'],
            chunk_info['detector_cols'],
            angles,
            source_distance,
            detector_distance
        )

        #chunk_proj_geom = astra.geom_2vec(chunk_proj_geom)

        return chunk_vol_geom, chunk_proj_geom

    def reconstruct_chunk(self, chunk_data, chunk_info, angles, center_shift=0):
        """Reconstruct a single chunk"""
        # Set up geometry
        print('Using center_shift:', center_shift)
        chunk_vol_geom_offset, chunk_proj_geom_offset = self.setup_geometry(chunk_info, angles, center_shift)

        print('Setup projection geometry: ', chunk_proj_geom_offset)
        print('Setup volume geometry: ', chunk_vol_geom_offset)

        #chunk_proj_geom_offset = astra.geom_postalignment(chunk_proj_geom, self.center_shift)
        #print("Chunk projection geometry after shift:", chunk_proj_geom_offset)

        # Create ASTRA objects
        proj_id = astra.create_projector('cuda3d', chunk_proj_geom_offset, chunk_vol_geom_offset)
        sino_id = astra.data3d.create('-proj3d', chunk_proj_geom_offset, chunk_data)
        recon_id = astra.data3d.create('-vol', chunk_vol_geom_offset)

        # Configure reconstruction
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = recon_id
        cfg['option'] = {'ShortScan': self.enable_short_scan}

        # Run reconstruction
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Get result and clean up
        chunk_result = astra.data3d.get(recon_id)

        # Clean up ASTRA objects
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(recon_id)
        astra.projector.delete(proj_id)

        return chunk_result