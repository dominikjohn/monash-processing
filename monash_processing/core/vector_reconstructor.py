import astra

from monash_processing.core.base_reconstructor import BaseReconstructor

class VectorReconstructor(BaseReconstructor):

    def setup_geometry(self, chunk_info, angles):
        # Completely override the geometry setup with vector-specific code
        # No need for super() here unless you want to use the parent's geometry somehow
        chunk_vol_geom = astra.create_vol_geom(
            chunk_info['detector_cols'],
            chunk_info['detector_cols'],
            chunk_info['chunk_rows']
        )
        chunk_proj_geom = astra.create_proj_geom(
            'cone',
            chunk_info['detector_cols'],
            chunk_info['chunk_rows'],
        )

        chunk_proj_geom = astra.geom_2vec(chunk_proj_geom, angles)

        return chunk_vol_geom, chunk_proj_geom

    def reconstruct_chunk(self, chunk_data, chunk_info, angles):
        """Reconstruct a single chunk"""
        # Set up geometry
        chunk_vol_geom, chunk_proj_geom = self.setup_geometry(chunk_info, angles)

        print('Applying center shift using ASTRA...')
        chunk_proj_geom_offset = astra.geom_postalignment(chunk_proj_geom, self.center_shift)

        # Create ASTRA objects
        proj_id = astra.create_projector('cuda3d', chunk_proj_geom_offset, chunk_vol_geom)
        sino_id = astra.data3d.create('-proj3d', chunk_proj_geom_offset, chunk_data)
        recon_id = astra.data3d.create('-vol', chunk_vol_geom)

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