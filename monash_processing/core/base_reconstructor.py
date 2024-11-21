import astra

class BaseReconstructor:
    def __init__(self, enable_short_scan=True):
        self.enable_short_scan = enable_short_scan

    def setup_geometry(self, chunk_info, angles):
        """Set up the reconstruction geometry"""
        chunk_vol_geom = astra.create_vol_geom(
            chunk_info['detector_cols'],
            chunk_info['detector_cols'],
            chunk_info['chunk_rows']
        )

        # Use very large source distance to approximate parallel beam
        source_distance = 1e8  # 100 million units
        detector_distance = 1e6  # 1 million units

        chunk_proj_geom = astra.create_proj_geom(
            'cone', 1.0, 1.0,
            chunk_info['chunk_rows'],
            chunk_info['detector_cols'],
            angles,
            source_distance,
            detector_distance
        )

        return chunk_vol_geom, chunk_proj_geom

    def reconstruct_chunk(self, chunk_data, chunk_info, angles):
        """Reconstruct a single chunk"""
        # Set up geometry
        chunk_vol_geom, chunk_proj_geom = self.setup_geometry(chunk_info, angles)

        # Create ASTRA objects
        proj_id = astra.create_projector('cuda3d', chunk_proj_geom, chunk_vol_geom)
        sino_id = astra.data3d.create('-proj3d', chunk_proj_geom, chunk_data)
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