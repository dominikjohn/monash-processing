import numpy as np
from tqdm import tqdm
import tifffile
import astra

class VolumeBuilder:
    def __init__(self, pixel_size, max_angle, channel, data_loader):
        """
        Args:
            pixel_size: Size of each pixel in physical units
            max_angle: Maximum angle in degrees
            channel: 'phase' or 'attenuation'
            data_loader: DataLoader instance
            n_angles: Number of projection angles
        """
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.max_angle_rad = np.deg2rad(max_angle)

    def load_projections(self):
        """Load all projections for the specified channel."""
        projections = []

        input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'att')
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))

        for tiff_file in tqdm(tiff_files, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_file)
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_file}: {str(e)}")

        return np.array(projections)

    def reconstruct(self):
        # Load projections
        projections = self.load_projections()

        angles = np.linspace(0, self.max_angle_rad, projections.shape[0])

        # Create geometries based on actual projection size
        height, width = projections[0].shape
        vol_geom = astra.create_vol_geom(width, width, height)
        proj_geom = astra.create_proj_geom('parallel3d', self.pixel_size, self.pixel_size, width, height, angles)

        # Create ASTRA projector
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        # Create sinogram data
        sino_id = astra.data3d.create('-sino', proj_geom, projections)

        # Create reconstruction volume
        rec_id = astra.data3d.create('-vol', vol_geom)

        # Create configuration
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ReconstructionDataId'] = rec_id

        # Create and run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Get the result and clean up
        result = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)

        print('Saving tomographic slices')
        for i in tqdm(range(result.shape[0])):
            self.data_loader.save_projection(
                channel=self.channel,
                angle_i=i,
                data=result[i]
            )

        return result