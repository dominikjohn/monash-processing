from pathlib import Path
import numpy as np
import astra
import logging
from typing import Optional, Dict, Union, Tuple


class VolumeBuilder:
    """Handles 3D reconstruction using ASTRA toolbox."""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Default reconstruction parameters
        self.defaults = {
            'algorithm': 'FBP',  # or 'SIRT3D', 'CGLS3D'
            'num_iterations': 100,  # for iterative methods
            'gpu_index': 0,
            'vol_shape': None,  # will be determined from projections
            'angles': None,  # will be set during reconstruction
        }

        # Update defaults with provided config
        self.defaults.update(self.config)

    def _create_geometries(self,
                           proj_shape: Tuple[int, int, int],
                           angles: np.ndarray) -> Tuple[int, int]:
        """Create ASTRA projection and volume geometries."""
        # proj_shape should be (num_angles, det_rows, det_cols)
        num_angles, det_rows, det_cols = proj_shape

        if self.defaults['vol_shape'] is None:
            # Default to volume with same size as detector
            vol_shape = (det_rows, det_cols, det_cols)
        else:
            vol_shape = self.defaults['vol_shape']

        # Create volume geometry
        vol_geom = astra.create_vol_geom(vol_shape)

        # Create projection geometry
        # Assuming cone beam geometry - modify if using parallel beam
        proj_geom = astra.create_proj_geom('cone', 1.0, 1.0,
                                           det_rows, det_cols,
                                           angles)

        return vol_geom, proj_geom

    def _setup_reconstruction(self,
                              projections: np.ndarray,
                              angles: np.ndarray) -> Tuple[int, int, int]:
        """Setup ASTRA objects for reconstruction."""
        # Create geometries
        vol_geom, proj_geom = self._create_geometries(projections.shape, angles)

        # Create ASTRA objects
        proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
        vol_id = astra.data3d.create('-vol', vol_geom)

        # Create algorithm
        alg = self.defaults['algorithm']
        cfg = astra.astra_dict(alg)
        cfg['ProjectionDataId'] = proj_id
        cfg['ReconstructionDataId'] = vol_id
        cfg['option'] = {
            'GPUindex': self.defaults['gpu_index']
        }

        if alg in ['SIRT3D', 'CGLS3D']:
            cfg['option']['ProjectorId'] = astra.create_projector('cuda3d', proj_geom, vol_geom)

        alg_id = astra.algorithm.create(cfg)

        return proj_id, vol_id, alg_id

    def reconstruct(self,
                    projections: np.ndarray,
                    angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct 3D volume from projections.

        Args:
            projections: Projection data with shape (num_angles, det_rows, det_cols)
            angles: Array of projection angles in radians. If None, assumes evenly
                   spaced angles over 180 degrees.

        Returns:
            Reconstructed volume as numpy array
        """
        try:
            if angles is None:
                # Assume evenly spaced angles over 180 degrees
                angles = np.linspace(0, np.pi, projections.shape[0])

            self.logger.info(f"Starting reconstruction with {len(angles)} projections")
            self.logger.info(f"Using algorithm: {self.defaults['algorithm']}")

            # Setup reconstruction
            proj_id, vol_id, alg_id = self._setup_reconstruction(projections, angles)

            # Run reconstruction
            if self.defaults['algorithm'] in ['SIRT3D', 'CGLS3D']:
                astra.algorithm.run(alg_id, self.defaults['num_iterations'])
                self.logger.info(f"Completed {self.defaults['num_iterations']} iterations")
            else:
                astra.algorithm.run(alg_id)

            # Get result
            volume = astra.data3d.get(vol_id)

            # Cleanup
            self._cleanup([proj_id, vol_id, alg_id])

            self.logger.info(f"Reconstruction completed. Volume shape: {volume.shape}")
            return volume

        except Exception as e:
            self.logger.error(f"Reconstruction failed: {str(e)}")
            raise

    def _cleanup(self, ids: list):
        """Clean up ASTRA objects."""
        for id in ids:
            astra.algorithm.delete(id)
            astra.data3d.delete(id)

    def set_config(self, config: Dict):
        """Update configuration parameters."""
        self.defaults.update(config)

    def get_available_algorithms(self) -> list:
        """Return list of available reconstruction algorithms."""
        return ['FBP', 'SIRT3D', 'CGLS3D']