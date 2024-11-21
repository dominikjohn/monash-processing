import numpy as np
import tomopy
from tqdm import tqdm

class RingFilter:
    """
    A class to apply ring artifact removal using TomoPy's remove_ring method.
    This filter should be applied to projections before reconstruction.
    """

    def __init__(self, center_x=None, center_y=None, thresh=300.0, thresh_max=300.0,
                 thresh_min=-100.0, theta_min=30, rwidth=30, int_mode='WRAP',
                 ncore=None, nchunk=None):
        """
        Initialize the ring filter with parameters for TomoPy's remove_ring function.

        Parameters
        ----------
        center_x : float, optional
            Abscissa location of center of rotation
        center_y : float, optional
            Ordinate location of center of rotation
        thresh : float, optional
            Maximum value of an offset due to a ring artifact
        thresh_max : float, optional
            Max value for portion of image to filter
        thresh_min : float, optional
            Min value for portion of image to filter
        theta_min : int, optional
            Features larger than twice this angle (degrees) will be considered ring artifacts
        rwidth : int, optional
            Maximum width of the rings to be filtered in pixels
        int_mode : str, optional
            'WRAP' for wrapping at 0 and 360 degrees, 'REFLECT' for reflective boundaries
        ncore : int, optional
            Number of cores that will be assigned to jobs
        nchunk : int, optional
            Chunk size for each core
        """
        self.center_x = center_x
        self.center_y = center_y
        self.thresh = thresh
        self.thresh_max = thresh_max
        self.thresh_min = thresh_min
        self.theta_min = theta_min
        self.rwidth = rwidth
        self.int_mode = int_mode
        self.ncore = ncore
        self.nchunk = nchunk

    def filter_projections(self, projections):
        """
        Apply ring artifact removal to projection data.

        Parameters
        ----------
        projections : ndarray
            3D array of projection data (angles, rows, cols)

        Returns
        -------
        ndarray
            Filtered projection data with ring artifacts removed
        """
        # Convert data type to float32 as required by TomoPy
        projections = projections.astype(np.float32)

        # Process each row of projections separately
        filtered_projections = np.zeros_like(projections)

        for row in tqdm(range(projections.shape[1]), desc="Applying ring filter"):
            # Extract sinogram for current row
            sinogram = projections[:, row, :]

            # Apply TomoPy's ring removal filter
            filtered_sinogram = tomopy.misc.corr.remove_ring(
                sinogram,
                center_x=self.center_x,
                center_y=self.center_y,
                thresh=self.thresh,
                thresh_max=self.thresh_max,
                thresh_min=self.thresh_min,
                theta_min=self.theta_min,
                rwidth=self.rwidth,
                int_mode=self.int_mode,
                ncore=self.ncore,
                nchunk=self.nchunk
            )

            filtered_projections[:, row, :] = filtered_sinogram

        return filtered_projections