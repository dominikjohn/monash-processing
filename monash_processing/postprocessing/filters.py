import numpy as np
import tomopy
from numpy.f2py.f2py2e import filter_files
from tqdm import tqdm

class RingFilter:
    """
    A class to apply ring artifact removal using TomoPy's remove_ring method.
    This filter should be applied to projections before reconstruction.
    """

    def __init__(self, center_x=None, center_y=None, thresh=300, thresh_max=1000,
                 thresh_min=-100, theta_min=30, rwidth=5, int_mode='REFLECT',
                 ncore=50, nchunk=None):
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

    def filter_slice(self, slice):
        return self.filter_volume(slice[np.newaxis, :, :])[0]

    def filter_volume(self, slices):
        """
        Apply ring artifact removal to projection data.

        Parameters
        ----------
        slices : ndarray
            3D array of reconstructed data

        Returns
        -------
        ndarray
            Filtered projection data with ring artifacts removed
        """
        # Convert data type to float32 as required by TomoPy
        slices = slices.astype(np.float32)
        print('Applying volume ring filter...')

        return tomopy.misc.corr.remove_ring(
            slices,
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