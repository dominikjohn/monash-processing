from monash_processing.core.data_loader import DataLoader
import numpy as np
import scipy.constants
from scipy.signal import savgol_filter
import pyamg
from scipy.sparse import diags

class EdgeCorrector:

    def __init__(self,
                 energy,
                 prop_distance,
                 pixel_size,
                 source_size_x,
                 source_size_y,
                 source_sample_dist,
                 sample_grid_dist,
                 grid_detector_dist,
                 data_loader: DataLoader):
        self.data_loader = data_loader
        self.energy = energy
        self.prop_distance = prop_distance
        self.pixel_size = pixel_size
        self.source_size_x = source_size_x
        self.source_size_y = source_size_y
        self.source_sample_dist = source_sample_dist
        self.sample_grid_dist = sample_grid_dist
        self.grid_detector_dist = grid_detector_dist
        self.wavevec = 2 * np.pi * self.energy / (
                scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)

        self.flatfields = self.data_loader.load_flat_fields()
        self.dark_current = self.data_loader.load_dark_currents()

    # Savitzy-Golay filter for smoother derivative
    def _savgol_filter(self, data, axis):
        return savgol_filter(data,
                             window_length=51,
                             polyorder=3,
                             deriv=1,
                             axis=axis,
                             delta=1)

    def calculate_correction(self, projection_i):
        # Load dx, dy
        dx = self.data_loader.load_processed_projection(projection_i, 'dx')
        dy = self.data_loader.load_processed_projection(projection_i, 'dy')

        umpa_slicing = np.s_[3:-3, 3:-3]  # set to actual size

        # Average mask positions to get an average detector intensity
        average_image = np.average(self.data_loader.load_projections(projection_i), axis=0)

        # Apply flat field correction
        average_image = (average_image - self.dark_current) / (self.flatfields - self.dark_current)
        f = average_image[umpa_slicing]

        # Calculate the correction
        # TODO check this is actually the correct axis
        d_dx = self._savgol_filter(dx, axis=0)
        d_dy = self._savgol_filter(dy, axis=1)

        blurring_prefactor = (self.sample_grid_dist + 2 * self.grid_detector_dist) / (2 * self.source_sample_dist)
        blurring_kernel_x = self.source_size_x * blurring_prefactor
        blurring_kernel_y = self.source_size_y * blurring_prefactor

        a = 0.5 * blurring_kernel_x^2 * self.prop_distance^2
        a_pr = 0.5 * blurring_kernel_y^2 * self.prop_distance^2

        d = self.prop_distance * d_dx / self.wavevec
        e = self.prop_distance * d_dy / self.wavevec

        g = 1 - self.prop_distance * (d_dx^2 + d_dy^2) / self.wavevec

        self._solve_2d_equation(f, d.shape[0], d.shape[1], self.pixel_size, a, a_pr, d, e, g)

    def _solve_2d_equation(self, f, nx, ny, h, a, a_pr, d, e, g, tol=1e-6, verbose=True):
        # Construct the matrix
        n = nx * ny

        g_flat = g.flatten()
        d_flat = d.flatten()
        e_flat = e.flatten()

        # Main diagonal (co efficient of u_i,j)
        main_diag = h ** 2 * g_flat - 2 * a - 2 * a_pr

        #TODO recheck coefficients

        # x-direction neighbors
        off_diag_x_m1 = h * d_flat / 2 + a  # coefficient of u_i-1,j
        off_diag_x_p1 = -h * d_flat / 2 + a  # coefficient of u_i+1,j

        # y-direction neighbors
        off_diag_y_m1 = h * e_flat / 2 + a_pr  # coefficient of u_i,j-1
        off_diag_y_p1 = -h * e_flat / 2 + a_pr  # coefficient of u_i,j+1

        # Build sparse matrix
        diagonals = [main_diag, off_diag_x_m1, off_diag_x_p1, off_diag_y_m1, off_diag_y_p1]
        offsets = [0, -1, 1, -ny, ny]
        A = diags(diagonals, offsets, shape=(n, n), format='csr')

        # Prepare right-hand side
        b = (h ** 2 * f).flatten()

        # More control over the solver
        ml = pyamg.ruge_stuben_solver(A,
                                      max_coarse=10,  # minimum size of coarsest grid
                                      coarse_solver='splu',  # use sparse LU on coarsest grid
                                      max_levels=10  # maximum number of levels
                                      )

        if verbose:
            residuals = []

            def callback(x):
                residuals.append(np.linalg.norm(b - A @ x))

            x = ml.solve(b, tol=tol, maxiter=500, callback=callback)

            print(f"Solver converged in {len(residuals)} iterations")
            print(f"Final residual: {residuals[-1]:.2e}")
        else:
            x = ml.solve(b, tol=tol, maxiter=500)

        return x.reshape(nx, ny)