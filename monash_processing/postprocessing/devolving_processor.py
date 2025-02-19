# ---------------------------------------------------------------------------------
# Written by Samantha Alloo // Date 23.10.2024
# This script contains code that will calculate the multimodal images of a sample
# using the various perspectives of the speckle-based X-ray imaging Fokker--Planck
# equation: 1) evolving and 2) devolving. Within, there is a single- and multiple-
# exposure algorithm for each perspective.
# ---------------------------------------------------------------------------------
# Importing required modules
import numpy as np
import os
import math
import scipy
from scipy import ndimage
from monash_processing.core.data_loader import DataLoader
from tqdm import tqdm
import h5py
from monash_processing.utils.utils import Utils
from pathlib import Path
import dask.bag as db

class DevolvingProcessor:

    def __init__(self, gamma, wavelength, prop, pixel_size, data_loader: DataLoader, flat_path):
        self.data_loader = data_loader
        self.save_dir = data_loader.get_save_path()
        self.gamma = gamma
        self.wavelength = wavelength
        self.prop = prop
        self.pixel_size = pixel_size
        self.flat_path = flat_path

    def load_pure_flat_field(self):
        """
        Loads and averages flat field images from H5 file.
        Returns the averaged flat field image.
        """
        # Read the H5 file
        with h5py.File(self.flat_path, 'r') as f:
            # Navigate to the dataset
            dataset = f['EXPERIMENT/SCANS/00_00/SAMPLE/DATA']

            # Load all images and compute the mean
            # Using [:] loads all data into memory
            flat_field = np.mean(dataset[:], axis=0)

            return flat_field

    def process_single_projection(self, i: int, pure_flat: np.ndarray, dark_current: np.ndarray, Ir: np.ndarray):
        """Process a single projection with the given parameters."""
        Is = (self.data_loader.load_projections(projection_i=i) - dark_current) / (pure_flat - dark_current)
        DF_atten, positive_D, negative_D, _ = self.Multiple_Devolving(Is, Ir, self.gamma, self.wavelength, self.prop,
                                                                      self.pixel_size)

        # Save results
        self.data_loader.save_tiff('DF_atten', i, DF_atten)
        self.data_loader.save_tiff('positive_D', i, positive_D)
        self.data_loader.save_tiff('negative_D', i, negative_D)

        return i  # Return the processed index for tracking

    def process_projections(self, num_projections: int, min_size_kb: int = 5, num_workers: int = 4):
        """
        Process projections using Dask bag map for parallel computation.
        Only processes files that haven't been processed yet or are incomplete.

        Args:
            num_projections: Total number of projections to process
            min_size_kb: Minimum file size in KB to consider a file as properly processed
            num_workers: Number of worker processes to use
        """
        # Check which files need processing
        to_process = Utils.check_existing_files(
            dir=Path(self.save_dir),
            num_angles=num_projections,
            min_size_kb=min_size_kb,
            channel='DF_atten'  # Using DF_atten as the reference channel
        )

        if not to_process:
            print("All projections have been processed already!")
            return

        print(f"Processing {len(to_process)} projections...")

        # Load common data needed for all projections
        pure_flat = self.load_pure_flat_field()
        dark_current = self.data_loader.load_flat_fields(dark=True)
        Ir = (self.data_loader.load_flat_fields()) / (pure_flat - dark_current)

        # Create a Dask bag from the indices to process
        bag = db.from_sequence(to_process, npartitions=num_workers)

        # Define a wrapper function that includes the constant parameters
        def process_wrapper(idx):
            return self.process_single_projection(
                i=idx,
                pure_flat=pure_flat,
                dark_current=dark_current,
                Ir=Ir
            )

        # Process using map
        print(f"Starting parallel processing with {num_workers} workers...")

        # Create progress bar
        pbar = tqdm(total=len(to_process), desc="Processing projections")

        # Process and update progress bar
        results = []
        for result in bag.map(process_wrapper).compute():
            results.append(result)
            pbar.update(1)

        pbar.close()
        print("Processing complete!")

        return results

    # ---------------------------------------------------------------------------------
    # Defining additional functions
    @staticmethod
    def kspace_kykx(image_shape: tuple, pixel_size: float = 1):
        # Multiply by 2pi for correct values, since DFT has 2pi in exponent
        rows = image_shape[0]
        columns = image_shape[1]
        ky = 2 * math.pi * scipy.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
        kx = 2 * math.pi * scipy.fft.fftfreq(columns,
                                             d=pixel_size)  # spatial frequencies relating to "columns" in real space
        return ky, kx

    @staticmethod
    def invLaplacian(image, pixel_size):
        # Need to mirror the image to enforce periodicity
        flip = np.concatenate((image, np.flipud(image)), axis=0)
        flip = np.concatenate((flip, np.fliplr(flip)), axis=1)

        ky, kx = DevolvingProcessor.kspace_kykx(flip.shape, pixel_size)
        ky2 = ky ** 2
        kx2 = kx ** 2

        kr2 = np.add.outer(ky2, kx2)
        regkr2 = 0.0001
        ftimage = np.fft.fft2(flip)
        regdiv = 1 / (kr2 + regkr2)
        invlapimageflip = -1 * np.fft.ifft2(regdiv * ftimage)

        row = int(image.shape[0])
        column = int(image.shape[1])

        invlap = np.real(invlapimageflip[0:row, 0:column])
        return invlap, regkr2

    @staticmethod
    def lowpass_2D(image, r, pixel_size):
        # -------------------------------------------------------------------
        # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, beyond some defined
        # spatial frequency r
        # DEFINITIONS
        # image: input image whos spatial frequencies you want to suppress
        # r: spatial frequency you want to suppress beyond [pixel number]
        # pixel_size: physical size of pixel [microns]
        # -------------------------------------------------------------------
        rows = image.shape[0]
        columns = image.shape[1]
        m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
        n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
        ky = (2 * math.pi * m)  # defined by row direction
        kx = (2 * math.pi * n)  # defined by column direction

        kx2 = kx ** 2
        ky2 = ky ** 2
        kr2 = np.add.outer(ky2, kx2)
        kr = np.sqrt(kr2)

        lowpass_2d = np.exp(-r * (kr ** 2))

        return lowpass_2d

    @staticmethod
    def highpass_2D(image, r, pixel_size):
        # -------------------------------------------------------------------
        # This function will generate a high-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
        # spatial frequency r
        # DEFINITIONS
        # image: input image whos spatial frequencies you want to suppress
        # r: spatial frequency you want to suppress beyond [pixel number]
        # pixel_size: physical size of pixel [microns]
        # -------------------------------------------------------------------
        rows = image.shape[0]
        columns = image.shape[1]
        m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
        n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
        ky = (2 * math.pi * m)  # defined by row direction
        kx = (2 * math.pi * n)  # defined by column direction

        kx2 = kx ** 2
        ky2 = ky ** 2
        kr2 = np.add.outer(ky2, kx2)
        kr = np.sqrt(kr2)

        highpass_2d = 1 - np.exp(-r * (kr ** 2))

        # plt.imshow(highpass_2d)
        # plt.title('High-Pass Filter 2D')
        # plt.colorbar()
        # plt.show()

        return highpass_2d

    @staticmethod
    def midpass_2D(image, r, pixel_size):
        # -------------------------------------------------------------------
        # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
        # spatial frequency r
        # DEFINITIONS
        # image: input image whos spatial frequencies you want to suppress
        # r: spatial frequency you want to suppress beyond [pixel number]
        # pixel_size: physical size of pixel [microns]
        # -------------------------------------------------------------------
        rows = image.shape[0]
        columns = image.shape[1]
        m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
        n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
        ky = (2 * math.pi * m)  # defined by row direction
        kx = (2 * math.pi * n)  # defined by column direction

        kx2 = kx ** 2
        ky2 = ky ** 2
        kr2 = np.add.outer(ky2, kx2)
        kr = np.sqrt(kr2)

        highpass_2d = 1 - np.exp(-r * (kr ** 2))

        C = np.zeros(columns, dtype=np.complex128)
        C = C + 0 + 1j
        ikx = kx * C  # (i) * spatial frequencies in x direction (along columns) - as complex numbers ( has "0" in the real components, and "kx" in the complex)
        denom = np.add.outer((-1 * ky), ikx)  # array with ikx - ky (DENOMINATOR)

        midpass_2d = np.divide(complex(1., 0.) * highpass_2d, denom, out=np.zeros_like(complex(1., 0.) * highpass_2d),
                               where=denom != 0)  # Setting output equal to zero where denominator equals zero

        # plt.imshow(np.real(midpass_2d))
        # plt.title('Mid-Pass Filter 2D')
        # plt.colorbar()
        # plt.show()

        return midpass_2d

    # 5) Multiple-exposure devolving speckle-based X-ray imaging Fokker--Planck perspective
    def Multiple_Devolving(self, Is, Ir, gamma, wavelength, prop, pixel_size):
        # ----------------------------------------------------------------
        # Is: list of sample-reference speckle fields (X-ray beam + mask + sample + detector) [array]
        # Ir: list of reference speckle fields (X-ray beam + mask + detector) [array]
        # gamma: ratio of real to imaginary refractive index coefficients of the sample
        # wavelength: wavelength of X-ray beam [microns]
        # prop: propagation distance between the sample and detector [microns]
        # pixel_size: pixel size of the detector [microns]
        # ----------------------------------------------------------------

        num_masks, rows, columns = Is.shape

        coeff_D = []  # Stores the Laplacian of the sample-reference speckle field
        coeff_dx = []  # Stores x-gradient of the sample-reference speckle field
        coeff_dy = []  # Stores y-gradient of the sample-reference speckle field
        lapacaian = []  # Stores the Laplacian term of the system of equations
        RHS = []  # Stores the right-hand side (RHS) of the system of equations

        # Arrays to store terms for QR decomposition
        coefficient_A = np.empty([int(num_masks), 4, int(rows), int(columns)])
        coefficient_b = np.empty([int(num_masks), 1, int(rows), int(columns)])

        # Loop over each mask to calculate and store the coefficients
        for i in range(num_masks):
            rhs = (1 / prop) * (Is[i, :, :] - Ir[i, :, :])  # Compute RHS for the system of equations
            lap = Is[i, :, :]  # Compute the Laplacian term (placeholder)
            deff = np.divide(ndimage.laplace(Is[i, :, :]),
                             pixel_size ** 2)  # Compute the Laplacian of the speckle field
            dy, dx = np.gradient(Is[i, :, :], pixel_size)  # Compute the x and y gradients of the speckle field
            dy_r = 2 * dy  # Adjust y-gradient term
            dx_r = 2 * dx  # Adjust x-gradient term

            # Append computed terms to the respective lists
            coeff_D.append(deff)
            coeff_dx.append(dx_r)
            coeff_dy.append(dy_r)
            lapacaian.append(lap)
            RHS.append(rhs)

        # Assemble the system of linear equations: Ax = b
        for n in range(len(coeff_dx)):
            coefficient_A[n, :, :, :] = np.array(
                [lapacaian[n], coeff_D[n], coeff_dx[n], coeff_dy[n]])  # Coefficient matrix
            coefficient_b[n, :, :, :] = RHS[n]  # RHS vector

        identity = np.identity(4)  # 4x4 identity matrix for Tikhonov Regularization
        alpha = np.std(coefficient_A) / 10000  # Optimal Tikhonov regularization parameter (tweak if system is unstable)
        reg = np.multiply(alpha, identity)  # Regularization matrix
        reg_repeat = np.repeat(reg, rows * columns).reshape(4, 4, rows,
                                                            columns)  # Repeat regularization for all pixel positions
        zero_repeat = np.zeros((4, 1, rows, columns))  # Zero matrix for regularizing RHS vector

        # Apply Tikhonov regularization to the system
        coefficient_A_reg = np.vstack([coefficient_A, reg_repeat])
        coefficient_b_reg = np.vstack([coefficient_b, zero_repeat])

        # Perform QR decomposition
        reg_Qr, reg_Rr = np.linalg.qr(coefficient_A_reg.transpose([2, 3, 0, 1]))

        # Solve the system using QR decomposition (solve Rx = Q^T b instead of inversion)
        reg_x = np.linalg.solve(reg_Rr, np.matmul(np.matrix.transpose(reg_Qr.transpose([2, 3, 1, 0])),
                                                  coefficient_b_reg.transpose([2, 3, 0, 1])))

        # Extract solution components
        lap_phiDF = reg_x[:, :, 0, 0]  # Laplacian term (Laplacian(1/wavenumber*Phi - D))
        DFqr = reg_x[:, :, 1, 0] / prop  # Dark-field (DF) term
        dxDF = reg_x[:, :, 2, 0] / prop  # Derivative of DF along x
        dyDF = reg_x[:, :, 3, 0] / prop  # Derivative of DF along y

        # Apply filtering to determine the true dark-field signal
        cutoff = 10  # Cutoff parameter for filtering (optimize this for best SNR or NIQE)

        # Compute Fourier transform of gradient terms
        i_dyDF = dyDF * (np.zeros((DFqr.shape), dtype=np.complex128) + 0 + 1j)
        insideft = dxDF + i_dyDF
        insideftm = np.concatenate((insideft, np.flipud(insideft)),
                                   axis=0)  # Mirror to enforce periodic boundary conditions
        ft_dx_idy = np.fft.fft2(insideftm)

        # Apply mid-pass and low-pass filters
        MP = self.midpass_2D(ft_dx_idy, cutoff, pixel_size)
        MP_deriv = MP * ft_dx_idy  # Mid-pass filtered derivative solution
        DFqrm = np.concatenate((DFqr, np.flipud(DFqr)), axis=0)
        ft_DFqr = np.fft.fft2(DFqrm)
        LP = self.lowpass_2D(ft_DFqr, cutoff, pixel_size)
        LP_DFqr = LP * ft_DFqr  # Low-pass filtered DF solution

        # Combine filtered solutions to obtain true dark-field signal -- aggregated dark-field
        combined = LP_DFqr + MP_deriv
        DF_filtered = np.fft.ifft2(combined)
        DF_filtered = np.real(DF_filtered[0:int(rows), :])  # Invert Fourier transform and keep real part

        # Compute phase shifts and attenuation term
        ref = Ir[0, :, :]
        sam = Is[0, :, :]
        lapphi = (ref - sam + prop ** 2 * np.divide(ndimage.laplace(DF_filtered * ref), pixel_size ** 2)) * (
                (2 * math.pi) / (wavelength * prop * ref))

        # Invert Laplacian to retrieve phase
        phi, reg = self.invLaplacian(lapphi, pixel_size)

        Iob = np.exp(2 * phi / gamma)  # Object's attenuation term

        # Corrected the aggregated dark-field image for X-ray attenuation in the sample
        DF_atten = np.real(DF_filtered / Iob)

        # Separate positive and negative values of dark-field signal and save
        positive_D = np.clip(DF_atten, 0, np.inf)
        negative_D = np.clip(DF_atten, -np.inf, 0)

        # Final output
        print('Multiple-exposure devolving SBXI Fokker-Planck inverse problem has been solved!')
        return DF_atten, positive_D, negative_D, Iob
