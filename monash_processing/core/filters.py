import numpy as np
from tqdm import tqdm
from monash_processing.core.data_loader import DataLoader
from scipy.ndimage import rotate

class ProcessingFilter:

    @staticmethod
    def beltran_two_material_filter(I, I_ref, A, mu_enc, mu_2, delta_enc, delta_2, p_size, distance):
        '''
        Beltran et al. 2D and 3D X-ray phase retrieval of multi-material objects using a single defocus distance (2010)
        :param I: intensity projection with sample
        :param I_ref: reference projection
        :param A: total thickness map
        :param mu_enc: mu of enclosing material (e.g. soft tissue)
        :param mu_2: mu of enclosed material (e.g. contrast agent)
        :param delta_enc: delta of enclosing material (e.g. soft tissue)
        :param delta_2: delta of enclosed material (e.g. contrast agent)
        :param p_size: pixel size
        :param distance: propagation distance
        :return: Thickness map
        '''

        # Get image dimensions
        ny, nx = I.shape

        # Calculate frequencies using fftfreq
        delta_x = p_size / (2 * np.pi)
        kx = np.fft.fftfreq(nx, d=delta_x)
        ky = np.fft.fftfreq(ny, d=delta_x)

        # Create 2D frequency grid
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_squared = kx_grid ** 2 + ky_grid ** 2

        image_fft = np.fft.fft2(I/(I_ref * np.exp(-mu_enc*A)))

        denom = (distance * (delta_2 - delta_enc) / (mu_2 - mu_enc)) * k_squared + 1
        filter = 1 / denom

        filtered_fft = image_fft * filter
        log_image = np.log(np.real(np.fft.ifft2(filtered_fft)))
        thickness = -log_image/(mu_2 - mu_enc)

        return thickness

    @staticmethod
    def croton_two_material_filter(I, I_ref, mu_enc, mu_2, delta_enc, delta_2, p_size, distance):
        '''
        Beltran et al. 2D and 3D X-ray phase retrieval of multi-material objects using a single defocus distance (2010)
        :param I: intensity projection with sample
        :param I_ref: reference projection
        :param mu_enc: mu of enclosing material (e.g. soft tissue)
        :param mu_2: mu of enclosed material (e.g. contrast agent)
        :param delta_enc: delta of enclosing material (e.g. soft tissue)
        :param delta_2: delta of enclosed material (e.g. contrast agent)
        :param p_size: pixel size
        :param distance: propagation distance
        :return: Thickness map
        '''

        # Get image dimensions
        ny, nx = I.shape

        # Calculate frequencies using fftfreq
        delta_x = p_size / (2 * np.pi)
        kx = np.fft.fftfreq(nx, d=delta_x)
        ky = np.fft.fftfreq(ny, d=delta_x)

        # Create 2D frequency grid
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_squared = kx_grid ** 2 + ky_grid ** 2

        image_fft = np.fft.fft2(I / I_ref)

        denom = (distance * (delta_2 - delta_enc) / (mu_2 - mu_enc)) * k_squared + 1
        filter = 1 / denom

        filtered_fft = image_fft * filter
        log_image = np.log(np.real(np.fft.ifft2(filtered_fft)))
        thickness = -log_image / (mu_2 - mu_enc)

        return thickness

    @staticmethod
    def croton_two_material_filter_umpa(T, mu_enc, mu_2, delta_enc, delta_2, p_size, distance):
        '''
        Beltran et al. 2D and 3D X-ray phase retrieval of multi-material objects using a single defocus distance (2010)
        :param I: intensity projection with sample
        :param I_ref: reference projection
        :param mu_enc: mu of enclosing material (e.g. soft tissue)
        :param mu_2: mu of enclosed material (e.g. contrast agent)
        :param delta_enc: delta of enclosing material (e.g. soft tissue)
        :param delta_2: delta of enclosed material (e.g. contrast agent)
        :param p_size: pixel size
        :param distance: propagation distance
        :return: Thickness map
        '''

        # Get image dimensions
        ny, nx = T.shape

        # Calculate frequencies using fftfreq
        delta_x = p_size / (2 * np.pi)
        kx = np.fft.fftfreq(nx, d=delta_x)
        ky = np.fft.fftfreq(ny, d=delta_x)

        # Create 2D frequency grid
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_squared = kx_grid ** 2 + ky_grid ** 2

        image_fft = np.fft.fft2(T)

        print((delta_2-delta_enc)/(mu_2-mu_enc))
        denom = (distance * (delta_2 - delta_enc) / (mu_2 - mu_enc)) * k_squared + 1
        filter = 1 / denom

        filtered_fft = image_fft * filter
        log_image = np.log(np.real(np.fft.ifft2(filtered_fft)))
        thickness = -log_image / (mu_2 - mu_enc)

        return thickness

    @staticmethod
    def beltran_two_material_filter_modified(I, I_ref, mu_enc, mu_2, delta_enc, delta_2, p_size, distance):
        '''
        Beltran et al. 2D and 3D X-ray phase retrieval of multi-material objects using a single defocus distance (2010)
        :param I: intensity projection with sample
        :param I_ref: reference projection
        :param A: total thickness map
        :param mu_enc: mu of enclosing material (e.g. soft tissue)
        :param mu_2: mu of enclosed material (e.g. contrast agent)
        :param delta_enc: delta of enclosing material (e.g. soft tissue)
        :param delta_2: delta of enclosed material (e.g. contrast agent)
        :param p_size: pixel size
        :param distance: propagation distance
        :return: Thickness map
        '''

        # Get image dimensions
        ny, nx = I.shape

        # Calculate frequencies using fftfreq
        delta_x = p_size / (2 * np.pi)
        kx = np.fft.fftfreq(nx, d=delta_x)
        ky = np.fft.fftfreq(ny, d=delta_x)

        # Create 2D frequency grid
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_squared = kx_grid ** 2 + ky_grid ** 2

        image_fft = np.fft.fft2(I / I_ref)

        denom = (distance * (delta_2 - delta_enc) / (mu_2 - mu_enc)) * k_squared + 1
        filter = 1 / denom

        filtered_fft = image_fft * filter
        log_image = -np.log(np.real(np.fft.ifft2(filtered_fft)))

        return log_image

    @staticmethod
    def get_thickness_map(volume, angle, psize, threshold=200):
        if angle != 0:
            volume = rotate(volume, angle, axes=(1, 2), order=0)
        return (volume > threshold).astype(int).sum(axis=1) * psize

    @staticmethod
    def process_thickness_projections(loader: DataLoader, volume, angles, psize, threshold, chunk_size=50):
        import cupy as cp
        from cupyx.scipy.ndimage import rotate

        # Process one angle at a time to minimize memory usage
        for i, angle in tqdm(enumerate(angles)):
            # Process in chunks
            chunks = [volume[i:i + chunk_size] for i in range(0, volume.shape[0], chunk_size)]
            angle_result = []

            for chunk in chunks:
                chunk_gpu = cp.asarray(chunk)
                if angle != 0:
                    rotated = rotate(chunk_gpu, angle, axes=(1, 2), order=0, reshape=False)
                    thickness = (rotated > threshold).sum(axis=1)
                else:
                    thickness = (chunk_gpu > threshold).sum(axis=1)

                # Move result back to CPU and clear GPU memory
                angle_result.append(cp.asnumpy(thickness))
                # Explicitly clear GPU memory
                del chunk_gpu
                if angle != 0:
                    del rotated
                del thickness
                cp.get_default_memory_pool().free_all_blocks()

            # Combine chunks and save
            combined_thickness = np.concatenate(angle_result) * psize
            loader.save_tiff('thickness', i, combined_thickness)
            # Clear chunk results
            del angle_result

    @staticmethod
    def process_thickness_projections(loader: DataLoader, volume, angles, psize, threshold):
        for i, angle in tqdm(enumerate(angles)):
            T = ProcessingFilter.get_thickness_map_gpu(volume, angle, psize, threshold)
            loader.save_tiff('thickness', i, T)