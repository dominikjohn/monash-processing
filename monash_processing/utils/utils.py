import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import shift as scipy_shift

class Utils:

    @staticmethod
    def select_areas(image):
        im = cv2.imread(image)

        roi1 = cv2.selectROI('Select left area', im, fromCenter=False)
        x, y, w, h = roi1
        area_left = np.s_[y:y + h, x:x + w]

        roi2 = cv2.selectROI('Select right area', im, fromCenter=False)
        x, y, w, h = roi2
        area_right = np.s_[y:y + h, x:x + w]

        cv2.destroyAllWindows()

        return area_left, area_right

    @staticmethod
    def check_existing_files(dir, num_angles, min_size_kb=5, subfolder=None, channel='phi', format='tif'):
        """
        Check which projection files need to be processed.

        Args:
            num_angles: Total number of projections
            min_size_kb: Minimum file size in KB to consider valid

        Returns:
            list: Indices of projections that need processing
        """
        to_process = []
        if subfolder is not None:
            results_dir = dir / subfolder / channel
        else:
            results_dir = dir / channel

        print("Checking existing files...")
        print('using directory ', results_dir)
        for angle_i in tqdm(range(num_angles), desc="Checking files"):
            file_path = results_dir / f'projection_{angle_i:04d}.{format}'

            # Check if file exists and is larger than min_size_kb
            needs_processing = (
                    not file_path.exists() or
                    file_path.stat().st_size < min_size_kb * 1024
            )

            if needs_processing:
                to_process.append(angle_i)

        print(f"\nFound {len(to_process)} projections that need processing:")
        if len(to_process) > 0:
            print(f"First few indices: {to_process[:5]}")
            if len(to_process) > 5:
                print(f"Last few indices: {to_process[-5:]}")

        return to_process

    @staticmethod
    def apply_centershift(projections, center_shift, cuda=True, batch_size=10, order=2):
        """
        Apply center shift to projections in batches to avoid GPU memory exhaustion.
        :param projections: 2D or 3D numpy array
        :param center_shift: float, shift in pixels
        :param cuda: bool, whether to use GPU
        :param batch_size: int, number of projections to process at once
        :return: shifted projections array
        """

        if center_shift == 0:
            print("Center shift is 0, skipping shift")
            return projections

        print(f"Applying center shift of {center_shift} pixels to projection data of shape {projections.shape}")

        # Get number of dimensions and create appropriate shift vector
        ndim = projections.ndim
        shift_vector = (0, center_shift) if ndim == 2 else (0, 0, center_shift)

        # If 2D or small enough, process normally
        if ndim == 2 or (not cuda):
            return scipy_shift(projections, shift_vector, mode='nearest', order=0)

        try:
            import cupy as cp
            from cupyx.scipy import ndimage

            # Process in batches
            result = np.zeros_like(projections)
            for i in tqdm(range(0, len(projections), batch_size), desc="Applying center shift"):
                batch = projections[i:i + batch_size]
                batch_gpu = cp.asarray(batch)

                # Process batch
                shifted = ndimage.shift(batch_gpu,
                                        shift=(0, 0, center_shift),
                                        mode='constant',
                                        order=order)

                # Store result and free GPU memory
                result[i:i + batch_size] = cp.asnumpy(shifted)
                del batch_gpu
                del shifted
                cp.get_default_memory_pool().free_all_blocks()
            return result

        except Exception as e:
            print(f"GPU shift failed: {str(e)}, falling back to CPU")
            return scipy_shift(projections, shift_vector, mode='nearest', order=0)