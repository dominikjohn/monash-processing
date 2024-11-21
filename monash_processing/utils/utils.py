import cv2
import numpy as np
import tqdm

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
    def check_existing_files(dir, num_angles, min_size_kb=5, channel='phi'):
        """
        Check which projection files need to be processed.

        Args:
            num_angles: Total number of projections
            min_size_kb: Minimum file size in KB to consider valid

        Returns:
            list: Indices of projections that need processing
        """
        to_process = []
        results_dir = dir / channel

        print("Checking existing files...")
        for angle_i in tqdm(range(num_angles), desc="Checking files"):
            file_path = results_dir / f'projection_{angle_i:04d}.tiff'

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