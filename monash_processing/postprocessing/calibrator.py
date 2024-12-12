import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tifffile
from tqdm import tqdm

materials = {
    'Ethanol': {
        'density': 0.789,
        'molecular_weight': 46.068,
        'electrons': 26,
        'composition': {'C': 2, 'H': 6, 'O': 1}
    },
    'PVC': {
        'density': 1.4,
        'molecular_weight': 62.5,
        'electrons': 32,
        'composition': {'C': 2, 'H': 3, 'Cl': 1}
    },
    'PTFE': {
        'density': 2.2,
        'molecular_weight': 100.02,
        'electrons': 48,
        'composition': {'C': 2, 'F': 4}
    },
    'POM': {
        'density': 1.41,
        'molecular_weight': 30.026,
        'electrons': 16,
        'composition': {'C': 1, 'H': 2, 'O': 1}
    },
    'PMMA': {
        'density': 1.18,
        'molecular_weight': 100.12,
        'electrons': 54,
        'composition': {'C': 5, 'H': 8, 'O': 2}
    },
}


class Calibrator:
    def __init__(self, data_loader, reference_material='PMMA'):
        """
        Initialize the calibrator with a data loader and reference material.

        Args:
            data_loader: DataLoaderDesy instance
            reference_material (str): Name of reference material (must be in materials dict)
        """
        self.data_loader = data_loader
        self.reference_material = reference_material

        if reference_material not in materials:
            raise ValueError(f"Reference material {reference_material} not found in materials dictionary")

        self.reference_density = self._calculate_electron_density(materials[reference_material])
        self.phase_dir = data_loader.get_save_path() / 'recon_phase'
        self.output_dir = data_loader.get_save_path() / 'recon_phase_calib'
        self.output_dir.mkdir(exist_ok=True)

        # Will store ROI selections
        self.material_roi = None
        self.air_roi = None
        self.material_value = None
        self.air_value = None

    def _calculate_electron_density(self, material_props):
        """Calculate electron density for a material in electrons/nm³."""
        avogadro_number = 6.022e23
        moles_per_cm3 = material_props['density'] / material_props['molecular_weight']
        molecules_per_cm3 = moles_per_cm3 * avogadro_number
        electrons_per_cm3 = molecules_per_cm3 * material_props['electrons']
        return electrons_per_cm3 / (10 ** 21)  # Convert to electrons/nm³

    def _load_phase_stack(self):
        """Load phase reconstruction stack."""
        tif_files = sorted(self.phase_dir.glob('*.tif*'))
        if not tif_files:
            raise FileNotFoundError(f"No TIFF files found in {self.phase_dir}")

        print(f"Loading {len(tif_files)} TIFF files...")
        first_img = tifffile.imread(tif_files[0])
        stack = np.zeros((len(tif_files), *first_img.shape), dtype=first_img.dtype)

        for i, tif_path in tqdm(enumerate(tif_files)):
            stack[i] = tifffile.imread(tif_path)

        return stack

    def _make_roi_selector(self, slice_data, title):
        """Create an interactive ROI selector window."""
        fig, ax = plt.subplots()
        ax.imshow(slice_data, cmap='gray')
        ax.set_title(title)

        roi = []

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            roi.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))
            plt.close()

        rs = RectangleSelector(ax, onselect, useblit=True, interactive=True)
        plt.show()

        return roi[0] if roi else None

    def _calculate_roi_mean(self, stack, slice_idx, roi, n_slices):
        """Calculate mean value in ROI over specified number of slices."""
        xmin, xmax, ymin, ymax = roi
        z_start = slice_idx
        z_end = min(slice_idx + n_slices, stack.shape[0])
        roi_mean = np.mean(stack[z_start:z_end, ymin:ymax, xmin:xmax])
        roi_std = np.std(stack[z_start:z_end, ymin:ymax, xmin:xmax])
        return roi_mean, roi_std

    def calibrate(self, slice_idx, n_slices=30):
        """
        Perform calibration by selecting ROIs and calculating scaling factors.

        Args:
            slice_idx (int): Index of slice to show for ROI selection
            n_slices (int): Number of slices to average above the selected slice
        """
        # Load phase stack
        phase_stack = self._load_phase_stack()

        # Select ROI for reference material
        print(f"\nSelect ROI for {self.reference_material}")
        self.material_roi = self._make_roi_selector(
            phase_stack[slice_idx],
            f"Select {self.reference_material} region"
        )

        if not self.material_roi:
            raise ValueError("No material ROI selected")

        # Select ROI for air
        print("\nSelect ROI for air")
        self.air_roi = self._make_roi_selector(
            phase_stack[slice_idx],
            "Select air region"
        )

        if not self.air_roi:
            raise ValueError("No air ROI selected")

        # Calculate mean values
        self.material_value, material_std = self._calculate_roi_mean(
            phase_stack, slice_idx, self.material_roi, n_slices
        )
        self.air_value, air_std = self._calculate_roi_mean(
            phase_stack, slice_idx, self.air_roi, n_slices
        )

        print(f"\nMeasured values:")
        print(f"{self.reference_material}: {self.material_value:.6f} ± {material_std:.6f}")
        print(f"Air: {self.air_value:.6f} ± {air_std:.6f}")

        # Calculate scaling factors
        scale = self.reference_density / (self.material_value - self.air_value)
        offset = -self.air_value * scale

        print(f"\nCalibration factors:")
        print(f"Scale: {scale:.6f}")
        print(f"Offset: {offset:.6f}")

        # Apply calibration to full stack
        print("\nApplying calibration to full stack...")
        calibrated_stack = phase_stack * scale + offset

        # Save calibrated stack
        print("\nSaving calibrated stack...")
        for i in tqdm(range(len(phase_stack))):
            tifffile.imwrite(
                self.output_dir / f'phase_{i:04d}.tif',
                calibrated_stack[i].astype(np.float32)
            )

        print(f"\nCalibrated stack saved to {self.output_dir}")
        return calibrated_stack