import numpy as np
from pathlib import Path
from skimage.io import imread
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from tqdm import tqdm
import xraylib
from monash_processing.postprocessing.binner import Binner


class CalibrationAnalysis:
    def __init__(self, materials=None, energy_keV=25):
        self._materials = materials if materials is not None else {}
        self.energy_keV = energy_keV
        self.last_results = None
        self.saved_rois = {}  # Dictionary to store ROIs by material index
        self.phase_stack = None
        self.att_stack = None

    @property
    def materials(self):
        """Get the current materials dictionary."""
        return self._materials

    @materials.setter
    def materials(self, new_materials):
        """Set new materials dictionary."""
        if not isinstance(new_materials, dict):
            raise TypeError("Materials must be a dictionary")
        self._materials = new_materials

    def update_material(self, name, properties):
        """
        Update or add a single material's properties.

        Args:
            name (str): Name of the material
            properties (dict): Material properties including density, molecular_weight,
                             electrons, and composition
        """
        required_keys = {'density', 'molecular_weight', 'electrons', 'composition'}
        if not all(key in properties for key in required_keys):
            raise ValueError(f"Material properties must include all of: {required_keys}")
        self._materials[name] = properties

    def remove_material(self, name):
        """Remove a material from the library."""
        if name in self._materials:
            del self._materials[name]
        else:
            raise KeyError(f"Material '{name}' not found in the library")

    @staticmethod
    def calculate_electron_density(density, molecular_weight, electrons_per_molecule):
        avogadro_number = 6.022e23
        moles_per_cm3 = density / molecular_weight
        molecules_per_cm3 = moles_per_cm3 * avogadro_number
        electrons_per_cm3 = molecules_per_cm3 * electrons_per_molecule
        electron_density_per_nm3 = electrons_per_cm3 / (10 ** 21)
        return electron_density_per_nm3

    @staticmethod
    def calculate_attenuation(composition, density, energy):
        """Calculate total attenuation coefficient for a material"""
        molecular_mass = sum(count * xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(element))
                             for element, count in composition.items())

        total_mass_atten = 0
        for element, count in composition.items():
            atomic_number = xraylib.SymbolToAtomicNumber(element)
            mass_atten = xraylib.CS_Total(atomic_number, energy)
            weight_fraction = (count * xraylib.AtomicWeight(atomic_number)) / molecular_mass
            total_mass_atten += mass_atten * weight_fraction

        return total_mass_atten * density

    def load_reconstruction_stacks(self, base_path, max_slices=None, bin_factor=4):
        base_path = Path(base_path)
        phase_dir = base_path / 'recon_phase'
        att_dir = base_path / 'recon_att'

        def process_directory(directory: Path) -> np.ndarray:
            binned_dir = directory / f'binned{bin_factor}'
            if binned_dir.exists():
                print(f"Loading existing {bin_factor}x binned data from {binned_dir}")
                source_dir = binned_dir
            else:
                print(f"No existing {bin_factor}x binned data found. Creating new binned data...")
                binner = Binner(directory)
                source_dir = binner.process_stack(bin_factor)

            tif_files = sorted(source_dir.glob('*.tif*'))
            first_img = imread(tif_files[0])
            stack = np.zeros((len(tif_files), *first_img.shape), dtype=first_img.dtype)
            for i, tif_path in tqdm(enumerate(tif_files)):
                if max_slices and i >= max_slices:
                    break
                stack[i] = imread(tif_path)
            return stack

        print("Processing phase reconstruction stack...")
        self.phase_stack = process_directory(phase_dir)
        print("\nProcessing attenuation reconstruction stack...")
        self.att_stack = process_directory(att_dir)

        return self.phase_stack, self.att_stack

    def get_roi_mean(self, stack, slice_idx, roi_coords, n_slices=100, mat_idx=None):
        xmin, xmax, ymin, ymax = roi_coords
        z_start = slice_idx
        z_end = min(slice_idx + n_slices, stack.shape[0])
        roi_mean = np.mean(stack[z_start:z_end, ymin:ymax, xmin:xmax])
        roi_std = np.std(stack[z_start:z_end, ymin:ymax, xmin:xmax])
        print(f"Material at slice {slice_idx}, ROI mean over slices {z_start}-{z_end}: {roi_mean:.6f} ± {roi_std:.6f}")

        self.last_results[mat_idx - 1] = [roi_mean, roi_std]
        self.saved_rois[mat_idx] = {
            'coords': roi_coords,
            'slice_idx': slice_idx
        }

    def make_onselect(self, stack, slice_idx, n_slices, mat_idx):
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            roi_coords = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
            self.get_roi_mean(stack, slice_idx, roi_coords, n_slices, mat_idx)

        return onselect

    def analyze_materials(self, material_slices, n_slices=30, use_att=False, phase_correction_factor=1.0,
                          font_params=None, overwrite=False):
        """
        Analyze materials with saved ROI support.

        Args:
            material_slices: List of slice indices to analyze
            n_slices: Number of slices to average over
            use_att: Whether to use attenuation data
            phase_correction_factor: Correction factor for phase data
            font_params (dict): Dictionary of font parameters
            overwrite (bool): If True, allow selection of new ROIs even if saved ones exist
        """
        if self.phase_stack is None or self.att_stack is None:
            raise ValueError("Please load reconstruction stacks first using load_reconstruction_stacks()")

        if font_params is None:
            font_params = {
                'title_size': 12,
                'label_size': 10,
                'tick_size': 10
            }

        self.last_results = [None] * len(material_slices)
        selection_stack = self.att_stack if use_att else self.phase_stack
        selection_type = "attenuation" if use_att else "phase contrast"

        for mat_idx, slice_idx in enumerate(material_slices, 1):
            if slice_idx >= selection_stack.shape[0]:
                print(f"Warning: Slice {slice_idx} is out of bounds (max: {selection_stack.shape[0] - 1})")
                continue

            # Check if we have saved ROIs and shouldn't overwrite
            if mat_idx in self.saved_rois and not overwrite:
                saved_roi = self.saved_rois[mat_idx]
                print(f"Using saved ROI for material {mat_idx}")
                self.get_roi_mean(selection_stack, saved_roi['slice_idx'],
                                  saved_roi['coords'], n_slices, mat_idx)

                # Optionally display the saved ROI
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(selection_stack[saved_roi['slice_idx']], cmap='grey')
                xmin, xmax, ymin, ymax = saved_roi['coords']
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.set_title(f'Material {mat_idx} (Slice {slice_idx}) - Saved ROI',
                             fontsize=font_params['title_size'])
                plt.show()
            else:
                # Allow new ROI selection
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(selection_stack[slice_idx], cmap='grey')
                ax.set_title(f'Material {mat_idx} (Slice {slice_idx}) - {selection_type}',
                             fontsize=font_params['title_size'])
                ax.tick_params(axis='both', labelsize=font_params['tick_size'])

                rs = RectangleSelector(ax, self.make_onselect(selection_stack, slice_idx, n_slices, mat_idx),
                                       useblit=True, interactive=True)
                plt.show()

        phase_results = []
        att_results = []

        # Process results for both phase and attenuation using saved ROIs
        if use_att:
            att_results = self.last_results.copy()
            for i, slice_idx in enumerate(material_slices):
                if i + 1 in self.saved_rois:
                    roi = self.saved_rois[i + 1]['coords']
                    xmin, xmax, ymin, ymax = roi
                    z_start = slice_idx
                    z_end = min(slice_idx + n_slices, self.phase_stack.shape[0])
                    roi_mean = np.mean(self.phase_stack[z_start:z_end, ymin:ymax, xmin:xmax])
                    roi_std = np.std(self.phase_stack[z_start:z_end, ymin:ymax, xmin:xmax])
                    phase_results.append([roi_mean * phase_correction_factor,
                                          roi_std * phase_correction_factor])
        else:
            phase_results = [[r[0] * phase_correction_factor, r[1] * phase_correction_factor]
                             for r in self.last_results.copy()]
            for i, slice_idx in enumerate(material_slices):
                if i + 1 in self.saved_rois:
                    roi = self.saved_rois[i + 1]['coords']
                    xmin, xmax, ymin, ymax = roi
                    z_start = slice_idx
                    z_end = min(slice_idx + n_slices, self.att_stack.shape[0])
                    roi_mean = np.mean(self.att_stack[z_start:z_end, ymin:ymax, xmin:xmax])
                    roi_std = np.std(self.att_stack[z_start:z_end, ymin:ymax, xmin:xmax])
                    att_results.append([roi_mean, roi_std])

        return phase_results, att_results

    def clear_saved_rois(self):
        """Clear all saved ROIs."""
        self.saved_rois = {}
        print("All saved ROIs have been cleared.")

    def calculate_theoretical_values(self):
        """Calculate theoretical electron densities and attenuations for all materials."""
        electron_densities = {}
        attenuations = {}

        for material, props in self.materials.items():
            electron_densities[material] = self.calculate_electron_density(
                props['density'],
                props['molecular_weight'],
                props['electrons']
            )
            attenuations[material] = self.calculate_attenuation(
                props['composition'],
                props['density'],
                self.energy_keV
            )

        return electron_densities, attenuations

    def plot_phase_vs_attenuation(self, phase_results, att_results, figsize=(10, 8), font_params=None):
        """
        Plot phase vs attenuation with customizable font parameters.

        Args:
            phase_results: List of phase measurement results
            att_results: List of attenuation measurement results
            figsize: Tuple of figure dimensions (width, height)
            font_params (dict): Dictionary of font parameters including:
                - 'title_size': Size of plot title
                - 'label_size': Size of axis labels
                - 'tick_size': Size of tick labels
                - 'legend_size': Size of legend text
                Default values will be used if not specified
        """
        # Set default font parameters if none provided
        if font_params is None:
            font_params = {
                'title_size': 12,
                'label_size': 10,
                'tick_size': 10,
                'legend_size': 10
            }

        # Create figure with specified size
        plt.figure(figsize=figsize)

        # Extract means and standard deviations for measured values
        phase_means = [res[0] for res in phase_results if res is not None]
        phase_stds = [res[1] for res in phase_results if res is not None]
        att_means = [res[0] for res in att_results if res is not None]
        att_stds = [res[1] for res in att_results if res is not None]

        # Calculate theoretical values
        electron_densities, attenuations = self.calculate_theoretical_values()

        # Define plot styles
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        markers = ['o', 's', '^', 'D', 'v', 'p']

        material_names = list(self.materials.keys())
        for i, (phase_mean, phase_std, att_mean, att_std) in enumerate(
                zip(phase_means, phase_stds, att_means, att_stds)):
            plt.errorbar(phase_mean, att_mean,
                         xerr=phase_std, yerr=att_std,
                         color=colors[i % len(colors)],
                         marker=markers[i % len(markers)],
                         markersize=8,
                         capsize=5,
                         capthick=1,
                         label=f'{material_names[i]} (measured)')

        # Plot theoretical values
        for i, (material, electron_density) in enumerate(electron_densities.items()):
            plt.scatter(electron_density, attenuations[material],
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        s=100,
                        label=f'{material} (theoretical)',
                        facecolors='none',
                        edgecolors=colors[i % len(colors)],
                        linewidth=2)

        plt.xlabel('Electron density [1/nm³]', fontsize=font_params['label_size'])
        plt.ylabel(f'Attenuation coefficient $\mu$ [1/cm]', fontsize=font_params['label_size'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=font_params['legend_size'])
        plt.tick_params(axis='both', labelsize=font_params['tick_size'])

        # Print the values
        self._print_results(phase_results, att_results, electron_densities, attenuations)

        plt.tight_layout()
        plt.show()

    def _print_results(self, phase_results, att_results, electron_densities, attenuations):
        print("\nMeasured values:")
        print("-" * 60)
        print(f"{'Material':<10} {'Phase Signal':<25} {'Attenuation Signal':<25}")
        print("-" * 60)
        for i, (phase, att) in enumerate(zip(phase_results, att_results)):
            if phase is not None and att is not None:
                print(f"Material {i + 1:<3} {phase[0]:>8.6f} ± {phase[1]:<10.6f} {att[0]:>8.6f} ± {att[1]:<10.6f}")

        print("\nTheoretical values:")
        print("-" * 50)
        print(f"{'Material':<10} {'Electron Density':<20} {'Attenuation':<20}")
        print(f"{'       ':<10} {'(electrons/nm³)':<20} {'(cm⁻¹)':<20}")
        print("-" * 50)
        for material in self.materials:
            print(f"{material:<10} {electron_densities[material]:,.2f}{' ' * 12} {attenuations[material]:,.4f}")