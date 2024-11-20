import numpy as np
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from pathlib import Path
from monash_processing.algorithms.phase_integration import PhaseIntegrator


class ParallelPhaseIntegrator:
    def __init__(self, energy, prop_distance, pixel_size, area_left, area_right, data_loader):
        self.integrator = PhaseIntegrator(
            energy, prop_distance, pixel_size,
            area_left, area_right, data_loader
        )
        self.data_loader = data_loader

    def check_existing_files(self, num_angles, min_size_kb=5):
        """
        Check which projection files need to be processed.

        Args:
            num_angles: Total number of projections
            min_size_kb: Minimum file size in KB to consider valid

        Returns:
            list: Indices of projections that need processing
        """
        to_process = []
        results_dir = self.data_loader.results_dir / 'phi'

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

    def integrate_parallel(self, num_angles, n_workers=None, min_size_kb=5):
        """
        Parallelize phase integration across angles using Dask.
        Only process files that are missing or too small.

        Args:
            num_angles: Total number of projections
            n_workers: Number of Dask workers to use
            min_size_kb: Minimum file size in KB to consider valid
        """
        # Check which files need processing
        to_process = self.check_existing_files(num_angles, min_size_kb)

        if not to_process:
            print("All files already processed successfully!")
            return

        print(f"\nStarting parallel processing for {len(to_process)} projections...")

        # Create a local cluster with custom settings
        cluster = LocalCluster(
            n_workers=n_workers,
            dashboard_address=None,
            processes=True
        )
        client = Client(cluster)

        try:
            print(f"Dask cluster initialized with {len(client.scheduler_info()['workers'])} workers")

            # Convert indices to process to a Dask bag
            angle_bag = db.from_sequence(to_process)

            # Map the integration function over the bag
            results = angle_bag.map(self.integrator.integrate_single)

            # Compute with progress bar
            print("Processing projections...")
            for _ in tqdm(results.compute(), total=len(to_process)):
                pass

            # Verify all files were processed correctly
            remaining = self.check_existing_files(num_angles, min_size_kb)
            if remaining:
                print(f"\nWarning: {len(remaining)} files still need processing after completion.")
                print("These might have failed during processing:")
                print(f"Indices: {remaining}")
            else:
                print("\nAll files processed successfully!")

        finally:
            client.close()
            cluster.close()