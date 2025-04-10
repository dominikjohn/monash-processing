import dask.bag as db
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from monash_processing.algorithms.phase_integration import PhaseIntegrator
from monash_processing.utils.utils import Utils

class ParallelPhaseIntegrator:
    def __init__(self, energy, prop_distance, pixel_size, area_left, area_right, data_loader, window_size=1, stitched=False, suffix=None, simple_format=True, slicing=None):
        self.integrator = PhaseIntegrator(
            energy, prop_distance, pixel_size,
            area_left, area_right, data_loader, window_size=window_size, stitched=stitched, suffix=suffix, simple_format=simple_format,
            slicing=slicing
        )
        self.data_loader = data_loader
        self.stitched = stitched
        self.suffix = suffix

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
        if self.stitched:
            if self.suffix is not None:
                channel_name = f'phi_stitched_{self.suffix}'
            else:
                channel_name = f'phi_stitched'
        else:
            if self.suffix is not None:
                channel_name = f'phi_{self.suffix}'
            else:
                channel_name = f'phi'

        to_process = Utils.check_existing_files(self.data_loader.results_dir, num_angles, min_size_kb=min_size_kb, subfolder=f'umpa_window{self.integrator.window_size}', channel=channel_name)

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
            remaining = Utils.check_existing_files(self.data_loader.results_dir, num_angles, min_size_kb=min_size_kb,
                                                    subfolder=f'umpa_window{self.integrator.window_size}',
                                                    channel=channel_name)
            if remaining:
                print(f"\nWarning: {len(remaining)} files still need processing after completion.")
                print("These might have failed during processing:")
                print(f"Indices: {remaining}")
            else:
                print("\nAll files processed successfully!")

        finally:
            client.close()
            cluster.close()