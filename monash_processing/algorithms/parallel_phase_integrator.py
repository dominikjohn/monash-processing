import numpy as np
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from monash_processing.core.phase_integrator import PhaseIntegrator

class ParallelPhaseIntegrator:
    def __init__(self, energy, prop_distance, pixel_size, area_left, area_right, data_loader):
        self.integrator = PhaseIntegrator(
            energy, prop_distance, pixel_size,
            area_left, area_right, data_loader
        )

    def integrate_parallel(self, num_angles, n_workers=None):
        """
        Parallelize phase integration across angles using Dask
        """
        # Create a local cluster with custom settings
        cluster = LocalCluster(
            n_workers=n_workers,
            dashboard_address=None,  # Disable dashboard
            processes=True
        )
        client = Client(cluster)

        try:
            print(f"Dask cluster initialized with {len(client.scheduler_info()['workers'])} workers")

            # Convert range to a Dask bag
            angle_bag = db.from_sequence(range(num_angles))

            # Map the integration function over the bag
            results = angle_bag.map(self.integrator.integrate_single)

            # Compute with progress bar
            print("Processing projections in parallel...")
            for _ in tqdm(results.compute(), total=num_angles):
                pass

        finally:
            client.close()
            cluster.close()