import numpy as np
import dask
from dask.distributed import Client
from tqdm.dask import tqdm
from monash_processing.core.phase_integrator import PhaseIntegrator

class ParallelPhaseIntegrator:
    def __init__(self, energy, prop_distance, pixel_size, area_left, area_right, data_loader):
        self.integrator = PhaseIntegrator(
            energy, prop_distance, pixel_size,
            area_left, area_right, data_loader
        )

    def integrate_parallel(self, num_angles):
        """
        Parallelize phase integration across angles using Dask
        """
        # Create a Dask client
        client = Client()

        try:
            # Create delayed objects for each angle
            delayed_tasks = [
                dask.delayed(self.integrator.integrate_single)(angle_i)
                for angle_i in range(num_angles)
            ]

            # Compute all tasks in parallel with progress bar
            results = dask.compute(
                delayed_tasks,
                scheduler='distributed',
                optimize_graph=True
            )

            return results

        finally:
            # Clean up client
            client.close()