from monash_processing.core.data_loader import DataLoader
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt
from monash_processing.postprocessing.devolving_processor import DevolvingProcessor
from monash_processing.utils.ImageViewer import ImageViewer as imshow

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_3H_Manual"
pixel_size = 1.444e-6 # m
wavelength = 5e-5 # µm
prop_distance = 0.155 # m

loader = DataLoader(scan_path, scan_name)
angles = np.mean(loader.load_angles(), axis=0)
gamma = 1000
i = 20
tikhonov = 1000
cutoff = 5
devolver = DevolvingProcessor(gamma, 5e-5, prop_distance*1e6, pixel_size*1e6, loader, '/data/mct/22203/Flatfields_340AM_Wed13Nov.h5')

pure_flat = devolver.load_pure_flat_field()
dark_current = loader.load_flat_fields(dark=True)
Ir = (loader.load_flat_fields()) / (pure_flat - dark_current)
D_plus, D_minus, D_atten = devolver.process_single_projection(i, pure_flat, dark_current, Ir, tikhonov=tikhonov, cutoff=cutoff)