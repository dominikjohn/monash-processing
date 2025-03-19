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
tikhonov = 10
cutoff = 0.1
flat_path = '/data/mct/22203/Flats_Day2LateArvo.h5'
#flat_path = '/data/mct/22203/Flatfields_340AM_Wed13Nov.h5'
devolver = DevolvingProcessor(gamma, 5e-5, prop_distance*1e6, pixel_size*1e6, loader, flat_path)

pure_flat = devolver.load_pure_flat_field()
dark_current = loader.load_flat_fields(dark=True)
Ir = (loader.load_flat_fields()) / (pure_flat - dark_current)
D_plus, D_minus, D_atten = devolver.process_single_projection(i, pure_flat, dark_current, Ir, tikhonov=tikhonov, cutoff=cutoff, return_images=True)

scan_name = "K3_1N"
loader = DataLoader(scan_path, scan_name)
angles = np.mean(loader.load_angles(), axis=0)
gamma = 1000
i = 20
tikhonov = 10
cutoff = 0.1
flat_path = '/data/mct/22203/Flats_Day2Noon.h5'
devolver = DevolvingProcessor(gamma, 5e-5, prop_distance*1e6, pixel_size*1e6, loader, flat_path)

pure_flat = devolver.load_pure_flat_field()
dark_current = loader.load_flat_fields(dark=True)
Ir = (loader.load_flat_fields()) / (pure_flat - dark_current)
D_plus_n, D_minus_n, D_atten_n = devolver.process_single_projection(i, pure_flat, dark_current, Ir, tikhonov=tikhonov, cutoff=cutoff, return_images=True)



vis_red = np.exp(-4*np.pi**2*D_plus*(prop_distance*1e6)**2/((pixel_size*1e6)**2))
vis_red_n = np.exp(-4*np.pi**2*D_plus_n*(prop_distance*1e6)**2/((pixel_size*1e6)**2))