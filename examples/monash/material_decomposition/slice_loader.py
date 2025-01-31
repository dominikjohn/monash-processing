import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm

# Directory containing the processed images
base = '/data/mct/22203/results/P6_Manual'
dir_att = Path(f'{base}/recon_att_paganin_pvc-ethanol')
#dir_phase = Path('/data/mct/22203/results/K3_3H_Manual/recon_phase')

# Get sorted list of all tif files
tif_files_att = sorted(list(dir_att.glob('*.tif*')))
#tif_files_phase = sorted(list(dir_phase.glob('*.tif*')))
num_files_att = len(tif_files_att)
#num_files_phase = len(tif_files_phase)

# Load first image to get dimensions
first_img = tifffile.imread(tif_files_att[0])
img_shape = first_img.shape

# Create empty array to hold all images
volume_att = np.empty((num_files_att, *img_shape), dtype=first_img.dtype)
#volume_phase = np.empty((num_files_att, *img_shape), dtype=first_img.dtype)
#volume_att = np.empty((num_files_phase, *img_shape), dtype=first_img.dtype)
tif_files_phase = tif_files_att
for i, (phase_path, att_path) in tqdm(enumerate(zip(tif_files_phase, tif_files_att))):
    #volume_phase[i] = tifffile.imread(phase_path)
    volume_att[i] = tifffile.imread(att_path)
