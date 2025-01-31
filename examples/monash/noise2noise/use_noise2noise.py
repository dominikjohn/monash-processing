import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import numpy as np

data_loader = DataLoader(Path("/data/mct/22203/"), "K3_3H")
angles = np.mean(loader.load_angles(), axis=0)
angle_step = np.diff(angles).mean()
print('Angle step:', angle_step)
index_0 = np.argmin(np.abs(angles - 0))
index_180 = np.argmin(np.abs(angles - 180))

imgs = data_loader.load_projections()

images_step0 = imgs[0, ...]

imgs1 = images_step0[:index_180,...]
imgs2 = images_step0[index_180:,...]

img_pair = ImagePair(imgs1, imgs2)

# Training
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8)
dataset = ImagePair(imgs1, imgs2)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(100):
    for img1, img2, mask in loader:
        optimizer.zero_grad()
        loss = torch.mean(((model(img1) - img2) ** 2) * mask)
        loss.backward()
        optimizer.step()