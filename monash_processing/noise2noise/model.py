import torch
import torch.nn as nn
from matplotlib.scale import scale_factory
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

class ImagePair(Dataset):
    def __init__(self, image_pairs1, image_pairs2, offset=1313):
        self.pairs1 = image_pairs1  # List of images from angle 1
        self.pairs2 = image_pairs2  # List of images from angle 2
        self.offset = offset

    def __getitem__(self, idx):
        img1 = torch.from_numpy(self.pairs1[idx])[None]
        img2 = torch.from_numpy(self.pairs2[idx])[None]
        img2 = TF.hflip(img2)

        w = img1.shape[2]
        overlap = w - self.offset
        start = (w - overlap) // 2
        end = start + overlap

        mask = torch.zeros_like(img1)
        mask[0, :, start:end] = 1

        return img1, img2, mask

class UNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels if i == 0 else 48, 48, 3, padding=1),
                          nn.LeakyReLU(0.1),
                          nn.MaxPool2d(2)) for i in range(6)
        ])

        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(96 if i == 0 else 144, 96, 3, padding=1),
                          nn.LeakyReLU(0.1),
                          nn.Upsample(scale_factor=2)) for i in range(5)
        ])

        self.final = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        skips = []
        for layer in self.down:
            skips.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([skips[-i - 1], x], 1))

        return self.final(x)