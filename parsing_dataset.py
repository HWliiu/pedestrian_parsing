import copy
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ParsingDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        data_path = os.path.join(root, "data")

        for root, _, files in os.walk(data_path):
            for file in files:
                data = os.path.join(root, file)
                if os.path.splitext(data)[1] == ".jpg":
                    mask = data.replace(".jpg", "_m.png")
                    self.data.append((data, mask))

    @staticmethod
    def remap_label(mask):
        new_mask = copy.deepcopy(mask)
        for i, lbl in enumerate([0, 9, 19, 29, 39, 50, 60, 62]):
            new_mask[mask == lbl] = i
        return new_mask

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        mask = self.remap_label(mask)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)

        return transformed["image"], transformed["mask"]

    def __len__(self):
        return len(self.data)
