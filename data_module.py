import random
from copy import deepcopy
from typing import Optional

import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from parsing_dataset import ParsingDataset
from reid_dataset import MSMT17, DukeMTMC, Market1501, ReidDataset


class DataModule(pl.LightningDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        train_transform = A.Compose(
            [
                A.Resize(256, 128),
                A.HorizontalFlip(p=0.5),
                A.RandomSizedCrop((128, 256), 256, 128, w2h_ratio=0.5),
                A.ColorJitter(
                    p=0.1, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                ),
                A.OneOf([A.Blur(p=0.5), A.Sharpen(p=0.5)], p=0.1),
                A.GaussNoise(p=0.1),
                A.Perspective(p=0.1),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
        val_transform = A.Compose(
            [
                A.Resize(256, 128),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
        test_transform = T.Compose([T.Resize((256, 128)), T.ToTensor()])

        dataset = ParsingDataset(
            "datasets/pedestrian_parsing_dataset", transform=train_transform
        )
        # split the dataset in train and val set
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)
        split_point = int(len(indices) * 0.8)
        self.train_dataset = Subset(deepcopy(dataset), indices[:split_point])
        self.val_dataset = Subset(deepcopy(dataset), indices[split_point:])
        self.val_dataset.dataset.transform = val_transform

        test_data = random.sample(MSMT17("datasets/msmt17").query, 64)
        self.test_dataset = ReidDataset(test_data, transform=test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
