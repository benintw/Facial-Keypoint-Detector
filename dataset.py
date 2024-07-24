
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from config import CONFIG
from data_augs import train_transforms, val_transforms


class FacialKeypointDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.category_names = [
            'left_eye_center_x',
            'left_eye_center_y',
            'right_eye_center_x',
            'right_eye_center_y',
            'left_eye_inner_corner_x',
            'left_eye_inner_corner_y',
            'left_eye_outer_corner_x',
            'left_eye_outer_corner_y',
            'right_eye_inner_corner_x',
            'right_eye_inner_corner_y',
            'right_eye_outer_corner_x',
            'right_eye_outer_corner_y',
            'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x',
            'left_eyebrow_outer_end_y',
            'right_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y',
            'right_eyebrow_outer_end_x',
            'right_eyebrow_outer_end_y',
            'nose_tip_x',
            'nose_tip_y',
            'mouth_left_corner_x',
            'mouth_left_corner_y',
            'mouth_right_corner_x',
            'mouth_right_corner_y',
            'mouth_center_top_lip_x',
            'mouth_center_top_lip_y',
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y'
            ]
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = self.df.iloc[idx, -1]
        image = image.split()
        image = np.array(image).astype(np.float32)  # shape: (9216,)

        if self.train:
            labels = np.array(self.df.iloc[idx, :-1].tolist())
            labels[np.isnan(labels)] = -1  # shape: (30,)
        else:
            # keypoints_dim = len(self.df.columns[:-1].tolist())
            labels = np.zeros(30)  # shape: (30,)

        ignore_indices = labels == -1
        labels = labels.reshape(15, 2)  # xy

        if self.transform:
            image = np.repeat(image.reshape(96, 96, -1), 3, 2).astype(
                np.uint8
            )  # shape: (96, 96, 3)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]  # shape: (3, 96, 96)
            labels = augmentations["keypoints"]  # shape: (30,)

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return image, labels.astype(np.float32)
