"""
dataset.py
Dataset pipeline with random cropping and geometric augmentations
"""

from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LowLightDenoiseDataset(Dataset):

    def __init__(
        self,
        gt_dir: Path,
        noisy_dir: Path,
        patch_size: int = 256,
        is_train: bool = True,
    ):
        self.gt_dir = Path(gt_dir)
        self.noisy_dir = Path(noisy_dir)
        self.patch_size = patch_size
        self.is_train = is_train

        self.ids = sorted([p.stem for p in self.gt_dir.glob("*.png")])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        gt_path = self.gt_dir / f"{img_id}.png"
        noisy_path = self.noisy_dir / f"{img_id}_noise.png"

        gt_img = (
            cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255.0
        )
        noisy_img = (
            cv2.cvtColor(cv2.imread(str(noisy_path)), cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255.0
        )

        h, w, _ = gt_img.shape

        if self.is_train:
            # Random Crop
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
            gt_patch = gt_img[y : y + self.patch_size, x : x + self.patch_size]
            noisy_patch = noisy_img[y : y + self.patch_size, x : x + self.patch_size]

            # Random Horizontal / Vertical Flips
            if random.random() > 0.5:
                gt_patch = np.fliplr(gt_patch)
                noisy_patch = np.fliplr(noisy_patch)
            if random.random() > 0.5:
                gt_patch = np.flipud(gt_patch)
                noisy_patch = np.flipud(noisy_patch)
        else:
            # Center crop for validation
            cy, cx = (h - self.patch_size) // 2, (w - self.patch_size) // 2
            gt_patch = gt_img[cy : cy + self.patch_size, cx : cx + self.patch_size]
            noisy_patch = noisy_img[
                cy : cy + self.patch_size, cx : cx + self.patch_size
            ]

        # Convert HWC to CHW PyTorch tensors
        gt_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(gt_patch, (2, 0, 1)))
        )
        noisy_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(noisy_patch, (2, 0, 1)))
        )

        return noisy_tensor, gt_tensor
